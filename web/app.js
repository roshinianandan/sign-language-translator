/* Sign Language Translator – browser version.
 * MediaPipe Hands (landmarks) + TensorFlow.js (your trained LSTM) + Web Speech API.
 * Feature extraction below MUST match utils.py exactly. */

const CFG = {
  SEQ: 30,               // frames per window (config.SEQUENCE_LENGTH)
  FEATS_PER_HAND: 65,    // 63 normalised landmarks + wrist x,y
  CONF: 0.85,            // config.CONFIDENCE_THRESHOLD
  STABLE: 5,             // consecutive agreeing predictions
  MIN_HAND: 0.4,         // fraction of window that must contain a hand
  COOLDOWN: 20,          // frames before another word is accepted
  PREDICT_EVERY: 2,      // run the model every N frames (saves battery)
  IDLE: 'neutral',
  DISPLAY: { hello:'Hello', thanks:'Thank you', yes:'Yes', no:'No',
             please:'Please', help:'Help', iloveyou:'I love you' },
};

const $ = (id) => document.getElementById(id);
const video = $('cam'), canvas = $('view'), ctx = canvas.getContext('2d');
const statusEl = $('status'), captionEl = $('caption'), currentEl = $('current');

let model = null, labels = [], hands = null, running = false;
const windowBuf = [], handFlags = [], recent = [];
let sentence = [], cooldown = 0, frameNo = 0, probs = [];
const src = document.createElement('canvas');      // mirrored frame fed to MediaPipe
const sctx = src.getContext('2d');

/* ---------- feature extraction (port of utils.py) ---------- */
function normaliseHand(lm) {
  const out = new Float32Array(CFG.FEATS_PER_HAND);
  if (!lm) return out;
  const w = lm[0];
  const rel = [];
  let maxAbs = 0;
  for (const p of lm) {
    const r = [p.x - w.x, p.y - w.y, p.z - w.z];
    rel.push(r);
    for (const v of r) maxAbs = Math.max(maxAbs, Math.abs(v));
  }
  const scale = maxAbs + 1e-6;
  let k = 0;
  for (const r of rel) for (const v of r) out[k++] = v / scale;   // row-major, like numpy flatten
  out[k++] = w.x; out[k++] = w.y;
  return out;
}

function extractKeypoints(res) {
  let left = null, right = null;
  if (res.multiHandLandmarks && res.multiHandedness) {
    res.multiHandLandmarks.forEach((lm, i) => {
      if (res.multiHandedness[i].label === 'Left') left = lm; else right = lm;
    });
  }
  const f = new Float32Array(CFG.FEATS_PER_HAND * 2);
  f.set(normaliseHand(left), 0);
  f.set(normaliseHand(right), CFG.FEATS_PER_HAND);
  return f;
}

/* ---------- speech ---------- */
function speak(text) {
  if (!text || !('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  speechSynthesis.speak(u);
}

/* ---------- UI ---------- */
function buildBars() {
  $('bars').innerHTML = labels.map((l) =>
    `<div class="row" data-l="${l}"><span>${l}</span><div class="track"><div class="fill"></div></div><span class="p">0</span></div>`
  ).join('');
}
function renderBars(current) {
  document.querySelectorAll('.row').forEach((row, i) => {
    const p = probs[i] || 0;
    row.querySelector('.fill').style.width = (p * 100).toFixed(0) + '%';
    row.querySelector('.p').textContent = p.toFixed(2);
    row.classList.toggle('hit', row.dataset.l === current);
  });
}
function renderCaption() {
  captionEl.textContent = sentence.length ? sentence.slice(-8).join(' ') : 'Tap Start and begin signing';
}

/* ---------- inference ---------- */
function predict() {
  const flat = new Float32Array(CFG.SEQ * CFG.FEATS_PER_HAND * 2);
  windowBuf.forEach((f, i) => flat.set(f, i * f.length));
  return tf.tidy(() => model.predict(tf.tensor(flat, [1, CFG.SEQ, CFG.FEATS_PER_HAND * 2])).dataSync());
}

function onResults(res) {
  ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);
  if (res.multiHandLandmarks) {
    for (const lm of res.multiHandLandmarks) {
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: '#22c55e', lineWidth: 3 });
      drawLandmarks(ctx, lm, { color: '#f43f5e', lineWidth: 1, radius: 3 });
    }
  }
  windowBuf.push(extractKeypoints(res));
  handFlags.push(!!(res.multiHandLandmarks && res.multiHandLandmarks.length));
  if (windowBuf.length > CFG.SEQ) { windowBuf.shift(); handFlags.shift(); }
  cooldown = Math.max(0, cooldown - 1);
  frameNo++;

  if (windowBuf.length < CFG.SEQ || frameNo % CFG.PREDICT_EVERY) return;

  let current = CFG.IDLE;
  const handRatio = handFlags.filter(Boolean).length / CFG.SEQ;
  if (handRatio >= CFG.MIN_HAND) {
    probs = Array.from(predict());
    let idx = 0;
    probs.forEach((p, i) => { if (p > probs[idx]) idx = i; });
    if (probs[idx] >= CFG.CONF) current = labels[idx];
  } else {
    probs = labels.map(() => 0);
  }
  recent.push(current);
  if (recent.length > CFG.STABLE) recent.shift();

  if (recent.length === CFG.STABLE && cooldown === 0 && current !== CFG.IDLE &&
      recent.every((r) => r === current)) {
    const text = CFG.DISPLAY[current] || current;
    if (sentence[sentence.length - 1] !== text) {
      sentence.push(text);
      if ($('auto').checked) speak(text);
      renderCaption();
    }
    cooldown = CFG.COOLDOWN;
  }
  currentEl.textContent = current === CFG.IDLE ? '–' : (CFG.DISPLAY[current] || current);
  renderBars(current);
}

/* ---------- camera loop ---------- */
async function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    sctx.save();
    sctx.scale(-1, 1);                                   // mirror, same as cv2.flip in Python
    sctx.drawImage(video, -src.width, 0, src.width, src.height);
    sctx.restore();
    await hands.send({ image: src });
  }
  requestAnimationFrame(loop);
}

async function startCamera() {
  $('start').disabled = true;
  statusEl.textContent = 'Starting camera…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    statusEl.textContent = 'Camera blocked or unavailable. Allow access and use HTTPS.';
    $('start').disabled = false;
    return;
  }
  src.width = canvas.width = video.videoWidth;
  src.height = canvas.height = video.videoHeight;
  speechSynthesis && speechSynthesis.speak(new SpeechSynthesisUtterance(''));   // unlock audio on iOS
  running = true;
  $('start').textContent = '⏸ Stop camera';
  $('start').disabled = false;
  statusEl.textContent = 'Translating… sign in front of the camera';
  loop();
}

function stopCamera() {
  running = false;
  (video.srcObject?.getTracks() || []).forEach((t) => t.stop());
  $('start').textContent = '▶ Start camera';
  statusEl.textContent = 'Stopped';
}

/* ---------- boot ---------- */
async function init() {
  try {
    labels = await (await fetch('model/labels.json')).json();
    model = await tf.loadLayersModel('model/model.json');
    tf.tidy(() => model.predict(tf.zeros([1, CFG.SEQ, CFG.FEATS_PER_HAND * 2])).dataSync()); // warm-up
  } catch (e) {
    statusEl.textContent = 'Model not found. Run convert_to_tfjs.py first (see README).';
    console.error(e);
    return;
  }
  probs = labels.map(() => 0);
  buildBars();

  hands = new Hands({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}` });
  hands.setOptions({ maxNumHands: 2, modelComplexity: 0, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5 });
  hands.onResults(onResults);

  statusEl.textContent = 'Ready';
  $('start').disabled = false;
}

$('start').onclick = () => (running ? stopCamera() : startCamera());
$('clear').onclick = () => { sentence = []; renderCaption(); };
$('speak').onclick = () => speak(sentence.join(' '));
init();
