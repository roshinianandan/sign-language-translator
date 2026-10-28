/* Sign Language Translator – browser version.
 * MediaPipe Hands + TensorFlow.js + Web Speech API.
 */

const CFG = {
  SEQ: 30,
  FEATS_PER_HAND: 65,
  CONF: 0.85,
  STABLE: 5,
  MIN_HAND: 0.4,
  COOLDOWN: 20,
  PREDICT_EVERY: 2,
  IDLE: "neutral",
  DISPLAY: {
    hello: "Hello",
    thanks: "Thank you",
    yes: "Yes",
    no: "No",
    please: "Please",
    help: "Help",
    iloveyou: "I love you"
  }
};

const $ = (id) => document.getElementById(id);

const video = $("cam");
const canvas = $("view");
const ctx = canvas.getContext("2d");
const statusEl = $("status");
const captionEl = $("caption");
const currentEl = $("current");

let model = null;
let labels = [];
let hands = null;
let running = false;
let mode = "model";
let lastAccepted = null;

const windowBuf = [];
const handFlags = [];
const recent = [];

let sentence = [];
let cooldown = 0;
let frameNo = 0;
let probs = [];

const src = document.createElement("canvas");
const sctx = src.getContext("2d");

/* ---------- Feature extraction ---------- */

function normaliseHand(lm) {
  const out = new Float32Array(CFG.FEATS_PER_HAND);

  if (!lm) return out;

  const w = lm[0];
  const rel = [];
  let maxAbs = 0;

  for (const p of lm) {
    const r = [
      p.x - w.x,
      p.y - w.y,
      p.z - w.z
    ];

    rel.push(r);

    for (const v of r) {
      maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }

  const scale = maxAbs + 1e-6;
  let k = 0;

  for (const r of rel) {
    for (const v of r) {
      out[k++] = v / scale;
    }
  }

  out[k++] = w.x;
  out[k++] = w.y;

  return out;
}

function extractKeypoints(res) {
  let left = null;
  let right = null;

  if (res.multiHandLandmarks && res.multiHandedness) {
    res.multiHandLandmarks.forEach((lm, i) => {
      if (res.multiHandedness[i].label === "Left") {
        left = lm;
      } else {
        right = lm;
      }
    });
  }

  const f = new Float32Array(CFG.FEATS_PER_HAND * 2);

  f.set(normaliseHand(left), 0);
  f.set(normaliseHand(right), CFG.FEATS_PER_HAND);

  return f;
}

/* ---------- Speech ---------- */

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;

  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;

  speechSynthesis.speak(u);
}

/* ---------- User interface ---------- */

function buildBars() {
  const bars = $("bars");
  if (!bars) return;

  bars.innerHTML = labels.map((l) => `
    <div class="row" data-l="${l}">
      <span>${l}</span>
      <div class="track">
        <div class="fill"></div>
      </div>
      <span class="p">0</span>
    </div>
  `).join("");
}

function renderBars(current) {
  document.querySelectorAll(".row").forEach((row, i) => {
    const p = probs[i] || 0;

    row.querySelector(".fill").style.width =
      (p * 100).toFixed(0) + "%";

    row.querySelector(".p").textContent = p.toFixed(2);

    row.classList.toggle("hit", row.dataset.l === current);
  });
}

function renderCaption() {
  if (!captionEl) return;

  captionEl.textContent = sentence.length
    ? sentence.slice(-8).join(" ")
    : "Tap Start and begin signing";
}

/* ---------- Gesture demo mode ---------- */

const RULE_STABLE = 8;

function buildGuide() {
  const g = $("guide");
  if (!g) return;

  g.hidden = false;

  g.innerHTML =
    "<h2>Show one of these gestures</h2>" +
    GESTURE_GUIDE.map((x) => `
      <div class="g" data-k="${x.key}">
        <b>${CFG.DISPLAY[x.key] || x.key}</b>
        <span>${x.how}</span>
      </div>
    `).join("");
}

function handleRules(res) {
  let label = null;

  if (
    res.multiHandLandmarks &&
    res.multiHandLandmarks.length
  ) {
    label = classifyGesture(
      res.multiHandLandmarks[0],
      canvas.width,
      canvas.height
    );
  }

  recent.push(label);

  if (recent.length > RULE_STABLE) {
    recent.shift();
  }

  if (recent.length === RULE_STABLE) {
    if (recent.every((r) => r === null)) {
      lastAccepted = null;
    } else if (
      label &&
      recent.every((r) => r === label) &&
      label !== lastAccepted
    ) {
      const text = CFG.DISPLAY[label] || label;

      sentence.push(text);
      lastAccepted = label;

      if ($("auto")?.checked) {
        speak(text);
      }

      renderCaption();
    }
  }

  currentEl.textContent = label
    ? CFG.DISPLAY[label] || label
    : "–";

  document.querySelectorAll(".g").forEach((el) => {
    el.classList.toggle("hit", el.dataset.k === label);
  });
}

/* ---------- Model prediction ---------- */

function predict() {
  const flat = new Float32Array(
    CFG.SEQ * CFG.FEATS_PER_HAND * 2
  );

  windowBuf.forEach((f, i) => {
    flat.set(f, i * f.length);
  });

  return tf.tidy(() =>
    model.predict(
      tf.tensor(flat, [
        1,
        CFG.SEQ,
        CFG.FEATS_PER_HAND * 2
      ])
    ).dataSync()
  );
}

/* ---------- MediaPipe results ---------- */

function onResults(res) {
  ctx.drawImage(
    res.image,
    0,
    0,
    canvas.width,
    canvas.height
  );

  if (res.multiHandLandmarks) {
    for (const lm of res.multiHandLandmarks) {
      drawConnectors(
        ctx,
        lm,
        HAND_CONNECTIONS,
        {
          color: "#22c55e",
          lineWidth: 3
        }
      );

      drawLandmarks(
        ctx,
        lm,
        {
          color: "#f43f5e",
          lineWidth: 1,
          radius: 3
        }
      );
    }
  }

  if (mode === "rules") {
    return handleRules(res);
  }

  windowBuf.push(extractKeypoints(res));

  handFlags.push(
    !!(
      res.multiHandLandmarks &&
      res.multiHandLandmarks.length
    )
  );

  if (windowBuf.length > CFG.SEQ) {
    windowBuf.shift();
    handFlags.shift();
  }

  cooldown = Math.max(0, cooldown - 1);
  frameNo++;

  if (
    windowBuf.length < CFG.SEQ ||
    frameNo % CFG.PREDICT_EVERY
  ) {
    return;
  }

  let current = CFG.IDLE;

  const handRatio =
    handFlags.filter(Boolean).length / CFG.SEQ;

  if (handRatio >= CFG.MIN_HAND) {
    probs = Array.from(predict());

    let idx = 0;

    probs.forEach((p, i) => {
      if (p > probs[idx]) {
        idx = i;
      }
    });

    if (probs[idx] >= CFG.CONF) {
      current = labels[idx];
    }
  } else {
    probs = labels.map(() => 0);
  }

  recent.push(current);

  if (recent.length > CFG.STABLE) {
    recent.shift();
  }

  if (
    recent.length === CFG.STABLE &&
    cooldown === 0 &&
    current !== CFG.IDLE &&
    recent.every((r) => r === current)
  ) {
    const text = CFG.DISPLAY[current] || current;

    if (sentence[sentence.length - 1] !== text) {
      sentence.push(text);

      if ($("auto")?.checked) {
        speak(text);
      }

      renderCaption();
    }

    cooldown = CFG.COOLDOWN;
  }

  currentEl.textContent =
    current === CFG.IDLE
      ? "–"
      : CFG.DISPLAY[current] || current;

  renderBars(current);
}

/* ---------- Camera loop ---------- */

async function loop() {
  if (!running) return;

  try {
    if (video.readyState >= 2) {
      sctx.save();
      sctx.scale(-1, 1);

      sctx.drawImage(
        video,
        -src.width,
        0,
        src.width,
        src.height
      );

      sctx.restore();

      await hands.send({ image: src });
    }
  } catch (error) {
    console.error("Camera processing error:", error);
    statusEl.textContent =
      "Camera processing failed. Please refresh the page.";
    stopCamera();
    return;
  }

  if (running) {
    requestAnimationFrame(loop);
  }
}

/* ---------- Start camera ---------- */

async function startCamera() {
  $("start").disabled = true;
  statusEl.textContent = "Starting camera…";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
  } catch (error) {
    console.error("Camera error:", error);

    statusEl.textContent =
      "Camera blocked or unavailable. Allow camera access and use HTTPS.";

    $("start").disabled = false;
    return;
  }

  src.width = canvas.width = video.videoWidth;
  src.height = canvas.height = video.videoHeight;

  if ("speechSynthesis" in window) {
    speechSynthesis.speak(
      new SpeechSynthesisUtterance("")
    );
  }

  running = true;
  $("start").textContent = "⏸ Stop camera";
  $("start").disabled = false;

  statusEl.textContent =
    mode === "rules"
      ? "Show a gesture and hold it for half a second"
      : "Translating… sign in front of the camera";

  loop();
}

function stopCamera() {
  running = false;

  const stream = video.srcObject;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  video.srcObject = null;
  $("start").textContent = "▶ Start camera";
  statusEl.textContent = "Stopped";
  $("start").disabled = false;
}

/* ---------- Initialize translator ---------- */

async function init() {
  statusEl.textContent = "Loading model…";

  try {
    const loadModel = async () => {
      const labelResponse = await fetch("model/labels.json");

      if (!labelResponse.ok) {
        throw new Error(
          `Could not load labels.json (${labelResponse.status})`
        );
      }

      labels = await labelResponse.json();

      if (typeof tf === "undefined") {
        throw new Error("TensorFlow.js did not load");
      }

      model = await tf.loadLayersModel("model/model.json");

      tf.tidy(() =>
        model.predict(
          tf.zeros([
            1,
            CFG.SEQ,
            CFG.FEATS_PER_HAND * 2
          ])
        ).dataSync()
      );

      mode = "model";
    };

    await Promise.race([
      loadModel(),
      new Promise((_, reject) => {
        setTimeout(() => {
          reject(
            new Error("Model loading timed out after 20 seconds")
          );
        }, 20000);
      })
    ]);
  } catch (error) {
    console.warn(
      "Using gesture demo mode because the trained model could not be loaded:",
      error
    );

    mode = "rules";
    model = null;
  }

  if (mode === "model") {
    probs = labels.map(() => 0);
    buildBars();
  } else {
    buildGuide();
  }

  try {
    if (typeof Hands === "undefined") {
      throw new Error(
        "MediaPipe Hands did not load. Check your internet connection."
      );
    }

    hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 0,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);
  } catch (error) {
    console.error(
      "Hand-tracking initialization failed:",
      error
    );

    statusEl.textContent =
      "Could not load hand tracking. Refresh the page and check your internet connection.";

    $("start").disabled = true;
    return;
  }

  statusEl.textContent =
    mode === "rules"
      ? "Ready - gesture demo mode (no trained model found)"
      : "Ready";

  $("start").disabled = false;
}

/* ---------- Main button events ---------- */

$("start").onclick = () => {
  if (running) {
    stopCamera();
  } else {
    startCamera();
  }
};

$("clear").onclick = () => {
  sentence = [];
  renderCaption();
};

$("speak").onclick = () => {
  speak(sentence.join(" "));
};

init();

/* ---------- Vocabulary search ---------- */

async function loadVocabulary() {
  const list = document.getElementById("wordList");
  const search = document.getElementById("wordSearch");
  const count = document.getElementById("wordCount");

  if (!list || !search || !count) return;

  try {
    const response = await fetch(
      "./sign_language_150_plus_vocabulary.csv"
    );

    if (!response.ok) {
      throw new Error("CSV file not found");
    }

    const csv = await response.text();

    // Parse CSV, including quoted fields.
    const rows = csv.trim().split(/\r?\n/).map((line) => {
      const values = [];
      let value = "";
      let quoted = false;

      for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (
          char === '"' &&
          line[i + 1] === '"' &&
          quoted
        ) {
          value += '"';
          i++;
        } else if (char === '"') {
          quoted = !quoted;
        } else if (char === "," && !quoted) {
          values.push(value);
          value = "";
        } else {
          value += char;
        }
      }

      values.push(value);
      return values;
    });

    const headers = rows.shift().map((h) => h.trim());

    const wordIndex = headers.indexOf("word_or_phrase");
    const categoryIndex = headers.indexOf("category");

    if (wordIndex < 0 || categoryIndex < 0) {
      throw new Error("CSV columns are missing");
    }

    const words = rows
      .filter((row) => row[wordIndex]?.trim())
      .map((row) => ({
        word: row[wordIndex].trim(),
        category: row[categoryIndex]?.trim() || "General"
      }));

    function renderWords() {
      const query = search.value.trim().toLowerCase();

      const filtered = words.filter((item) =>
        item.word.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query)
      );

      count.textContent = `${filtered.length} words found`;
      list.replaceChildren();

      filtered.forEach((item) => {
        const card = document.createElement("div");
        card.className = "vocabulary-card";

        const word = document.createElement("strong");
        word.textContent = item.word;

        const category = document.createElement("small");
        category.textContent = item.category;

        const button = document.createElement("button");
        button.textContent = "Show Gesture";
        button.className = "gesture-button";
        button.type = "button";

        button.addEventListener("click", () => {
          showGesture(item.word);
        });

        card.append(word, category, button);
        list.appendChild(card);
      });
    }

    search.addEventListener("input", renderWords);
    renderWords();
  } catch (error) {
    console.error("Vocabulary loading failed:", error);

    count.textContent =
      "Unable to load vocabulary. Check the CSV path and browser console.";
  }
}

loadVocabulary();

/* ---------- Gesture reference popup ---------- */

function showGesture(word) {
  let modal = document.getElementById("gestureModal");

  if (!modal) {
    modal = document.createElement("div");
    modal.id = "gestureModal";
    modal.className = "gesture-modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");

    document.body.appendChild(modal);
  }

  const slug = word
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

  modal.replaceChildren();

  const content = document.createElement("div");
  content.className = "gesture-content";

  const title = document.createElement("h2");
  title.textContent = word;

  const image = document.createElement("img");
  image.src = `./gestures/${slug}.jpg`;
  image.alt = `ASL sign reference for ${word}`;

  const message = document.createElement("p");
  message.textContent = "Loading sign reference…";

  image.onload = () => {
    message.textContent = "ASL sign reference";
  };

  image.onerror = () => {
    image.hidden = true;
    message.textContent =
      "No gesture image has been added for this word yet.";
  };

  const source = document.createElement("a");
  source.href = "https://aslsignbank.haskins.yale.edu/";
  source.target = "_blank";
  source.rel = "noopener noreferrer";
  source.textContent = "Find this sign in ASL Signbank";

  const close = document.createElement("button");
  close.textContent = "Close";
  close.className = "gesture-button";
  close.type = "button";
  close.addEventListener("click", () => modal.remove());

  content.append(title, image, message, source, close);
  modal.appendChild(content);

  close.focus();
}

/* Close the gesture popup using Escape. */
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document.getElementById("gestureModal")?.remove();
  }
});
