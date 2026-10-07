# Real-Time Assistive Sign Language Translator (MediaPipe + TensorFlow)

Translates dynamic hand signs from a webcam into on-screen captions and synthesized speech,
helping deaf / hard-of-hearing users communicate with people who don't know sign language.

## Pipeline

```
Webcam -> MediaPipe Hands (2 x 21 landmarks)
       -> Normalisation (wrist-relative, scale-invariant, + wrist position) = 130 features/frame
       -> Sliding window of 30 frames
       -> Conv1D + BiLSTM + LSTM classifier (TensorFlow/Keras)
       -> Temporal smoothing (confidence threshold, N-frame agreement, cooldown)
       -> Sentence builder -> on-screen captions + offline TTS (pyttsx3)
```

Why landmarks instead of raw images? They are tiny (130 floats vs. thousands of pixels), robust to
lighting / background / skin tone, run in real time on a CPU, and protect privacy (no images stored).

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt                     # Python 3.10 / 3.11 recommended
```

## Usage
1. **Collect data** (60 samples per sign; vary distance, angle, speed, and signers):
   `python collect_data.py`
   Record `neutral` too (hands down, idle, small random movements) so the model can say "no sign".
2. **Train**: `python train.py` -> `models/sign_model.keras`, `reports/confusion_matrix.png`, `reports/metrics.txt`
3. **Run live**: `python realtime.py`  (Q quit, C clear, S speak, M toggle auto-speak)

## Adding your own signs
Edit `ACTIONS`/`DISPLAY` in `config.py`, record the new sign, and retrain.

## Tuning tips
| Problem | Fix |
|---|---|
| Random words appear | raise `CONFIDENCE_THRESHOLD`, `STABLE_FRAMES`; record more `neutral` data |
| Words are slow / missed | lower `STABLE_FRAMES` or `COOLDOWN_FRAMES` |
| Poor accuracy on new users | record multiple signers, lighting and camera positions |
| Left/right confusion | the frame is mirrored in both collection and inference; don't change one only |

## Limitations & future work
- Recognises isolated signs from a small vocabulary, not full sign-language grammar.
- Hands only; adding facial expression and body pose (MediaPipe Holistic) improves ISL/ASL coverage.
- Next steps: continuous sign recognition (CTC / Transformer), fingerspelling alphabet module,
  TFLite on Android (`models/sign_model.tflite`), multilingual TTS, web demo with MediaPipe JS.

## Project structure
```
config.py        settings, vocabulary         utils.py     MediaPipe + feature extraction
collect_data.py  dataset recorder             train.py     training, evaluation, TFLite export
realtime.py      live translator + speech     data/ models/ reports/
```

## Mobile web app (runs in the phone browser)

The `web/` folder is a static site: camera -> MediaPipe Hands (JS) -> your model in TensorFlow.js
-> captions + speech (Web Speech API). Nothing is uploaded; all processing happens on the phone.

1. Train as usual (`collect_data.py`, `train.py`).
2. Convert the model: `pip install tensorflowjs` then `python convert_to_tfjs.py`
   (writes `web/model/model.json`, weights and `labels.json`).
3. Host the `web/` folder over **HTTPS** (phones only allow camera access on HTTPS):
   - **Netlify Drop**: drag the `web` folder onto https://app.netlify.com/drop -> instant public link
   - or **GitHub Pages**: push `web/` contents to a repo, Settings -> Pages
4. Open the link on your phone, tap **Start camera**, allow access.

Local test on a computer: `cd web && python -m http.server 8000` -> http://localhost:8000
(localhost is allowed to use the camera without HTTPS).

Notes: if you change the sign list or thresholds in `config.py`, mirror the values in the `CFG`
block at the top of `web/app.js`. The page loads TensorFlow.js and MediaPipe from the jsDelivr CDN,
so the phone needs internet on first load.

## Instant gesture demo (no recording or training)

`python demo_rules.py` recognises 7 simple hand gestures straight away using rules on the MediaPipe
landmarks (`rules.py`). The phone app (`web/`) switches to the same mode automatically when
`web/model/model.json` is missing, so it works as soon as you host the folder.

| Word | Gesture |
|---|---|
| Hello | Open palm, all 5 fingers up |
| Yes | Thumbs up |
| No | Thumbs down |
| I love you | Thumb + index + little finger out |
| Thank you | Peace sign (index + middle) |
| Please | Point up with index finger only |
| Help | Closed fist, thumb tucked in |

These are easy static handshapes, not the real ASL versions of most words (only "I love you" matches ASL).
For real ASL signs, including moving ones, record your own data and use the trained model (`realtime.py`).
Thresholds are at the top of `rules.py` and `web/rules.js` (keep both in sync).
