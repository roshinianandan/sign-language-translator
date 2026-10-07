"""Real-time sign language translation with on-screen captions and speech.

Keys:  Q quit | C clear sentence | S speak sentence | M toggle auto-speak
"""
import json
import queue
import threading
from collections import deque, Counter
import cv2
import numpy as np
import tensorflow as tf
from config import *
from utils import create_detector, detect, draw_landmarks, extract_keypoints, hand_present


class Speaker:
    """Non-blocking text-to-speech (pyttsx3 works offline)."""
    def __init__(self):
        self.q = queue.Queue()
        try:
            import pyttsx3  # noqa
            self.ok = True
            threading.Thread(target=self._run, daemon=True).start()
        except Exception:
            self.ok = False

    def _run(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        while True:
            engine.say(self.q.get())
            engine.runAndWait()

    def say(self, text):
        if self.ok and text:
            self.q.put(text)


def draw_ui(frame, probs, labels, sentence, current, auto_speak):
    h, w = frame.shape[:2]
    # probability bars
    for i, (lab, p) in enumerate(zip(labels, probs)):
        y = 30 + i * 26
        cv2.rectangle(frame, (10, y), (10 + int(p * 180), y + 18),
                      (0, 200, 0) if lab == current else (180, 120, 0), -1)
        cv2.putText(frame, f"{lab} {p:.2f}", (200, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # caption bar
    cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
    text = " ".join(sentence[-8:]) or "..."
    cv2.putText(frame, text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"auto-speak: {'ON' if auto_speak else 'OFF'}", (w - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    model = tf.keras.models.load_model(MODEL_DIR / "sign_model.keras")
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    detector = create_detector()
    speaker = Speaker()

    window = deque(maxlen=SEQUENCE_LENGTH)
    hand_flags = deque(maxlen=SEQUENCE_LENGTH)
    recent = deque(maxlen=STABLE_FRAMES)
    sentence, cooldown, auto_speak = [], 0, True
    probs, current = np.zeros(len(labels)), IDLE_LABEL

    cap = cv2.VideoCapture(CAMERA_INDEX)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        results = detect(frame, detector)
        draw_landmarks(frame, results)
        window.append(extract_keypoints(results))
        hand_flags.append(hand_present(results))
        cooldown = max(0, cooldown - 1)

        if len(window) == SEQUENCE_LENGTH:
            if np.mean(hand_flags) >= MIN_HAND_FRAMES:
                probs = model.predict(np.expand_dims(np.array(window), 0), verbose=0)[0]
                idx = int(probs.argmax())
                current = labels[idx] if probs[idx] >= CONFIDENCE_THRESHOLD else IDLE_LABEL
            else:
                probs, current = np.zeros(len(labels)), IDLE_LABEL
            recent.append(current)

            # accept a word only if predictions are stable and cooldown is over
            if len(recent) == STABLE_FRAMES and cooldown == 0:
                word, count = Counter(recent).most_common(1)[0]
                if count == STABLE_FRAMES and word != IDLE_LABEL:
                    text = DISPLAY.get(word, word)
                    if not sentence or sentence[-1] != text:
                        sentence.append(text)
                        if auto_speak:
                            speaker.say(text)
                    cooldown = COOLDOWN_FRAMES

        draw_ui(frame, probs, labels, sentence, current, auto_speak)
        cv2.imshow("Sign Language Translator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
        elif key == ord("s"):
            speaker.say(" ".join(sentence))
        elif key == ord("m"):
            auto_speak = not auto_speak

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
