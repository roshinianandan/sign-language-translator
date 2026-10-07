"""Instant demo - no recording or training needed.

    python demo_rules.py

Show a gesture to the camera and hold it for about half a second.
Keys: Q quit | C clear | S speak sentence | M toggle auto-speak
"""
from collections import deque
import cv2
from config import DISPLAY, CAMERA_INDEX
from utils import create_detector, detect, draw_landmarks
from rules import classify, GUIDE
from speech import Speaker

STABLE_FRAMES = 8


def put(frame, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    detector = create_detector()
    speaker = Speaker()
    recent, sentence, last, auto = deque(maxlen=STABLE_FRAMES), [], None, True

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = detect(frame, detector)
        draw_landmarks(frame, res)

        label = None
        if res.multi_hand_landmarks:
            lm = [(p.x, p.y, p.z) for p in res.multi_hand_landmarks[0].landmark]
            label = classify(lm, w, h)
        recent.append(label)

        if len(recent) == STABLE_FRAMES:
            if all(r is None for r in recent):
                last = None                      # hand released -> same word may repeat
            elif label and all(r == label for r in recent) and label != last:
                text = DISPLAY[label]
                sentence.append(text)
                last = label
                if auto:
                    speaker.say(text)

        # overlay: guide, current sign, caption
        for k, (key, how) in enumerate(GUIDE):
            put(frame, f"{DISPLAY[key]}: {how}", (10, 22 + 20 * k), 0.5,
                (0, 255, 0) if key == label else (220, 220, 220))
        put(frame, DISPLAY.get(label, "-"), (10, h - 80), 1.2, (0, 255, 255), 3)
        cv2.rectangle(frame, (0, h - 55), (w, h), (30, 30, 30), -1)
        put(frame, " ".join(sentence[-8:]) or "...", (12, h - 18), 0.9, (255, 255, 255), 2)
        cv2.imshow("Sign translator - gesture demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
        elif key == ord("s"):
            speaker.say(" ".join(sentence))
        elif key == ord("m"):
            auto = not auto
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
