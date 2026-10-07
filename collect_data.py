"""Record landmark sequences from the webcam.

Usage:
    python collect_data.py                    # record every action in config.ACTIONS
    python collect_data.py --actions hello yes --sequences 40

Controls: SPACE = start, Q = quit.  Existing samples are kept (new ones appended).
"""
import argparse
import time
import cv2
import numpy as np
from config import *
from utils import create_detector, detect, draw_landmarks, extract_keypoints


def banner(frame, text, color=(0, 255, 255), y=40):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", nargs="+", default=ACTIONS)
    ap.add_argument("--sequences", type=int, default=SEQUENCES_PER_ACTION)
    args = ap.parse_args()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    detector = create_detector()

    for action in args.actions:
        out = DATA_DIR / action
        out.mkdir(parents=True, exist_ok=True)
        start_idx = len(list(out.glob("*.npy")))

        # wait for the user
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            frame = cv2.flip(frame, 1)
            banner(frame, f"Next sign: {action.upper()}  (SPACE to start, Q to quit)")
            banner(frame, f"Existing samples: {start_idx}", (255, 255, 255), 80)
            cv2.imshow("Collect", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows(); return

        for n in range(args.sequences):
            # short pause so the signer can reset
            t0 = time.time()
            while time.time() - t0 < 1.2:
                ok, frame = cap.read()
                frame = cv2.flip(frame, 1)
                banner(frame, f"{action.upper()}  #{n + 1}/{args.sequences}  GET READY", (0, 165, 255))
                cv2.imshow("Collect", frame)
                cv2.waitKey(1)

            seq = []
            while len(seq) < SEQUENCE_LENGTH:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results = detect(frame, detector)
                draw_landmarks(frame, results)
                seq.append(extract_keypoints(results))
                banner(frame, f"RECORDING {action.upper()}  #{n + 1}/{args.sequences}", (0, 255, 0))
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
            np.save(out / f"{start_idx + n}.npy", np.array(seq, dtype=np.float32))

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Now run: python train.py")


if __name__ == "__main__":
    main()
