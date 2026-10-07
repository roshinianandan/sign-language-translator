"""MediaPipe helpers: landmark detection, normalisation and drawing."""
import cv2
import numpy as np
import mediapipe as mp
from config import FEATURES_PER_HAND

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def create_detector():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


def detect(frame_bgr, detector):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return detector.process(rgb)


def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


def _normalise_hand(arr):
    """Translation + scale invariant features, plus absolute wrist position."""
    if not arr.any():
        return np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    wrist = arr[0].copy()
    rel = arr - wrist
    scale = np.max(np.abs(rel)) + 1e-6
    rel = rel / scale
    return np.concatenate([rel.flatten(), wrist[:2]]).astype(np.float32)


def extract_keypoints(results):
    """Return a (130,) vector: [left hand | right hand]. Zeros if a hand is missing.

    NOTE: the frame must be mirrored (cv2.flip) identically during data
    collection and inference so handedness labels stay consistent.
    """
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            if hd.classification[0].label == "Left":
                left = arr
            else:
                right = arr
    return np.concatenate([_normalise_hand(left), _normalise_hand(right)])


def hand_present(results):
    return bool(results.multi_hand_landmarks)
