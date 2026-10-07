"""Rule-based hand-gesture classifier (no training or recorded data needed).

Works on one hand's 21 MediaPipe landmarks. Orientation-independent finger tests
(distance from the wrist) decide which fingers are extended; thumbs up/down also
use the image direction. Mirror of web/rules.js - keep the two in sync.
"""
import math

# (word key, gesture to show to the camera)
GUIDE = [
    ("hello",    "Open palm - all 5 fingers up"),
    ("yes",      "Thumbs up (fist, thumb up)"),
    ("no",       "Thumbs down (fist, thumb down)"),
    ("iloveyou", "Thumb + index + little finger out"),
    ("thanks",   "Peace sign (index + middle finger)"),
    ("please",   "Point up with index finger only"),
    ("help",     "Closed fist (thumb tucked in)"),
]

FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))   # (pip, tip) for index, middle, ring, pinky
FINGER_RATIO = 1.2     # tip must be this much farther from the wrist than the pip
THUMB_RATIO = 0.65     # thumb tip distance from index knuckle / palm size


def _d(a, b, w, h):
    return math.hypot((a[0] - b[0]) * w, (a[1] - b[1]) * h)


def finger_states(lm, w, h):
    wrist = lm[0]
    ext = [_d(lm[t], wrist, w, h) > FINGER_RATIO * _d(lm[p], wrist, w, h) for p, t in FINGERS]
    palm = _d(wrist, lm[9], w, h) + 1e-6
    thumb = (_d(lm[4], lm[5], w, h) / palm > THUMB_RATIO and
             _d(lm[4], lm[17], w, h) > _d(lm[3], lm[17], w, h))
    return thumb, ext


def classify(lm, w, h):
    """lm: list of 21 (x, y, z) normalised points. Returns a word key or None."""
    thumb, (i, m, r, p) = finger_states(lm, w, h)
    if i and m and r and p:
        return "hello"
    if thumb and i and p and not m and not r:
        return "iloveyou"
    if i and m and not r and not p:
        return "thanks"
    if i and not (m or r or p):
        return "please"
    if not (i or m or r or p):
        if thumb:
            dx = (lm[4][0] - lm[2][0]) * w
            dy = (lm[4][1] - lm[2][1]) * h
            if abs(dy) > 1.2 * abs(dx):
                return "yes" if dy < 0 else "no"
            return None
        return "help"
    return None
