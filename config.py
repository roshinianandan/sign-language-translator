"""Central configuration for the sign language translator."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Signs to recognise. "neutral" = hands down / resting; never added to the sentence.
ACTIONS = ["neutral", "hello", "thanks", "yes", "no", "please", "help", "iloveyou"]
IDLE_LABEL = "neutral"

# Text shown / spoken for each label
DISPLAY = {
    "hello": "Hello", "thanks": "Thank you", "yes": "Yes", "no": "No",
    "please": "Please", "help": "Help", "iloveyou": "I love you",
}

SEQUENCE_LENGTH = 30          # frames per sign (~1 second at 30 FPS)
SEQUENCES_PER_ACTION = 60     # samples to record per sign
FEATURES_PER_HAND = 65        # 21*3 normalised landmarks + wrist (x, y)
NUM_FEATURES = FEATURES_PER_HAND * 2

# Inference
CONFIDENCE_THRESHOLD = 0.85   # minimum softmax probability to accept a sign
STABLE_FRAMES = 8             # last N predictions must agree
MIN_HAND_FRAMES = 0.4         # fraction of window that must contain a hand
COOLDOWN_FRAMES = 20          # frames to wait before accepting a new word
CAMERA_INDEX = 0
