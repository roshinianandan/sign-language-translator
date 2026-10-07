"""Convert the trained Keras model to TensorFlow.js format for the web app.

    pip install tensorflowjs        # ideally in its own virtualenv
    python convert_to_tfjs.py

Writes web/model/model.json (+ weight shards) and web/model/labels.json.
"""
import shutil
import subprocess
import sys
import tensorflow as tf
from config import MODEL_DIR, ROOT

OUT = ROOT / "web" / "model"
OUT.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(MODEL_DIR / "sign_model.keras")
h5 = MODEL_DIR / "sign_model.h5"
model.save(h5)

exe = shutil.which("tensorflowjs_converter")
if not exe:
    sys.exit("tensorflowjs_converter not found. Run: pip install tensorflowjs")
subprocess.check_call([exe, "--input_format=keras", str(h5), str(OUT)])
shutil.copy(MODEL_DIR / "labels.json", OUT / "labels.json")
print(f"Done. Web model written to {OUT}")
