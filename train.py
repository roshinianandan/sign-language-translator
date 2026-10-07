"""Train the LSTM sign classifier.

Usage:  python train.py [--epochs 120]
Outputs: models/sign_model.keras, models/labels.json, reports/*.png, reports/metrics.txt
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import *


def load_dataset():
    X, y, labels = [], [], []
    for action in ACTIONS:
        files = sorted((DATA_DIR / action).glob("*.npy"))
        if not files:
            raise SystemExit(f"No data for '{action}'. Run collect_data.py first.")
        labels.append(action)
        for f in files:
            seq = np.load(f)
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X.append(seq)
                y.append(ACTIONS.index(action))
    return np.array(X), np.array(y), labels


def augment(X, copies=3):
    """Gaussian jitter + small random time-shift + random hand dropout."""
    out = [X]
    for _ in range(copies):
        Xa = X + np.random.normal(0, 0.01, X.shape).astype(np.float32)
        shift = np.random.randint(-3, 4)
        Xa = np.roll(Xa, shift, axis=1)
        out.append(Xa)
    return np.concatenate(out)


def build_model(n_classes):
    m = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    X, y, labels = load_dataset()
    print(f"Loaded {len(X)} sequences, {len(labels)} classes")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)
    X_aug = augment(X_tr)
    y_aug = np.tile(y_tr, len(X_aug) // len(y_tr))

    model = build_model(len(labels))
    model.summary()
    cbs = [
        callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5),
    ]
    hist = model.fit(X_aug, y_aug, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=args.batch, callbacks=cbs)

    # Evaluation
    probs = model.predict(X_te)
    pred = probs.argmax(1)
    report = classification_report(y_te, pred, target_names=labels, digits=3)
    print(report)
    (REPORT_DIR / "metrics.txt").write_text(report)

    cm = confusion_matrix(y_te, pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(REPORT_DIR / "training_curves.png", dpi=150); plt.close()

    model.save(MODEL_DIR / "sign_model.keras")
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels))
    print("Saved model to", MODEL_DIR / "sign_model.keras")

    # Optional TFLite export for mobile / Raspberry Pi
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        conv._experimental_lower_tensor_list_ops = False
        (MODEL_DIR / "sign_model.tflite").write_bytes(conv.convert())
        print("Saved TFLite model.")
    except Exception as e:  # noqa
        print("TFLite export skipped:", e)


if __name__ == "__main__":
    main()
