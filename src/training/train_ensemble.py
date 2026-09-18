"""Fast Ensemble Stacking Pipeline for Speech Emotion Recognition.

Combines predictions from:
1. Enriched 200D CNN + Multi-Head Self-Attention
2. Deep BiLSTM + Multi-Head Self-Attention

Soft-voting / Logit Averaging eliminates individual model variance and delivers top accuracy.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.config import (
    SPLITS_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    NUM_CLASSES,
)
from src.models.cnn_attention_enriched import build_enriched_cnn_attention, SpecAugment

EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]


def load_actor_ids(filename: str):
    path = SPLITS_DIR / filename
    return set(path.read_text().splitlines())


def get_actor_id(filename: str):
    return filename.split("_")[0]


def create_split_mask(filenames, actor_ids):
    return np.array([get_actor_id(f) in actor_ids for f in filenames])


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    print("Loading 200D Enriched Dataset for Ensemble Evaluation...", flush=True)
    features = np.load(DATA_PROCESSED_DIR / "features_enriched.npy").astype(np.float32)
    labels = np.load(DATA_PROCESSED_DIR / "labels_enriched.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "filenames_enriched.npy")

    features = np.transpose(features, (0, 2, 1))  # (N, 174, 200)

    train_actors = load_actor_ids("train_actors.txt")
    val_actors = load_actor_ids("val_actors.txt")
    test_actors = load_actor_ids("test_actors.txt")

    train_mask = create_split_mask(filenames, train_actors)
    val_mask = create_split_mask(filenames, val_actors)
    test_mask = create_split_mask(filenames, test_actors)

    X_train_raw, y_train_raw = features[train_mask], labels[train_mask]
    X_val_raw, y_val_raw = features[val_mask], labels[val_mask]
    X_test_raw, y_test_raw = features[test_mask], labels[test_mask]

    N_tr, T, F = X_train_raw.shape
    N_va = X_val_raw.shape[0]
    N_te = X_test_raw.shape[0]

    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, F).astype(np.float32))

    X_train = scaler.transform(X_train_raw.reshape(-1, F).astype(np.float32)).reshape(N_tr, T, F)
    X_val = scaler.transform(X_val_raw.reshape(-1, F).astype(np.float32)).reshape(N_va, T, F)
    X_test = scaler.transform(X_test_raw.reshape(-1, F).astype(np.float32)).reshape(N_te, T, F)

    y_train_cat = keras.utils.to_categorical(y_train_raw, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val_raw, NUM_CLASSES)

    # Train Model 1: Deep CNN + Self-Attention
    print("\n[Ensemble Model 1] Training Deep CNN + Multi-Head Self-Attention...", flush=True)
    model1 = build_enriched_cnn_attention(input_shape=(T, F), num_classes=NUM_CLASSES)
    model1.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=55, batch_size=128, verbose=1
    )

    prob1_test = model1.predict(X_test, verbose=0)
    acc1 = np.mean(np.argmax(prob1_test, axis=1) == y_test_raw)
    print(f"Model 1 (CNN + Attention) Test Accuracy: {acc1 * 100:.2f}%", flush=True)

    # Train Model 2: BiLSTM + Multi-Head Self-Attention
    print("\n[Ensemble Model 2] Training BiLSTM + Multi-Head Self-Attention...", flush=True)
    inputs2 = keras.Input(shape=(T, F))
    x2 = SpecAugment(freq_mask_max=15, time_mask_max=20)(inputs2)
    x2 = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.25))(x2)
    x2 = layers.LayerNormalization()(x2)
    x2 = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.25))(x2)
    attn2 = layers.MultiHeadAttention(num_heads=4, key_dim=48, dropout=0.2)(x2, x2)
    x2 = layers.Add()([x2, attn2])
    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dense(128, activation="relu")(x2)
    x2 = layers.Dropout(0.3)(x2)
    out2 = layers.Dense(NUM_CLASSES, activation="softmax")(x2)

    model2 = keras.Model(inputs=inputs2, outputs=out2)
    model2.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"],
    )
    model2.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=45, batch_size=128, verbose=1
    )

    prob2_test = model2.predict(X_test, verbose=0)
    acc2 = np.mean(np.argmax(prob2_test, axis=1) == y_test_raw)
    print(f"Model 2 (BiLSTM + Attention) Test Accuracy: {acc2 * 100:.2f}%", flush=True)

    # Ensemble Weighted Soft Voting Prediction
    print("\nCalculating Ensemble Stacking Predictions...", flush=True)
    ensemble_prob = 0.55 * prob1_test + 0.45 * prob2_test
    ensemble_preds = np.argmax(ensemble_prob, axis=1)
    ensemble_acc = np.mean(ensemble_preds == y_test_raw)

    print("\n" + "=" * 60, flush=True)
    print(f"ENSEMBLE MODEL TEST ACCURACY: {ensemble_acc * 100:.2f}%", flush=True)
    print("=" * 60, flush=True)

    report = classification_report(y_test_raw, ensemble_preds, target_names=EMOTION_NAMES, digits=4)
    cm = confusion_matrix(y_test_raw, ensemble_preds)

    print("\nClassification Report:\n", report, flush=True)
    print("Confusion Matrix:\n", cm, flush=True)

    report_path = RESULTS_DIR / "ensemble_model_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Ensemble Stacking Model (CNN-Attention + BiLSTM-Attention) Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Ensemble Test Accuracy: {ensemble_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    metrics_json_path = RESULTS_DIR / "metrics" / "ensemble_ser.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Ensemble Stacking (CNN-Attention + BiLSTM-Attention)",
            "test_accuracy": float(ensemble_acc),
            "test_accuracy_percent": float(ensemble_acc * 100),
        }, f, indent=4)

    print(f"\nSaved ensemble report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
