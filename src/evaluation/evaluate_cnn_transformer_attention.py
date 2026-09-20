import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.cnn_transformer_attention import build_cnn_transformer_attention


# =========================
# Configuration
# =========================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed/logmel"
MODEL_PATH = "models/cnn_transformer_attention/best_model.keras"
RESULTS_DIR = "results/cnn_transformer_attention"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# Load Test Data
# =========================

print("\nLoading test data...")

X_test = np.load(
    os.path.join(DATA_DIR, "test_features.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "test_labels.npy")
)

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)


# =========================
# Add Channel Dimension
# =========================

if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]

print("Test input shape:", X_test.shape)


# =========================
# Load Best Model
# =========================

print("\nLoading best CNN + Transformer + Attention model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TransformerEncoder": 
        __import__(
            "src.models.cnn_transformer_attention",
            fromlist=["TransformerEncoder"]
        ).TransformerEncoder
    }
)

print("Model loaded successfully.")


# =========================
# Final Test Evaluation
# =========================

print("\nRunning final evaluation on unseen TEST set...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=32,
    verbose=1
)


# =========================
# Predictions
# =========================

y_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(y_prob, axis=1)


# =========================
# Metrics
# =========================

accuracy = accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================
# Classification Report
# =========================

class_names = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# =========================
# Print Results
# =========================

print("\n" + "=" * 60)
print("CNN + TRANSFORMER + ATTENTION")
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss           : {test_loss:.4f}")
print(f"Test Accuracy       : {accuracy * 100:.2f}%")

print(f"Macro Precision     : {macro_precision * 100:.2f}%")
print(f"Macro Recall        : {macro_recall * 100:.2f}%")
print(f"Macro F1            : {macro_f1 * 100:.2f}%")

print(f"Weighted Precision  : {weighted_precision * 100:.2f}%")
print(f"Weighted Recall     : {weighted_recall * 100:.2f}%")
print(f"Weighted F1        : {weighted_f1 * 100:.2f}%")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)


# =========================
# Save Metrics
# =========================

results = {
    "model": "CNN + Transformer + Attention",
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "classification_report": report,
    "confusion_matrix": cm.tolist()
}

results_path = os.path.join(
    RESULTS_DIR,
    "test_results.json"
)

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print(results_path)

print("\nEvaluation completed successfully.")