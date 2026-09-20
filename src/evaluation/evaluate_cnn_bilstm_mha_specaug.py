import os
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.models.cnn_bilstm_mha_specaug import (
    build_cnn_bilstm_mha_specaug
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/processed/logmel"

MODEL_PATH = (
    "models/cnn_bilstm_mha_specaug/"
    "best_model.keras"
)

RESULTS_DIR = (
    "results/cnn_bilstm_mha_specaug"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Load Test Data
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "test_features.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "test_labels.npy"
    )
)

print(
    "Test features:",
    X_test.shape
)

print(
    "Test labels:",
    y_test.shape
)


# ============================================================
# Prepare Test Data
# ============================================================

X_test = X_test.astype(
    np.float32
)

y_test = y_test.astype(
    np.int32
)

# Add channel dimension
# (N, 64, 174)
# ->
# (N, 64, 174, 1)

X_test = X_test[
    ...,
    np.newaxis
]

print(
    "Test input shape:",
    X_test.shape
)


# ============================================================
# Load Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Model loaded successfully."
)


# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# Evaluate Test Set
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON UNSEEN TEST SET")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=32,
    verbose=1
)


# ============================================================
# Predictions
# ============================================================

print("\nGenerating predictions...")

y_probability = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# Classification Metrics
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# Classification Report
# ============================================================

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

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# Final Results
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss       : {test_loss:.4f}"
)

print(
    f"Test Accuracy   : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : "
    f"{precision * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{recall * 100:.2f}%"
)

print(
    f"Macro F1-Score  : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# Save Results
# ============================================================

results = {
    "model": (
        "CNN + BiLSTM + Multi-Head "
        "Self-Attention + SpecAugment"
    ),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "test_samples": int(len(y_test))
}


results_path = os.path.join(
    RESULTS_DIR,
    "test_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# Save Confusion Matrix
# ============================================================

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


print("\nResults saved to:")
print(results_path)

print("\nConfusion matrix saved to:")
print(cm_path)

print("\nTest evaluation completed successfully.")