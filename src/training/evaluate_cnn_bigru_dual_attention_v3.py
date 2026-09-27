import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.models.cnn_bigru_dual_attention_v3 import build_model


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "models/cnn_bigru_dual_attention_v3/"
    "best_model.keras"
)

X_TEST_PATH = (
    "data/processed/logmel/test_features.npy"
)

Y_TEST_PATH = (
    "data/processed/logmel/test_labels.npy"
)

RESULT_DIR = (
    "results/cnn_bigru_dual_attention_v3"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]


print("=" * 70)
print("CNN + BiGRU + DUAL ATTENTION V3")
print("FINAL TEST EVALUATION")
print("=" * 70)

print("Test samples:", len(X_test))
print("Test shape:", X_test.shape)


# ============================================================
# BUILD EXACT ARCHITECTURE
# ============================================================

model = build_model()

model.load_weights(
    MODEL_PATH
)

print(
    "\nBest model weights loaded successfully."
)


# ============================================================
# PREDICTION
# ============================================================

y_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# MACRO METRICS
# ============================================================

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# WEIGHTED METRICS
# ============================================================

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy       : "
    f"{accuracy * 100:.2f}%"
)

print("\nMacro Metrics")

print(
    f"Precision           : "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Recall              : "
    f"{recall_macro * 100:.2f}%"
)

print(
    f"F1-Score            : "
    f"{f1_macro * 100:.2f}%"
)

print("\nWeighted Metrics")

print(
    f"Precision           : "
    f"{precision_weighted * 100:.2f}%"
)

print(
    f"Recall              : "
    f"{recall_weighted * 100:.2f}%"
)

print(
    f"F1-Score            : "
    f"{f1_weighted * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
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

print(
    "\n" + "=" * 70
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 70
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    "=" * 70
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 70
)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.npy"
    ),
    cm
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    os.path.join(
        RESULT_DIR,
        "test_metrics.txt"
    ),
    "w"
) as f:

    f.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro Precision: "
        f"{precision_macro * 100:.2f}%\n"
    )

    f.write(
        f"Macro Recall: "
        f"{recall_macro * 100:.2f}%\n"
    )

    f.write(
        f"Macro F1: "
        f"{f1_macro * 100:.2f}%\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{precision_weighted * 100:.2f}%\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{recall_weighted * 100:.2f}%\n"
    )

    f.write(
        f"Weighted F1: "
        f"{f1_weighted * 100:.2f}%\n"
    )


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)