"""
Evaluate CNN + BiGRU + Temporal Statistics V4

Uses:
- Test Log-Mel features
- Best validation-accuracy checkpoint
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

IMPORTANT:
The test set is used only for final evaluation.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.cnn_bigru_temporal_stats_v4 import build_model


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLASSES = 6

N_MELS = 64
TIME_FRAMES = 174

BATCH_SIZE = 32

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]


# ============================================================
# PATHS
# ============================================================

TEST_X_PATH = (
    "data/processed/logmel/test_features.npy"
)

TEST_Y_PATH = (
    "data/processed/logmel/test_labels.npy"
)

BEST_MODEL_PATH = (
    "models/cnn_bigru_temporal_stats_v4/"
    "best_model.keras"
)

RESULTS_DIR = (
    "results/cnn_bigru_temporal_stats_v4"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CNN + BiGRU + TEMPORAL STATISTICS V4")
print("FINAL TEST EVALUATION")
print("=" * 70)

print(f"Seed        : {SEED}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Input shape : ({N_MELS}, {TIME_FRAMES}, 1)")


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test = np.load(TEST_X_PATH)
y_test = np.load(TEST_Y_PATH)

print(
    f"Original X_test shape: {X_test.shape}"
)

print(
    f"Original y_test shape: {y_test.shape}"
)


# ============================================================
# PREPARE INPUT
# ============================================================

if X_test.ndim == 3:

    X_test = X_test[..., np.newaxis]


X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int32)


print(
    f"Final X_test shape   : {X_test.shape}"
)

print(
    f"Final y_test shape   : {y_test.shape}"
)


# ============================================================
# CHECK DATA
# ============================================================

print("\nTest labels:")
print(
    np.unique(
        y_test,
        return_counts=True
    )
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING V4 MODEL")
print("=" * 70)

model = build_model(
    input_shape=(
        N_MELS,
        TIME_FRAMES,
        1
    ),
    num_classes=NUM_CLASSES
)

print(
    f"Model parameters: {model.count_params():,}"
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST CHECKPOINT")
print("=" * 70)

print(
    f"Checkpoint: {BEST_MODEL_PATH}"
)

if not os.path.exists(BEST_MODEL_PATH):

    raise FileNotFoundError(
        f"\nBest model not found:\n"
        f"{BEST_MODEL_PATH}\n"
        f"\nPlease make sure training completed successfully."
    )


model.load_weights(
    BEST_MODEL_PATH
)

print("Best checkpoint loaded successfully.")


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# KERAS TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss              : {test_loss:.4f}"
)

print(
    f"Test Accuracy          : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision        : {precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall           : {recall_macro * 100:.2f}%"
)

print(
    f"Macro F1-score         : {f1_macro * 100:.2f}%"
)

print(
    f"Weighted Precision     : {precision_weighted * 100:.2f}%"
)

print(
    f"Weighted Recall        : {recall_weighted * 100:.2f}%"
)

print(
    f"Weighted F1-score      : {f1_weighted * 100:.2f}%"
)

print("=" * 70)


# ============================================================
# PER-CLASS REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "Rows = True Class"
)

print(
    "Columns = Predicted Class"
)

print()

print(
    "          "
    + " ".join(
        f"{name[:3]:>6}"
        for name in CLASS_NAMES
    )
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:>8} "
        + " ".join(
            f"{value:6d}"
            for value in row
        )
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CNN + BiGRU + Temporal Statistics V4\n"
    )

    f.write(
        "Final Test Evaluation\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: {precision_macro:.6f}\n"
    )

    f.write(
        f"Macro Recall: {recall_macro:.6f}\n"
    )

    f.write(
        f"Macro F1: {f1_macro:.6f}\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{precision_weighted:.6f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{recall_weighted:.6f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{f1_weighted:.6f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(report)

    f.write(
        "\n\nConfusion Matrix\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        np.array2string(cm)
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

np.save(
    CONFUSION_PATH,
    cm
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"Classification report saved to:"
)

print(
    REPORT_PATH
)

print(
    f"\nConfusion matrix saved to:"
)

print(
    CONFUSION_PATH
)

print("=" * 70)