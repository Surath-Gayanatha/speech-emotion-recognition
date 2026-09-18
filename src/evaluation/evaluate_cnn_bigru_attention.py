"""
Final Evaluation - CNN + BiGRU + Multi-Head Attention
======================================================

Dataset:
    CREMA-D

Feature:
    Log-Mel Spectrogram
    64 Mel bands x 174 time frames

Purpose:
    Load the best trained checkpoint and evaluate it
    on the unseen test set.

IMPORTANT:
    - This script does NOT train the model.
    - Test data is used only for final evaluation.
    - Best checkpoint is selected using validation accuracy.
    - SpecAugment is disabled automatically during inference.

Outputs:
    - Test Loss
    - Test Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1
    - Weighted Precision
    - Weighted Recall
    - Weighted F1
    - Classification Report
    - Confusion Matrix
    - Predictions
    - Prediction Probabilities
    - Final Metrics JSON
"""

import os
import json
import random

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Import the custom SpecAugment layer used during training.
# This is required when loading the saved model.
from src.training.train_cnn_bigru_attention import SpecAugment


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/logmel"

MODEL_DIR = "models/cnn_bigru_attention"

RESULTS_DIR = "results/cnn_bigru_attention"

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

BATCH_SIZE = 32

NUM_CLASSES = 6

INPUT_SHAPE = (64, 174, 1)

EXPECTED_TEST_SHAPE = (
    1229,
    64,
    174
)

EMOTION_NAMES = [
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD",
]


# ============================================================
# 3. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 4. HEADER
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("CNN + BiGRU + MULTI-HEAD ATTENTION")
print("=" * 75)

print("\nIMPORTANT:")
print("This script does NOT train the model.")
print("The best validation checkpoint is loaded.")
print("The unseen test set is used only for final evaluation.")


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 75)
print("CHECKING REQUIRED FILES")
print("=" * 75)

required_files = [
    CHECKPOINT_PATH,
    os.path.join(DATA_DIR, "test_features.npy"),
    os.path.join(DATA_DIR, "test_labels.npy"),
]

for file_path in required_files:

    print(f"\nChecking: {file_path}")

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print("  [OK] Found")


# ============================================================
# 6. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING TEST DATA")
print("=" * 75)

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
    f"Test features shape: {X_test.shape}"
)

print(
    f"Test labels shape:   {y_test.shape}"
)


# ============================================================
# 7. VALIDATE TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("VALIDATING TEST DATA")
print("=" * 75)

if X_test.shape != EXPECTED_TEST_SHAPE:

    raise ValueError(
        f"Unexpected test feature shape.\n"
        f"Expected: {EXPECTED_TEST_SHAPE}\n"
        f"Actual:   {X_test.shape}"
    )


if len(X_test) != len(y_test):

    raise ValueError(
        "Test features and test labels have different lengths."
    )


y_test = y_test.astype(
    np.int32
)


if not np.all(
    np.isin(
        y_test,
        np.arange(NUM_CLASSES)
    )
):

    raise ValueError(
        "Invalid test labels found."
    )


if not np.isfinite(X_test).all():

    raise ValueError(
        "NaN or Inf found in test features."
    )


print("Test shape validation: PASSED")
print("Test label validation: PASSED")
print("Numerical validation: PASSED")


# ============================================================
# 8. ADD CHANNEL DIMENSION
# ============================================================

print("\n" + "=" * 75)
print("PREPARING MODEL INPUT")
print("=" * 75)

if X_test.ndim == 3:

    X_test = np.expand_dims(
        X_test,
        axis=-1
    )

elif X_test.ndim != 4:

    raise ValueError(
        f"Unexpected test dimensions: {X_test.shape}"
    )


X_test = X_test.astype(
    np.float32,
    copy=False
)


print(
    f"Final X_test shape: {X_test.shape}"
)


# ============================================================
# 9. TEST CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TEST CLASS DISTRIBUTION")
print("=" * 75)

unique, counts = np.unique(
    y_test,
    return_counts=True
)

for class_id, count in zip(
    unique,
    counts
):

    emotion = EMOTION_NAMES[
        int(class_id)
    ]

    print(
        f"Class {class_id} "
        f"({emotion}): {count}"
    )


# ============================================================
# 10. LOAD BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("LOADING BEST CHECKPOINT")
print("=" * 75)

checkpoint_size = os.path.getsize(
    CHECKPOINT_PATH
)

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
)

print(
    f"Checkpoint size: "
    f"{checkpoint_size / (1024 * 1024):.2f} MB"
)


if checkpoint_size == 0:

    raise RuntimeError(
        "Checkpoint file is empty."
    )


# ============================================================
# LOAD MODEL
# ============================================================

best_model = tf.keras.models.load_model(
    CHECKPOINT_PATH,
    custom_objects={
        "SpecAugment": SpecAugment
    },
    compile=True
)


print(
    "\n[OK] Best checkpoint loaded successfully."
)

print(
    f"Model parameters: "
    f"{best_model.count_params():,}"
)


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("MODEL INFORMATION")
print("=" * 75)

print(
    f"Model name: {best_model.name}"
)

print(
    f"Input shape: {best_model.input_shape}"
)

print(
    f"Output shape: {best_model.output_shape}"
)

print(
    f"Parameters: {best_model.count_params():,}"
)


# ============================================================
# 12. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST SET EVALUATION")
print("=" * 75)

print(
    "\nEvaluating on unseen test set..."
)

print(
    "Test set is NOT used for training or model selection."
)

test_results = best_model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


test_loss = float(
    test_results[0]
)

test_accuracy = float(
    test_results[1]
)


print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 13. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING TEST PREDICTIONS")
print("=" * 75)

y_prob = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


y_pred = np.argmax(
    y_prob,
    axis=1
)


print(
    f"Predictions shape: {y_pred.shape}"
)

print(
    f"Probability shape: {y_prob.shape}"
)


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


(
    macro_precision,
    macro_recall,
    macro_f1,
    _
) = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


(
    weighted_precision,
    weighted_recall,
    weighted_f1,
    _
) = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 15. PRINT FINAL METRICS
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST METRICS")
print("=" * 75)

print(
    f"\nAccuracy:           "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision:    "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall:       "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1:           "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Weighted Precision: "
    f"{weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall:    "
    f"{weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1:        "
    f"{weighted_f1 * 100:.2f}%"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=EMOTION_NAMES,
    digits=4,
    zero_division=0
)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "final_test_classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CNN + BiGRU + Multi-Head Attention\n"
    )

    f.write(
        "FINAL UNSEEN TEST EVALUATION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{weighted_precision:.6f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{weighted_recall:.6f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{weighted_f1:.6f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(report)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


cm_path = os.path.join(
    RESULTS_DIR,
    "final_test_confusion_matrix.npy"
)


np.save(
    cm_path,
    cm
)


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    RESULTS_DIR,
    "test_predictions.npy"
)

probabilities_path = os.path.join(
    RESULTS_DIR,
    "test_probabilities.npy"
)


np.save(
    predictions_path,
    y_pred
)

np.save(
    probabilities_path,
    y_prob
)


# ============================================================
# 19. SAVE FINAL METRICS JSON
# ============================================================

final_metrics = {

    "model":
        "CNN + BiGRU + Multi-Head Attention",

    "feature":
        "Log-Mel Spectrogram",

    "input_shape":
        list(INPUT_SHAPE),

    "num_classes":
        NUM_CLASSES,

    "emotion_classes":
        EMOTION_NAMES,

    "test_samples":
        int(len(y_test)),

    "seed":
        SEED,

    "batch_size":
        BATCH_SIZE,

    "test_loss":
        test_loss,

    "test_accuracy":
        accuracy,

    "test_accuracy_percent":
        accuracy * 100,

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1),

    "checkpoint":
        CHECKPOINT_PATH,

    "test_used_for_training":
        False,

    "test_used_for_model_selection":
        False,
}


metrics_path = os.path.join(
    RESULTS_DIR,
    "final_test_metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("FINAL EVALUATION COMPLETE")
print("=" * 75)

print(
    f"\nFINAL TEST ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"FINAL MACRO F1: "
    f"{macro_f1 * 100:.2f}%"
)

print("\nSaved files:")

print(
    f"1. {report_path}"
)

print(
    f"2. {cm_path}"
)

print(
    f"3. {predictions_path}"
)

print(
    f"4. {probabilities_path}"
)

print(
    f"5. {metrics_path}"
)

print("\nDone.")