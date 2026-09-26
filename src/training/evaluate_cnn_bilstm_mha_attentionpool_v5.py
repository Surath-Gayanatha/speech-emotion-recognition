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

from src.models.cnn_bilstm_mha_attentionpool_v5 import (
    build_cnn_bilstm_mha_attentionpool_v5
)


# ============================================================
# CONFIG
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

TEST_X_PATH = "data/processed/logmel/test_features.npy"
TEST_Y_PATH = "data/processed/logmel/test_labels.npy"

BEST_MODEL_PATH = (
    "models/cnn_bilstm_mha_attentionpool_v5/"
    "best_model.keras"
)

RESULTS_DIR = (
    "results/cnn_bilstm_mha_attentionpool_v5"
)


# ============================================================
# SEED
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CNN + BiLSTM + MHA + ATTENTION POOLING V5")
print("FINAL TEST EVALUATION")
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test = np.load(TEST_X_PATH)
y_test = np.load(TEST_Y_PATH)

print("Original X_test:", X_test.shape)
print("Original y_test:", y_test.shape)


# ============================================================
# PREPARE DATA
# ============================================================

if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int32)

print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING V5 MODEL")
print("=" * 70)

model = build_cnn_bilstm_mha_attentionpool_v5(
    input_shape=(N_MELS, TIME_FRAMES, 1),
    num_classes=NUM_CLASSES
)

print(
    f"Model parameters: {model.count_params():,}"
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST V5 CHECKPOINT")
print("=" * 70)

print(BEST_MODEL_PATH)

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"Best model not found: {BEST_MODEL_PATH}"
    )

model.load_weights(
    BEST_MODEL_PATH
)

print("Best checkpoint loaded successfully.")


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TEST LOSS + ACCURACY
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

print("\nGenerating predictions...")

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
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL V5 TEST RESULTS")
print("=" * 70)

print(f"Test Loss          : {test_loss:.4f}")
print(f"Test Accuracy      : {accuracy * 100:.2f}%")
print(f"Macro Precision    : {macro_precision * 100:.2f}%")
print(f"Macro Recall       : {macro_recall * 100:.2f}%")
print(f"Macro F1-score     : {macro_f1 * 100:.2f}%")
print(f"Weighted Precision : {weighted_precision * 100:.2f}%")
print(f"Weighted Recall    : {weighted_recall * 100:.2f}%")
print(f"Weighted F1-score  : {weighted_f1 * 100:.2f}%")

print("=" * 70)


# ============================================================
# CLASSIFICATION REPORT
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

print("Rows = True Class")
print("Columns = Predicted Class\n")

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
# SAVE RESULTS
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CNN + BiLSTM + MHA + Attention Pooling V5\n"
    )

    f.write(
        "Final Test Evaluation\n"
    )

    f.write("=" * 70 + "\n\n")

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

    f.write("Classification Report\n")
    f.write("=" * 70 + "\n")
    f.write(report)

    f.write("\n\nConfusion Matrix\n")
    f.write("=" * 70 + "\n")
    f.write(np.array2string(cm))


np.save(
    cm_path,
    cm
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("V5 EVALUATION COMPLETED")
print("=" * 70)

print("Classification report:")
print(report_path)

print("\nConfusion matrix:")
print(cm_path)

print("=" * 70)