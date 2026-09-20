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

from src.models.cnn_bigru_attention_fusion import (
    build_cnn_bigru_attention_fusion,
    FeatureFusion
)


# ============================================================
# Paths
# ============================================================

LOGMEL_DIR = "data/processed/logmel"
MFCC_DIR = "data/processed/mfcc"

MODEL_PATH = (
    "models/cnn_bigru_attention_fusion/best_model.keras"
)

RESULTS_DIR = "results/cnn_bigru_attention_fusion"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load Test Data
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test_logmel = np.load(
    os.path.join(
        LOGMEL_DIR,
        "test_features.npy"
    )
)

X_test_mfcc = np.load(
    os.path.join(
        MFCC_DIR,
        "test_features.npy"
    )
)

y_test = np.load(
    os.path.join(
        LOGMEL_DIR,
        "test_labels.npy"
    )
)

print("Log-Mel test shape:", X_test_logmel.shape)
print("MFCC test shape   :", X_test_mfcc.shape)
print("Labels shape      :", y_test.shape)


# ============================================================
# Verify MFCC and Log-Mel Labels
# ============================================================

mfcc_test_labels = np.load(
    os.path.join(
        MFCC_DIR,
        "test_labels.npy"
    )
)

if not np.array_equal(
    y_test,
    mfcc_test_labels
):
    raise ValueError(
        "Log-Mel and MFCC test labels do not match."
    )

print("\nTest labels verified successfully.")


# ============================================================
# Add Channel Dimension
# ============================================================

X_test_logmel = X_test_logmel[..., np.newaxis]
X_test_mfcc = X_test_mfcc[..., np.newaxis]

X_test_logmel = X_test_logmel.astype(np.float32)
X_test_mfcc = X_test_mfcc.astype(np.float32)

y_test = y_test.astype(np.int32)


print("\nAfter adding channel dimension:")
print("Log-Mel:", X_test_logmel.shape)
print("MFCC   :", X_test_mfcc.shape)


# ============================================================
# Load Best Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "FeatureFusion": FeatureFusion
    }
)

print("Model loaded successfully.")


# ============================================================
# Prepare Test Inputs
# ============================================================

test_inputs = {
    "logmel_input": X_test_logmel,
    "mfcc_input": X_test_mfcc
}


# ============================================================
# Prediction
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST PREDICTION")
print("=" * 70)

y_prob = model.predict(
    test_inputs,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ============================================================
# Calculate Metrics
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
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# Print Final Test Results
# ============================================================

print("\n" + "=" * 70)
print("CNN + BiGRU + ATTENTION FEATURE FUSION")
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision     : {precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall        : {recall_macro * 100:.2f}%"
)

print(
    f"Macro F1-Score      : {f1_macro * 100:.2f}%"
)

print(
    f"Weighted Precision  : {precision_weighted * 100:.2f}%"
)

print(
    f"Weighted Recall     : {recall_weighted * 100:.2f}%"
)

print(
    f"Weighted F1-Score   : {f1_weighted * 100:.2f}%"
)


# ============================================================
# Classification Report
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report_text)


# ============================================================
# Confusion Matrix
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# Save Results
# ============================================================

results = {
    "model": "CNN + BiGRU + Attention Feature Fusion",

    "test_accuracy": float(accuracy),

    "macro_precision": float(
        precision_macro
    ),

    "macro_recall": float(
        recall_macro
    ),

    "macro_f1": float(
        f1_macro
    ),

    "weighted_precision": float(
        precision_weighted
    ),

    "weighted_recall": float(
        recall_weighted
    ),

    "weighted_f1": float(
        f1_weighted
    ),

    "classification_report": report,

    "confusion_matrix": cm.tolist()
}


output_path = os.path.join(
    RESULTS_DIR,
    "test_results.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# Completion Message
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("Results saved to:")
print(output_path)

print("\nTest evaluation finished successfully.")