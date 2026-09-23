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

from src.models.cnn_se_bilstm_attention import build_cnn_se_bilstm_attention


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FEATURES = "data/processed/logmel/test_features.npy"
TEST_LABELS = "data/processed/logmel/test_labels.npy"

MODEL_PATH = "models/cnn_se_bilstm_attention/best_model.keras"

RESULTS_DIR = "results/cnn_se_bilstm_attention"

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CNN + SE + BiLSTM + Multi-Head Attention")
    print("TEST SET EVALUATION")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    X_test = np.load(TEST_FEATURES)
    y_test = np.load(TEST_LABELS)

    print(f"Test features shape: {X_test.shape}")
    print(f"Test labels shape   : {y_test.shape}")

    # Add channel dimension
    if X_test.ndim == 3:
        X_test = X_test[..., np.newaxis]

    print(f"Model input shape    : {X_test.shape}")

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "SEBlock": __import__(
                "src.models.cnn_se_bilstm_attention",
                fromlist=["SEBlock"]
            ).SEBlock
        }
    )

    print(f"Model loaded from:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nRunning prediction on unseen test set...")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    y_prob = model.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    y_pred = np.argmax(y_prob, axis=1)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Test Loss           : {test_loss:.4f}")
    print(f"Test Accuracy       : {accuracy * 100:.2f}%")
    print(f"Macro Precision     : {macro_precision * 100:.2f}%")
    print(f"Macro Recall        : {macro_recall * 100:.2f}%")
    print(f"Macro F1            : {macro_f1 * 100:.2f}%")
    print(f"Weighted Precision  : {weighted_precision * 100:.2f}%")
    print(f"Weighted Recall     : {weighted_recall * 100:.2f}%")
    print(f"Weighted F1        : {weighted_f1 * 100:.2f}%")

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report_text)

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "model": "CNN_SE_BiLSTM_Attention",
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "test_samples": int(len(y_test)),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    results_path = os.path.join(
        RESULTS_DIR,
        "test_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    cm_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.npy"
    )

    np.save(cm_path, cm)

    print("\nResults saved to:")
    print(results_path)

    print("\nConfusion matrix saved to:")
    print(cm_path)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()