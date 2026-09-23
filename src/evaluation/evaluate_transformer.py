"""
Evaluate Advanced Acoustic Conformer Transformer
=================================================

Rebuilds the exact Transformer architecture and loads the
saved best checkpoint weights before evaluating on the
unseen CREMA-D test set.

Test:
    data/processed/logmel/test_features.npy
    data/processed/logmel/test_labels.npy

Weights:
    models/advanced_transformer/best_model.keras

Results:
    results/advanced_transformer/test_results.json
    results/advanced_transformer/confusion_matrix.png
    results/advanced_transformer/classification_report.txt
"""

import os
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

from src.models.transformer import build_transformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/advanced_transformer/best_model.keras"

DATA_DIR = "data/processed/logmel"

RESULTS_DIR = "results/advanced_transformer"

TEST_FEATURES = os.path.join(
    DATA_DIR,
    "test_features.npy"
)

TEST_LABELS = os.path.join(
    DATA_DIR,
    "test_labels.npy"
)

RESULTS_JSON = os.path.join(
    RESULTS_DIR,
    "test_results.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

EMOTION_LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADVANCED ACOUSTIC CONFORMER TRANSFORMER")
    print("TEST SET EVALUATION")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    if not os.path.exists(TEST_FEATURES):
        raise FileNotFoundError(
            f"Test features not found:\n{TEST_FEATURES}"
        )

    if not os.path.exists(TEST_LABELS):
        raise FileNotFoundError(
            f"Test labels not found:\n{TEST_LABELS}"
        )

    X_test = np.load(TEST_FEATURES)
    y_test = np.load(TEST_LABELS)

    print(
        f"Original X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    # ========================================================
    # TRANSPOSE
    # ========================================================
    #
    # Stored Log-Mel:
    #     (samples, 64, 174)
    #
    # Transformer:
    #     (samples, 174, 64)
    # ========================================================

    if X_test.ndim != 3:
        raise ValueError(
            f"Expected 3D test features, "
            f"got {X_test.shape}"
        )

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )

    X_test = X_test.astype(
        np.float32
    )

    y_test = y_test.astype(
        np.int32
    )

    print(
        f"Transformer X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    # ========================================================
    # CHECK LABELS
    # ========================================================

    unique_labels = np.unique(y_test)

    print(
        f"Test labels found: {unique_labels}"
    )

    if not np.all(
        np.isin(
            unique_labels,
            np.arange(
                len(EMOTION_LABELS)
            )
        )
    ):
        raise ValueError(
            "Unexpected test label values."
        )

    # ========================================================
    # BUILD EXACT MODEL ARCHITECTURE
    # ========================================================

    print("\n" + "=" * 70)
    print("BUILDING TRANSFORMER ARCHITECTURE")
    print("=" * 70)

    model = build_transformer(
        input_shape=(174, 64),
        num_classes=6,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=2,
        conv_kernel_size=5,
        dropout=0.15,
        head_dropout=0.25,
    )

    print(
        "\nModel architecture rebuilt successfully."
    )

    print(
        f"Model name      : {model.name}"
    )

    print(
        f"Total parameters: {model.count_params():,}"
    )

    # ========================================================
    # LOAD SAVED WEIGHTS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING SAVED BEST WEIGHTS")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Saved model file not found:\n{MODEL_PATH}"
        )

    print(
        f"Checkpoint: {MODEL_PATH}"
    )

    try:

        # First try loading weights directly.
        model.load_weights(
            MODEL_PATH
        )

        print(
            "\nSaved weights loaded successfully."
        )

    except Exception as first_error:

        print(
            "\nDirect weight loading failed."
        )

        print(
            "Trying checkpoint loading through Keras..."
        )

        try:

            loaded_model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    "MultiScaleConvBlock":
                        __import__(
                            "src.models.transformer",
                            fromlist=[
                                "MultiScaleConvBlock"
                            ]
                        ).MultiScaleConvBlock,

                    "DepthwiseConvPositionalEncoding":
                        __import__(
                            "src.models.transformer",
                            fromlist=[
                                "DepthwiseConvPositionalEncoding"
                            ]
                        ).DepthwiseConvPositionalEncoding,

                    "ConformerBlock":
                        __import__(
                            "src.models.transformer",
                            fromlist=[
                                "ConformerBlock"
                            ]
                        ).ConformerBlock,

                    "MultiHeadAttentionPooling":
                        __import__(
                            "src.models.transformer",
                            fromlist=[
                                "MultiHeadAttentionPooling"
                            ]
                        ).MultiHeadAttentionPooling,

                    "TemporalStatsPooling":
                        __import__(
                            "src.models.transformer",
                            fromlist=[
                                "TemporalStatsPooling"
                            ]
                        ).TemporalStatsPooling,
                },
                compile=False,
            )

            model.set_weights(
                loaded_model.get_weights()
            )

            print(
                "\nSaved model weights loaded successfully."
            )

        except Exception as second_error:

            print("\nERROR: Could not load saved weights.")
            print("\nFirst error:")
            print(first_error)
            print("\nSecond error:")
            print(second_error)

            raise RuntimeError(
                "The saved checkpoint could not be loaded "
                "into the rebuilt Transformer architecture."
            )

    # ========================================================
    # PREDICTION
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1,
    )

    print(
        f"Prediction shape: {probabilities.shape}"
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    print(
        f"Predicted labels shape: {y_pred.shape}"
    )

    # ========================================================
    # METRICS
    # ========================================================

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

    # ========================================================
    # TEST RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Test Samples       : {len(y_test)}"
    )

    print(
        f"Test Accuracy      : "
        f"{accuracy * 100:.2f}%"
    )

    print("\nMacro Average:")

    print(
        f"Precision          : "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Recall             : "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"F1-score           : "
        f"{f1_macro * 100:.2f}%"
    )

    print("\nWeighted Average:")

    print(
        f"Precision          : "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Recall             : "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"F1-score           : "
        f"{f1_weighted * 100:.2f}%"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(
            len(EMOTION_LABELS)
        ),
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Advanced Acoustic Conformer Transformer\n"
        )

        f.write(
            "CREMA-D Test Set Classification Report\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(
            len(EMOTION_LABELS)
        )
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Advanced Acoustic Conformer Transformer\n"
        "Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(EMOTION_LABELS)
    )

    plt.xticks(
        tick_marks,
        EMOTION_LABELS,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        EMOTION_LABELS
    )

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    plt.ylabel(
        "True Emotion"
    )

    plt.xlabel(
        "Predicted Emotion"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SAVE JSON
    # ========================================================

    results = {

        "model":
            "Advanced Acoustic Conformer Transformer",

        "model_name":
            model.name,

        "test_samples":
            int(len(y_test)),

        "test_accuracy":
            float(accuracy),

        "test_accuracy_percent":
            float(accuracy * 100),

        "macro_precision":
            float(precision_macro),

        "macro_recall":
            float(recall_macro),

        "macro_f1":
            float(f1_macro),

        "weighted_precision":
            float(precision_weighted),

        "weighted_recall":
            float(recall_weighted),

        "weighted_f1":
            float(f1_weighted),

        "confusion_matrix":
            cm.tolist(),

        "emotion_labels":
            EMOTION_LABELS
    }

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"\nTest Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1      : "
        f"{f1_macro * 100:.2f}%"
    )

    print("\nSaved files:")

    print(
        f"1. {RESULTS_JSON}"
    )

    print(
        f"2. {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"3. {CLASSIFICATION_REPORT_PATH}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()