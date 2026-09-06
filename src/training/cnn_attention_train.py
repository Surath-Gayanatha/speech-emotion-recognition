import os
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DATA_PROCESSED_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
)

from src.models.cnn_attention import build_cnn_attention
from src.data.preprocessing import normalize_features


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_actor_ids(path):
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def extract_actor_id(filename):
    return filename.split("_")[0]


def main():

    set_seed()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading features...")

    features = np.load(
        DATA_PROCESSED_DIR / "features.npy"
    ).astype(np.float32)

    labels = np.load(
        DATA_PROCESSED_DIR / "labels.npy"
    )

    filenames = np.load(
        DATA_PROCESSED_DIR / "filenames.npy",
        allow_pickle=True
    )

    print("Original feature shape:", features.shape)
    print("Labels shape:", labels.shape)

    # ---------------------------------------------------------
    # Convert:
    # (samples, 120, 174)
    #
    # to:
    # (samples, 174, 120)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # ACTOR-LEVEL SPLIT
    # ---------------------------------------------------------

    train_actors = read_actor_ids(
        SPLITS_DIR / "train_actors.txt"
    )

    val_actors = read_actor_ids(
        SPLITS_DIR / "val_actors.txt"
    )

    test_actors = read_actor_ids(
        SPLITS_DIR / "test_actors.txt"
    )

    features = normalize_features(features, filenames, train_actors)

    print("Model input shape:", features.shape)

    filename_actor_ids = np.array(
        [
            extract_actor_id(str(name))
            for name in filenames
        ]
    )

    train_mask = np.isin(
        filename_actor_ids,
        list(train_actors)
    )

    val_mask = np.isin(
        filename_actor_ids,
        list(val_actors)
    )

    test_mask = np.isin(
        filename_actor_ids,
        list(test_actors)
    )

    X_train = features[train_mask]
    y_train = labels[train_mask]

    X_val = features[val_mask]
    y_val = labels[val_mask]

    X_test = features[test_mask]
    y_test = labels[test_mask]

    print("\nDataset split:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    # ---------------------------------------------------------
    # CLASS DISTRIBUTION
    # ---------------------------------------------------------

    print("\nTraining class distribution:")

    unique_classes, class_counts = np.unique(
        y_train,
        return_counts=True
    )

    for cls, count in zip(
        unique_classes,
        class_counts
    ):
        print(
            f"Class {cls}: {count} samples"
        )

    # ---------------------------------------------------------
    # CLASS WEIGHTS
    # IMPORTANT:
    # Calculated ONLY from training data
    # ---------------------------------------------------------

    classes = np.unique(y_train)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(
            classes,
            class_weights_array
        )
    }

    print("\nClass weights:")
    print(class_weights)

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    model = build_cnn_attention(
        input_shape=X_train.shape[1:],
        num_classes=NUM_CLASSES
    )

    print("\nModel Summary:")
    model.summary()

    # ---------------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------------

    checkpoint_path = (
        MODELS_DIR /
        "cnn_attention_v3_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    print(
        "\nStarting CNN + Attention V3 training...\n"
    )

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=60,

        batch_size=32,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    # ---------------------------------------------------------
    # LOAD BEST VALIDATION MODEL
    # ---------------------------------------------------------

    print(
        "\nLoading best validation model..."
    )

    best_model = tf.keras.models.load_model(
        checkpoint_path
    )

    # ---------------------------------------------------------
    # SAVE BEST MODEL AS V3
    # ---------------------------------------------------------

    final_model_path = (
        MODELS_DIR /
        "cnn_attention_v3_final.keras"
    )

    best_model.save(
        final_model_path
    )

    print(
        "\nBest V3 model saved to:"
    )

    print(final_model_path)

    # ---------------------------------------------------------
    # TEST EVALUATION
    #
    # TEST SET IS USED ONLY HERE
    # ---------------------------------------------------------

    print(
        "\nEvaluating BEST MODEL on TEST set..."
    )

    test_loss, test_accuracy = (
        best_model.evaluate(
            X_test,
            y_test,
            verbose=1
        )
    )

    print("\n==============================")
    print("CNN + Attention V3 TEST RESULTS")
    print("==============================")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    y_probability = best_model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        y_probability,
        axis=1
    )

    emotion_names = [
        "Anger",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad"
    ]

    # ---------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=emotion_names,
        digits=4
    )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------

    report_path = (
        RESULTS_DIR /
        "cnn_attention_v3_classification_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "CNN + Attention V3 - Test Results\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Test Loss: {test_loss:.4f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(report)

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(str(cm))

    # ---------------------------------------------------------
    # SAVE TRAINING HISTORY
    # ---------------------------------------------------------

    history_path = (
        RESULTS_DIR /
        "cnn_attention_v3_history.npz"
    )

    np.savez(
        history_path,

        loss=np.array(
            history.history["loss"]
        ),

        accuracy=np.array(
            history.history["accuracy"]
        ),

        val_loss=np.array(
            history.history["val_loss"]
        ),

        val_accuracy=np.array(
            history.history["val_accuracy"]
        )
    )

    print("\nResults saved:")

    print(report_path)

    print(history_path)

    print(
        "\n=============================="
    )

    print(
        "CNN + Attention V3 TRAINING "
        "COMPLETED"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()