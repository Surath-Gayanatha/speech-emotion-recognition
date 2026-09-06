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

from src.models.cnn_bilstm_attention import (
    build_cnn_bilstm_attention
)
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

    print("=" * 60)
    print("CNN + BiLSTM + Attention TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load processed features
    # ---------------------------------------------------------

    print("\nLoading processed features...")

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

    print("Original features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # ---------------------------------------------------------
    # Convert:
    # (samples, 120, 174)
    #
    # to:
    # (samples, 174, 120)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Load actor-level splits
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

    # ---------------------------------------------------------
    # Extract actor IDs from filenames
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Create datasets
    # ---------------------------------------------------------

    X_train = features[train_mask]
    y_train = labels[train_mask]

    X_val = features[val_mask]
    y_val = labels[val_mask]

    X_test = features[test_mask]
    y_test = labels[test_mask]

    print("\nDataset split:")
    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape
    )

    # ---------------------------------------------------------
    # Class weights
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
    # Build model
    # ---------------------------------------------------------

    print("\nBuilding CNN + BiLSTM + Attention model...")

    model = build_cnn_bilstm_attention(
        input_shape=X_train.shape[1:],
        num_classes=NUM_CLASSES
    )

    print("\nModel Summary:")
    model.summary()

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    checkpoint_path = (
        MODELS_DIR /
        "cnn_bilstm_attention_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
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
    # Train
    # ---------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

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
    # Save final model
    # ---------------------------------------------------------

    final_model_path = (
        MODELS_DIR /
        "cnn_bilstm_attention_final.keras"
    )

    model.save(final_model_path)

    print("\nFinal model saved to:")
    print(final_model_path)

    # ---------------------------------------------------------
    # Final TEST evaluation
    # ---------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\n")
    print("=" * 60)
    print("CNN + BiLSTM + Attention TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    emotion_names = [
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
        labels=list(range(NUM_CLASSES)),
        target_names=emotion_names,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print("Confusion Matrix:")
    print(cm)

    # ---------------------------------------------------------
    # Save classification report
    # ---------------------------------------------------------

    report_path = (
        RESULTS_DIR /
        "cnn_bilstm_attention_classification_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "CNN + BiLSTM + Attention "
            "Test Results\n"
        )

        f.write(
            "================================\n\n"
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
    # Save training history
    # ---------------------------------------------------------

    history_path = (
        RESULTS_DIR /
        "cnn_bilstm_attention_history.npz"
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

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nSaved files:")

    print(final_model_path)
    print(report_path)
    print(history_path)


if __name__ == "__main__":
    main()