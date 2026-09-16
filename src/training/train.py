import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    SPLITS_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    NUM_CLASSES,
)

from src.models.cnn1d import build_cnn1d


EMOTION_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]


def load_actor_ids(filename):
    path = SPLITS_DIR / filename
    return set(path.read_text().splitlines())


def get_actor_id(filename):
    return filename.split("_")[0]


def create_split_mask(filenames, actor_ids):
    return np.array([
        get_actor_id(filename) in actor_ids
        for filename in filenames
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn1d")
    parser.add_argument("--config", default="configs/cnn1d.yaml")
    args = parser.parse_args()

    if args.model != "cnn1d":
        raise ValueError("This training script currently supports cnn1d only.")

    # Reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Create output folders
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data
    print("Loading processed features...")

    features = np.load(DATA_PROCESSED_DIR / "features.npy")
    labels = np.load(DATA_PROCESSED_DIR / "labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "filenames.npy")

    print(f"Original features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # ---------------------------------------------------------
    # IMPORTANT:
    # Extracted features are:
    # (samples, n_features, time_steps)
    #
    # CNN expects:
    # (samples, time_steps, n_features)
    # ---------------------------------------------------------

    features = np.transpose(features, (0, 2, 1))

    print(f"CNN input shape: {features.shape}")

    # Load actor-level splits
    train_actors = load_actor_ids("train_actors.txt")
    val_actors = load_actor_ids("val_actors.txt")
    test_actors = load_actor_ids("test_actors.txt")

    # Create masks
    train_mask = create_split_mask(filenames, train_actors)
    val_mask = create_split_mask(filenames, val_actors)
    test_mask = create_split_mask(filenames, test_actors)

    X_train = features[train_mask]
    y_train = labels[train_mask]

    X_val = features[val_mask]
    y_val = labels[val_mask]

    X_test = features[test_mask]
    y_test = labels[test_mask]

    print()
    print("Dataset split:")
    print(f"Train: {X_train.shape} | Labels: {y_train.shape}")
    print(f"Val:   {X_val.shape} | Labels: {y_val.shape}")
    print(f"Test:  {X_test.shape} | Labels: {y_test.shape}")

    # Build CNN1D
    print()
    print("Building CNN1D model...")

    model = build_cnn1d(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=NUM_CLASSES,
    )

    model.summary()

    # Callbacks
    checkpoint_path = MODELS_DIR / "cnn1d_best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    print()
    print("=" * 60)
    print("STARTING CNN1D TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = MODELS_DIR / "cnn1d_final.keras"
    model.save(final_model_path)

    # Evaluate on test set
    print()
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predictions
    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=EMOTION_NAMES,
        digits=4,
    )

    print()
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(cm)

    # Save metrics
    report_path = RESULTS_DIR / "cnn1d_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CNN1D Speech Emotion Recognition Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write("\nConfusion Matrix\n")
        f.write("=" * 60 + "\n")
        f.write(str(cm))

    # Save training history
    history_path = RESULTS_DIR / "cnn1d_history.npz"

    np.savez(
        history_path,
        accuracy=np.array(history.history["accuracy"]),
        val_accuracy=np.array(history.history["val_accuracy"]),
        loss=np.array(history.history["loss"]),
        val_loss=np.array(history.history["val_loss"]),
    )

    print()
    print("Training completed successfully.")
    print(f"Best model: {checkpoint_path}")
    print(f"Final model: {final_model_path}")
    print(f"Results: {report_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()