import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    SPLITS_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    NUM_CLASSES,
)

from src.models.bilstm import build_bilstm
from src.models.cnn1d import build_cnn1d
from src.models.lstm import build_lstm
from src.models.mlp import build_mlp


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
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8")
    return str(filename).split("_")[0]


def create_split_mask(filenames, actor_ids):
    return np.array([
        get_actor_id(filename) in actor_ids
        for filename in filenames
    ])


def normalize_feature_splits(splits):
    """Standardize sequence features using training samples only."""
    train_features = splits[0][0]
    feature_mean = train_features.mean(axis=(0, 1), keepdims=True)
    feature_std = train_features.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    return [
        ((split_features - feature_mean) / feature_std, split_labels)
        for split_features, split_labels in splits
    ]


def load_config(config_path):
    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def load_data():
    features = np.load(DATA_PROCESSED_DIR / "features.npy")
    labels = np.load(DATA_PROCESSED_DIR / "labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "filenames.npy")

    features = np.transpose(features, (0, 2, 1))
    train_actors = load_actor_ids("train_actors.txt")
    val_actors = load_actor_ids("val_actors.txt")
    test_actors = load_actor_ids("test_actors.txt")

    masks = [
        create_split_mask(filenames, actors)
        for actors in (train_actors, val_actors, test_actors)
    ]
    splits = [
        (features[mask], labels[mask])
        for mask in masks
    ]

    return normalize_feature_splits(splits)


def build_model(model_name, input_shape):
    builders = {
        "mlp": build_mlp,
        "cnn1d": build_cnn1d,
        "lstm": build_lstm,
        "bilstm": build_bilstm,
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {', '.join(builders)}")
    if model_name == "mlp" and len(input_shape) > 1:
        input_shape = (input_shape[0] * input_shape[1],)
    return builders[model_name](input_shape=input_shape, num_classes=NUM_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn1d", "lstm", "bilstm"], default="cnn1d")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config_path = args.config or f"configs/{args.model}.yaml"
    config = load_config(config_path)
    if config.get("model") != args.model:
        raise ValueError(f"Config {config_path} is for {config.get('model')!r}, not {args.model!r}.")

    # Reproducibility
    seed = int(config.get("seed", RANDOM_SEED))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create output folders
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data
    print("Loading processed features...")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    if args.model == "mlp":
        X_train = X_train.reshape(len(X_train), -1)
        X_val = X_val.reshape(len(X_val), -1)
        X_test = X_test.reshape(len(X_test), -1)
    print(f"Features shape: {X_train.shape[1:]}")

    print()
    print("Dataset split:")
    print(f"Train: {X_train.shape} | Labels: {y_train.shape}")
    print(f"Val:   {X_val.shape} | Labels: {y_val.shape}")
    print(f"Test:  {X_test.shape} | Labels: {y_test.shape}")

    # Build the selected architecture using the shared actor-level split.
    print()
    print(f"Building {args.model.upper()} model...")

    model = build_model(args.model, X_train.shape[1:])
    if hasattr(model.optimizer, "learning_rate"):
        model.optimizer.learning_rate.assign(float(config.get("learning_rate", 0.001)))

    model.summary()

    # Callbacks
    model_dir = MODELS_DIR / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{args.model}_best.keras"

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config.get("early_stopping_patience", 8)),
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
    print(f"STARTING {args.model.upper()} TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(config.get("epochs", 60)),
        batch_size=int(config.get("batch_size", 32)),
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = model_dir / f"{args.model}_final.keras"
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
    report_path = RESULTS_DIR / f"{args.model}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{args.model.upper()} Speech Emotion Recognition Results\n")
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
    history_path = RESULTS_DIR / f"{args.model}_history.npz"

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