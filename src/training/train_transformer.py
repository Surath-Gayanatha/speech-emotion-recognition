import json
import math
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from src.models.transformer import build_transformer
from src.config import (
    ROOT_DIR,
    DATA_PROCESSED_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    NUM_CLASSES,
    EMOTION_LABELS
)


# ============================================================
# Load Configuration
# ============================================================

CONFIG_PATH = ROOT_DIR / "configs" / "transformer.yaml"

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "model": {
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 256,
            "num_layers": 2,
            "conv_kernel_size": 5,
            "dropout": 0.15,
            "head_dropout": 0.20
        },
        "training": {
            "batch_size": 64,
            "epochs": 80,
            "learning_rate": 5.0e-4,
            "min_lr": 1.0e-6,
            "warmup_epochs": 5,
            "weight_decay": 1.0e-3,
            "label_smoothing": 0.1,
            "patience": 18,
            "spec_augment": {
                "time_mask_max_size": 12,
                "time_masks": 2,
                "freq_mask_max_size": 12,
                "freq_masks": 2
            }
        }
    }

MODEL_CFG = config.get("model", {})
TRAIN_CFG = config.get("training", {})
SPEC_CFG = TRAIN_CFG.get("spec_augment", {})

SEED = 42
BATCH_SIZE = TRAIN_CFG.get("batch_size", 64)
EPOCHS = TRAIN_CFG.get("epochs", 80)
PATIENCE = TRAIN_CFG.get("patience", 18)
INIT_LR = float(TRAIN_CFG.get("learning_rate", 5.0e-4))
MIN_LR = float(TRAIN_CFG.get("min_lr", 1.0e-6))
WARMUP_EPOCHS = TRAIN_CFG.get("warmup_epochs", 5)
LABEL_SMOOTHING = float(TRAIN_CFG.get("label_smoothing", 0.1))
WEIGHT_DECAY = float(TRAIN_CFG.get("weight_decay", 1.0e-3))

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Warmup + Cosine Decay LR Schedule
# ============================================================

def get_lr_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        return INIT_LR * float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
    else:
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
        return MIN_LR + 0.5 * (INIT_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


# ============================================================
# Dynamic SpecAugment + Audio Mixup Data Generator
# ============================================================

class DynamicAudioDataGenerator(keras.utils.Sequence):
    """Generates augmented acoustic sequence batches with SpecAugment and Audio Mixup."""

    def __init__(self, X, y, batch_size=64, shuffle=True, is_training=True, alpha=0.2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_training = is_training
        self.alpha = alpha
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx].copy()

        if self.is_training:
            # 1. SpecAugment
            X_batch = self.apply_spec_augment(X_batch)

            # 2. Mixup Augmentation
            if self.alpha > 0 and np.random.rand() < 0.5 and len(X_batch) > 1:
                lam = np.random.beta(self.alpha, self.alpha)
                perm = np.random.permutation(len(X_batch))
                X_batch = lam * X_batch + (1.0 - lam) * X_batch[perm]
                y_batch = lam * y_batch + (1.0 - lam) * y_batch[perm]

        return X_batch, y_batch

    def apply_spec_augment(self, X_batch):
        n_samples, time_steps, n_features = X_batch.shape
        for i in range(n_samples):
            if np.random.rand() < 0.7:
                # Time Masking
                for _ in range(SPEC_CFG.get("time_masks", 2)):
                    t_size = np.random.randint(1, SPEC_CFG.get("time_mask_max_size", 12) + 1)
                    t_start = np.random.randint(0, max(1, time_steps - t_size))
                    X_batch[i, t_start:t_start + t_size, :] = 0.0

                # Frequency Channel Masking
                for _ in range(SPEC_CFG.get("freq_masks", 2)):
                    f_size = np.random.randint(1, SPEC_CFG.get("freq_mask_max_size", 12) + 1)
                    f_start = np.random.randint(0, max(1, n_features - f_size))
                    X_batch[i, :, f_start:f_start + f_size] = 0.0

        return X_batch


# ============================================================
# Main Training Function
# ============================================================

def main():
    print("==================================================")
    print("ADVANCED FINE-TUNED TRANSFORMER FOR HIGH ACCURACY")
    print("==================================================")

    # 1. Load Processed Data
    features = np.load(DATA_PROCESSED_DIR / "features.npy")
    labels = np.load(DATA_PROCESSED_DIR / "labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "filenames.npy")

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # 2. Load Actor Splits
    def load_actor_file(fname):
        return set((SPLITS_DIR / fname).read_text().splitlines())

    train_actors = load_actor_file("train_actors.txt")
    val_actors = load_actor_file("val_actors.txt")
    test_actors = load_actor_file("test_actors.txt")

    def get_actor_id(f):
        return str(f).split("_")[0]

    actors = np.array([get_actor_id(f) for f in filenames])

    train_mask = np.array([a in train_actors for a in actors])
    val_mask = np.array([a in val_actors for a in actors])
    test_mask = np.array([a in test_actors for a in actors])

    X_train = features[train_mask]
    y_train = labels[train_mask]

    X_val = features[val_mask]
    y_val = labels[val_mask]

    X_test = features[test_mask]
    y_test = labels[test_mask]

    print("\nDataset split sizes:")
    print(f"Train     : {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test      : {X_test.shape[0]} samples")

    # 3. Feature Standardization
    scaler = StandardScaler()
    n_train, time_steps, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    X_train_2d = scaler.fit_transform(X_train.reshape(-1, n_features))
    X_val_2d = scaler.transform(X_val.reshape(-1, n_features))
    X_test_2d = scaler.transform(X_test.reshape(-1, n_features))

    X_train = X_train_2d.reshape(n_train, time_steps, n_features)
    X_val = X_val_2d.reshape(n_val, time_steps, n_features)
    X_test = X_test_2d.reshape(n_test, time_steps, n_features)

    # Convert targets to one-hot for label smoothing loss
    y_train_oh = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_oh = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_oh = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    # 4. Data Generators
    train_gen = DynamicAudioDataGenerator(
        X_train, y_train_oh, batch_size=BATCH_SIZE, shuffle=True, is_training=True, alpha=0.2
    )
    val_gen = DynamicAudioDataGenerator(
        X_val, y_val_oh, batch_size=BATCH_SIZE, shuffle=False, is_training=False
    )

    # 5. Build Advanced Architecture
    model = build_transformer(
        input_shape=(time_steps, n_features),
        num_classes=NUM_CLASSES,
        embed_dim=MODEL_CFG.get("embed_dim", 128),
        num_heads=MODEL_CFG.get("num_heads", 4),
        ff_dim=MODEL_CFG.get("ff_dim", 256),
        num_layers=MODEL_CFG.get("num_layers", 2),
        conv_kernel_size=MODEL_CFG.get("conv_kernel_size", 5),
        dropout=MODEL_CFG.get("dropout", 0.15),
        head_dropout=MODEL_CFG.get("head_dropout", 0.20)
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=INIT_LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    model.summary()

    # 6. Callbacks
    model_dir = MODELS_DIR / "transformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "transformer_model.keras"

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    lr_scheduler = keras.callbacks.LearningRateScheduler(get_lr_schedule, verbose=1)

    # 7. Model Training
    print("\nStarting Advanced Transformer training...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        verbose=1
    )

    # 8. Evaluation & Test Prediction
    print("\nEvaluating fine-tuned Transformer on Test Split...\n")

    if checkpoint_path.exists():
        try:
            model.load_weights(checkpoint_path)
        except Exception:
            pass

    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n==============================================")
    print("FINE-TUNED TRANSFORMER TEST RESULTS")
    print("==============================================")
    print(f"Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("==============================================\n")

    report_str = classification_report(
        y_test,
        y_pred,
        target_names=list(EMOTION_LABELS.keys()),
        zero_division=0
    )
    print("Classification Report:")
    print(report_str)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 9. Save Metrics
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "transformer.json"

    metrics_data = {
        "model": "transformer",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "test_samples": int(len(y_test))
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics to: {metrics_path}")

    # Export report text
    report_file = RESULTS_DIR / "transformer_classification_report.txt"
    report_content = f"Advanced Fine-Tuned Transformer - Test Results\n{'=' * 40}\n\n"
    report_content += f"Test Accuracy: {acc * 100:.2f}%\n"
    report_content += f"Test F1-Score: {f1:.4f}\n\n"
    report_content += f"Classification Report:\n{report_str}\n\n"
    report_content += f"Confusion Matrix:\n{cm}\n"
    report_file.write_text(report_content)
    print(f"Saved classification report to: {report_file}")

    # Plot & Save Confusion Matrix Image
    cm_dir = RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_path = cm_dir / "transformer.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(EMOTION_LABELS.keys()),
        yticklabels=list(EMOTION_LABELS.keys())
    )
    plt.title(f"Fine-Tuned Transformer Confusion Matrix (Acc: {acc * 100:.2f}%)")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to: {cm_path}")


if __name__ == "__main__":
    main()