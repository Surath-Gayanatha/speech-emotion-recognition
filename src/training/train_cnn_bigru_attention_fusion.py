import os
import json
import random

import numpy as np
import tensorflow as tf

from src.models.cnn_bigru_attention_fusion import (
    build_cnn_bigru_attention_fusion
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR_LOGMEL = "data/processed/logmel"
DATA_DIR_MFCC = "data/processed/mfcc"

MODEL_DIR = "models/cnn_bigru_attention_fusion"
RESULTS_DIR = "results/cnn_bigru_attention_fusion"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 60


# ============================================================
# Load Log-Mel Data
# ============================================================

print("\n" + "=" * 70)
print("LOADING LOG-MEL FEATURES")
print("=" * 70)

X_train_logmel = np.load(
    os.path.join(DATA_DIR_LOGMEL, "train_features.npy")
)

X_val_logmel = np.load(
    os.path.join(DATA_DIR_LOGMEL, "validation_features.npy")
)

X_test_logmel = np.load(
    os.path.join(DATA_DIR_LOGMEL, "test_features.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR_LOGMEL, "train_labels.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR_LOGMEL, "validation_labels.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR_LOGMEL, "test_labels.npy")
)

print("Train Log-Mel:", X_train_logmel.shape)
print("Validation Log-Mel:", X_val_logmel.shape)
print("Test Log-Mel:", X_test_logmel.shape)


# ============================================================
# Load MFCC Data
# ============================================================

print("\n" + "=" * 70)
print("LOADING MFCC FEATURES")
print("=" * 70)

X_train_mfcc = np.load(
    os.path.join(DATA_DIR_MFCC, "train_features.npy")
)

X_val_mfcc = np.load(
    os.path.join(DATA_DIR_MFCC, "validation_features.npy")
)

X_test_mfcc = np.load(
    os.path.join(DATA_DIR_MFCC, "test_features.npy")
)

print("Train MFCC:", X_train_mfcc.shape)
print("Validation MFCC:", X_val_mfcc.shape)
print("Test MFCC:", X_test_mfcc.shape)


# ============================================================
# Verify Labels
# ============================================================

if not np.array_equal(
    y_train,
    np.load(os.path.join(DATA_DIR_MFCC, "train_labels.npy"))
):
    raise ValueError("Train labels between Log-Mel and MFCC do not match.")

if not np.array_equal(
    y_val,
    np.load(os.path.join(DATA_DIR_MFCC, "validation_labels.npy"))
):
    raise ValueError("Validation labels between Log-Mel and MFCC do not match.")

if not np.array_equal(
    y_test,
    np.load(os.path.join(DATA_DIR_MFCC, "test_labels.npy"))
):
    raise ValueError("Test labels between Log-Mel and MFCC do not match.")

print("\nLabel alignment verified successfully.")


# ============================================================
# Add Channel Dimension
# ============================================================

X_train_logmel = X_train_logmel[..., np.newaxis]
X_val_logmel = X_val_logmel[..., np.newaxis]
X_test_logmel = X_test_logmel[..., np.newaxis]

X_train_mfcc = X_train_mfcc[..., np.newaxis]
X_val_mfcc = X_val_mfcc[..., np.newaxis]
X_test_mfcc = X_test_mfcc[..., np.newaxis]


print("\nAfter adding channel dimension:")

print("Train Log-Mel:", X_train_logmel.shape)
print("Train MFCC:", X_train_mfcc.shape)


# ============================================================
# Memory Optimization
# ============================================================

X_train_logmel = X_train_logmel.astype(np.float32)
X_val_logmel = X_val_logmel.astype(np.float32)

X_train_mfcc = X_train_mfcc.astype(np.float32)
X_val_mfcc = X_val_mfcc.astype(np.float32)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)


# ============================================================
# Create Training Dataset
# ============================================================

train_inputs = {
    "logmel_input": X_train_logmel,
    "mfcc_input": X_train_mfcc
}

val_inputs = {
    "logmel_input": X_val_logmel,
    "mfcc_input": X_val_mfcc
}


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_inputs,
        y_train
    )
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(y_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_inputs,
        y_val
    )
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Build Model
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CNN + BiGRU + ATTENTION FEATURE FUSION MODEL")
print("=" * 70)

model = build_cnn_bigru_attention_fusion()

model.summary()


# ============================================================
# Callback Paths
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

history_path = os.path.join(
    RESULTS_DIR,
    "history.json"
)

config_path = os.path.join(
    RESULTS_DIR,
    "config.json"
)

best_result_path = os.path.join(
    RESULTS_DIR,
    "best_validation_result.json"
)


# ============================================================
# Callbacks
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        reduce_lr,
        early_stopping
    ],
    verbose=1
)


# ============================================================
# Save Training History
# ============================================================

history_dict = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        history_dict,
        file,
        indent=4
    )


# ============================================================
# Find Best Epoch
# ============================================================

val_accuracy_history = history.history["val_accuracy"]

best_epoch_index = int(
    np.argmax(val_accuracy_history)
)

best_epoch = best_epoch_index + 1

best_val_accuracy = float(
    val_accuracy_history[best_epoch_index]
)

train_accuracy_at_best = float(
    history.history["accuracy"][best_epoch_index]
)

train_loss_at_best = float(
    history.history["loss"][best_epoch_index]
)

val_loss_at_best = float(
    history.history["val_loss"][best_epoch_index]
)

train_val_gap = (
    train_accuracy_at_best -
    best_val_accuracy
)


# ============================================================
# Save Best Validation Result
# ============================================================

best_result = {
    "model": "CNN + BiGRU + Attention Feature Fusion",
    "best_epoch": best_epoch,
    "train_accuracy": train_accuracy_at_best,
    "validation_accuracy": best_val_accuracy,
    "train_loss": train_loss_at_best,
    "validation_loss": val_loss_at_best,
    "train_validation_gap": train_val_gap
}

with open(
    best_result_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        best_result,
        file,
        indent=4
    )


# ============================================================
# Save Configuration
# ============================================================

config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "logmel_shape": [64, 174],
    "mfcc_shape": [120, 174],
    "fused_shape": [184, 174],
    "num_classes": 6,
    "test_used_during_training": False,
    "feature_fusion": "Log-Mel + MFCC"
}

with open(
    config_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        config,
        file,
        indent=4
    )


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Epoch          : {best_epoch}"
)

print(
    f"Train Accuracy      : "
    f"{train_accuracy_at_best * 100:.2f}%"
)

print(
    f"Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Train Loss          : "
    f"{train_loss_at_best:.4f}"
)

print(
    f"Validation Loss     : "
    f"{val_loss_at_best:.4f}"
)

print(
    f"Train-Val Gap       : "
    f"{train_val_gap * 100:.2f} percentage points"
)

print("\nBest model saved to:")
print(checkpoint_path)

print("\nHistory saved to:")
print(history_path)

print("\nBest validation result saved to:")
print(best_result_path)

print(
    "\nIMPORTANT: "
    "The test set was NOT used during training or model selection."
)

print("\nTraining completed successfully.")