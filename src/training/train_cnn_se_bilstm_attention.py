import os
import json
import random

import numpy as np
import tensorflow as tf

from src.models.cnn_se_bilstm_attention import (
    build_cnn_se_bilstm_attention
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed/logmel"

MODEL_DIR = "models/cnn_se_bilstm_attention"
RESULTS_DIR = "results/cnn_se_bilstm_attention"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 60


# ============================================================
# SpecAugment
# ============================================================

def spec_augment(
    spectrogram,
    freq_mask_param=8,
    time_mask_param=18
):

    # (64, 174, 1) -> (64, 174)
    spectrogram = tf.squeeze(
        spectrogram,
        axis=-1
    )

    freq_size = tf.shape(spectrogram)[0]
    time_size = tf.shape(spectrogram)[1]

    # --------------------------------------------------------
    # Frequency Mask
    # --------------------------------------------------------

    freq_width = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=freq_mask_param + 1,
        dtype=tf.int32
    )

    max_freq_start = tf.maximum(
        1,
        freq_size - freq_width + 1
    )

    freq_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=max_freq_start,
        dtype=tf.int32
    )

    freq_mask = tf.concat(
        [
            tf.ones(
                [freq_start, time_size],
                dtype=spectrogram.dtype
            ),

            tf.zeros(
                [freq_width, time_size],
                dtype=spectrogram.dtype
            ),

            tf.ones(
                [
                    freq_size -
                    freq_start -
                    freq_width,
                    time_size
                ],
                dtype=spectrogram.dtype
            )
        ],
        axis=0
    )

    spectrogram = spectrogram * freq_mask

    # --------------------------------------------------------
    # Time Mask
    # --------------------------------------------------------

    time_width = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=time_mask_param + 1,
        dtype=tf.int32
    )

    max_time_start = tf.maximum(
        1,
        time_size - time_width + 1
    )

    time_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=max_time_start,
        dtype=tf.int32
    )

    time_mask = tf.concat(
        [
            tf.ones(
                [
                    freq_size,
                    time_start
                ],
                dtype=spectrogram.dtype
            ),

            tf.zeros(
                [
                    freq_size,
                    time_width
                ],
                dtype=spectrogram.dtype
            ),

            tf.ones(
                [
                    freq_size,
                    time_size -
                    time_start -
                    time_width
                ],
                dtype=spectrogram.dtype
            )
        ],
        axis=1
    )

    spectrogram = spectrogram * time_mask

    # (64, 174) -> (64, 174, 1)
    spectrogram = tf.expand_dims(
        spectrogram,
        axis=-1
    )

    return spectrogram


# ============================================================
# Load Log-Mel Data
# ============================================================

print("\n" + "=" * 70)
print("LOADING LOG-MEL DATA")
print("=" * 70)

X_train = np.load(
    os.path.join(
        DATA_DIR,
        "train_features.npy"
    )
)

X_val = np.load(
    os.path.join(
        DATA_DIR,
        "validation_features.npy"
    )
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        "train_labels.npy"
    )
)

y_val = np.load(
    os.path.join(
        DATA_DIR,
        "validation_labels.npy"
    )
)

print("Train features:", X_train.shape)
print("Validation features:", X_val.shape)

print("Train labels:", y_train.shape)
print("Validation labels:", y_val.shape)


# ============================================================
# Data Types
# ============================================================

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)


# ============================================================
# Add Channel Dimension
# ============================================================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print("\nAfter adding channel dimension:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)


# ============================================================
# Training Dataset
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)


def training_augmentation(x, y):

    x = spec_augment(
        x,
        freq_mask_param=8,
        time_mask_param=18
    )

    return x, y


train_dataset = train_dataset.shuffle(
    buffer_size=len(y_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    training_augmentation,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Validation Dataset
# ============================================================

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
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
print("BUILDING CNN + SE + BiLSTM + MHA MODEL")
print("=" * 70)

model = build_cnn_se_bilstm_attention()

model.summary()


# ============================================================
# Output Paths
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
# Start Training
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
# Save History
# ============================================================

history_dict = {
    key: [
        float(value)
        for value in values
    ]
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

val_accuracy_history = history.history[
    "val_accuracy"
]

best_epoch_index = int(
    np.argmax(
        val_accuracy_history
    )
)

best_epoch = best_epoch_index + 1

best_val_accuracy = float(
    val_accuracy_history[
        best_epoch_index
    ]
)

train_accuracy_at_best = float(
    history.history["accuracy"][
        best_epoch_index
    ]
)

train_loss_at_best = float(
    history.history["loss"][
        best_epoch_index
    ]
)

val_loss_at_best = float(
    history.history["val_loss"][
        best_epoch_index
    ]
)

train_val_gap = (
    train_accuracy_at_best -
    best_val_accuracy
)


# ============================================================
# Save Best Validation Result
# ============================================================

best_result = {
    "model": (
        "CNN + SE Channel Attention + "
        "BiLSTM + Multi-Head Self-Attention + SpecAugment"
    ),
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
    "input_shape": [64, 174],
    "num_classes": 6,
    "specaugment": True,
    "frequency_mask": 8,
    "time_mask": 18,
    "test_used_during_training": False
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
# Final Training Summary
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
    "The test set was NOT used during training "
    "or model selection."
)

print("\nTraining completed successfully.")