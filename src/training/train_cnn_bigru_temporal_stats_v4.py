"""
V4: CNN + BiGRU + Multi-Head Attention
    + Temporal Attention Pooling
    + Temporal Statistics Pooling

Training configuration:
- Log-Mel spectrogram features
- Actor-independent train/validation split
- SpecAugment
- AdamW
- ReduceLROnPlateau
- EarlyStopping
- Best checkpoint based on validation accuracy
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from src.models.cnn_bigru_temporal_stats_v4 import build_model


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 60

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6

# Log-Mel input
N_MELS = 64
TIME_FRAMES = 174


# ============================================================
# PATHS
# ============================================================

TRAIN_X = "data/processed/logmel/train_features.npy"
TRAIN_Y = "data/processed/logmel/train_labels.npy"

VAL_X = "data/processed/logmel/validation_features.npy"
VAL_Y = "data/processed/logmel/validation_labels.npy"

MODEL_DIR = "models/cnn_bigru_temporal_stats_v4"
RESULTS_DIR = "results/cnn_bigru_temporal_stats_v4"

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")

HISTORY_PATH = os.path.join(RESULTS_DIR, "history.npz")


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("CNN + BiGRU + TEMPORAL STATISTICS V4")
print("=" * 70)

print(f"Seed           : {SEED}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Epochs         : {EPOCHS}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"Weight decay   : {WEIGHT_DECAY}")
print(f"Input shape    : ({N_MELS}, {TIME_FRAMES}, 1)")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

X_train = np.load(TRAIN_X)
y_train = np.load(TRAIN_Y)

X_val = np.load(VAL_X)
y_val = np.load(VAL_Y)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape  : {X_val.shape}")
print(f"y_val shape  : {y_val.shape}")


# ============================================================
# PREPARE INPUT SHAPE
# ============================================================

# Existing Log-Mel features are expected to be:
# (samples, 64, 174)
#
# CNN expects:
# (samples, 64, 174, 1)

if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]

if X_val.ndim == 3:
    X_val = X_val[..., np.newaxis]

print(f"\nFinal X_train shape: {X_train.shape}")
print(f"Final X_val shape  : {X_val.shape}")


# ============================================================
# LABEL HANDLING
# ============================================================

print("\nUnique training labels:", np.unique(y_train))
print("Unique validation labels:", np.unique(y_val))

# Convert labels to integer if necessary
y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)


# ============================================================
# SPEC AUGMENTATION
# ============================================================

def spec_augment(
    spectrogram,
    max_freq_mask=8,
    max_time_mask=18
):
    """
    Apply SpecAugment to one Log-Mel spectrogram.

    Input:
        (64, 174, 1)

    Returns:
        Augmented spectrogram with same shape.
    """

    x = spectrogram.copy()

    n_mels = x.shape[0]
    n_frames = x.shape[1]

    # --------------------------------------------------------
    # Frequency masking
    # --------------------------------------------------------

    freq_width = np.random.randint(
        0,
        min(max_freq_mask, n_mels) + 1
    )

    if freq_width > 0:
        freq_start = np.random.randint(
            0,
            n_mels - freq_width + 1
        )

        x[
            freq_start:freq_start + freq_width,
            :,
            :
        ] = 0.0

    # --------------------------------------------------------
    # Time masking
    # --------------------------------------------------------

    time_width = np.random.randint(
        0,
        min(max_time_mask, n_frames) + 1
    )

    if time_width > 0:
        time_start = np.random.randint(
            0,
            n_frames - time_width + 1
        )

        x[
            :,
            time_start:time_start + time_width,
            :
        ] = 0.0

    return x


# ============================================================
# TF DATASET
# ============================================================

def augment_sample(x, y):
    """
    Apply SpecAugment only to training samples.
    """

    x = tf.numpy_function(
        func=spec_augment,
        inp=[x],
        Tout=tf.float32
    )

    x.set_shape((N_MELS, TIME_FRAMES, 1))

    return x, y


# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train.astype(np.float32), y_train)
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    augment_sample,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# Validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val.astype(np.float32), y_val)
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING V4 MODEL")
print("=" * 70)

model = build_model(
    input_shape=(N_MELS, TIME_FRAMES, 1),
    num_classes=NUM_CLASSES
)

model.summary()

print("\nTotal parameters:", model.count_params())


# ============================================================
# OPTIMIZER
# ============================================================

try:

    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

except AttributeError:

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=8,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAINING
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
# SAVE FINAL MODEL
# ============================================================

model.save(FINAL_MODEL_PATH)

print("\nFinal model saved to:")
print(FINAL_MODEL_PATH)

print("\nBest model saved to:")
print(BEST_MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.savez(
    HISTORY_PATH,
    accuracy=np.array(history.history["accuracy"]),
    val_accuracy=np.array(history.history["val_accuracy"]),
    loss=np.array(history.history["loss"]),
    val_loss=np.array(history.history["val_loss"])
)

print("\nTraining history saved to:")
print(HISTORY_PATH)


# ============================================================
# BEST EPOCH ANALYSIS
# ============================================================

train_acc = np.array(history.history["accuracy"])
val_acc = np.array(history.history["val_accuracy"])

train_loss = np.array(history.history["loss"])
val_loss = np.array(history.history["val_loss"])

best_epoch = int(np.argmax(val_acc)) + 1

best_train_acc = train_acc[best_epoch - 1]
best_val_acc = val_acc[best_epoch - 1]

best_train_loss = train_loss[best_epoch - 1]
best_val_loss = val_loss[best_epoch - 1]

train_val_gap = (
    best_train_acc - best_val_acc
) * 100


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"Total epochs trained : {len(train_acc)}")
print(f"Best epoch           : {best_epoch}")

print(
    f"Train Accuracy       : {best_train_acc * 100:.2f}%"
)

print(
    f"Validation Accuracy  : {best_val_acc * 100:.2f}%"
)

print(
    f"Train Loss           : {best_train_loss:.4f}"
)

print(
    f"Validation Loss      : {best_val_loss:.4f}"
)

print(
    f"Train-Val Gap        : {train_val_gap:.2f} percentage points"
)

print("=" * 70)