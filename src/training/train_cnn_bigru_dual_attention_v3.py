import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

from tensorflow.keras.optimizers import AdamW

from src.models.cnn_bigru_dual_attention_v3 import build_model


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 60

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

MODEL_DIR = "models/cnn_bigru_dual_attention_v3"
RESULT_DIR = "results/cnn_bigru_dual_attention_v3"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD LOG-MEL DATA
# ============================================================

X_train = np.load(
    "data/processed/logmel/train_features.npy"
)

y_train = np.load(
    "data/processed/logmel/train_labels.npy"
)

X_val = np.load(
    "data/processed/logmel/validation_features.npy"
)

y_val = np.load(
    "data/processed/logmel/validation_labels.npy"
)


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]

if X_val.ndim == 3:
    X_val = X_val[..., np.newaxis]


print("=" * 70)
print("CNN + BiGRU + DUAL ATTENTION V3")
print("=" * 70)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# ============================================================
# SPEC AUGMENTATION
# ============================================================

def spec_augment(x):

    x = tf.identity(x)

    # --------------------------------------------------------
    # Frequency masking
    # --------------------------------------------------------

    freq_mask = tf.random.uniform(
        [],
        minval=0,
        maxval=9,
        dtype=tf.int32
    )

    freq_start = tf.random.uniform(
        [],
        minval=0,
        maxval=tf.shape(x)[0] - freq_mask + 1,
        dtype=tf.int32
    )

    freq_mask_tensor = tf.concat(
        [
            tf.ones(
                (
                    freq_start,
                    tf.shape(x)[1],
                    1
                )
            ),

            tf.zeros(
                (
                    freq_mask,
                    tf.shape(x)[1],
                    1
                )
            ),

            tf.ones(
                (
                    tf.shape(x)[0]
                    - freq_start
                    - freq_mask,
                    tf.shape(x)[1],
                    1
                )
            )
        ],
        axis=0
    )

    x = x * freq_mask_tensor


    # --------------------------------------------------------
    # Time masking
    # --------------------------------------------------------

    time_mask = tf.random.uniform(
        [],
        minval=0,
        maxval=19,
        dtype=tf.int32
    )

    time_start = tf.random.uniform(
        [],
        minval=0,
        maxval=tf.shape(x)[1] - time_mask + 1,
        dtype=tf.int32
    )

    time_mask_tensor = tf.concat(
        [
            tf.ones(
                (
                    tf.shape(x)[0],
                    time_start,
                    1
                )
            ),

            tf.zeros(
                (
                    tf.shape(x)[0],
                    time_mask,
                    1
                )
            ),

            tf.ones(
                (
                    tf.shape(x)[0],
                    tf.shape(x)[1]
                    - time_start
                    - time_mask,
                    1
                )
            )
        ],
        axis=1
    )

    x = x * time_mask_tensor

    return x


# ============================================================
# TRAIN DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)

train_ds = train_ds.shuffle(
    buffer_size=len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    lambda x, y: (
        spec_augment(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# VALIDATION DATASET
# ============================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
        y_val
    )
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

model = build_model()

model.summary()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
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

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_DIR,
        "best_model.keras"
    ),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=False,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        reduce_lr,
        early_stop
    ]
)


# ============================================================
# SAVE HISTORY
# ============================================================

np.savez(
    os.path.join(
        RESULT_DIR,
        "history.npz"
    ),
    **history.history
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

best_epoch = (
    int(
        np.argmax(
            history.history["val_accuracy"]
        )
    )
    + 1
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_train_accuracy = (
    history.history["accuracy"]
    [best_epoch - 1]
)

best_train_loss = (
    history.history["loss"]
    [best_epoch - 1]
)

best_val_loss = (
    history.history["val_loss"]
    [best_epoch - 1]
)

gap = (
    best_train_accuracy
    - best_val_accuracy
)


print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Train Accuracy: "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Train Loss: "
    f"{best_train_loss:.4f}"
)

print(
    f"Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Train-Val Gap: "
    f"{gap * 100:.2f} percentage points"
)

print(
    "Best model saved to:"
)

print(
    os.path.join(
        MODEL_DIR,
        "best_model.keras"
    )
)