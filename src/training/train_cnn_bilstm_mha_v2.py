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

from src.models.cnn_bilstm_mha_v2 import build_model


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 60

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

MODEL_DIR = "models/cnn_bilstm_mha_v2"
RESULT_DIR = "results/cnn_bilstm_mha_v2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATA
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
# CHANNEL DIMENSION
# ============================================================

if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]

if X_val.ndim == 3:
    X_val = X_val[..., np.newaxis]


print("=" * 70)
print("CNN + BiLSTM + MHA V2")
print("=" * 70)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)


# ============================================================
# SPEC AUGMENTATION
# ============================================================

def spec_augment(x):
    x = tf.identity(x)

    # Frequency masking
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

    mask_freq = tf.concat([
        tf.ones((freq_start, tf.shape(x)[1], 1)),
        tf.zeros((freq_mask, tf.shape(x)[1], 1)),
        tf.ones((
            tf.shape(x)[0] - freq_start - freq_mask,
            tf.shape(x)[1],
            1
        ))
    ], axis=0)

    x = x * mask_freq

    # Time masking
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

    mask_time = tf.concat([
        tf.ones((tf.shape(x)[0], time_start, 1)),
        tf.zeros((tf.shape(x)[0], time_mask, 1)),
        tf.ones((
            tf.shape(x)[0],
            tf.shape(x)[1] - time_start - time_mask,
            1
        ))
    ], axis=1)

    x = x * mask_time

    return x


# ============================================================
# DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_ds = train_ds.shuffle(
    len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    lambda x, y: (spec_augment(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_ds = val_ds.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# MODEL
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


model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    os.path.join(
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


print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

best_epoch = int(
    np.argmax(
        history.history["val_accuracy"]
    )
) + 1

best_val = max(
    history.history["val_accuracy"]
)

print("Best Epoch:", best_epoch)
print(
    "Best Validation Accuracy:",
    f"{best_val * 100:.2f}%"
)