"""
CNN + Transformer + Multi-Head Attention
Training Pipeline for CREMA-D Speech Emotion Recognition.

Important experimental controls:
- Actor-independent CREMA-D split is preserved.
- Log-Mel features are already normalized using training data only.
- No second normalization is applied.
- Validation data is used for model selection.
- Test data is NOT used during training or model selection.
- Random seed is fixed for reproducibility.
"""

import os
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.cnn_transformer_attention import (
    build_cnn_transformer_attention
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "data/processed/logmel"

MODEL_DIR = "models/cnn_transformer_attention"
RESULT_DIR = "results/cnn_transformer_attention"

BATCH_SIZE = 32
EPOCHS = 60

NUM_CLASSES = 6
INPUT_SHAPE = (64, 174, 1)

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING LOG-MEL DATA")
print("=" * 70)

X_train = np.load(
    os.path.join(DATA_DIR, "train_features.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "train_labels.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "validation_features.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "validation_labels.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "test_features.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "test_labels.npy")
)


print(f"Train features : {X_train.shape}")
print(f"Train labels   : {y_train.shape}")
print(f"Validation     : {X_val.shape}")
print(f"Val labels     : {y_val.shape}")
print(f"Test features  : {X_test.shape}")
print(f"Test labels    : {y_test.shape}")


# ============================================================
# DATA VALIDATION
# ============================================================

assert X_train.shape == (5147, 64, 174)
assert X_val.shape == (1066, 64, 174)
assert X_test.shape == (1229, 64, 174)

assert y_train.shape == (5147,)
assert y_val.shape == (1066,)
assert y_test.shape == (1229,)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

assert y_train.min() >= 0
assert y_train.max() < NUM_CLASSES

assert y_val.min() >= 0
assert y_val.max() < NUM_CLASSES

assert y_test.min() >= 0
assert y_test.max() < NUM_CLASSES

print("\nData validation: PASSED")


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"\nFinal train shape : {X_train.shape}")
print(f"Final val shape   : {X_val.shape}")
print(f"Final test shape  : {X_test.shape}")


# ============================================================
# NO SECOND NORMALIZATION
# ============================================================

print(
    "\nUsing existing Log-Mel normalization."
)

print(
    "Normalization was calculated from training data only."
)

print(
    "No second normalization is applied during training."
)


# ============================================================
# DATASETS
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_ds = train_ds.shuffle(
    buffer_size=len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    AUTOTUNE
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CNN + TRANSFORMER + ATTENTION MODEL")
print("=" * 70)

model = build_cnn_transformer_attention(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = keras.optimizers.AdamW(
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

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

history_path = os.path.join(
    RESULT_DIR,
    "history.json"
)

config_path = os.path.join(
    RESULT_DIR,
    "config.json"
)

best_result_path = os.path.join(
    RESULT_DIR,
    "best_validation_result.json"
)


checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "dataset": "CREMA-D",

    "input_representation": "Log-Mel Spectrogram",

    "input_shape": [64, 174, 1],

    "train_samples": 5147,
    "validation_samples": 1066,
    "test_samples": 1229,

    "num_classes": 6,

    "architecture": (
        "CNN + Transformer Encoder x2 "
        "+ Multi-Head Attention"
    ),

    "cnn_filters": [32, 64, 128],

    "transformer_blocks": 2,

    "transformer_embedding_dimension": 256,

    "transformer_attention_heads": 8,

    "transformer_feed_forward_dimension": 512,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "optimizer": "AdamW",

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "loss": "Sparse Categorical Crossentropy",

    "seed": SEED,

    "normalization": (
        "Precomputed training-data-only "
        "per-Mel-band normalization"
    ),

    "test_usage": (
        "Test set excluded from training, "
        "hyperparameter tuning and model selection"
    ),

    "early_stopping_patience": 10,

    "reduce_lr_patience": 4,

    "reduce_lr_factor": 0.5,

    "minimum_learning_rate": 1e-6
}


with open(config_path, "w") as f:
    json.dump(
        config,
        f,
        indent=2
    )


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        reduce_lr,
        early_stopping
    ],
    verbose=1
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_dict = {}

for key, values in history.history.items():
    history_dict[key] = [
        float(value)
        for value in values
    ]


with open(history_path, "w") as f:
    json.dump(
        history_dict,
        f,
        indent=2
    )


# ============================================================
# BEST VALIDATION EPOCH
# ============================================================

best_epoch_index = int(
    np.argmax(
        history_dict["val_accuracy"]
    )
)

best_epoch = best_epoch_index + 1

best_train_accuracy = (
    history_dict["accuracy"][best_epoch_index]
)

best_val_accuracy = (
    history_dict["val_accuracy"][best_epoch_index]
)

best_train_loss = (
    history_dict["loss"][best_epoch_index]
)

best_val_loss = (
    history_dict["val_loss"][best_epoch_index]
)

train_val_gap = (
    best_train_accuracy -
    best_val_accuracy
)


best_result = {
    "best_epoch": best_epoch,

    "train_accuracy": float(
        best_train_accuracy
    ),

    "validation_accuracy": float(
        best_val_accuracy
    ),

    "train_loss": float(
        best_train_loss
    ),

    "validation_loss": float(
        best_val_loss
    ),

    "train_validation_gap": float(
        train_val_gap
    )
}


with open(
    best_result_path,
    "w"
) as f:

    json.dump(
        best_result,
        f,
        indent=2
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

model.save(
    final_model_path
)


# ============================================================
# FINAL TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Epoch       : {best_epoch}"
)

print(
    f"Train Accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Acc.  : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Train Loss       : "
    f"{best_train_loss:.4f}"
)

print(
    f"Validation Loss  : "
    f"{best_val_loss:.4f}"
)

print(
    f"Train-Val Gap    : "
    f"{train_val_gap * 100:.2f} percentage points"
)

print(
    "\nBest checkpoint:"
)

print(
    checkpoint_path
)

print(
    "\nTraining history:"
)

print(
    history_path
)

print(
    "\nTest set was NOT used for training "
    "or model selection."
)

print(
    "\nUse the best checkpoint for the "
    "final unseen test evaluation."
)