"""
Training Script
Advanced Acoustic Conformer Transformer
Speech Emotion Recognition - CREMA-D

Input:
    Log-Mel Spectrogram
    Original shape: (samples, 64, 174)
    Transformer shape: (samples, 174, 64)

Important:
    - Train / validation are used for model development
    - Test set is NOT used during training
    - Log-Mel features are already normalized using training statistics
"""

import os
import json
import random

import numpy as np
import tensorflow as tf

from src.models.transformer import build_transformer


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/logmel"

RESULTS_DIR = "results/advanced_transformer"

MODEL_DIR = "models/advanced_transformer"

EPOCHS = 60
BATCH_SIZE = 32

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6

EMOTION_NAMES = [
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD"
]


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
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


print("Original shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# 4. DATA TYPES
# ============================================================

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)


# ============================================================
# 5. TRANSPOSE FOR TEMPORAL TRANSFORMER
# ============================================================

# Original:
# (samples, mel_bins, time_frames)
#
# Transformer:
# (samples, time_frames, mel_bins)

X_train = np.transpose(
    X_train,
    (0, 2, 1)
)

X_val = np.transpose(
    X_val,
    (0, 2, 1)
)


print("\nTransformer input shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# ============================================================
# 6. CREATE TF.DATA DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)


train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 7. BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING ADVANCED ACOUSTIC CONFORMER TRANSFORMER")
print("=" * 70)


model = build_transformer(
    input_shape=(174, 64),
    num_classes=NUM_CLASSES,
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    conv_kernel_size=5,
    dropout=0.15,
    head_dropout=0.25
)


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 9. COMPILE
# ============================================================

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 11. CALLBACKS
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
    patience=8,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 12. SAVE CONFIGURATION
# ============================================================

config = {
    "model": "Advanced Acoustic Conformer Transformer",

    "input_shape": [174, 64],

    "num_classes": NUM_CLASSES,

    "embed_dim": 128,

    "num_heads": 4,

    "ff_dim": 256,

    "num_layers": 2,

    "conv_kernel_size": 5,

    "dropout": 0.15,

    "head_dropout": 0.25,

    "optimizer": "AdamW",

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "seed": SEED,

    "dataset": "CREMA-D",

    "feature": "Log-Mel Spectrogram",

    "augmentation": "None"
}


with open(
    config_path,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Seed:", SEED)


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
# 14. SAVE HISTORY
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
    "w"
) as f:

    json.dump(
        history_dict,
        f,
        indent=4
    )


# ============================================================
# 15. BEST VALIDATION RESULT
# ============================================================

best_epoch = int(
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)

best_val_accuracy = float(
    np.max(
        history.history["val_accuracy"]
    )
)

best_train_accuracy = float(
    history.history["accuracy"][best_epoch - 1]
)

best_val_loss = float(
    history.history["val_loss"][best_epoch - 1]
)

best_train_loss = float(
    history.history["loss"][best_epoch - 1]
)

gap = (
    best_train_accuracy
    -
    best_val_accuracy
)


best_result = {

    "best_epoch": best_epoch,

    "train_accuracy": best_train_accuracy,

    "validation_accuracy": best_val_accuracy,

    "train_loss": best_train_loss,

    "validation_loss": best_val_loss,

    "train_validation_gap": gap

}


best_result_path = os.path.join(
    RESULTS_DIR,
    "best_validation_result.json"
)


with open(
    best_result_path,
    "w"
) as f:

    json.dump(
        best_result,
        f,
        indent=4
    )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Epoch           : {best_epoch}"
)

print(
    f"Train Accuracy       : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy  : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Train Loss           : "
    f"{best_train_loss:.4f}"
)

print(
    f"Validation Loss      : "
    f"{best_val_loss:.4f}"
)

print(
    f"Train-Val Gap        : "
    f"{gap * 100:.2f} percentage points"
)

print("\nBest model:")
print(checkpoint_path)

print("\nHistory:")
print(history_path)