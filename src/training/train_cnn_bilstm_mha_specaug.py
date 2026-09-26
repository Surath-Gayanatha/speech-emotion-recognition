"""
CNN + BiLSTM + Multi-Head Attention + SpecAugment
Experiment 1:
Reduced SpecAugment strength
Frequency Mask = 4
Time Mask = 10

IMPORTANT:
This experiment uses a SEPARATE results directory.
The original baseline result:
results/cnn_bilstm_mha_specaug
is NOT overwritten.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "data/processed/logmel"

# IMPORTANT:
# Separate experiment folder.
# Baseline 59.89% remains untouched.
MODEL_DIR = "models/cnn_bilstm_mha_specaug_exp1_4_10"
RESULTS_DIR = "results/cnn_bilstm_mha_specaug_exp1_4_10"

BATCH_SIZE = 32
EPOCHS = 60

INPUT_SHAPE = (64, 174, 1)
NUM_CLASSES = 6

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# ============================================================
# EXPERIMENT 1 - REDUCED SPECAUGMENT
# ============================================================

FREQ_MASK_PARAM = 4
TIME_MASK_PARAM = 10


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
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]

if X_val.ndim == 3:
    X_val = X_val[..., np.newaxis]

if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# SPEC AUGMENTATION
# ============================================================

def spec_augment(
    x,
    freq_mask_param=FREQ_MASK_PARAM,
    time_mask_param=TIME_MASK_PARAM
):
    """
    Apply frequency masking and time masking.

    This augmentation is applied ONLY to training data.
    Validation and test data remain unchanged.
    """

    # --------------------------------------------------------
    # Frequency masking
    # --------------------------------------------------------

    freq_mask = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=freq_mask_param + 1,
        dtype=tf.int32
    )

    freq_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=tf.maximum(
            1,
            tf.shape(x)[0] - freq_mask + 1
        ),
        dtype=tf.int32
    )

    freq_indices = tf.range(
        tf.shape(x)[0]
    )

    freq_mask_bool = tf.logical_and(
        freq_indices >= freq_start,
        freq_indices < freq_start + freq_mask
    )

    freq_mask_bool = tf.reshape(
        freq_mask_bool,
        [-1, 1, 1]
    )

    x = tf.where(
        freq_mask_bool,
        tf.zeros_like(x),
        x
    )

    # --------------------------------------------------------
    # Time masking
    # --------------------------------------------------------

    time_mask = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=time_mask_param + 1,
        dtype=tf.int32
    )

    time_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=tf.maximum(
            1,
            tf.shape(x)[1] - time_mask + 1
        ),
        dtype=tf.int32
    )

    time_indices = tf.range(
        tf.shape(x)[1]
    )

    time_mask_bool = tf.logical_and(
        time_indices >= time_start,
        time_indices < time_start + time_mask
    )

    time_mask_bool = tf.reshape(
        time_mask_bool,
        [1, -1, 1]
    )

    x = tf.where(
        time_mask_bool,
        tf.zeros_like(x),
        x
    )

    return x


# ============================================================
# DATASET CREATION
# ============================================================

print("\nCreating TensorFlow datasets...")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def training_augmentation(x, y):

    x = spec_augment(
        x,
        freq_mask_param=FREQ_MASK_PARAM,
        time_mask_param=TIME_MASK_PARAM
    )

    return x, y


train_dataset = (
    train_dataset
    .shuffle(
        buffer_size=len(X_train),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        training_augmentation,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# VALIDATION DATASET
# ============================================================

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# MODEL
# ============================================================

def build_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
):

    inputs = layers.Input(
        shape=input_shape,
        name="logmel_input"
    )

    # ========================================================
    # CNN BLOCK 1
    # ========================================================

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.15)(x)


    # ========================================================
    # CNN BLOCK 2
    # ========================================================

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.20)(x)


    # ========================================================
    # CNN BLOCK 3
    # ========================================================

    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)


    # ========================================================
    # CONVERT CNN FEATURES TO SEQUENCE
    # ========================================================

    x = layers.Permute(
        (2, 1, 3)
    )(x)

    shape = x.shape

    time_steps = shape[1]
    feature_dim = shape[2] * shape[3]

    x = layers.Reshape(
        (time_steps, feature_dim)
    )(x)


    # ========================================================
    # BiLSTM
    # ========================================================

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.20
        )
    )(x)


    # ========================================================
    # MULTI-HEAD ATTENTION
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32,
        dropout=0.10
    )(
        x,
        x
    )

    # Residual connection
    x = layers.Add()(
        [x, attention_output]
    )

    x = layers.LayerNormalization()(x)


    # ========================================================
    # FEED FORWARD NETWORK
    # ========================================================

    ffn = layers.Dense(
        256,
        activation=tf.keras.activations.gelu
    )(x)

    ffn = layers.Dropout(
        0.20
    )(ffn)

    ffn = layers.Dense(
        256
    )(ffn)

    # Residual connection
    x = layers.Add()(
        [x, ffn]
    )

    x = layers.LayerNormalization()(x)


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    x = layers.GlobalAveragePooling1D()(x)


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.40
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)


    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiLSTM_MHA_SpecAugment_EXP1"
    )

    return model


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = build_model()

model.summary()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.AdamW(
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

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)


checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# SAVE EXPERIMENT CONFIG
# ============================================================

config = {
    "experiment": "EXP1_REDUCED_SPECAUGMENT",

    "seed": SEED,

    "input_shape": list(INPUT_SHAPE),

    "num_classes": NUM_CLASSES,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "optimizer": "AdamW",

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "specaugment": True,

    "frequency_mask": FREQ_MASK_PARAM,

    "time_mask": TIME_MASK_PARAM,

    "architecture": (
        "CNN32-64-128 + BiLSTM128 + "
        "MultiHeadAttention8 + FFN + GAP + Dense128"
    ),

    "dataset": "CREMA-D",

    "representation": "Log-Mel Spectrogram",

    "test_used_during_training": False
}


with open(
    os.path.join(
        RESULTS_DIR,
        "config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING EXPERIMENT 1")
print("=" * 70)

print("Frequency Mask :", FREQ_MASK_PARAM)
print("Time Mask      :", TIME_MASK_PARAM)
print("Results Folder :", RESULTS_DIR)
print("Model Folder   :", MODEL_DIR)

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
# SAVE HISTORY
# ============================================================

history_dict = history.history

with open(
    os.path.join(
        RESULTS_DIR,
        "history.json"
    ),
    "w"
) as f:

    json.dump(
        {
            key: [
                float(v)
                for v in values
            ]
            for key, values
            in history_dict.items()
        },
        f,
        indent=4
    )


# ============================================================
# BEST VALIDATION EPOCH
# ============================================================

val_accuracies = history_dict["val_accuracy"]

best_epoch = int(
    np.argmax(val_accuracies)
)

best_val_accuracy = float(
    val_accuracies[best_epoch]
)

best_train_accuracy = float(
    history_dict["accuracy"][best_epoch]
)

best_train_loss = float(
    history_dict["loss"][best_epoch]
)

best_val_loss = float(
    history_dict["val_loss"][best_epoch]
)

train_val_gap = (
    best_train_accuracy -
    best_val_accuracy
)


# ============================================================
# PRINT TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Epoch          : {best_epoch + 1}"
)

print(
    f"Train Accuracy      : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Train Loss          : "
    f"{best_train_loss:.4f}"
)

print(
    f"Validation Loss     : "
    f"{best_val_loss:.4f}"
)

print(
    f"Train-Val Gap       : "
    f"{train_val_gap * 100:.2f} percentage points"
)


# ============================================================
# SAVE BEST VALIDATION RESULT
# ============================================================

best_validation_result = {

    "best_epoch": best_epoch + 1,

    "train_accuracy": best_train_accuracy,

    "validation_accuracy": best_val_accuracy,

    "train_loss": best_train_loss,

    "validation_loss": best_val_loss,

    "train_val_gap": train_val_gap,

    "frequency_mask": FREQ_MASK_PARAM,

    "time_mask": TIME_MASK_PARAM,

    "test_used_during_training": False
}


with open(
    os.path.join(
        RESULTS_DIR,
        "best_validation_result.json"
    ),
    "w"
) as f:

    json.dump(
        best_validation_result,
        f,
        indent=4
    )


# ============================================================
# IMPORTANT
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "Test set was NOT used during training or model selection."
)

print(
    "The original baseline results/cnn_bilstm_mha_specaug "
    "folder was NOT overwritten."
)

print(
    f"Experiment results saved to: {RESULTS_DIR}"
)

print(
    f"Best model saved to: {checkpoint_path}"
)

print("\nDo NOT run test evaluation yet.")
print("First compare this experiment's validation accuracy")
print("against the baseline validation accuracy of 62.76%.")