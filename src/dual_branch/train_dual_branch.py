"""
Dual-Branch Speech Emotion Recognition
=======================================

MFCC Branch:
    MFCC -> Conv1D -> Conv1D -> BiLSTM -> Attention

Log-Mel Branch:
    Log-Mel -> ResCNN -> SE -> Feature Projection

Fusion:
    MFCC + Log-Mel
        -> BiGRU
        -> Multi-Head Attention
        -> Global Average Pooling
        -> Dense
        -> Dropout
        -> 6-class Softmax

Dataset:
    CREMA-D

Evaluation:
    Actor-independent
    Test set NOT used during training/model selection

All model layers are trained from scratch.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)
from tensorflow.keras.optimizers import AdamW


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "data/processed/dual_branch"
MODEL_DIR = "models/dual_branch"
RESULTS_DIR = "results/dual_branch"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 60

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.40
L2_REG = 1e-4

LABEL_SMOOTHING = 0.05

NUM_CLASSES = 6

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 75)
print("DUAL-BRANCH SPEECH EMOTION RECOGNITION")
print("=" * 75)

print(f"TensorFlow version : {tf.__version__}")
print(f"Seed               : {SEED}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Epochs             : {EPOCHS}")
print(f"Learning rate      : {LEARNING_RATE}")
print(f"Weight decay       : {WEIGHT_DECAY}")
print(f"Dropout            : {DROPOUT}")
print(f"L2 regularization  : {L2_REG}")
print(f"Label smoothing    : {LABEL_SMOOTHING}")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)


def load_split(split):

    mfcc = np.load(
        os.path.join(DATA_DIR, f"{split}_mfcc.npy")
    )

    logmel = np.load(
        os.path.join(DATA_DIR, f"{split}_logmel.npy")
    )

    labels = np.load(
        os.path.join(DATA_DIR, f"{split}_labels.npy")
    )

    filenames = np.load(
        os.path.join(DATA_DIR, f"{split}_filenames.npy"),
        allow_pickle=True
    )

    return mfcc, logmel, labels, filenames


train_mfcc, train_logmel, train_labels, train_filenames = load_split(
    "train"
)

val_mfcc, val_logmel, val_labels, val_filenames = load_split(
    "validation"
)

test_mfcc, test_logmel, test_labels, test_filenames = load_split(
    "test"
)


print("\nOriginal feature shapes:")

print(f"Train MFCC    : {train_mfcc.shape}")
print(f"Train Log-Mel : {train_logmel.shape}")

print(f"Val MFCC      : {val_mfcc.shape}")
print(f"Val Log-Mel   : {val_logmel.shape}")

print(f"Test MFCC     : {test_mfcc.shape}")
print(f"Test Log-Mel  : {test_logmel.shape}")


# ============================================================
# 4. TRANSPOSE FEATURES
# ============================================================
#
# Current:
#   MFCC    = (N, 120, 174)
#   Log-Mel = (N, 64, 174)
#
# Conv1D expects:
#   (N, time, channels)
#
# Therefore:
#   MFCC    -> (N, 174, 120)
#   Log-Mel -> (N, 174, 64)
# ============================================================

print("\nTransposing features for Conv1D...")

train_mfcc = np.transpose(train_mfcc, (0, 2, 1))
val_mfcc = np.transpose(val_mfcc, (0, 2, 1))
test_mfcc = np.transpose(test_mfcc, (0, 2, 1))

train_logmel = np.transpose(train_logmel, (0, 2, 1))
val_logmel = np.transpose(val_logmel, (0, 2, 1))
test_logmel = np.transpose(test_logmel, (0, 2, 1))


print("\nAfter transpose:")

print(f"Train MFCC    : {train_mfcc.shape}")
print(f"Train Log-Mel : {train_logmel.shape}")

print(f"Val MFCC      : {val_mfcc.shape}")
print(f"Val Log-Mel   : {val_logmel.shape}")

print(f"Test MFCC     : {test_mfcc.shape}")
print(f"Test Log-Mel  : {test_logmel.shape}")


# ============================================================
# 5. CONVERT TO FLOAT32
# ============================================================

train_mfcc = train_mfcc.astype(np.float32)
val_mfcc = val_mfcc.astype(np.float32)
test_mfcc = test_mfcc.astype(np.float32)

train_logmel = train_logmel.astype(np.float32)
val_logmel = val_logmel.astype(np.float32)
test_logmel = test_logmel.astype(np.float32)


# ============================================================
# 6. ONE-HOT LABELS
# ============================================================

train_labels_cat = tf.keras.utils.to_categorical(
    train_labels,
    NUM_CLASSES
)

val_labels_cat = tf.keras.utils.to_categorical(
    val_labels,
    NUM_CLASSES
)

test_labels_cat = tf.keras.utils.to_categorical(
    test_labels,
    NUM_CLASSES
)


# ============================================================
# 7. SPEC AUGMENTATION
# ============================================================

class SpecAugment(layers.Layer):
    """
    Applies frequency and time masking.

    Training only.
    """

    def __init__(
        self,
        frequency_mask=8,
        time_mask=18,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.frequency_mask = frequency_mask
        self.time_mask = time_mask

    def call(self, inputs, training=None):

        if not training:
            return inputs

        x = inputs

        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        features = tf.shape(x)[2]

        # ----------------------------------------------------
        # Frequency masking
        # ----------------------------------------------------

        freq_width = tf.random.uniform(
            [],
            minval=0,
            maxval=self.frequency_mask + 1,
            dtype=tf.int32
        )

        freq_start_max = tf.maximum(
            features - freq_width,
            1
        )

        freq_start = tf.random.uniform(
            [],
            minval=0,
            maxval=freq_start_max,
            dtype=tf.int32
        )

        freq_indices = tf.range(features)

        freq_mask = tf.logical_and(
            freq_indices >= freq_start,
            freq_indices < freq_start + freq_width
        )

        freq_mask = tf.cast(freq_mask, x.dtype)

        freq_mask = 1.0 - freq_mask

        freq_mask = tf.reshape(
            freq_mask,
            [1, 1, -1]
        )

        x = x * freq_mask

        # ----------------------------------------------------
        # Time masking
        # ----------------------------------------------------

        time_width = tf.random.uniform(
            [],
            minval=0,
            maxval=self.time_mask + 1,
            dtype=tf.int32
        )

        time_start_max = tf.maximum(
            time_steps - time_width,
            1
        )

        time_start = tf.random.uniform(
            [],
            minval=0,
            maxval=time_start_max,
            dtype=tf.int32
        )

        time_indices = tf.range(time_steps)

        time_mask_tensor = tf.logical_and(
            time_indices >= time_start,
            time_indices < time_start + time_width
        )

        time_mask_tensor = tf.cast(
            time_mask_tensor,
            x.dtype
        )

        time_mask_tensor = 1.0 - time_mask_tensor

        time_mask_tensor = tf.reshape(
            time_mask_tensor,
            [1, -1, 1]
        )

        x = x * time_mask_tensor

        return x


# ============================================================
# 8. SE BLOCK
# ============================================================

def se_block(
    x,
    reduction=8,
    name="se"
):

    channels = x.shape[-1]

    if channels is None:
        raise ValueError("Channel dimension must be defined.")

    se = layers.GlobalAveragePooling1D(
        name=f"{name}_gap"
    )(x)

    se = layers.Dense(
        max(channels // reduction, 8),
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f"{name}_dense1"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{name}_dense2"
    )(se)

    se = layers.Reshape(
        (1, channels),
        name=f"{name}_reshape"
    )(se)

    x = layers.Multiply(
        name=f"{name}_scale"
    )([x, se])

    return x


# ============================================================
# 9. RESIDUAL CNN BLOCK
# ============================================================

def residual_cnn_block(
    x,
    filters,
    kernel_size=3,
    dropout=0.2,
    name="res"
):

    shortcut = x

    # --------------------------------------------------------
    # First convolution
    # --------------------------------------------------------

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f"{name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.Activation(
        "relu",
        name=f"{name}_relu1"
    )(x)

    x = layers.Dropout(
        dropout,
        name=f"{name}_drop1"
    )(x)

    # --------------------------------------------------------
    # Second convolution
    # --------------------------------------------------------

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f"{name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    # --------------------------------------------------------
    # Projection shortcut if needed
    # --------------------------------------------------------

    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv1D(
            filters,
            1,
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f"{name}_shortcut"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_shortcut_bn"
        )(shortcut)

    # --------------------------------------------------------
    # Residual addition
    # --------------------------------------------------------

    x = layers.Add(
        name=f"{name}_add"
    )([x, shortcut])

    x = layers.Activation(
        "relu",
        name=f"{name}_relu2"
    )(x)

    return x


# ============================================================
# 10. MFCC BRANCH
# ============================================================

def build_mfcc_branch(inputs):

    x = SpecAugment(
        frequency_mask=8,
        time_mask=18,
        name="mfcc_specaugment"
    )(inputs)

    # --------------------------------------------------------
    # CNN block 1
    # --------------------------------------------------------

    x = layers.Conv1D(
        64,
        5,
        padding="same",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="mfcc_conv1"
    )(x)

    x = layers.BatchNormalization(
        name="mfcc_bn1"
    )(x)

    x = layers.Activation(
        "relu",
        name="mfcc_relu1"
    )(x)

    x = layers.MaxPooling1D(
        2,
        name="mfcc_pool1"
    )(x)

    x = layers.Dropout(
        0.20,
        name="mfcc_dropout1"
    )(x)

    # --------------------------------------------------------
    # CNN block 2
    # --------------------------------------------------------

    x = layers.Conv1D(
        96,
        3,
        padding="same",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="mfcc_conv2"
    )(x)

    x = layers.BatchNormalization(
        name="mfcc_bn2"
    )(x)

    x = layers.Activation(
        "relu",
        name="mfcc_relu2"
    )(x)

    x = layers.MaxPooling1D(
        2,
        name="mfcc_pool2"
    )(x)

    x = layers.Dropout(
        0.20,
        name="mfcc_dropout2"
    )(x)

    # --------------------------------------------------------
    # BiLSTM
    # --------------------------------------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            96,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(L2_REG)
        ),
        name="mfcc_bilstm"
    )(x)

    x = layers.LayerNormalization(
        name="mfcc_lstm_norm"
    )(x)

    # --------------------------------------------------------
    # Self attention
    # --------------------------------------------------------

    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.15,
        name="mfcc_attention"
    )

    attn_output = attention(
        x,
        x
    )

    x = layers.Add(
        name="mfcc_attention_residual"
    )([x, attn_output])

    x = layers.LayerNormalization(
        name="mfcc_attention_norm"
    )(x)

    return x


# ============================================================
# 11. LOG-MEL BRANCH
# ============================================================

def build_logmel_branch(inputs):

    x = SpecAugment(
        frequency_mask=6,
        time_mask=18,
        name="logmel_specaugment"
    )(inputs)

    # --------------------------------------------------------
    # Initial projection
    # --------------------------------------------------------

    x = layers.Conv1D(
        64,
        5,
        padding="same",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="logmel_initial_conv"
    )(x)

    x = layers.BatchNormalization(
        name="logmel_initial_bn"
    )(x)

    x = layers.Activation(
        "relu",
        name="logmel_initial_relu"
    )(x)

    # --------------------------------------------------------
    # Residual block 1
    # --------------------------------------------------------

    x = residual_cnn_block(
        x,
        filters=64,
        dropout=0.15,
        name="logmel_res1"
    )

    x = layers.MaxPooling1D(
        2,
        name="logmel_pool1"
    )(x)

    # --------------------------------------------------------
    # Residual block 2
    # --------------------------------------------------------

    x = residual_cnn_block(
        x,
        filters=96,
        dropout=0.15,
        name="logmel_res2"
    )

    x = layers.MaxPooling1D(
        2,
        name="logmel_pool2"
    )(x)

    # --------------------------------------------------------
    # Residual block 3
    # --------------------------------------------------------

    x = residual_cnn_block(
        x,
        filters=128,
        dropout=0.20,
        name="logmel_res3"
    )

    # --------------------------------------------------------
    # SE attention
    # --------------------------------------------------------

    x = se_block(
        x,
        reduction=8,
        name="logmel_se"
    )

    x = layers.LayerNormalization(
        name="logmel_norm"
    )(x)

    return x


# ============================================================
# 12. BUILD COMPLETE MODEL
# ============================================================

def build_dual_branch_model():

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    mfcc_input = layers.Input(
        shape=(174, 120),
        name="mfcc_input"
    )

    logmel_input = layers.Input(
        shape=(174, 64),
        name="logmel_input"
    )

    # --------------------------------------------------------
    # Branches
    # --------------------------------------------------------

    mfcc_features = build_mfcc_branch(
        mfcc_input
    )

    logmel_features = build_logmel_branch(
        logmel_input
    )

    # --------------------------------------------------------
    # Match sequence lengths
    #
    # MFCC branch after 2 pooling layers:
    # 174 -> 87 -> 43
    #
    # Log-Mel branch after 2 pooling layers:
    # 174 -> 87 -> 43
    #
    # Therefore both have same time dimension.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Feature fusion
    # --------------------------------------------------------

    fused = layers.Concatenate(
        axis=-1,
        name="feature_fusion"
    )([
        mfcc_features,
        logmel_features
    ])

    fused = layers.LayerNormalization(
        name="fusion_norm"
    )(fused)

    fused = layers.Dropout(
        0.25,
        name="fusion_dropout"
    )(fused)

    # --------------------------------------------------------
    # BiGRU
    # --------------------------------------------------------

    fused = layers.Bidirectional(
        layers.GRU(
            128,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(L2_REG)
        ),
        name="fusion_bigru"
    )(fused)

    fused = layers.LayerNormalization(
        name="bigru_norm"
    )(fused)

    # --------------------------------------------------------
    # Multi-Head Attention
    # --------------------------------------------------------

    mha = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.15,
        name="fusion_mha"
    )

    attention_output = mha(
        fused,
        fused
    )

    fused = layers.Add(
        name="mha_residual"
    )([
        fused,
        attention_output
    ])

    fused = layers.LayerNormalization(
        name="mha_norm"
    )(fused)

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    pooled = layers.GlobalAveragePooling1D(
        name="global_average_pool"
    )(fused)

    # --------------------------------------------------------
    # Dense classifier
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_128"
    )(pooled)

    x = layers.BatchNormalization(
        name="classifier_bn"
    )(x)

    x = layers.Dropout(
        DROPOUT,
        name="classifier_dropout"
    )(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.25,
        name="classifier_dropout2"
    )(x)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = Model(
        inputs=[
            mfcc_input,
            logmel_input
        ],
        outputs=output,
        name="DualBranch_MFCC_LogMel_SER"
    )

    return model


# ============================================================
# 13. BUILD MODEL
# ============================================================

print("\n" + "=" * 75)
print("BUILDING MODEL")
print("=" * 75)

model = build_dual_branch_model()

model.summary()


# ============================================================
# 14. COMPILE
# ============================================================

optimizer = AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

loss = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=LABEL_SMOOTHING
)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(
            name="accuracy"
        )
    ]
)


# ============================================================
# 15. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

history_path = os.path.join(
    RESULTS_DIR,
    "history.json"
)

csv_log_path = os.path.join(
    RESULTS_DIR,
    "training_log.csv"
)

callbacks = [

    ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    CSVLogger(
        csv_log_path,
        append=False
    )
]


# ============================================================
# 16. TRAINING DATASET
# ============================================================

print("\n" + "=" * 75)
print("CREATING TF.DATA DATASETS")
print("=" * 75)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {
            "mfcc_input": train_mfcc,
            "logmel_input": train_logmel
        },
        train_labels_cat
    )
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_labels_cat),
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
        {
            "mfcc_input": val_mfcc,
            "logmel_input": val_logmel
        },
        val_labels_cat
    )
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 17. SAVE CONFIG
# ============================================================

config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "l2_regularization": L2_REG,
    "label_smoothing": LABEL_SMOOTHING,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "mfcc_input_shape": [174, 120],
    "logmel_input_shape": [174, 64],
    "optimizer": "AdamW",
    "specaugment": True,
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 8
    },
    "reduce_lr": {
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-6
    },
    "test_used_during_training": False,
    "architecture": (
        "MFCC CNN + BiLSTM + Attention + "
        "LogMel ResCNN + SE + "
        "Feature Fusion + BiGRU + MultiHeadAttention"
    )
}

config_path = os.path.join(
    RESULTS_DIR,
    "config.json"
)

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
# 18. TRAIN
# ============================================================

print("\n" + "=" * 75)
print("STARTING TRAINING")
print("=" * 75)

print("\nIMPORTANT:")
print("Test data is NOT used during training.")
print("Only training + validation data are used.")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 19. SAVE HISTORY
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
# 20. FIND BEST EPOCH
# ============================================================

val_accuracy_history = history.history.get(
    "val_accuracy",
    []
)

val_loss_history = history.history.get(
    "val_loss",
    []
)

train_accuracy_history = history.history.get(
    "accuracy",
    []
)

train_loss_history = history.history.get(
    "loss",
    []
)

if val_loss_history:

    best_epoch = int(
        np.argmin(val_loss_history)
    ) + 1

    best_val_loss = float(
        val_loss_history[best_epoch - 1]
    )

    best_val_accuracy = float(
        val_accuracy_history[best_epoch - 1]
    )

    best_train_accuracy = float(
        train_accuracy_history[best_epoch - 1]
    )

    best_train_loss = float(
        train_loss_history[best_epoch - 1]
    )

    train_val_gap = (
        best_train_accuracy -
        best_val_accuracy
    ) * 100

else:

    best_epoch = -1
    best_val_loss = 0
    best_val_accuracy = 0
    best_train_accuracy = 0
    best_train_loss = 0
    train_val_gap = 0


# ============================================================
# 21. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETED")
print("=" * 75)

print(f"Best Epoch          : {best_epoch}")
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
    f"{train_val_gap:.2f} percentage points"
)

print("\nBest model saved to:")
print(best_model_path)

print("\nHistory saved to:")
print(history_path)

print("\nTraining log saved to:")
print(csv_log_path)

print("\nConfiguration saved to:")
print(config_path)

print("\n" + "=" * 75)
print("NEXT STEP: VALIDATION ANALYSIS")
print("=" * 75)