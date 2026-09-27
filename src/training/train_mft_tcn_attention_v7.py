import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.models.mft_tcn_attention_v2 import (
    build_mft_tcn_attention_v2
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = "data/mft_processed"
SPLIT_DIR = "data/splits"

MODEL_DIR = "models/mft_tcn_attention"
RESULTS_DIR = "results/mft_tcn_attention"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT NAME
# ============================================================

EXPERIMENT_NAME = "MFT_TCN_Attention_07"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    EXPERIMENT_NAME + ".keras"
)

RESULT_PATH = os.path.join(
    RESULTS_DIR,
    EXPERIMENT_NAME + "_Results.txt"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    EXPERIMENT_NAME + "_History.csv"
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    EXPERIMENT_NAME + "_ConfusionMatrix.npy"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 12

NUM_CLASSES = 6
RANDOM_SEED = 42

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001


# ============================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================

AUGMENT_TRAINING = True

# Maximum number of consecutive time frames to mask
TIME_MASK_MAX = 20

# Maximum number of Log-Mel feature bands to mask
FREQ_MASK_MAX = 8

# MFT feature layout:
#
# 0   - 39   : MFCC
# 40  - 79   : Delta MFCC
# 80  - 119  : Delta-Delta MFCC
# 120 - 183  : Log-Mel
# 184 - 195  : Chroma
# 196        : RMS
# 197        : ZCR
# 198        : Spectral Centroid
# 199        : Spectral Bandwidth
# 200        : Spectral Rolloff

LOG_MEL_START = 120
LOG_MEL_END = 184


# ============================================================
# SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MFT TCN + ATTENTION - EXPERIMENT 07")
print("=" * 60)

print("\nLoading MFT features...")

features = np.load(
    os.path.join(
        FEATURE_DIR,
        "features.npy"
    )
)

labels = np.load(
    os.path.join(
        FEATURE_DIR,
        "labels.npy"
    )
)

actors = np.load(
    os.path.join(
        FEATURE_DIR,
        "actors.npy"
    )
)

print("Features:", features.shape)
print("Labels:", labels.shape)
print("Actors:", actors.shape)


# ============================================================
# LOAD ACTOR SPLITS
# ============================================================

train_actors = np.loadtxt(
    os.path.join(
        SPLIT_DIR,
        "train_actors.txt"
    ),
    dtype=int
)

val_actors = np.loadtxt(
    os.path.join(
        SPLIT_DIR,
        "val_actors.txt"
    ),
    dtype=int
)

test_actors = np.loadtxt(
    os.path.join(
        SPLIT_DIR,
        "test_actors.txt"
    ),
    dtype=int
)


# ============================================================
# ACTOR-BASED SPLIT
# ============================================================

train_mask = np.isin(
    actors,
    train_actors
)

val_mask = np.isin(
    actors,
    val_actors
)

test_mask = np.isin(
    actors,
    test_actors
)


X_train = features[train_mask].copy()
y_train = labels[train_mask].copy()

X_val = features[val_mask].copy()
y_val = labels[val_mask].copy()

X_test = features[test_mask].copy()
y_test = labels[test_mask].copy()


print("\nActor-based split:")

print(
    "Training samples   :",
    len(X_train)
)

print(
    "Validation samples :",
    len(X_val)
)

print(
    "Test samples       :",
    len(X_test)
)


# ============================================================
# STANDARD SCALER
# FIT ON TRAINING DATA ONLY
# ============================================================

print("\nApplying StandardScaler...")

num_train, time_steps, num_features = X_train.shape

scaler = StandardScaler()

X_train_flat = X_train.reshape(
    -1,
    num_features
)

X_val_flat = X_val.reshape(
    -1,
    num_features
)

X_test_flat = X_test.reshape(
    -1,
    num_features
)

# IMPORTANT:
# The scaler is fitted ONLY on the training data.

scaler.fit(X_train_flat)


X_train = scaler.transform(
    X_train_flat
).reshape(
    num_train,
    time_steps,
    num_features
).astype(np.float32)


X_val = scaler.transform(
    X_val_flat
).reshape(
    X_val.shape[0],
    time_steps,
    num_features
).astype(np.float32)


X_test = scaler.transform(
    X_test_flat
).reshape(
    X_test.shape[0],
    time_steps,
    num_features
).astype(np.float32
)


print("✓ StandardScaler fitted on training data only")


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_mft_sample(x):

    """
    Apply MFT-compatible training augmentation.

    Augmentation:
        1. Temporal masking
        2. Log-Mel frequency masking

    This function is applied ONLY to training samples.
    Validation and test data remain unchanged.
    """

    x = x.copy()

    # --------------------------------------------------------
    # TEMPORAL MASKING
    # --------------------------------------------------------

    if TIME_MASK_MAX > 0:

        max_mask = min(
            TIME_MASK_MAX,
            x.shape[0]
        )

        mask_length = np.random.randint(
            1,
            max_mask + 1
        )

        max_start = x.shape[0] - mask_length

        if max_start > 0:

            start = np.random.randint(
                0,
                max_start + 1
            )

            x[
                start:start + mask_length,
                :
            ] = 0.0

    # --------------------------------------------------------
    # LOG-MEL FREQUENCY MASKING
    # --------------------------------------------------------

    log_mel_features = LOG_MEL_END - LOG_MEL_START

    if log_mel_features > 0:

        max_mask = min(
            FREQ_MASK_MAX,
            log_mel_features
        )

        mask_width = np.random.randint(
            1,
            max_mask + 1
        )

        max_start = (
            LOG_MEL_END
            - mask_width
        )

        start = np.random.randint(
            LOG_MEL_START,
            max_start + 1
        )

        x[
            :,
            start:start + mask_width
        ] = 0.0

    return x


# ============================================================
# CREATE TRAINING DATASET
# ============================================================

def create_training_dataset(X, y):

    def generator():

        for i in range(len(X)):

            sample = X[i].copy()

            if AUGMENT_TRAINING:

                sample = augment_mft_sample(
                    sample
                )

            yield (
                sample.astype(np.float32),
                np.int32(y[i])
            )

    dataset = tf.data.Dataset.from_generator(

        generator,

        output_signature=(

            tf.TensorSpec(
                shape=(
                    time_steps,
                    num_features
                ),
                dtype=tf.float32
            ),

            tf.TensorSpec(
                shape=(),
                dtype=tf.int32
            )
        )
    )

    dataset = dataset.shuffle(
        buffer_size=len(X),
        seed=RANDOM_SEED,
        reshuffle_each_iteration=True
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CREATE VALIDATION / TEST DATASETS
# NO AUGMENTATION
# ============================================================

def create_evaluation_dataset(X, y):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            X,
            y
        )
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_dataset = create_training_dataset(
    X_train,
    y_train
)

val_dataset = create_evaluation_dataset(
    X_val,
    y_val
)

test_dataset = create_evaluation_dataset(
    X_test,
    y_test
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

classes = np.arange(
    NUM_CLASSES
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        class_weights_array
    )
}


print("\nClass weights:")

for cls, weight in class_weights.items():

    print(
        f"Class {cls} "
        f"({CLASS_NAMES[cls]}): "
        f"{weight:.4f}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding model...")

model = build_mft_tcn_attention_v2(
    input_shape=(
        time_steps,
        num_features
    ),
    num_classes=NUM_CLASSES
)

model.summary()


# ============================================================
# OPTIMIZER
# ============================================================

try:

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0
    )

except AttributeError:

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizer,

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

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

    monitor="val_accuracy",

    mode="max",

    patience=PATIENCE,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print("\nTraining augmentation:")
print("  Temporal masking: ON")
print("  Log-Mel masking:  ON")

print("\nClass weighting: ON")

print("\nStarting training...\n")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        reduce_lr,
        early_stopping
    ]
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    np.arange(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# BEST EPOCH
# ============================================================

best_epoch_index = np.argmax(
    history.history["val_accuracy"]
)

best_epoch = (
    best_epoch_index + 1
)

best_train_accuracy = (
    history.history["accuracy"]
    [best_epoch_index]
)

best_val_accuracy = (
    history.history["val_accuracy"]
    [best_epoch_index]
)


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

print(
    "\nThe test set is evaluated only "
    "after model selection."
)

test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

y_probability = model.predict(

    test_dataset,

    verbose=1
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

test_precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


test_recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


test_f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    target_names=CLASS_NAMES,

    digits=2,

    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred
)

np.save(
    CONFUSION_PATH,
    cm
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"\nBest Epoch: "
    f"{best_epoch}"
)

print(
    f"Training Accuracy: "
    f"{best_train_accuracy:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Precision: "
    f"{test_precision:.4f}"
)

print(
    f"Test Recall: "
    f"{test_recall:.4f}"
)

print(
    f"Test F1-score: "
    f"{test_f1:.4f}"
)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE RESULT FILE
# ============================================================

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 60 + "\n")

    f.write(
        "MFT TCN + ATTENTION - "
        "EXPERIMENT 07\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Experiment: "
        f"{EXPERIMENT_NAME}\n\n"
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    f.write("DATASET\n")
    f.write("-" * 60 + "\n")

    f.write(
        "Dataset: CREMA-D\n"
    )

    f.write(
        "Split method: Actor-based split\n"
    )

    f.write(
        f"Total samples: "
        f"{len(features)}\n"
    )

    f.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(X_val)}\n"
    )

    f.write(
        f"Test samples: "
        f"{len(X_test)}\n\n"
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    f.write(
        "FEATURE CONFIGURATION\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        f"Input shape: "
        f"{features.shape[1:]}\n"
    )

    f.write(
        "Number of features: 201\n"
    )

    f.write(
        "MFCC features: 40\n"
    )

    f.write(
        "Delta features: 40\n"
    )

    f.write(
        "Delta-Delta features: 40\n"
    )

    f.write(
        "Log-Mel features: 64\n"
    )

    f.write(
        "Additional spectral features: 17\n\n"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    f.write(
        "MODEL CONFIGURATION\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        "Model: MFT TCN + "
        "Multi-Head Attention V2\n"
    )

    f.write(
        "Feature projection: Dense(160)\n"
    )

    f.write(
        "TCN filters: 160\n"
    )

    f.write(
        "TCN kernel size: 3\n"
    )

    f.write(
        "TCN dilation rates: "
        "1, 2, 4, 8, 16\n"
    )

    f.write(
        "Multi-Head Attention heads: 8\n"
    )

    f.write(
        "Attention key dimension: 20\n"
    )

    f.write(
        "Feed-forward dimension: 320\n"
    )

    f.write(
        "Pooling: Average + Max + Attention\n\n"
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    f.write(
        "TRAINING CONFIGURATION\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Maximum epochs: "
        f"{EPOCHS}\n"
    )

    f.write(
        f"Learning rate: "
        f"{LEARNING_RATE}\n"
    )

    f.write(
        f"Weight decay: "
        f"{WEIGHT_DECAY}\n"
    )

    f.write(
        f"Random seed: "
        f"{RANDOM_SEED}\n"
    )

    f.write(
        "Optimizer: AdamW\n"
    )

    f.write(
        "Gradient clipping: 1.0\n"
    )

    f.write(
        "Class weighting: Enabled\n"
    )

    f.write(
        "Temporal masking: Enabled\n"
    )

    f.write(
        "Log-Mel frequency masking: Enabled\n"
    )

    f.write(
        f"Time mask maximum: "
        f"{TIME_MASK_MAX}\n"
    )

    f.write(
        f"Frequency mask maximum: "
        f"{FREQ_MASK_MAX}\n"
    )

    f.write(
        "Validation/test augmentation: Disabled\n\n"
    )


    # --------------------------------------------------------
    # BEST RESULTS
    # --------------------------------------------------------

    f.write(
        "BEST RESULTS\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Training Accuracy: "
        f"{best_train_accuracy:.4f}\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{best_val_accuracy:.4f}\n\n"
    )


    # --------------------------------------------------------
    # TEST RESULTS
    # --------------------------------------------------------

    f.write(
        "FINAL TEST RESULTS\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Precision: "
        f"{test_precision:.4f}\n"
    )

    f.write(
        f"Test Recall: "
        f"{test_recall:.4f}\n"
    )

    f.write(
        f"Test F1-score: "
        f"{test_f1:.4f}\n\n"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write("-" * 60 + "\n")

    f.write(report)

    f.write("\n\n")


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        np.array2string(cm)
    )

    f.write("\n")


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nResults:")
print(RESULT_PATH)

print("\nHistory:")
print(HISTORY_PATH)

print("\nConfusion matrix:")
print(CONFUSION_PATH)

print("\n" + "=" * 60)
print("EXPERIMENT 07 COMPLETED SUCCESSFULLY")
print("=" * 60)