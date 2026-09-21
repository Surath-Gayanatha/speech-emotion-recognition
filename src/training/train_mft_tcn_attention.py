import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.models.mft_tcn_attention import build_mft_tcn_attention


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

EXPERIMENT_NAME = "MFT_TCN_Attention_01"

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
EPOCHS = 50
PATIENCE = 10

NUM_CLASSES = 6
RANDOM_SEED = 42

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0001


# ============================================================
# RANDOM SEEDS
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
print("MFT TCN + ATTENTION - EXPERIMENT 01")
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
# CREATE ACTOR-BASED SPLIT
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


X_train = features[train_mask]
y_train = labels[train_mask]

X_val = features[val_mask]
y_val = labels[val_mask]

X_test = features[test_mask]
y_test = labels[test_mask]


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
# STANDARD SCALING
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


# FIT ONLY ON TRAINING DATA

scaler.fit(X_train_flat)


X_train = scaler.transform(
    X_train_flat
).reshape(
    num_train,
    time_steps,
    num_features
)


X_val = scaler.transform(
    X_val_flat
).reshape(
    X_val.shape[0],
    time_steps,
    num_features
)


X_test = scaler.transform(
    X_test_flat
).reshape(
    X_test.shape[0],
    time_steps,
    num_features
)


print("Scaling completed.")


# ============================================================
# TF.DATA DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    seed=RANDOM_SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MFT TCN + Attention model...")

model = build_mft_tcn_attention(
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
        weight_decay=WEIGHT_DECAY
    )

except AttributeError:

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizer,

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
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

print("\nStarting training...\n")

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

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
# FIND BEST VALIDATION EPOCH
# ============================================================

best_epoch_index = np.argmax(
    history.history["val_accuracy"]
)

best_epoch = best_epoch_index + 1

best_train_accuracy = (
    history.history["accuracy"]
    [best_epoch_index]
)

best_val_accuracy = (
    history.history["val_accuracy"]
    [best_epoch_index]
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)


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
# TEST METRICS
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
    f"\nBest Epoch: {best_epoch}"
)

print(
    f"Training Accuracy at Best Epoch: "
    f"{best_train_accuracy:.4f}"
)

print(
    f"Best Validation Accuracy: "
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
# SAVE RESULTS
# ============================================================

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 60 + "\n")
    f.write("MFT TCN + ATTENTION - EXPERIMENT RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Experiment: {EXPERIMENT_NAME}\n\n"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    f.write("DATASET\n")
    f.write("-" * 60 + "\n")

    f.write("Dataset: CREMA-D\n")
    f.write("Split method: Actor-based split\n")
    f.write(f"Total samples: {len(features)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Validation samples: {len(X_val)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    f.write("FEATURE CONFIGURATION\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Input shape: {features.shape[1:]}\n"
    )

    f.write("Number of features: 201\n\n")

    f.write("Features:\n")
    f.write("- MFCC\n")
    f.write("- MFCC Delta\n")
    f.write("- MFCC Delta-Delta\n")
    f.write("- Log-Mel Spectrogram\n")
    f.write("- Chroma\n")
    f.write("- RMS Energy\n")
    f.write("- Zero Crossing Rate\n")
    f.write("- Spectral Centroid\n")
    f.write("- Spectral Bandwidth\n")
    f.write("- Spectral Rolloff\n\n")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    f.write("MODEL CONFIGURATION\n")
    f.write("-" * 60 + "\n")

    f.write("Model: MFT TCN + Multi-Head Attention\n")
    f.write("Feature projection: Dense(128)\n")
    f.write("TCN filters: 128\n")
    f.write("TCN kernel size: 3\n")
    f.write("TCN dilation rates: 1, 2, 4, 8\n")
    f.write("Multi-Head Attention heads: 8\n")
    f.write("Attention key dimension: 16\n")
    f.write(
        "Pooling: Average + Max + Attention Pooling\n"
    )
    f.write(
        "Classification head: Dense(128) -> Dense(64) -> Dense(6)\n\n"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 60 + "\n")

    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Maximum epochs: {EPOCHS}\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )
    f.write(
        f"Weight decay: {WEIGHT_DECAY}\n"
    )
    f.write(
        f"Early stopping patience: {PATIENCE}\n\n"
    )

    # --------------------------------------------------------
    # BEST RESULTS
    # --------------------------------------------------------

    f.write("BEST TRAINING / VALIDATION RESULTS\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Best Epoch: {best_epoch}\n"
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
    # TEST
    # --------------------------------------------------------

    f.write("FINAL TEST RESULTS\n")
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
    # REPORT
    # --------------------------------------------------------

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")

    f.write(report)

    f.write("\n\n")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    f.write("CONFUSION MATRIX\n")
    f.write("-" * 60 + "\n")

    f.write(
        np.array2string(cm)
    )

    f.write("\n")


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

print("\nExperiment completed successfully.")