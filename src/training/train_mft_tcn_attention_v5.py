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

from src.models.mft_tcn_attention_v5 import (
    build_mft_tcn_attention_v5
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "MFT_TCN_Attention_05"

RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 12

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

NUM_CLASSES = 6


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = "data/mft_processed"
SPLIT_DIR = "data/splits"

MODEL_DIR = "models/mft_tcn_attention"
RESULTS_DIR = "results/mft_tcn_attention"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

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
print("MFT TCN + ATTENTION - EXPERIMENT 05")
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

print(
    "Features:",
    features.shape
)

print(
    "Labels:",
    labels.shape
)

print(
    "Actors:",
    actors.shape
)


# ============================================================
# LOAD FIXED ACTOR SPLITS
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
# ACTOR SPLIT
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
# STANDARD SCALER
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

scaler.fit(
    X_train_flat
)

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


# ============================================================
# DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)

train_dataset = train_dataset.shuffle(
    len(X_train),
    seed=RANDOM_SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
        y_val
    )
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_test,
        y_test
    )
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MFT TCN Attention V5...")

model = build_mft_tcn_attention_v5(
    input_shape=(
        time_steps,
        num_features
    ),
    num_classes=NUM_CLASSES
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LOSS
# ============================================================

loss=tf.keras.losses.SparseCategoricalCrossentropy(),


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
model.summary()


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
    ],
    verbose=1
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
    history.history[
        "val_accuracy"
    ]
)

best_epoch = (
    best_epoch_index + 1
)

best_train_accuracy = (
    history.history[
        "accuracy"
    ][best_epoch_index]
)

best_val_accuracy = (
    history.history[
        "val_accuracy"
    ][best_epoch_index]
)


# ============================================================
# TEST
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
# REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
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
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MFT TCN ATTENTION V5 RESULTS")
print("=" * 60)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Training Accuracy: "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Test Recall: "
    f"{test_recall * 100:.2f}%"
)

print(
    f"Test F1 Score: "
    f"{test_f1 * 100:.2f}%"
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
    f.write(
        "MFT TCN + ATTENTION - EXPERIMENT 05\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Experiment: {EXPERIMENT_NAME}\n\n"
    )

    f.write("DATASET\n")
    f.write("-" * 60 + "\n")
    f.write("Dataset: CREMA-D\n")
    f.write(
        "Split method: Actor-based split\n"
    )
    f.write(
        f"Total samples: {len(features)}\n"
    )
    f.write(
        f"Training samples: {len(X_train)}\n"
    )
    f.write(
        f"Validation samples: {len(X_val)}\n"
    )
    f.write(
        f"Test samples: {len(X_test)}\n\n"
    )

    f.write("MODEL CONFIGURATION\n")
    f.write("-" * 60 + "\n")
    f.write(
        "Model: MFT TCN + Gated Attention V5\n"
    )
    f.write(
        "Feature projection: Dense(160)\n"
    )
    f.write(
        "TCN filters: 160\n"
    )
    f.write(
        "TCN dilation rates: 1, 2, 4, 8, 16\n"
    )
    f.write(
        "TCN gating: Yes\n"
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
        "Pooling: Average + Max + Attention\n"
    )
    f.write(
        "Label smoothing: 0.05\n\n"
    )

    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 60 + "\n")
    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )
    f.write(
        f"Maximum epochs: {EPOCHS}\n"
    )
    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )
    f.write(
        f"Weight decay: {WEIGHT_DECAY}\n"
    )
    f.write(
        f"Random seed: {RANDOM_SEED}\n\n"
    )

    f.write("BEST RESULTS\n")
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

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(report)

    f.write("\n\nCONFUSION MATRIX\n")
    f.write("-" * 60 + "\n")
    f.write(
        np.array2string(cm)
    )

    f.write("\n")


# ============================================================
# FILES
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

print("\nExperiment completed successfully.")