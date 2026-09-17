import os
import sys
import random

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from src.models.mft_transformer import (
    build_mft_transformer,
    FeatureProjection,
    PositionalEmbedding,
    MFTTransformerBlock,
    AttentionPooling
)

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = "data/mft_processed"
SPLIT_DIR = "data/splits"

MODEL_DIR = "models/mft_transformer"
RESULTS_DIR = "results/mft_transformer"

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10

NUM_CLASSES = 6
RANDOM_SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

try:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
except Exception:
    pass

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# EXPERIMENT NAME
# ============================================================

# IMPORTANT:
# Change this value manually for each experiment.

EXPERIMENT_NAME = "MFT_Transformer_FineTune_02"

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

# ============================================================
# LOAD FEATURES
# ============================================================

print("\n" + "=" * 60)
print("LOADING MFT FEATURES")
print("=" * 60)

X = np.load(
    os.path.join(FEATURE_DIR, "features.npy")
)

y = np.load(
    os.path.join(FEATURE_DIR, "labels.npy")
)

actors = np.load(
    os.path.join(FEATURE_DIR, "actors.npy")
)

print(f"Features shape : {X.shape}")
print(f"Labels shape   : {y.shape}")
print(f"Actors shape   : {actors.shape}")

# ============================================================
# LOAD ACTOR SPLITS
# ============================================================

def load_actor_ids(filename):

    path = os.path.join(
        SPLIT_DIR,
        filename
    )

    with open(path, "r") as f:

        actor_ids = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    return np.array(actor_ids)


train_actors = load_actor_ids(
    "train_actors.txt"
)

val_actors = load_actor_ids(
    "val_actors.txt"
)

test_actors = load_actor_ids(
    "test_actors.txt"
)

print("\nActor split sizes:")
print(
    f"Train actors      : {len(train_actors)}"
)
print(
    f"Validation actors : {len(val_actors)}"
)
print(
    f"Test actors       : {len(test_actors)}"
)

# ============================================================
# CREATE ACTOR-BASED SPLITS
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

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nDataset split:")
print(
    f"X_train : {X_train.shape}"
)
print(
    f"X_val   : {X_val.shape}"
)
print(
    f"X_test  : {X_test.shape}"
)

# ============================================================
# STANDARDIZATION
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZING FEATURES")
print("=" * 60)

num_features = X_train.shape[-1]

scaler = StandardScaler()

# Fit ONLY on training data

X_train_2d = X_train.reshape(
    -1,
    num_features
)

scaler.fit(
    X_train_2d
)

X_train = scaler.transform(
    X_train_2d
).reshape(
    X_train.shape
)

X_val = scaler.transform(
    X_val.reshape(
        -1,
        num_features
    )
).reshape(
    X_val.shape
)

X_test = scaler.transform(
    X_test.reshape(
        -1,
        num_features
    )
).reshape(
    X_test.shape
)

print("Standardization complete.")

# ============================================================
# CHECK DATA
# ============================================================

print("\nChecking data...")

print(
    "NaN in train:",
    np.isnan(X_train).sum()
)

print(
    "NaN in validation:",
    np.isnan(X_val).sum()
)

print(
    "NaN in test:",
    np.isnan(X_test).sum()
)

print(
    "Inf in train:",
    np.isinf(X_train).sum()
)

print(
    "Inf in validation:",
    np.isinf(X_val).sum()
)

print(
    "Inf in test:",
    np.isinf(X_test).sum()
)

# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print("\n" + "=" * 60)
print("CREATING DATASETS")
print("=" * 60)

train_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (X_train, y_train)
    )
    .shuffle(
        buffer_size=len(X_train),
        seed=RANDOM_SEED,
        reshuffle_each_iteration=True
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (X_val, y_val)
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)

test_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (X_test, y_test)
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)

# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING MFT TRANSFORMER")
print("=" * 60)

model = build_mft_transformer(
    input_shape=X_train.shape[1:],
    num_classes=NUM_CLASSES,
    embed_dim=128,
    num_heads=8,
    ff_dim=256,
    num_layers=4,
    dropout=0.15
)

model.summary()

# ============================================================
# COMPILE
# ============================================================

print("\n" + "=" * 60)
print("COMPILING MODEL")
print("=" * 60)

try:

    optimizer = keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=1e-4
    )

except AttributeError:

    optimizer = keras.optimizers.Adam(
        learning_rate=2e-4
    )

# ============================================================
# LABEL SMOOTHING
# ============================================================

loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING MFT TRANSFORMER TRAINING")
print("=" * 60)

print(f"\nExperiment: {EXPERIMENT_NAME}")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)

model.save(
    MODEL_PATH
)

print(
    f"Model saved to:\n{MODEL_PATH}"
)

# ============================================================
# EVALUATE TEST SET
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_prob = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]

report = classification_report(
    y_test,
    y_pred,
    target_names=emotion_names,
    zero_division=0
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ============================================================
# TRAINING SUMMARY
# ============================================================

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(
    f"Best epoch       : {best_epoch}"
)

print(
    f"Best val accuracy: {best_val_accuracy:.4f}"
)

print(
    f"Test accuracy    : {accuracy:.4f}"
)

print(
    f"Test F1-score    : {f1:.4f}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAVING EXPERIMENT RESULTS")
print("=" * 60)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MFT TRANSFORMER - EXPERIMENT RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Experiment: {EXPERIMENT_NAME}\n\n"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    f.write(
        "DATASET\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        "Dataset: CREMA-D\n"
    )

    f.write(
        "Split method: Actor-based split\n"
    )

    f.write(
        f"Total samples: {len(X)}\n"
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

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    f.write(
        "FEATURE CONFIGURATION\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Input shape: {X_train.shape[1:]}\n"
    )

    f.write(
        "Number of features: 201\n\n"
    )

    f.write(
        "Features:\n"
    )

    f.write(
        "- MFCC\n"
    )

    f.write(
        "- MFCC Delta\n"
    )

    f.write(
        "- MFCC Delta-Delta\n"
    )

    f.write(
        "- Log-Mel Spectrogram\n"
    )

    f.write(
        "- Chroma\n"
    )

    f.write(
        "- RMS Energy\n"
    )

    f.write(
        "- Zero Crossing Rate\n"
    )

    f.write(
        "- Spectral Centroid\n"
    )

    f.write(
        "- Spectral Bandwidth\n"
    )

    f.write(
        "- Spectral Rolloff\n\n"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    f.write(
        "MODEL CONFIGURATION\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        "Model: MFT Transformer\n"
    )

    f.write(
        "Embedding dimension: 128\n"
    )

    f.write(
        "Attention heads: 8\n"
    )

    f.write(
        "Transformer blocks: 4\n"
    )

    f.write(
        "Feed-forward dimension: 256\n"
    )

    f.write(
        "Transformer dropout: 0.15\n"
    )

    f.write(
        f"Total parameters: {model.count_params():,}\n\n"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    f.write(
        "TRAINING CONFIGURATION\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Maximum epochs: {EPOCHS}\n"
    )

    f.write(
        "Initial learning rate: 0.0002\n"
    )

    f.write(
        "Weight decay: 0.0001\n"
    )

    f.write(
        f"Early stopping patience: {PATIENCE}\n"
    )

    f.write(
        "ReduceLROnPlateau factor: 0.5\n\n"
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    f.write(
        "BEST TRAINING RESULT\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}\n\n"
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    f.write(
        "FINAL TEST RESULTS\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1-score : {f1:.4f}\n\n"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        report
    )

    f.write(
        "\n\n"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        np.array2string(cm)
    )

    f.write(
        "\n\n"
    )

    # --------------------------------------------------------
    # Experiment notes
    # --------------------------------------------------------

    f.write(
        "EXPERIMENT NOTES\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        "This experiment uses the MFT Transformer model "
        "with independently extracted multi-feature "
        "acoustic representations.\n"
    )

    f.write(
        "The dataset uses an actor-based train, "
        "validation and test split.\n"
    )

    f.write(
        "StandardScaler was fitted only on the training data.\n"
    )

    f.write(
        "The best model was selected using validation accuracy.\n"
    )

print(
    f"\nResults saved to:\n{RESULT_PATH}"
)

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = np.column_stack([
    np.arange(
        1,
        len(
            history.history["accuracy"]
        ) + 1
    ),

    history.history["accuracy"],

    history.history["val_accuracy"],

    history.history["loss"],

    history.history["val_loss"]
])

np.savetxt(
    HISTORY_PATH,
    history_data,
    delimiter=",",
    header="epoch,accuracy,val_accuracy,loss,val_loss",
    comments="",
    fmt="%.6f"
)

print(
    f"Training history saved to:\n{HISTORY_PATH}"
)

print("\nTraining complete.")