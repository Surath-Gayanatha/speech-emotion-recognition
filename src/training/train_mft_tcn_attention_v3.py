import os
import csv
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.mft_tcn_attention_v3 import (
    build_mft_tcn_attention_v3
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 32
EPOCHS = 60

INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 12


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = "data/mft_processed"
SPLIT_DIR = "data/splits"

MODEL_DIR = "models/mft_tcn_attention"

RESULT_DIR = "results/mft_tcn_attention"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "MFT_TCN_Attention_03.keras"
)

RESULT_PATH = os.path.join(
    RESULT_DIR,
    "MFT_TCN_Attention_03_Results.txt"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "MFT_TCN_Attention_03_History.csv"
)

CONFUSION_PATH = os.path.join(
    RESULT_DIR,
    "MFT_TCN_Attention_03_ConfusionMatrix.npy"
)


# ============================================================
# LOAD DATA
# ============================================================

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
# LOAD EXISTING ACTOR SPLITS
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
# ACTOR-BASED SPLIT
# ============================================================

unique_actors = np.unique(actors)

print("\nActors:")
print(unique_actors)

rng = np.random.default_rng(SEED)

shuffled_actors = unique_actors.copy()

rng.shuffle(shuffled_actors)

n_actors = len(shuffled_actors)

train_actor_count = int(
    n_actors * 0.70
)

val_actor_count = int(
    n_actors * 0.15
)

train_actors = shuffled_actors[
    :train_actor_count
]

val_actors = shuffled_actors[
    train_actor_count:
    train_actor_count + val_actor_count
]

test_actors = shuffled_actors[
    train_actor_count + val_actor_count:
]


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


print("\nDataset split:")

print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nStandardizing features...")

mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

std = np.where(
    std < 1e-8,
    1.0,
    std
)

X_train = (
    X_train - mean
) / std

X_val = (
    X_val - mean
) / std

X_test = (
    X_test - mean
) / std


# ============================================================
# MODEL
# ============================================================

print("\nBuilding MFT TCN Attention V3...")

model = build_mft_tcn_attention_v3(
    input_shape=X_train.shape[1:],
    num_classes=6
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=INITIAL_LR,
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

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# FIND BEST EPOCH
# ============================================================

best_epoch_index = int(
    np.argmax(
        history.history["val_accuracy"]
    )
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
# SAVE HISTORY CSV
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss",
        "learning_rate"
    ])

    for i in range(
        len(history.history["loss"])
    ):

        if "learning_rate" in history.history:

            lr = history.history[
                "learning_rate"
            ][i]

        else:

            lr = INITIAL_LR

        writer.writerow([
            i + 1,
            history.history["accuracy"][i],
            history.history["val_accuracy"][i],
            history.history["loss"][i],
            history.history["val_loss"][i],
            lr
        ])


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TCNBlock": __import__(
            "src.models.mft_tcn_attention_v3",
            fromlist=["TCNBlock"]
        ).TCNBlock,

        "GatedFusion": __import__(
            "src.models.mft_tcn_attention_v3",
            fromlist=["GatedFusion"]
        ).GatedFusion,

        "AttentionPooling": __import__(
            "src.models.mft_tcn_attention_v3",
            fromlist=["AttentionPooling"]
        ).AttentionPooling
    }
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nEvaluating test set...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    predictions
)

test_precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]

report = classification_report(
    y_test,
    predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

np.save(
    CONFUSION_PATH,
    cm
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_PATH,
    "w"
) as file:

    file.write(
        "MFT_TCN_Attention_03\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        "Architecture:\n"
    )

    file.write(
        "Multi-Scale TCN + Gated Fusion + "
        "Multi-Head Attention\n\n"
    )

    file.write(
        f"Input shape: {X_train.shape[1:]}\n"
    )

    file.write(
        "MFT features: 201\n"
    )

    file.write(
        "Number of classes: 6\n\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Maximum epochs: {EPOCHS}\n"
    )

    file.write(
        f"Initial learning rate: {INITIAL_LR}\n"
    )

    file.write(
        f"Weight decay: {WEIGHT_DECAY}\n\n"
    )

    file.write(
        f"Best Epoch: {best_epoch}\n"
    )

    file.write(
        f"Best Training Accuracy: "
        f"{best_train_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "TEST RESULTS\n"
    )

    file.write(
        "========================================\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Precision: "
        f"{test_precision * 100:.2f}%\n"
    )

    file.write(
        f"Test Recall: "
        f"{test_recall * 100:.2f}%\n"
    )

    file.write(
        f"Test F1 Score: "
        f"{test_f1 * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "========================================\n"
    )

    file.write(report)

    file.write(
        "\n\nCONFUSION MATRIX\n"
    )

    file.write(
        "========================================\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("MFT TCN ATTENTION V3 RESULTS")
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

print("=" * 60)

print(
    "\nSaved model:",
    MODEL_PATH
)

print(
    "Saved results:",
    RESULT_PATH
)

print(
    "Saved history:",
    HISTORY_PATH
)

print(
    "Saved confusion matrix:",
    CONFUSION_PATH
)