import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.vit_ser import (
    build_vit_ser,
    PatchEmbedding,
    ClassTokenPositionEmbedding,
    TransformerEncoder
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = "data/vit_processed"
SPLIT_DIR = "data/splits"
MODEL_DIR = "models/vit_ser"

FEATURE_PATH = os.path.join(FEATURE_DIR, "features.npy")
LABEL_PATH = os.path.join(FEATURE_DIR, "labels.npy")
ACTOR_PATH = os.path.join(FEATURE_DIR, "actors.npy")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vit_ser_model.keras"
)

BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 12

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading features...")

X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
actors = np.load(ACTOR_PATH)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Actors shape:", actors.shape)


# ============================================================
# LOAD ACTOR SPLITS
# ============================================================

def load_actor_ids(filename):
    path = os.path.join(SPLIT_DIR, filename)

    actor_ids = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                actor_ids.append(int(line))
            except ValueError:
                continue

    return set(actor_ids)


train_actors = load_actor_ids("train_actors.txt")
val_actors = load_actor_ids("val_actors.txt")
test_actors = load_actor_ids("test_actors.txt")

print("\nActor split:")
print("Train actors:", sorted(train_actors))
print("Validation actors:", sorted(val_actors))
print("Test actors:", sorted(test_actors))


# ============================================================
# CREATE ACTOR-BASED MASKS
# ============================================================

train_mask = np.isin(actors, list(train_actors))
val_mask = np.isin(actors, list(val_actors))
test_mask = np.isin(actors, list(test_actors))


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nDataset sizes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# ============================================================
# STANDARDIZATION
# TRAINING DATA ONLY
# ============================================================

print("\nStandardizing features...")

train_mean = np.mean(
    X_train,
    axis=(0, 2),
    keepdims=True
)

train_std = np.std(
    X_train,
    axis=(0, 2),
    keepdims=True
)

train_std = np.maximum(train_std, 1e-6)

X_train = (
    X_train - train_mean
) / train_std

X_val = (
    X_val - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nAfter adding channel dimension:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print("\nCreating datasets...")

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .shuffle(
        buffer_size=len(X_train),
        reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# BUILD VISION TRANSFORMER
# ============================================================

print("\nBuilding Vision Transformer...")

model = build_vit_ser(
    input_shape=(128, 256, 1),
    num_classes=6,
    patch_size=16,
    embed_dim=128,
    num_heads=8,
    ff_dim=256,
    num_layers=4,
    dropout=0.15
)

model.summary()


# ============================================================
# OPTIMIZER
# ============================================================

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
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=PATIENCE,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    mode="max",
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING VISION TRANSFORMER TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PatchEmbedding": PatchEmbedding,
        "ClassTokenPositionEmbedding": ClassTokenPositionEmbedding,
        "TransformerEncoder": TransformerEncoder
    }
)
print("Best model loaded from:")
print(MODEL_PATH)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_prob = model.predict(
    test_dataset,
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


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


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

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=emotion_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 60)

print("Model saved at:")
print(MODEL_PATH)