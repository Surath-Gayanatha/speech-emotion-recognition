import os
import numpy as np
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
    PatchEmbedding,
    ClassTokenPositionEmbedding,
    TransformerEncoder
)


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = "data/vit_processed"
SPLIT_DIR = "data/splits"
MODEL_PATH = "models/vit_ser/vit_ser_model.keras"

FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "features.npy"
)

LABEL_PATH = os.path.join(
    FEATURE_DIR,
    "labels.npy"
)

ACTOR_PATH = os.path.join(
    FEATURE_DIR,
    "actors.npy"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading test data...")

X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
actors = np.load(ACTOR_PATH)

print("Features:", X.shape)
print("Labels:", y.shape)
print("Actors:", actors.shape)


# ============================================================
# LOAD TEST ACTORS
# ============================================================

def load_actor_ids(filename):

    path = os.path.join(
        SPLIT_DIR,
        filename
    )

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


test_actors = load_actor_ids(
    "test_actors.txt"
)

print("\nTest actors:")
print(sorted(test_actors))


# ============================================================
# CREATE TEST SET
# ============================================================

test_mask = np.isin(
    actors,
    list(test_actors)
)

X_test = X[test_mask]
y_test = y[test_mask]

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# STANDARDIZATION
# ============================================================
#
# IMPORTANT:
# We need to use the SAME training statistics that were used
# during the original training.
#
# These are recalculated here from the training actors only.
# ============================================================

train_actors = load_actor_ids(
    "train_actors.txt"
)

train_mask = np.isin(
    actors,
    list(train_actors)
)

X_train = X[train_mask]

print("\nCalculating training normalization statistics...")

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

train_std = np.maximum(
    train_std,
    1e-6
)


# ============================================================
# STANDARDIZE TEST DATA
# ============================================================

X_test = (
    X_test - train_mean
) / train_std


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

X_test = X_test[..., np.newaxis]

print(
    "X_test after channel dimension:",
    X_test.shape
)


# ============================================================
# LOAD BEST SAVED MODEL
# ============================================================

print("\nLoading saved ViT model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PatchEmbedding": PatchEmbedding,
        "ClassTokenPositionEmbedding": ClassTokenPositionEmbedding,
        "TransformerEncoder": TransformerEncoder
    },
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_prob = model.predict(
    X_test,
    batch_size=32,
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
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("VISION TRANSFORMER TEST RESULTS")
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

print("\n")
print("=" * 60)
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

print("\n")
print("=" * 60)
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

print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)