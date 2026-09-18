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

from sklearn.preprocessing import StandardScaler

from src.models.bat_transformer import build_bat_transformer

from src.config import (
    DATA_PROCESSED_DIR,
    SPLITS_DIR,
    MODELS_DIR
)


# ============================================================
# Settings
# ============================================================

SEED = 42

BATCH_SIZE = 32

EPOCHS = 80

PATIENCE = 10


np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load data
# ============================================================

features = np.load(
    DATA_PROCESSED_DIR / "features.npy"
)

labels = np.load(
    DATA_PROCESSED_DIR / "labels.npy"
)

filenames = np.load(
    DATA_PROCESSED_DIR / "filenames.npy"
)


print("Features:", features.shape)
print("Labels:", labels.shape)


# ============================================================
# Actor split
# ============================================================

def load_actor_file(filename):

    path = SPLITS_DIR / filename

    return set(
        path.read_text().splitlines()
    )


train_actors = load_actor_file(
    "train_actors.txt"
)

val_actors = load_actor_file(
    "val_actors.txt"
)

test_actors = load_actor_file(
    "test_actors.txt"
)


# ============================================================
# Actor IDs
# ============================================================

def get_actor_id(filename):

    return str(filename).split("_")[0]


actors = np.array([
    get_actor_id(f)
    for f in filenames
])


# ============================================================
# Masks
# ============================================================

train_mask = np.array([
    actor in train_actors
    for actor in actors
])

val_mask = np.array([
    actor in val_actors
    for actor in actors
])

test_mask = np.array([
    actor in test_actors
    for actor in actors
])


# ============================================================
# Split data
# ============================================================

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
# Standardization
# ============================================================

scaler = StandardScaler()

n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]

time_steps = X_train.shape[1]
n_features = X_train.shape[2]


X_train_2d = X_train.reshape(
    -1,
    n_features
)

X_val_2d = X_val.reshape(
    -1,
    n_features
)

X_test_2d = X_test.reshape(
    -1,
    n_features
)


# Fit ONLY on training data

X_train_2d = scaler.fit_transform(
    X_train_2d
)

X_val_2d = scaler.transform(
    X_val_2d
)

X_test_2d = scaler.transform(
    X_test_2d
)


# Restore sequence shape

X_train = X_train_2d.reshape(
    n_train,
    time_steps,
    n_features
)

X_val = X_val_2d.reshape(
    n_val,
    time_steps,
    n_features
)

X_test = X_test_2d.reshape(
    n_test,
    time_steps,
    n_features
)


print("\nAfter standardization:")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# Build model
# ============================================================

model = build_bat_transformer(
    input_shape=(
        time_steps,
        n_features
    )
)


model.summary()


# ============================================================
# Callbacks
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=PATIENCE,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# Train
# ============================================================

print(
    "\nStarting BAT Transformer training...\n"
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# Test
# ============================================================

print(
    "\nGenerating test predictions...\n"
)


y_probability = model.predict(
    X_test,
    verbose=1
)


y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# Metrics
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


print("\n==============================")
print("BAT TRANSFORMER TEST RESULTS")
print("==============================")


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
# Classification report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# Save model
# ============================================================

model_dir = MODELS_DIR / "bat_transformer"

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


model.save(
    model_dir / "bat_transformer_model.keras"
)


print(
    f"\nModel saved to: "
    f"{model_dir / 'bat_transformer_model.keras'}"
)