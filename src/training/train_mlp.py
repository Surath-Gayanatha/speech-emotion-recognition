import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from src.models.mlp import build_mlp
from src.config import DATA_PROCESSED_DIR, SPLITS_DIR, MODELS_DIR


# -----------------------------
# Settings
# -----------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Load data
# -----------------------------
features = np.load(DATA_PROCESSED_DIR / "features.npy")
labels = np.load(DATA_PROCESSED_DIR / "labels.npy")
filenames = np.load(DATA_PROCESSED_DIR / "filenames.npy")


print("Features:", features.shape)
print("Labels:", labels.shape)


# -----------------------------
# Load actor splits
# -----------------------------
def load_actor_file(filename):
    path = SPLITS_DIR / filename
    return set(path.read_text().splitlines())


train_actors = load_actor_file("train_actors.txt")
val_actors = load_actor_file("val_actors.txt")
test_actors = load_actor_file("test_actors.txt")


# -----------------------------
# Get actor ID from filename
# -----------------------------
def get_actor_id(filename):
    return str(filename).split("_")[0]


actors = np.array([get_actor_id(f) for f in filenames])


# -----------------------------
# Create train/validation/test
# -----------------------------
train_mask = np.array([actor in train_actors for actor in actors])
val_mask = np.array([actor in val_actors for actor in actors])
test_mask = np.array([actor in test_actors for actor in actors])


X_train = features[train_mask]
y_train = labels[train_mask]

X_val = features[val_mask]
y_val = labels[val_mask]

X_test = features[test_mask]
y_test = labels[test_mask]


print("\nDataset split:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# -----------------------------
# Flatten features for MLP
# -----------------------------
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


print("\nFlattened input shape:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# -----------------------------
# Standardization
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# -----------------------------
# Build MLP
# -----------------------------
model = build_mlp(
    input_shape=(X_train.shape[1],)
)


model.summary()


# -----------------------------
# Early stopping
# -----------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    mode="max",
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# -----------------------------
# Test prediction
# -----------------------------
y_probability = model.predict(X_test)

y_pred = np.argmax(y_probability, axis=1)


# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

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
print("MLP TEST RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# -----------------------------
# Classification report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------
# Save model
# -----------------------------
model_dir = MODELS_DIR / "mlp"
model_dir.mkdir(parents=True, exist_ok=True)

model.save(model_dir / "mlp_model.keras")

print(
    f"\nModel saved to: {model_dir / 'mlp_model.keras'}"
)