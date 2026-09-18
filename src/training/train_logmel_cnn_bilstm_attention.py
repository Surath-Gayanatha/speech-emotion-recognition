import os
import json
import random

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from src.models.logmel_cnn_bilstm_attention import (
    build_logmel_cnn_bilstm_attention,
    SelfAttention
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/logmel"
RESULTS_DIR = "results/logmel_cnn_bilstm_attention"

EPOCHS = 60
BATCH_SIZE = 32

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6

EMOTION_NAMES = [
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD"
]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def load_npy(filename):
    """
    Load a NumPy file and print its shape.
    """

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\n"
            f"\nPlease check the Log-Mel data directory."
        )

    data = np.load(path)

    print(f"{filename:<30} {data.shape}")

    return data


# ============================================================
# 4. LOAD LOG-MEL DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING LOG-MEL DATA")
print("=" * 70)

print(f"Data directory: {DATA_DIR}\n")

X_train = load_npy("train_features.npy")
y_train = load_npy("train_labels.npy")

X_val = load_npy("validation_features.npy")
y_val = load_npy("validation_labels.npy")

X_test = load_npy("test_features.npy")
y_test = load_npy("test_labels.npy")


# ============================================================
# 5. BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DATA")
print("=" * 70)

expected_train = (5147, 64, 174)
expected_val = (1066, 64, 174)
expected_test = (1229, 64, 174)

print(f"Expected train shape: {expected_train}")
print(f"Actual train shape:   {X_train.shape}")

print(f"\nExpected validation shape: {expected_val}")
print(f"Actual validation shape:   {X_val.shape}")

print(f"\nExpected test shape: {expected_test}")
print(f"Actual test shape:   {X_test.shape}")


if X_train.shape[1:] != (64, 174):
    raise ValueError(
        f"Unexpected training feature shape: {X_train.shape}"
    )

if X_val.shape[1:] != (64, 174):
    raise ValueError(
        f"Unexpected validation feature shape: {X_val.shape}"
    )

if X_test.shape[1:] != (64, 174):
    raise ValueError(
        f"Unexpected test feature shape: {X_test.shape}"
    )


if len(X_train) != len(y_train):
    raise ValueError("Train features and labels have different lengths.")

if len(X_val) != len(y_val):
    raise ValueError("Validation features and labels have different lengths.")

if len(X_test) != len(y_test):
    raise ValueError("Test features and labels have different lengths.")


print("\nData shape validation: PASSED")


# ============================================================
# 6. LABEL VALIDATION
# ============================================================

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)

all_valid_labels = np.arange(NUM_CLASSES)

if not np.all(np.isin(y_train, all_valid_labels)):
    raise ValueError("Invalid labels found in training data.")

if not np.all(np.isin(y_val, all_valid_labels)):
    raise ValueError("Invalid labels found in validation data.")

if not np.all(np.isin(y_test, all_valid_labels)):
    raise ValueError("Invalid labels found in test data.")

print("Label validation: PASSED")


# ============================================================
# 7. ADD CHANNEL DIMENSION
# ============================================================

print("\n" + "=" * 70)
print("PREPARING MODEL INPUT")
print("=" * 70)

if X_train.ndim == 3:

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

elif X_train.ndim != 4:

    raise ValueError(
        f"Unexpected input dimensions: {X_train.shape}"
    )


print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")


# ============================================================
# 8. CHECK DATA VALUES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATA VALUES")
print("=" * 70)

print(
    f"Train range: "
    f"{X_train.min():.4f} → {X_train.max():.4f}"
)

print(
    f"Validation range: "
    f"{X_val.min():.4f} → {X_val.max():.4f}"
)

print(
    f"Test range: "
    f"{X_test.min():.4f} → {X_test.max():.4f}"
)

if not np.isfinite(X_train).all():
    raise ValueError("NaN or Inf found in training data.")

if not np.isfinite(X_val).all():
    raise ValueError("NaN or Inf found in validation data.")

if not np.isfinite(X_test).all():
    raise ValueError("NaN or Inf found in test data.")

print("Numerical validation: PASSED")


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for class_id, count in zip(unique, counts):

    emotion = EMOTION_NAMES[int(class_id)]

    print(
        f"Class {class_id} ({emotion}): {count}"
    )


# ============================================================
# 10. CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("COMPUTING CLASS WEIGHTS")
print("=" * 70)

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(classes, class_weights_array)
}

for class_id, weight in class_weights.items():

    emotion = EMOTION_NAMES[class_id]

    print(
        f"Class {class_id} ({emotion}): "
        f"{weight:.4f}"
    )


# ============================================================
# 11. BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LOG-MEL CNN + BiLSTM + SELF-ATTENTION")
print("=" * 70)

model = build_logmel_cnn_bilstm_attention()

model.summary()


# ============================================================
# 12. COMPILE MODEL
# ============================================================

print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)

print("Optimizer: AdamW")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")


# ============================================================
# 13. CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    RESULTS_DIR,
    "best_logmel_cnn_bilstm_attention.keras"
)

history_path = os.path.join(
    RESULTS_DIR,
    "training_history.npz"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

print("\n" + "=" * 70)
print("SAVING TRAINING HISTORY")
print("=" * 70)

np.savez(
    history_path,
    **history.history
)

print(
    f"Training history saved to:\n"
    f"{history_path}"
)


# ============================================================
# 16. TRAINING SUMMARY
# ============================================================

train_accuracy_history = history.history["accuracy"]
val_accuracy_history = history.history["val_accuracy"]

train_loss_history = history.history["loss"]
val_loss_history = history.history["val_loss"]

best_epoch = int(
    np.argmax(val_accuracy_history) + 1
)

best_train_accuracy = float(
    train_accuracy_history[best_epoch - 1]
)

best_val_accuracy = float(
    val_accuracy_history[best_epoch - 1]
)

best_train_loss = float(
    train_loss_history[best_epoch - 1]
)

best_val_loss = float(
    val_loss_history[best_epoch - 1]
)

generalization_gap = (
    best_train_accuracy -
    best_val_accuracy
)


print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"Total epochs completed: {len(train_loss_history)}")
print(f"Best validation epoch: {best_epoch}")

print(
    f"Train Accuracy at best epoch: "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy at best epoch: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Train Loss at best epoch: "
    f"{best_train_loss:.4f}"
)

print(
    f"Validation Loss at best epoch: "
    f"{best_val_loss:.4f}"
)

print(
    f"Generalization Gap: "
    f"{generalization_gap * 100:.2f} percentage points"
)


# ============================================================
# 17. LOAD BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST CHECKPOINT")
print("=" * 70)

if not os.path.exists(checkpoint_path):

    raise FileNotFoundError(
        f"Best model checkpoint was not created:\n"
        f"{checkpoint_path}"
    )

best_model = tf.keras.models.load_model(
    checkpoint_path,
    custom_objects={
        "SelfAttention": SelfAttention
    },
    compile=True
)

print("Best model loaded successfully.")


# ============================================================
# 18. VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

val_loss, val_accuracy = best_model.evaluate(
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    f"\nValidation Loss: "
    f"{val_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# 19. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print("Evaluating on unseen test set...")

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 20. TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_prob = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ============================================================
# 21. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# 22. PRINT TEST METRICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST METRICS")
print("=" * 70)

print(
    f"Accuracy:           "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision:    "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall:       "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1:           "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Weighted Precision: "
    f"{weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall:    "
    f"{weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1:        "
    f"{weighted_f1 * 100:.2f}%"
)


# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=EMOTION_NAMES,
    digits=4,
    zero_division=0
)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Log-Mel CNN + BiLSTM + Self-Attention\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Best Epoch: {best_epoch}\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.6f}\n"
    )

    f.write(
        f"Test Loss: "
        f"{test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{weighted_precision:.6f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{weighted_recall:.6f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{weighted_f1:.6f}\n"
    )

    f.write(
        f"\nGeneralization Gap: "
        f"{generalization_gap:.6f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write("-" * 70 + "\n")

    f.write(report)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)

print(
    f"Confusion matrix saved to:\n"
    f"{cm_path}"
)


# ============================================================
# 25. SAVE METRICS JSON
# ============================================================

summary = {

    "model":
        "Log-Mel CNN + BiLSTM + Self-Attention",

    "feature":
        "Log-Mel Spectrogram",

    "input_shape":
        [64, 174, 1],

    "num_classes":
        NUM_CLASSES,

    "emotion_classes":
        EMOTION_NAMES,

    "seed":
        SEED,

    "epochs_configured":
        EPOCHS,

    "epochs_completed":
        len(train_loss_history),

    "best_epoch":
        best_epoch,

    "batch_size":
        BATCH_SIZE,

    "optimizer":
        "AdamW",

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "best_train_accuracy":
        best_train_accuracy,

    "best_validation_accuracy":
        best_val_accuracy,

    "generalization_gap":
        generalization_gap,

    "validation_loss":
        float(val_loss),

    "validation_accuracy":
        float(val_accuracy),

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1)
}


summary_path = os.path.join(
    RESULTS_DIR,
    "metrics.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nFinal Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Final Macro F1: "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"\nResults directory:\n"
    f"{RESULTS_DIR}"
)

print("\nSaved files:")

print(
    "1. best_logmel_cnn_bilstm_attention.keras"
)

print(
    "2. training_history.npz"
)

print(
    "3. classification_report.txt"
)

print(
    "4. confusion_matrix.npy"
)

print(
    "5. metrics.json"
)

print("\nDone.")