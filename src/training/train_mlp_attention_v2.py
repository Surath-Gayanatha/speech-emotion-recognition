import os
import random
import csv

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.mlp_attention_v2 import (
    build_mlp_attention_v2
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

FEATURE_DIR = "data/mlp_attention_processed"
SPLIT_DIR = "data/splits"

MODEL_DIR = "models/mlp_attention"
RESULT_DIR = "results/mlp_attention"

BATCH_SIZE = 64

EPOCHS = 60

PATIENCE = 12

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6

LABEL_SMOOTHING = 0.05

AUGMENT = True

TIME_MASK_MAX = 20

FREQ_MASK_MAX = 8


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
# OUTPUT PATHS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "MLP_Attention_V2.keras"
)

RESULT_PATH = os.path.join(
    RESULT_DIR,
    "MLP_Attention_V2_Results.txt"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "MLP_Attention_V2_History.csv"
)

CONFUSION_PATH = os.path.join(
    RESULT_DIR,
    "MLP_Attention_V2_ConfusionMatrix.npy"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("MLP + MULTI-HEAD TEMPORAL ATTENTION V2")
print("=" * 70)

print("\nLoading features...")

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

print("\nLoading fixed actor splits...")

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
# ACTOR-LEVEL SPLIT
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


X_train = features[
    train_mask
].copy()

y_train = labels[
    train_mask
].copy()


X_val = features[
    val_mask
].copy()

y_val = labels[
    val_mask
].copy()


X_test = features[
    test_mask
].copy()

y_test = labels[
    test_mask
].copy()


print("\nActor-level split:")

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
# TRAIN DATA ONLY
# ============================================================

print(
    "\nFitting StandardScaler on training data only..."
)

num_train = X_train.shape[0]

time_steps = X_train.shape[1]

num_features = X_train.shape[2]


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
).astype(
    np.float32
)


X_val = scaler.transform(
    X_val_flat
).reshape(
    X_val.shape[0],
    time_steps,
    num_features
).astype(
    np.float32
)


X_test = scaler.transform(
    X_test_flat
).reshape(
    X_test.shape[0],
    time_steps,
    num_features
).astype(
    np.float32
)


print(
    "✓ Train-only normalization completed"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

classes = np.arange(
    NUM_CLASSES
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)


class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        weights
    )
}


print("\nClass weights:")

for cls, weight in class_weights.items():

    print(
        f"{CLASS_NAMES[cls]}: "
        f"{weight:.4f}"
    )


# ============================================================
# ONE-HOT LABELS
# ============================================================

y_train_onehot = tf.keras.utils.to_categorical(
    y_train,
    NUM_CLASSES
)

y_val_onehot = tf.keras.utils.to_categorical(
    y_val,
    NUM_CLASSES
)

y_test_onehot = tf.keras.utils.to_categorical(
    y_test,
    NUM_CLASSES
)


# ============================================================
# SPEC AUGMENTATION
# ============================================================

def augment_sample(x):

    x = x.copy()

    # --------------------------------------------------------
    # TEMPORAL MASK
    # --------------------------------------------------------

    max_width = min(
        TIME_MASK_MAX,
        x.shape[0]
    )

    width = np.random.randint(
        1,
        max_width + 1
    )

    start = np.random.randint(
        0,
        x.shape[0] - width + 1
    )

    x[
        start:start + width,
        :
    ] = 0.0


    # --------------------------------------------------------
    # FREQUENCY MASK
    #
    # 0-63     Log-Mel
    # 64-127   Delta
    # 128-191  Delta-Delta
    # --------------------------------------------------------

    max_width = min(
        FREQ_MASK_MAX,
        64
    )

    width = np.random.randint(
        1,
        max_width + 1
    )

    start = np.random.randint(
        0,
        64 - width + 1
    )


    x[
        :,
        start:start + width
    ] = 0.0


    x[
        :,
        64 + start:64 + start + width
    ] = 0.0


    x[
        :,
        128 + start:128 + start + width
    ] = 0.0


    return x


# ============================================================
# TRAINING GENERATOR
# ============================================================

def training_generator():

    while True:

        indices = np.random.permutation(
            len(X_train)
        )

        for index in indices:

            sample = X_train[
                index
            ].copy()

            label = y_train[
                index
            ]


            if AUGMENT:

                sample = augment_sample(
                    sample
                )


            # One-hot target
            target = tf.keras.utils.to_categorical(
                label,
                NUM_CLASSES
            ).astype(
                np.float32
            )


            # Sample weight
            sample_weight = np.float32(
                class_weights[
                    int(label)
                ]
            )


            yield (
                sample.astype(
                    np.float32
                ),
                target,
                sample_weight
            )


# ============================================================
# TRAIN DATASET
# ============================================================

train_dataset = tf.data.Dataset.from_generator(

    training_generator,

    output_signature=(

        tf.TensorSpec(
            shape=(
                time_steps,
                num_features
            ),
            dtype=tf.float32
        ),

        tf.TensorSpec(
            shape=(
                NUM_CLASSES,
            ),
            dtype=tf.float32
        ),

        tf.TensorSpec(
            shape=(),
            dtype=tf.float32
        )
    )
)


train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# VALIDATION DATASET
# ============================================================

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
        y_val_onehot
    )
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_test,
        y_test_onehot
    )
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print(
    "\nBuilding MLP + Multi-Head Temporal Attention V2..."
)


model = build_mlp_attention_v2(
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

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    clipnorm=1.0
)


# ============================================================
# LOSS
# ============================================================

loss_function = (
    tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizer,

    loss=loss_function,

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

print("\n" + "=" * 70)
print("STARTING V2 TRAINING")
print("=" * 70)

print("\nConfiguration:")
print("  Log-Mel + Delta + Delta-Delta")
print("  192 features/frame")
print("  200 frames")
print("  Multi-head temporal attention")
print("  Attention + mean pooling")
print("  Class weighting")
print("  SpecAugment")
print("  Label smoothing: 0.05")
print("  Gradient clipping: 1.0")
print("  Fixed actor split")
print("  Train-only normalization")
print()


steps_per_epoch = int(
    np.ceil(
        len(X_train) /
        BATCH_SIZE
    )
)


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    steps_per_epoch=steps_per_epoch,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# BEST EPOCH
# ============================================================

best_index = int(
    np.argmax(
        history.history[
            "val_accuracy"
        ]
    )
)


best_epoch = best_index + 1


best_train_accuracy = (
    history.history[
        "accuracy"
    ][best_index]
)


best_val_accuracy = (
    history.history[
        "val_accuracy"
    ][best_index]
)


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "epoch",
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss"
    ])


    for i in range(
        len(
            history.history[
                "loss"
            ]
        )
    ):

        writer.writerow([
            i + 1,
            history.history[
                "accuracy"
            ][i],
            history.history[
                "val_accuracy"
            ][i],
            history.history[
                "loss"
            ][i],
            history.history[
                "val_loss"
            ][i]
        ])


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL V2 TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict(
    test_dataset,
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

report = classification_report(
    y_test,
    predictions,
    target_names=CLASS_NAMES,
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
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MLP + MULTI-HEAD TEMPORAL ATTENTION V2\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "ARCHITECTURE\n"
    )

    file.write(
        "Dense Projection + Multi-Head Temporal "
        "Attention + Attention/Mean Pooling + MLP\n\n"
    )

    file.write(
        "FEATURE PIPELINE\n"
    )

    file.write(
        "64 Log-Mel + 64 Delta + 64 Delta-Delta\n"
    )

    file.write(
        "Features per frame: 192\n"
    )

    file.write(
        "Maximum frames: 200\n"
    )

    file.write(
        "Silence trimming: top_db=30\n"
    )

    file.write(
        "Actor-level split: enabled\n"
    )

    file.write(
        "Train-only normalization: enabled\n"
    )

    file.write(
        "SpecAugment: enabled\n"
    )

    file.write(
        "Class weighting: enabled\n\n"
    )

    file.write(
        "TRAINING CONFIGURATION\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Epochs: {EPOCHS}\n"
    )

    file.write(
        f"Initial learning rate: {LEARNING_RATE}\n"
    )

    file.write(
        f"Weight decay: {WEIGHT_DECAY}\n"
    )

    file.write(
        f"Label smoothing: {LABEL_SMOOTHING}\n"
    )

    file.write(
        f"Random seed: {SEED}\n\n"
    )

    file.write(
        "BEST RESULTS\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        f"Best Epoch: {best_epoch}\n"
    )

    file.write(
        f"Training Accuracy: "
        f"{best_train_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "TEST RESULTS\n"
    )

    file.write(
        "=" * 60 + "\n"
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
        f"Test F1-score: "
        f"{test_f1 * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nCONFUSION MATRIX\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MLP + TEMPORAL ATTENTION V2 RESULTS")
print("=" * 70)

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
    f"Test F1-score: "
    f"{test_f1 * 100:.2f}%"
)

print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved results:")
print(RESULT_PATH)

print("\nSaved history:")
print(HISTORY_PATH)

print("\nSaved confusion matrix:")
print(CONFUSION_PATH)

print("\nExperiment completed.")