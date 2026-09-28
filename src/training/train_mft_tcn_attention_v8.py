"""
Train MFT-TCN Attention V8.

V8 uses the same 192-D input representation as the team's BiLSTM +
Attention experiment:

    64 Log-Mel + 64 Delta + 64 Delta-Delta

The architecture remains TCN + Multi-Head Attention, so the model
comparison focuses more directly on architecture.

Run from repository root:
    python src/training/train_mft_tcn_attention_v8.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.models.mft_tcn_attention_v8 import build_mft_tcn_attention_v8


FEATURE_DIR = Path("data/mft_tcn_v8_processed")
SPLIT_DIR = Path("data/splits")

MODEL_DIR = Path("models/mft_tcn_attention_v8")
RESULTS_DIR = Path("results/mft_tcn_attention_v8")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "MFT_TCN_Attention_V8"

MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.keras"
RESULT_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_Results.txt"
HISTORY_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_History.csv"
CONFUSION_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_ConfusionMatrix.npy"

BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 12
NUM_CLASSES = 6
RANDOM_SEED = 42
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

AUGMENT_TRAINING = True
TIME_MASK_MAX = 20
FREQ_MASK_MAX = 8

N_MELS = 64
FEAT_DIM = 192
LOG_MEL_START = 0
LOG_MEL_END = 64

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)


def load_data():
    features = np.load(FEATURE_DIR / "features.npy")
    labels = np.load(FEATURE_DIR / "labels.npy")
    actors = np.load(FEATURE_DIR / "actors.npy")

    train_actors = np.loadtxt(
        SPLIT_DIR / "train_actors.txt",
        dtype=int,
    )
    val_actors = np.loadtxt(
        SPLIT_DIR / "val_actors.txt",
        dtype=int,
    )
    test_actors = np.loadtxt(
        SPLIT_DIR / "test_actors.txt",
        dtype=int,
    )

    train_mask = np.isin(actors, train_actors)
    val_mask = np.isin(actors, val_actors)
    test_mask = np.isin(actors, test_actors)

    X_train = features[train_mask].copy()
    y_train = labels[train_mask].copy()

    X_val = features[val_mask].copy()
    y_val = labels[val_mask].copy()

    X_test = features[test_mask].copy()
    y_test = labels[test_mask].copy()

    return (
        features,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


def standardize_train_only(X_train, X_val, X_test):
    num_train, time_steps, num_features = X_train.shape

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, num_features))

    X_train = scaler.transform(
        X_train.reshape(-1, num_features)
    ).reshape(
        num_train,
        time_steps,
        num_features,
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val.reshape(-1, num_features)
    ).reshape(
        X_val.shape[0],
        time_steps,
        num_features,
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test.reshape(-1, num_features)
    ).reshape(
        X_test.shape[0],
        time_steps,
        num_features,
    ).astype(np.float32)

    return X_train, X_val, X_test


def augment_sample(x):
    x = x.copy()

    # Temporal masking across all 192 features.
    mask_length = np.random.randint(
        1,
        min(TIME_MASK_MAX, x.shape[0]) + 1,
    )

    max_start = x.shape[0] - mask_length

    if max_start >= 0:
        start = np.random.randint(0, max_start + 1)
        x[start:start + mask_length, :] = 0.0

    # Frequency masking: apply the same mel-band mask to
    # Log-Mel, Delta and Delta-Delta blocks.
    mask_width = np.random.randint(
        1,
        min(FREQ_MASK_MAX, N_MELS) + 1,
    )

    start = np.random.randint(
        LOG_MEL_START,
        LOG_MEL_END - mask_width + 1,
    )

    for block_start in (0, 64, 128):
        x[:, block_start + start:block_start + start + mask_width] = 0.0

    return x


def create_training_dataset(X, y, time_steps):
    def generator():
        for i in range(len(X)):
            sample = X[i].copy()

            if AUGMENT_TRAINING:
                sample = augment_sample(sample)

            yield (
                sample.astype(np.float32),
                np.int32(y[i]),
            )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(time_steps, FEAT_DIM),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int32,
            ),
        ),
    )

    return (
        dataset
        .shuffle(
            buffer_size=len(X),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def create_eval_dataset(X, y):
    return (
        tf.data.Dataset.from_tensor_slices((X, y))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def main():
    print("=" * 70)
    print("MFT-TCN + MHA V8")
    print("192-D Log-Mel + Delta + Delta-Delta input")
    print("=" * 70)

    (
        all_features,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = load_data()

    print("\nRaw feature shape:", all_features.shape)
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    if X_train.shape[-1] != FEAT_DIM:
        raise ValueError(
            f"Expected {FEAT_DIM} input features, "
            f"but found {X_train.shape[-1]}."
        )

    X_train, X_val, X_test = standardize_train_only(
        X_train,
        X_val,
        X_test,
    )

    time_steps = X_train.shape[1]

    train_dataset = create_training_dataset(
        X_train,
        y_train,
        time_steps,
    )
    val_dataset = create_eval_dataset(X_val, y_val)
    test_dataset = create_eval_dataset(X_test, y_test)

    classes = np.arange(NUM_CLASSES)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(
            classes,
            class_weights_array,
        )
    }

    print("\nClass weights:")
    for cls, weight in class_weights.items():
        print(
            f"  {cls} ({CLASS_NAMES[cls]}): "
            f"{weight:.4f}"
        )

    print("\nBuilding model...")

    model = build_mft_tcn_attention_v8(
        input_shape=(time_steps, FEAT_DIM),
        num_classes=NUM_CLASSES,
    )

    model.summary()

    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        )
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            checkpoint,
            reduce_lr,
            early_stopping,
        ],
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(
        0,
        "epoch",
        np.arange(1, len(history_df) + 1),
    )
    history_df.to_csv(HISTORY_PATH, index=False)

    best_epoch_index = int(
        np.argmax(history.history["val_accuracy"])
    )
    best_epoch = best_epoch_index + 1

    best_train_accuracy = float(
        history.history["accuracy"][best_epoch_index]
    )
    best_val_accuracy = float(
        history.history["val_accuracy"][best_epoch_index]
    )

    print("\nLoading best saved model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "TCNBlockV8": __import__(
                "src.models.mft_tcn_attention_v8",
                fromlist=["TCNBlockV8"],
            ).TCNBlockV8,
            "AttentionPoolingV8": __import__(
                "src.models.mft_tcn_attention_v8",
                fromlist=["AttentionPoolingV8"],
            ).AttentionPoolingV8,
        },
    )

    print("\nFinal test evaluation...")
    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    probabilities = model.predict(
        test_dataset,
        verbose=1,
    )
    y_pred = np.argmax(probabilities, axis=1)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    np.save(CONFUSION_PATH, cm)

    result_text = f"""
======================================================================
MFT-TCN + MHA V8 RESULTS
======================================================================

DATASET
----------------------------------------------------------------------
Dataset: CREMA-D
Split: Team actor-level split
Total samples: {len(all_features)}
Training samples: {len(X_train)}
Validation samples: {len(X_val)}
Test samples: {len(X_test)}

FEATURE CONFIGURATION
----------------------------------------------------------------------
Representation: Log-Mel + Delta + Delta-Delta
Log-Mel dimensions: 64
Delta dimensions: 64
Delta-Delta dimensions: 64
Total feature dimensions: 192
Sample rate: 16000 Hz
N_FFT: 640
Hop length: 320
Silence trimming: top_db=30
Maximum time steps: {time_steps}
StandardScaler: fitted on training data only

MODEL
----------------------------------------------------------------------
Architecture: TCN + Multi-Head Self-Attention
TCN filters: 160
TCN kernel size: 3
TCN dilation rates: 1, 2, 4, 8, 16
MHA heads: 8
MHA key dimension: 20
Feed-forward dimension: 320
Pooling: Average + Max + Attention
Classifier: Dense 128 -> Dense 64 -> Softmax 6

TRAINING
----------------------------------------------------------------------
Batch size: {BATCH_SIZE}
Maximum epochs: {EPOCHS}
Learning rate: {LEARNING_RATE}
Weight decay: {WEIGHT_DECAY}
Optimizer: AdamW
Gradient clipping: 1.0
Class weighting: enabled
Temporal masking: enabled
Frequency masking: enabled
Validation/test augmentation: disabled
Random seed: {RANDOM_SEED}

BEST VALIDATION RESULT
----------------------------------------------------------------------
Best epoch: {best_epoch}
Training accuracy: {best_train_accuracy:.4f}
Validation accuracy: {best_val_accuracy:.4f}

FINAL TEST RESULT
----------------------------------------------------------------------
Test loss: {test_loss:.4f}
Test accuracy: {test_accuracy:.4f}
Weighted precision: {precision:.4f}
Weighted recall: {recall:.4f}
Weighted F1-score: {f1:.4f}

CLASSIFICATION REPORT
----------------------------------------------------------------------
{report}

CONFUSION MATRIX
----------------------------------------------------------------------
{cm}
"""

    RESULT_PATH.write_text(
        result_text.strip() + "\n",
        encoding="utf-8",
    )

    print("\n" + result_text)
    print("\nSaved:")
    print("  Model:", MODEL_PATH)
    print("  Results:", RESULT_PATH)
    print("  History:", HISTORY_PATH)
    print("  Confusion matrix:", CONFUSION_PATH)


if __name__ == "__main__":
    main()
