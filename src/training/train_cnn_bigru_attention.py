"""
Train CNN + BiGRU + Attention on normalized Log-Mel features.

Dataset:
    CREMA-D

Splits:
    Train      : 5147 samples
    Validation : 1066 samples
    Test       : 1229 samples

Feature representation:
    Log-Mel Spectrogram
    64 Mel bands x 174 time frames

Important:
    Log-Mel features are already normalized using training-data-only
    mean/std during feature extraction.

Therefore, this training script does NOT perform normalization again.

Test data is loaded only for shape verification and is NOT used for
training, tuning, or model selection.
"""

from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.cnn_bigru_attention import build_cnn_bigru_attention


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = Path("data/processed/logmel")

MODEL_DIR = Path("models/cnn_bigru_attention")
RESULT_DIR = Path("results/cnn_bigru_attention")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 60

# Classification
NUM_CLASSES = 6

# Log-Mel input
INPUT_SHAPE = (64, 174, 1)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load preprocessed Log-Mel features and labels."""

    print("=" * 70)
    print("LOADING NORMALIZED LOG-MEL DATA")
    print("=" * 70)

    train_x = np.load(
        DATA_DIR / "train_features.npy"
    )

    train_y = np.load(
        DATA_DIR / "train_labels.npy"
    )

    val_x = np.load(
        DATA_DIR / "validation_features.npy"
    )

    val_y = np.load(
        DATA_DIR / "validation_labels.npy"
    )

    test_x = np.load(
        DATA_DIR / "test_features.npy"
    )

    test_y = np.load(
        DATA_DIR / "test_labels.npy"
    )

    print(
        f"Train features      : {train_x.shape}"
    )

    print(
        f"Train labels        : {train_y.shape}"
    )

    print(
        f"Validation features : {val_x.shape}"
    )

    print(
        f"Validation labels   : {val_y.shape}"
    )

    print(
        f"Test features       : {test_x.shape}"
    )

    print(
        f"Test labels         : {test_y.shape}"
    )

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    )


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
):
    """Validate shapes and labels before training."""

    print()
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    expected_feature_shape = (64, 174)

    if train_x.shape[1:] != expected_feature_shape:
        raise ValueError(
            f"Unexpected train shape: {train_x.shape}"
        )

    if val_x.shape[1:] != expected_feature_shape:
        raise ValueError(
            f"Unexpected validation shape: {val_x.shape}"
        )

    if test_x.shape[1:] != expected_feature_shape:
        raise ValueError(
            f"Unexpected test shape: {test_x.shape}"
        )

    if len(train_x) != len(train_y):
        raise ValueError(
            "Train features and labels have different lengths."
        )

    if len(val_x) != len(val_y):
        raise ValueError(
            "Validation features and labels have different lengths."
        )

    if len(test_x) != len(test_y):
        raise ValueError(
            "Test features and labels have different lengths."
        )

    if not np.isfinite(train_x).all():
        raise ValueError(
            "NaN or Inf values found in training features."
        )

    if not np.isfinite(val_x).all():
        raise ValueError(
            "NaN or Inf values found in validation features."
        )

    if not np.isfinite(test_x).all():
        raise ValueError(
            "NaN or Inf values found in test features."
        )

    all_labels = np.concatenate(
        [
            train_y,
            val_y,
            test_y,
        ]
    )

    unique_labels = set(
        all_labels.tolist()
    )

    expected_labels = set(
        range(NUM_CLASSES)
    )

    if not unique_labels.issubset(
        expected_labels
    ):
        raise ValueError(
            f"Unexpected labels: {unique_labels}"
        )

    print("Feature shapes       : PASSED")
    print("Feature/label sizes  : PASSED")
    print("NaN/Inf check        : PASSED")
    print("Label check          : PASSED")

    print()
    print("DATA VALIDATION PASSED")


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(features):
    """
    Add channel dimension.

    Input:
        (N, 64, 174)

    Output:
        (N, 64, 174, 1)
    """

    features = features.astype(
        np.float32,
        copy=False
    )

    if features.ndim == 3:

        features = np.expand_dims(
            features,
            axis=-1
        )

    elif features.ndim != 4:

        raise ValueError(
            f"Unexpected feature dimensions: "
            f"{features.shape}"
        )

    return features


# ============================================================
# SPEC AUGMENTATION
# ============================================================

class SpecAugment(layers.Layer):
    """
    SpecAugment for Log-Mel spectrograms.

    Applies:
        - Frequency masking
        - Time masking

    The augmentation is automatically disabled during validation
    and inference.
    """

    def __init__(
        self,
        freq_mask_param=8,
        time_mask_param=18,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(
        self,
        inputs,
        training=None
    ):

        if training is False:
            return inputs

        x = inputs

        n_mels = tf.shape(x)[1]
        time_steps = tf.shape(x)[2]

        # ----------------------------------------------------
        # Frequency masking
        # ----------------------------------------------------

        freq_width = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=self.freq_mask_param + 1,
            dtype=tf.int32
        )

        freq_start_max = tf.maximum(
            n_mels - freq_width + 1,
            1
        )

        freq_start = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=freq_start_max,
            dtype=tf.int32
        )

        freq_positions = tf.range(
            n_mels
        )

        freq_mask = tf.logical_or(
            freq_positions < freq_start,
            freq_positions >= (
                freq_start + freq_width
            )
        )

        freq_mask = tf.cast(
            freq_mask,
            x.dtype
        )

        freq_mask = tf.reshape(
            freq_mask,
            [1, n_mels, 1, 1]
        )

        x = x * freq_mask

        # ----------------------------------------------------
        # Time masking
        # ----------------------------------------------------

        time_width = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=self.time_mask_param + 1,
            dtype=tf.int32
        )

        time_start_max = tf.maximum(
            time_steps - time_width + 1,
            1
        )

        time_start = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=time_start_max,
            dtype=tf.int32
        )

        time_positions = tf.range(
            time_steps
        )

        time_mask = tf.logical_or(
            time_positions < time_start,
            time_positions >= (
                time_start + time_width
            )
        )

        time_mask = tf.cast(
            time_mask,
            x.dtype
        )

        time_mask = tf.reshape(
            time_mask,
            [1, 1, time_steps, 1]
        )

        x = x * time_mask

        return x


# ============================================================
# DATASETS
# ============================================================

def create_datasets(
    train_x,
    train_y,
    val_x,
    val_y
):
    """Create TensorFlow training and validation datasets."""

    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            train_x,
            train_y
        )
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (
            val_x,
            val_y
        )
    )

    train_ds = (
        train_ds
        .shuffle(
            buffer_size=len(train_x),
            seed=SEED,
            reshuffle_each_iteration=True
        )
        .batch(
            BATCH_SIZE,
            drop_remainder=False
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    val_ds = (
        val_ds
        .batch(
            BATCH_SIZE,
            drop_remainder=False
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    return (
        train_ds,
        val_ds
    )


# ============================================================
# BUILD TRAINING MODEL
# ============================================================

def build_training_model():
    """
    Build CNN + BiGRU + Attention model with SpecAugment.

    Architecture:
        Log-Mel
          ↓
        SpecAugment
          ↓
        CNN
          ↓
        BiGRU
          ↓
        Multi-Head Attention
          ↓
        Global Average Pooling
          ↓
        Dense
          ↓
        Softmax
    """

    base_model = build_cnn_bigru_attention(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES
    )

    inputs = keras.Input(
        shape=INPUT_SHAPE,
        name="logmel_input"
    )

    augmented = SpecAugment(
        freq_mask_param=8,
        time_mask_param=18,
        name="spec_augment"
    )(inputs)

    outputs = base_model(
        augmented
    )

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_bigru_attention_specaugment"
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history):
    """Save training history as JSON."""

    history_path = (
        RESULT_DIR / "history.json"
    )

    history_dict = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_dict,
            file,
            indent=4
        )

    print(
        f"History saved to: {history_path}"
    )


# ============================================================
# SAVE TRAINING CONFIGURATION
# ============================================================

def save_config():
    """Save experiment configuration."""

    config = {
        "model": "CNN + BiGRU + Multi-Head Attention",
        "feature_type": "Log-Mel Spectrogram",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "specaugment": True,
        "frequency_mask": 8,
        "time_mask": 18,
        "early_stopping_patience": 8,
        "reduce_lr_patience": 4,
        "reduce_lr_factor": 0.5,
        "minimum_learning_rate": 1e-6,
        "seed": SEED,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }

    config_path = (
        RESULT_DIR / "config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print(
        f"Configuration saved to: {config_path}"
    )


# ============================================================
# BEST VALIDATION RESULT
# ============================================================

def report_best_validation_result(
    history
):
    """Report best validation performance."""

    val_accuracy = history.history.get(
        "val_accuracy",
        []
    )

    train_accuracy = history.history.get(
        "accuracy",
        []
    )

    if not val_accuracy:
        return

    best_index = int(
        np.argmax(val_accuracy)
    )

    best_epoch = (
        best_index + 1
    )

    best_val = float(
        val_accuracy[best_index]
    )

    corresponding_train = float(
        train_accuracy[best_index]
    )

    gap = (
        corresponding_train
        - best_val
    )

    print()
    print("=" * 70)
    print("BEST VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Best Epoch       : {best_epoch}"
    )

    print(
        f"Train Accuracy   : "
        f"{corresponding_train * 100:.2f}%"
    )

    print(
        f"Validation Acc.  : "
        f"{best_val * 100:.2f}%"
    )

    print(
        f"Train-Val Gap    : "
        f"{gap * 100:.2f} percentage points"
    )

    # Save summary
    summary = {
        "best_epoch": best_epoch,
        "train_accuracy": corresponding_train,
        "validation_accuracy": best_val,
        "train_validation_gap": gap,
    }

    summary_path = (
        RESULT_DIR / "best_validation_result.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"Best validation summary saved to: "
        f"{summary_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed()

    print()
    print("=" * 70)
    print("CREMA-D CNN + BiGRU + ATTENTION TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Save configuration
    # --------------------------------------------------------

    save_config()

    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------

    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y
    ) = load_data()

    # --------------------------------------------------------
    # 3. Validate data
    # --------------------------------------------------------

    validate_data(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y
    )

    # --------------------------------------------------------
    # 4. Prepare input
    # --------------------------------------------------------

    train_x = prepare_input(
        train_x
    )

    val_x = prepare_input(
        val_x
    )

    test_x = prepare_input(
        test_x
    )

    print()
    print("=" * 70)
    print("INPUT SHAPES")
    print("=" * 70)

    print(
        f"Train      : {train_x.shape}"
    )

    print(
        f"Validation : {val_x.shape}"
    )

    print(
        f"Test       : {test_x.shape}"
    )

    # --------------------------------------------------------
    # 5. Create datasets
    # --------------------------------------------------------

    (
        train_ds,
        val_ds
    ) = create_datasets(
        train_x,
        train_y,
        val_x,
        val_y
    )

    # --------------------------------------------------------
    # 6. Build model
    # --------------------------------------------------------

    model = build_training_model()

    print()
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    model.summary()

    # --------------------------------------------------------
    # 7. Callbacks
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR / "best_model.keras"
    )

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # 8. Start training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------------
    # 9. Save history
    # --------------------------------------------------------

    save_history(
        history
    )

    # --------------------------------------------------------
    # 10. Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR / "final_model.keras"
    )

    model.save(
        final_model_path
    )

    print()
    print(
        f"Final model saved to: "
        f"{final_model_path}"
    )

    print(
        f"Best model saved to : "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # 11. Report validation result
    # --------------------------------------------------------

    report_best_validation_result(
        history
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # Test data is intentionally NOT evaluated here.
    #
    # The validation set should be used for:
    #     - hyperparameter tuning
    #     - architecture comparison
    #     - model selection
    #
    # After the final configuration is frozen,
    # evaluate the best checkpoint once on the unseen test set.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print()
    print(
        "Test set was NOT used for training "
        "or model selection."
    )

    print(
        "Use the best checkpoint for final "
        "unseen test evaluation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()