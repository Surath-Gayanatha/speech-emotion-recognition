"""
1D CNN for Speech Emotion Recognition.

The model learns local spectral-temporal patterns from
MFCC + delta + delta-delta sequences.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


def build_cnn1d(
    input_shape: tuple,
    num_classes: int = NUM_CLASSES
) -> keras.Model:

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(
            64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.Conv1D(
            64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv1D(
            128,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.Conv1D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.30),

        # Block 3
        layers.Conv1D(
            256,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling1D(),

        # Classification head
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.50),

        layers.Dense(
            num_classes,
            activation="softmax"
        ),
    ], name="cnn1d")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model