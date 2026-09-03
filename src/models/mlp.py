"""MLP baseline for Speech Emotion Recognition.

Architectural role in the comparison: treats the MFCC(+delta) matrix as a
flattened feature vector, discarding temporal order entirely. This makes it
a genuine baseline for testing whether sequential/local structure (as
modeled by 1D CNN / LSTM / BiLSTM) actually helps -- not just "a smaller
network," but a structurally different assumption (no time-awareness).
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


def build_mlp(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """input_shape: (n_features * MAX_PAD_LEN,) after flattening upstream."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ], name="mlp_baseline")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
