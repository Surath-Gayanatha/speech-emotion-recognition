"""LSTM for Speech Emotion Recognition.

Architectural role: models the MFCC sequence as an ordered time series,
maintaining a hidden state that captures how emotional cues (pitch, energy,
spectral shape) evolve *forward in time* across an utterance. Unlike the
1D CNN, it can in principle model long-range dependencies across the whole
clip rather than only local windows. Unlike BiLSTM, it only ever sees past
context when producing each timestep's representation.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


def build_lstm(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """input_shape: (time_steps, n_features)."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="lstm")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
