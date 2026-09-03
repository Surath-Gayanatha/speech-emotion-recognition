"""1D CNN for Speech Emotion Recognition.

Architectural role: convolves filters along the time axis of the MFCC
sequence to learn local spectral-temporal patterns (short bursts of
pitch/energy change) without modeling long-range sequential dependency
the way LSTM/BiLSTM do. Distinguishes itself from the MLP by preserving
time structure, and from LSTM/BiLSTM by only capturing *local* rather than
*long-range* temporal context.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


def build_cnn1d(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """input_shape: (time_steps, n_features) -- MFCC transposed so time is axis 0."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn1d")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
