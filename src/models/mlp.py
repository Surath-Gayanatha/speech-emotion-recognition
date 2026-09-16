"""MLP baseline for Speech Emotion Recognition."""

from tensorflow import keras
from tensorflow.keras import layers
from src.config import NUM_CLASSES


def build_mlp(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ], name="mlp_baseline")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model