"""MLP baseline for Speech Emotion Recognition."""

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.config import NUM_CLASSES


def build_mlp(
    input_shape: tuple,
    num_classes: int = NUM_CLASSES
) -> keras.Model:

    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mlp_regularized"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0003
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model