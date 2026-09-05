"""
CNN + Self-Attention for Speech Emotion Recognition.

Input:
    (time_steps, n_features)

CNN layers learn local spectral-temporal patterns.
Self-attention learns which time steps are important
for emotion classification.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


def build_cnn_attention(
    input_shape: tuple,
    num_classes: int = NUM_CLASSES
) -> keras.Model:

    inputs = keras.Input(shape=input_shape)

    # ---------------------------------------------------------
    # Input normalization
    # ---------------------------------------------------------

    x = layers.LayerNormalization()(inputs)

    # ---------------------------------------------------------
    # CNN Block 1
    # ---------------------------------------------------------

    x = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2
    )(x)

    x = layers.Dropout(0.20)(x)

    # ---------------------------------------------------------
    # CNN Block 2
    # ---------------------------------------------------------

    x = layers.Conv1D(
        128,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2
    )(x)

    x = layers.Dropout(0.25)(x)

    # ---------------------------------------------------------
    # CNN feature projection
    # ---------------------------------------------------------

    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    # ---------------------------------------------------------
    # Self-Attention
    # ---------------------------------------------------------

    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.10
    )(
        query=x,
        value=x,
        key=x
    )

    # Residual connection
    x = layers.Add()(
        [x, attention_output]
    )

    x = layers.LayerNormalization()(x)

    # Feed-forward projection after attention
    ff = layers.Dense(
        256,
        activation="relu"
    )(x)

    ff = layers.Dropout(0.20)(ff)

    ff = layers.Dense(
        128
    )(ff)

    x = layers.Add()(
        [x, ff]
    )

    x = layers.LayerNormalization()(x)

    # ---------------------------------------------------------
    # Global representation
    # ---------------------------------------------------------

    x = layers.GlobalAveragePooling1D()(x)

    # ---------------------------------------------------------
    # Classification head
    # ---------------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.40)(x)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_attention"
    )

    # ---------------------------------------------------------
    # Compile
    # ---------------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model