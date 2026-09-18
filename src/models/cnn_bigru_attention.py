"""CNN + BiGRU + Attention for Log-Mel Speech Emotion Recognition."""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_bigru_attention(
    input_shape=(64, 174, 1),
    num_classes=6,
):
    """Build CNN + BiGRU + Multi-Head Attention model."""

    inputs = keras.Input(
        shape=input_shape,
        name="logmel_input",
    )

    # ========================================================
    # CNN FEATURE EXTRACTION
    # ========================================================

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv2d_32",
    )(inputs)

    x = layers.BatchNormalization(
        name="bn_32"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool_32",
    )(x)

    x = layers.Dropout(
        0.20,
        name="dropout_32",
    )(x)

    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv2d_64",
    )(x)

    x = layers.BatchNormalization(
        name="bn_64"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool_64",
    )(x)

    x = layers.Dropout(
        0.25,
        name="dropout_64",
    )(x)

    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv2d_128",
    )(x)

    x = layers.BatchNormalization(
        name="bn_128"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool_128",
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_128",
    )(x)

    # ========================================================
    # CNN FEATURE MAP → TEMPORAL SEQUENCE
    # ========================================================

    x = layers.Permute(
        (2, 1, 3),
        name="time_frequency_permute",
    )(x)

    # After 3 pooling layers:
    # 64 → 32 → 16 → 8
    # 174 → 87 → 43 → 21
    #
    # Therefore:
    # time steps = 21
    # features per timestep = 8 × 128 = 1024

    x = layers.Reshape(
        (21, 1024),
        name="sequence_reshape",
    )(x)

    # ========================================================
    # BIDIRECTIONAL GRU
    # ========================================================

    x = layers.Bidirectional(
        layers.GRU(
            128,
            return_sequences=True,
            dropout=0.20,
        ),
        name="bigru",
    )(x)

    # ========================================================
    # MULTI-HEAD SELF ATTENTION
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.10,
        name="self_attention",
    )(x, x)

    # Residual connection
    x = layers.Add(
        name="attention_residual"
    )([x, attention_output])

    x = layers.LayerNormalization(
        name="attention_layer_norm"
    )(x)

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="dense_128",
    )(x)

    x = layers.Dropout(
        0.40,
        name="classification_dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output",
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_bigru_attention",
    )

    # ========================================================
    # COMPILATION
    # ========================================================

    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model