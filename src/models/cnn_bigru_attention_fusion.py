import tensorflow as tf
from tensorflow.keras import layers, Model


class FeatureFusion(layers.Layer):
    def call(self, inputs):
        logmel, mfcc = inputs
        return tf.concat([logmel, mfcc], axis=1)


def build_cnn_bigru_attention_fusion(
    input_shape=(184, 174, 1),
    num_classes=6
):
    """
    CNN + BiGRU + Multi-Head Attention
    using fused Log-Mel + MFCC features.

    Input:
        Log-Mel: 64 x 174
        MFCC:    120 x 174

    Fused:
        184 x 174
    """

    logmel_input = layers.Input(
        shape=(64, 174, 1),
        name="logmel_input"
    )

    mfcc_input = layers.Input(
        shape=(120, 174, 1),
        name="mfcc_input"
    )

    # -------------------------------------------------
    # Feature Fusion
    # -------------------------------------------------

    logmel = layers.Reshape(
        (64, 174)
    )(logmel_input)

    mfcc = layers.Reshape(
        (120, 174)
    )(mfcc_input)

    fused = FeatureFusion(name="feature_fusion")(
        [logmel, mfcc]
    )

    x = layers.Reshape(
        (184, 174, 1),
        name="fused_reshape"
    )(fused)

    # -------------------------------------------------
    # CNN Block 1
    # -------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.15)(x)

    # -------------------------------------------------
    # CNN Block 2
    # -------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.20)(x)

    # -------------------------------------------------
    # CNN Block 3
    # -------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)

    # -------------------------------------------------
    # Convert CNN feature maps to sequence
    # -------------------------------------------------

    x = layers.Permute(
        (2, 1, 3)
    )(x)

    shape = x.shape

    x = layers.Reshape(
        (shape[1], shape[2] * shape[3])
    )(x)

    # -------------------------------------------------
    # BiGRU
    # -------------------------------------------------

    x = layers.Bidirectional(
        layers.GRU(
            128,
            return_sequences=True,
            dropout=0.20
        ),
        name="bidirectional_gru"
    )(x)

    # -------------------------------------------------
    # Multi-Head Self Attention
    # -------------------------------------------------

    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.10,
        name="self_attention"
    )(x, x)

    # Residual connection
    x = layers.Add()(
        [x, attention_output]
    )

    x = layers.LayerNormalization()(x)

    # -------------------------------------------------
    # Global pooling
    # -------------------------------------------------

    x = layers.GlobalAveragePooling1D()(x)

    # -------------------------------------------------
    # Classification Head
    # -------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.40)(x)

    output = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = Model(
        inputs=[logmel_input, mfcc_input],
        outputs=output,
        name="CNN_BiGRU_Attention_FeatureFusion"
    )

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model