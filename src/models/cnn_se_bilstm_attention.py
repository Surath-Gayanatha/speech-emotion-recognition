import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# Squeeze-and-Excitation Block
# ============================================================

class SEBlock(layers.Layer):

    def __init__(
        self,
        channels,
        reduction=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.channels = channels
        self.reduction = reduction

        self.global_pool = layers.GlobalAveragePooling2D()

        self.dense1 = layers.Dense(
            max(channels // reduction, 8),
            activation="relu"
        )

        self.dense2 = layers.Dense(
            channels,
            activation="sigmoid"
        )

        self.reshape = layers.Reshape(
            (1, 1, channels)
        )

    def call(self, inputs):

        # Channel descriptor
        x = self.global_pool(inputs)

        # Bottleneck
        x = self.dense1(x)

        # Channel attention weights
        x = self.dense2(x)

        # Restore spatial dimensions
        x = self.reshape(x)

        # Re-weight feature maps
        return inputs * x


# ============================================================
# CNN + SE + BiLSTM + Multi-Head Attention
# ============================================================

def build_cnn_se_bilstm_attention(
    input_shape=(64, 174, 1),
    num_classes=6
):

    inputs = layers.Input(
        shape=input_shape,
        name="logmel_input"
    )

    # ========================================================
    # CNN Block 1
    # ========================================================

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = SEBlock(32)(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.15)(x)


    # ========================================================
    # CNN Block 2
    # ========================================================

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = SEBlock(64)(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.20)(x)


    # ========================================================
    # CNN Block 3
    # ========================================================

    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = SEBlock(128)(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)


    # ========================================================
    # Convert CNN Feature Maps to Sequence
    # ========================================================

    # Shape after CNN:
    # (8, 21, 128)

    x = layers.Permute(
        (2, 1, 3)
    )(x)

    # (21, 8, 128) -> (21, 1024)

    x = layers.Reshape(
        (21, 8 * 128)
    )(x)


    # ========================================================
    # Feature Projection
    # ========================================================

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.20)(x)


    # ========================================================
    # BiLSTM
    # ========================================================

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.20
        )
    )(x)


    # ========================================================
    # Multi-Head Self Attention
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32,
        dropout=0.10
    )(
        x,
        x
    )

    # Residual connection
    x = layers.Add()(
        [
            x,
            attention_output
        ]
    )

    x = layers.LayerNormalization()(x)


    # ========================================================
    # Feed Forward Network
    # ========================================================

    ffn = layers.Dense(
        256,
        activation=tf.keras.activations.gelu
    )(x)

    ffn = layers.Dropout(0.20)(ffn)

    ffn = layers.Dense(
        256
    )(ffn)

    x = layers.Add()(
        [
            x,
            ffn
        ]
    )

    x = layers.LayerNormalization()(x)


    # ========================================================
    # Global Temporal Pooling
    # ========================================================

    x = layers.GlobalAveragePooling1D()(x)


    # ========================================================
    # Classification Head
    # ========================================================

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)


    # ========================================================
    # Build Model
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_SE_BiLSTM_Attention"
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-4
    )


    # ========================================================
    # Compile
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    return model