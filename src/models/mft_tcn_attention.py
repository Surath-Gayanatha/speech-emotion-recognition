import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# TCN BLOCK
# ============================================================

class TCNBlock(layers.Layer):
    """
    Temporal Convolutional Network block.

    Uses dilated Conv1D layers to capture temporal patterns
    at different time scales.
    """

    def __init__(
        self,
        filters=128,
        kernel_size=3,
        dilation_rate=1,
        dropout=0.20,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout

        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )

        self.norm1 = layers.LayerNormalization()

        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )

        self.norm2 = layers.LayerNormalization()

        self.dropout = layers.Dropout(dropout)

        self.activation = layers.Activation("gelu")

        self.residual_projection = None

    def build(self, input_shape):

        if input_shape[-1] != self.filters:
            self.residual_projection = layers.Conv1D(
                self.filters,
                kernel_size=1,
                padding="same"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):

        residual = inputs

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)

        if self.residual_projection is not None:
            residual = self.residual_projection(residual)

        return x + residual

    def get_config(self):

        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout": self.dropout_rate
        })

        return config


# ============================================================
# ATTENTION POOLING
# ============================================================

class AttentionPooling(layers.Layer):
    """
    Learns which time steps are important.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(1)

    def call(self, inputs):

        scores = self.score_layer(inputs)

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        weighted = inputs * weights

        return tf.reduce_sum(
            weighted,
            axis=1
        )

    def get_config(self):

        return super().get_config()


# ============================================================
# MFT TCN + ATTENTION MODEL
# ============================================================

def build_mft_tcn_attention(
    input_shape=(256, 201),
    num_classes=6
):

    inputs = layers.Input(
        shape=input_shape,
        name="mft_input"
    )

    # --------------------------------------------------------
    # FEATURE PROJECTION
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        name="feature_projection"
    )(inputs)

    x = layers.LayerNormalization(
        name="input_normalization"
    )(x)

    x = layers.Dropout(
        0.15,
        name="input_dropout"
    )(x)

    # --------------------------------------------------------
    # TCN BLOCKS
    # --------------------------------------------------------

    x = TCNBlock(
        filters=128,
        kernel_size=3,
        dilation_rate=1,
        dropout=0.15,
        name="tcn_block_1"
    )(x)

    x = TCNBlock(
        filters=128,
        kernel_size=3,
        dilation_rate=2,
        dropout=0.15,
        name="tcn_block_2"
    )(x)

    x = TCNBlock(
        filters=128,
        kernel_size=3,
        dilation_rate=4,
        dropout=0.15,
        name="tcn_block_3"
    )(x)

    x = TCNBlock(
        filters=128,
        kernel_size=3,
        dilation_rate=8,
        dropout=0.15,
        name="tcn_block_4"
    )(x)

    # --------------------------------------------------------
    # MULTI-HEAD SELF ATTENTION
    # --------------------------------------------------------

    attention_input = layers.LayerNormalization(
        name="attention_normalization"
    )(x)

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=16,
        dropout=0.10,
        name="multi_head_attention"
    )(
        attention_input,
        attention_input
    )

    x = layers.Add(
        name="attention_residual"
    )([x, attention_output])

    # --------------------------------------------------------
    # SECOND NORMALIZATION
    # --------------------------------------------------------

    x = layers.LayerNormalization(
        name="post_attention_normalization"
    )(x)

    # --------------------------------------------------------
    # TWO TYPES OF TEMPORAL POOLING
    # --------------------------------------------------------

    avg_pool = layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    max_pool = layers.GlobalMaxPooling1D(
        name="global_max_pooling"
    )(x)

    attention_pool = AttentionPooling(
        name="attention_pooling"
    )(x)

    # Combine all temporal representations

    x = layers.Concatenate(
        name="hybrid_pooling"
    )([
        avg_pool,
        max_pool,
        attention_pool
    ])

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="gelu",
        name="dense_128"
    )(x)

    x = layers.Dropout(
        0.30,
        name="classifier_dropout_1"
    )(x)

    x = layers.Dense(
        64,
        activation="gelu",
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.20,
        name="classifier_dropout_2"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="MFT_TCN_Attention"
    )

    return model