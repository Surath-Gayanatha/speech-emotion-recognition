import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# TCN BLOCK
# ============================================================

class TCNBlockV2(layers.Layer):

    def __init__(
        self,
        filters=160,
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
                filters=self.filters,
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

class AttentionPoolingV2(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(1)

    def call(self, inputs):

        scores = self.score_layer(inputs)

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        return tf.reduce_sum(
            inputs * weights,
            axis=1
        )

    def get_config(self):

        return super().get_config()


# ============================================================
# MFT TCN + ATTENTION V2
# ============================================================

def build_mft_tcn_attention_v2(
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
        160,
        name="feature_projection"
    )(inputs)

    x = layers.LayerNormalization(
        name="input_normalization"
    )(x)

    x = layers.Dropout(
        0.20,
        name="input_dropout"
    )(x)

    # --------------------------------------------------------
    # TCN BLOCKS
    # --------------------------------------------------------

    dilation_rates = [1, 2, 4, 8, 16]

    for i, dilation in enumerate(dilation_rates):

        x = TCNBlockV2(
            filters=160,
            kernel_size=3,
            dilation_rate=dilation,
            dropout=0.20,
            name=f"tcn_block_{i + 1}"
        )(x)

    # --------------------------------------------------------
    # MULTI-HEAD SELF ATTENTION
    # --------------------------------------------------------

    attention_input = layers.LayerNormalization(
        name="attention_normalization"
    )(x)

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=20,
        dropout=0.10,
        name="multi_head_attention"
    )(
        attention_input,
        attention_input
    )

    x = layers.Add(
        name="attention_residual"
    )([
        x,
        attention_output
    ])

    # --------------------------------------------------------
    # FEED-FORWARD REFINEMENT
    # --------------------------------------------------------

    ff_input = layers.LayerNormalization(
        name="ff_normalization"
    )(x)

    ff = layers.Dense(
        320,
        activation="gelu",
        name="ff_dense_1"
    )(ff_input)

    ff = layers.Dropout(
        0.20,
        name="ff_dropout"
    )(ff)

    ff = layers.Dense(
        160,
        name="ff_dense_2"
    )(ff)

    x = layers.Add(
        name="ff_residual"
    )([
        x,
        ff
    ])

    # --------------------------------------------------------
    # HYBRID TEMPORAL POOLING
    # --------------------------------------------------------

    avg_pool = layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    max_pool = layers.GlobalMaxPooling1D(
        name="global_max_pooling"
    )(x)

    attention_pool = AttentionPoolingV2(
        name="attention_pooling"
    )(x)

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
        0.35,
        name="classifier_dropout_1"
    )(x)

    x = layers.Dense(
        64,
        activation="gelu",
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.25,
        name="classifier_dropout_2"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="MFT_TCN_Attention_V2"
    )