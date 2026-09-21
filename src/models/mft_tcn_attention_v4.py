import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# TCN BLOCK
# ============================================================

class TCNBlock(layers.Layer):

    def __init__(
        self,
        filters,
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

        self.norm = layers.LayerNormalization()

        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )

        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )

        self.activation = layers.Activation("gelu")

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        self.residual_projection = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same"
        )

    def call(self, inputs, training=None):

        residual = self.residual_projection(inputs)

        x = self.norm(inputs)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout1(
            x,
            training=training
        )

        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout2(
            x,
            training=training
        )

        return residual + x

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
# TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(1)

    def call(self, inputs):

        scores = self.score_layer(inputs)

        scores = tf.nn.softmax(
            scores,
            axis=1
        )

        return inputs * scores

    def get_config(self):

        return super().get_config()


# ============================================================
# ATTENTION POOLING
# ============================================================

class AttentionPooling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(1)

    def call(self, inputs):

        scores = self.score_layer(inputs)

        scores = tf.nn.softmax(
            scores,
            axis=1
        )

        weighted = inputs * scores

        return tf.reduce_sum(
            weighted,
            axis=1
        )

    def get_config(self):

        return super().get_config()


# ============================================================
# BUILD MODEL
# ============================================================

def build_mft_tcn_attention_v4(
    input_shape=(256, 201),
    num_classes=6
):

    inputs = layers.Input(
        shape=input_shape,
        name="mft_features"
    )

    # --------------------------------------------------------
    # FEATURE PROJECTION
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        name="feature_projection"
    )(inputs)

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.20)(x)

    # --------------------------------------------------------
    # TEMPORAL FEATURE EXTRACTION
    # --------------------------------------------------------

    x = TCNBlock(
        filters=128,
        dilation_rate=1,
        dropout=0.20,
        name="tcn_dilation_1"
    )(x)

    x = TCNBlock(
        filters=128,
        dilation_rate=2,
        dropout=0.20,
        name="tcn_dilation_2"
    )(x)

    x = TCNBlock(
        filters=128,
        dilation_rate=4,
        dropout=0.20,
        name="tcn_dilation_4"
    )(x)

    # --------------------------------------------------------
    # TEMPORAL ATTENTION
    # --------------------------------------------------------

    temporal_attention = TemporalAttention(
        name="temporal_attention"
    )(x)

    x = layers.Add(
        name="temporal_attention_residual"
    )([
        x,
        temporal_attention
    ])

    # --------------------------------------------------------
    # MULTI-HEAD SELF ATTENTION
    # --------------------------------------------------------

    attention_norm = layers.LayerNormalization(
        name="mha_norm"
    )(x)

    self_attention = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=16,
        dropout=0.10,
        name="multi_head_attention"
    )(
        attention_norm,
        attention_norm
    )

    x = layers.Add(
        name="mha_residual"
    )([
        x,
        self_attention
    ])

    # --------------------------------------------------------
    # FEED-FORWARD REFINEMENT
    # --------------------------------------------------------

    ff_norm = layers.LayerNormalization(
        name="ff_norm"
    )(x)

    ff = layers.Dense(
        256,
        activation="gelu"
    )(ff_norm)

    ff = layers.Dropout(0.25)(ff)

    ff = layers.Dense(
        128
    )(ff)

    x = layers.Add(
        name="ff_residual"
    )([
        x,
        ff
    ])

    # --------------------------------------------------------
    # HYBRID POOLING
    # --------------------------------------------------------

    avg_pool = layers.GlobalAveragePooling1D(
        name="average_pooling"
    )(x)

    attention_pool = AttentionPooling(
        name="attention_pooling"
    )(x)

    pooled = layers.Concatenate(
        name="hybrid_pooling"
    )([
        avg_pool,
        attention_pool
    ])

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="gelu"
    )(pooled)

    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        64,
        activation="gelu"
    )(x)

    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    return Model(
        inputs,
        outputs,
        name="MFT_TCN_Attention_V4"
    )