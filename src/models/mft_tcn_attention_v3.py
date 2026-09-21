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
        dropout=0.15,
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

        residual = inputs

        x = self.norm(inputs)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout2(x, training=training)

        residual = self.residual_projection(residual)

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
# GATED MULTI-SCALE FUSION
# ============================================================

class GatedFusion(layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters

        self.projection = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same"
        )

        self.gate = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same",
            activation="sigmoid"
        )

    def call(self, inputs):

        x = self.projection(inputs)

        gate = self.gate(inputs)

        # Gate controls how much transformed information
        # is retained versus the original information.
        return gate * x + (1.0 - gate) * inputs[:, :, :self.filters]

    def get_config(self):

        config = super().get_config()

        config.update({
            "filters": self.filters
        })

        return config


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

def build_mft_tcn_attention_v3(
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
        160,
        name="feature_projection"
    )(inputs)

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.15)(x)

    # --------------------------------------------------------
    # LOCAL TEMPORAL PATH
    # dilation = 1, 2
    # --------------------------------------------------------

    local = x

    local = TCNBlock(
        filters=160,
        dilation_rate=1,
        dropout=0.15,
        name="local_tcn_1"
    )(local)

    local = TCNBlock(
        filters=160,
        dilation_rate=2,
        dropout=0.15,
        name="local_tcn_2"
    )(local)

    # --------------------------------------------------------
    # LONG-RANGE TEMPORAL PATH
    # dilation = 4, 8, 16
    # --------------------------------------------------------

    long_range = x

    long_range = TCNBlock(
        filters=160,
        dilation_rate=4,
        dropout=0.15,
        name="long_tcn_4"
    )(long_range)

    long_range = TCNBlock(
        filters=160,
        dilation_rate=8,
        dropout=0.15,
        name="long_tcn_8"
    )(long_range)

    long_range = TCNBlock(
        filters=160,
        dilation_rate=16,
        dropout=0.15,
        name="long_tcn_16"
    )(long_range)

    # --------------------------------------------------------
    # MULTI-SCALE FUSION
    # --------------------------------------------------------

    fused = layers.Concatenate(
        name="multi_scale_concat"
    )([
        local,
        long_range
    ])

    fused = GatedFusion(
        filters=160,
        name="gated_fusion"
    )(fused)

    # --------------------------------------------------------
    # MULTI-HEAD ATTENTION
    # --------------------------------------------------------

    attention_input = layers.LayerNormalization(
        name="attention_norm"
    )(fused)

    attention = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=20,
        dropout=0.10,
        name="multi_head_attention"
    )(
        attention_input,
        attention_input
    )

    fused = layers.Add()([
        fused,
        attention
    ])

    # --------------------------------------------------------
    # FEED FORWARD REFINEMENT
    # --------------------------------------------------------

    ff_norm = layers.LayerNormalization()(fused)

    ff = layers.Dense(
        320,
        activation="gelu"
    )(ff_norm)

    ff = layers.Dropout(0.20)(ff)

    ff = layers.Dense(
        160
    )(ff)

    fused = layers.Add()([
        fused,
        ff
    ])

    # --------------------------------------------------------
    # HYBRID POOLING
    # --------------------------------------------------------

    avg_pool = layers.GlobalAveragePooling1D(
        name="average_pooling"
    )(fused)

    max_pool = layers.GlobalMaxPooling1D(
        name="max_pooling"
    )(fused)

    attention_pool = AttentionPooling(
        name="attention_pooling"
    )(fused)

    pooled = layers.Concatenate(
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
        activation="gelu"
    )(pooled)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(
        64,
        activation="gelu"
    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    return Model(
        inputs,
        outputs,
        name="MFT_TCN_Attention_V3"
    )