import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# TEMPORAL GATED TCN BLOCK
# ============================================================

class GatedTCNBlock(layers.Layer):

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

        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )

        self.gate_conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            activation="sigmoid"
        )

        self.dropout = layers.Dropout(
            dropout
        )

        self.residual_projection = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same"
        )

    def call(self, inputs, training=None):

        residual = self.residual_projection(
            inputs
        )

        x = self.norm(inputs)

        features = self.conv(x)

        features = tf.nn.gelu(
            features
        )

        gate = self.gate_conv(x)

        x = features * gate

        x = self.dropout(
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
# ATTENTION POOLING
# ============================================================

class AttentionPooling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(
            1
        )

    def call(self, inputs):

        scores = self.score_layer(
            inputs
        )

        scores = tf.nn.softmax(
            scores,
            axis=1
        )

        return tf.reduce_sum(
            inputs * scores,
            axis=1
        )

    def get_config(self):

        return super().get_config()


# ============================================================
# BUILD V5
# ============================================================

def build_mft_tcn_attention_v5(
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

    x = layers.Dropout(
        0.15
    )(x)

    # --------------------------------------------------------
    # MULTI-SCALE GATED TCN
    # --------------------------------------------------------

    x = GatedTCNBlock(
        filters=160,
        dilation_rate=1,
        dropout=0.20,
        name="gated_tcn_1"
    )(x)

    x = GatedTCNBlock(
        filters=160,
        dilation_rate=2,
        dropout=0.20,
        name="gated_tcn_2"
    )(x)

    x = GatedTCNBlock(
        filters=160,
        dilation_rate=4,
        dropout=0.20,
        name="gated_tcn_4"
    )(x)

    x = GatedTCNBlock(
        filters=160,
        dilation_rate=8,
        dropout=0.20,
        name="gated_tcn_8"
    )(x)

    x = GatedTCNBlock(
        filters=160,
        dilation_rate=16,
        dropout=0.20,
        name="gated_tcn_16"
    )(x)

    # --------------------------------------------------------
    # MULTI-HEAD SELF ATTENTION
    # --------------------------------------------------------

    attention_input = layers.LayerNormalization(
        name="attention_norm"
    )(x)

    attention = layers.MultiHeadAttention(
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
        attention
    ])

    # --------------------------------------------------------
    # FEED FORWARD REFINEMENT
    # --------------------------------------------------------

    ff_input = layers.LayerNormalization(
        name="ff_norm"
    )(x)

    ff = layers.Dense(
        320,
        activation="gelu"
    )(ff_input)

    ff = layers.Dropout(
        0.25
    )(ff)

    ff = layers.Dense(
        160
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

    average_pool = layers.GlobalAveragePooling1D(
        name="average_pooling"
    )(x)

    max_pool = layers.GlobalMaxPooling1D(
        name="max_pooling"
    )(x)

    attention_pool = AttentionPooling(
        name="attention_pooling"
    )(x)

    pooled = layers.Concatenate(
        name="hybrid_pooling"
    )([
        average_pool,
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

    x = layers.Dropout(
        0.35
    )(x)

    x = layers.Dense(
        64,
        activation="gelu"
    )(x)

    x = layers.Dropout(
        0.25
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    return Model(
        inputs,
        outputs,
        name="MFT_TCN_Attention_V5"
    )