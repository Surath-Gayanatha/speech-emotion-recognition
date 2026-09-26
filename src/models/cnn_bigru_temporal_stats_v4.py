import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# TEMPORAL ATTENTION POOLING
# ============================================================

class TemporalAttentionPooling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.score_layer = layers.Dense(1)

        super().build(input_shape)

    def call(self, inputs):

        # inputs:
        # (batch, time, features)

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


# ============================================================
# TEMPORAL STATISTICS POOLING
# ============================================================

class TemporalStatisticsPooling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):

        # Mean across time
        mean = tf.reduce_mean(
            inputs,
            axis=1
        )

        # Standard deviation across time
        std = tf.math.reduce_std(
            inputs,
            axis=1
        )

        # Concatenate mean + std
        return tf.concat(
            [mean, std],
            axis=-1
        )


# ============================================================
# CNN BLOCK
# ============================================================

def cnn_block(
    x,
    filters,
    dropout_rate
):

    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(
        dropout_rate
    )(x)

    return x


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    input_shape=(64, 174, 1),
    num_classes=6
):

    inputs = layers.Input(
        shape=input_shape
    )

    # ========================================================
    # CNN FEATURE EXTRACTION
    # ========================================================

    x = cnn_block(
        inputs,
        filters=32,
        dropout_rate=0.15
    )

    x = cnn_block(
        x,
        filters=64,
        dropout_rate=0.20
    )

    x = cnn_block(
        x,
        filters=128,
        dropout_rate=0.25
    )

    # ========================================================
    # CNN → TEMPORAL SEQUENCE
    # ========================================================

    # Before:
    # (batch, 8, 21, 128)

    x = layers.Permute(
        (2, 1, 3)
    )(x)

    # After:
    # (batch, 21, 8, 128)

    x = layers.Reshape(
        (21, 1024)
    )(x)

    # ========================================================
    # FEATURE PROJECTION
    # ========================================================

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(
        0.20
    )(x)

    # ========================================================
    # BIDIRECTIONAL GRU
    # ========================================================

    x = layers.Bidirectional(
        layers.GRU(
            128,
            return_sequences=True,
            dropout=0.20
        )
    )(x)

    # Output:
    # (batch, 21, 256)

    # ========================================================
    # MULTI-HEAD SELF ATTENTION
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

    x = layers.Add()([
        x,
        attention_output
    ])

    x = layers.LayerNormalization()(x)

    # ========================================================
    # DUAL TEMPORAL POOLING
    # ========================================================

    # Branch 1:
    # Attention-based temporal representation

    attention_features = TemporalAttentionPooling(
        name="attention_pooling"
    )(x)

    # Branch 2:
    # Mean + Standard deviation representation

    statistical_features = TemporalStatisticsPooling(
        name="statistics_pooling"
    )(x)

    # ========================================================
    # FUSION
    # ========================================================

    x = layers.Concatenate(
        name="temporal_fusion"
    )([
        attention_features,
        statistical_features
    ])

    # Attention branch = 256
    # Statistics branch = 512
    # Total = 768

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.35
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiGRU_TemporalStats_V4"
    )

    return model