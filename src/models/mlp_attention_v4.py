import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class TemporalAttention(layers.Layer):
    """
    Learnable temporal attention pooling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(
            1,
            activation="tanh"
        )

    def call(self, x):

        scores = self.score_layer(x)

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        context = tf.reduce_sum(
            x * weights,
            axis=1
        )

        return context


def build_mlp_attention_v4(
    input_shape,
    num_classes=6
):

    inputs = layers.Input(
        shape=input_shape,
        name="input_features"
    )

    # ========================================================
    # FEATURE PROJECTION
    # ========================================================

    x = layers.Dense(
        160,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="feature_projection"
    )(inputs)

    x = layers.LayerNormalization(
        name="projection_norm"
    )(x)

    x = layers.Dropout(
        0.15,
        name="projection_dropout"
    )(x)

    # ========================================================
    # MULTI-HEAD TEMPORAL ATTENTION
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=40,
        dropout=0.10,
        name="multi_head_temporal_attention"
    )(
        query=x,
        value=x,
        key=x
    )

    x = layers.Add(
        name="attention_residual"
    )(
        [x, attention_output]
    )

    x = layers.LayerNormalization(
        name="attention_norm"
    )(x)

    # ========================================================
    # TEMPORAL ATTENTION POOLING
    # ========================================================

    attention_context = TemporalAttention(
        name="temporal_attention"
    )(x)

    # ========================================================
    # GLOBAL MEAN POOLING
    # ========================================================

    mean_context = layers.GlobalAveragePooling1D(
        name="mean_pooling"
    )(x)

    # ========================================================
    # FUSION
    # ========================================================

    x = layers.Concatenate(
        name="attention_mean_fusion"
    )(
        [
            attention_context,
            mean_context
        ]
    )

    # ========================================================
    # CLASSIFICATION MLP
    # ========================================================

    x = layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1.5e-4),
        name="dense_256"
    )(x)

    x = layers.BatchNormalization(
        name="bn_256"
    )(x)

    x = layers.Dropout(
        0.35,
        name="dropout_256"
    )(x)

    x = layers.Dense(
        128,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1.5e-4),
        name="dense_128"
    )(x)

    x = layers.BatchNormalization(
        name="bn_128"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_128"
    )(x)

    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1.5e-4),
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.25,
        name="dropout_64"
    )(x)

    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="mlp_temporal_attention_v4"
    )

    return model