import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class TemporalAttention(layers.Layer):
    """
    Learns the importance of each time frame and
    produces a weighted temporal representation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(
            1,
            activation="tanh"
        )

    def call(self, inputs):

        # inputs:
        # (batch, time_steps, features)

        scores = self.score_layer(inputs)

        # (batch, time_steps, 1)
        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        # Weighted sum across time
        context = tf.reduce_sum(
            inputs * weights,
            axis=1
        )

        return context


def build_mlp_attention(
    input_shape,
    num_classes=6
):

    inputs = layers.Input(
        shape=input_shape,
        name="input_features"
    )

    # ========================================================
    # FRAME-LEVEL FEATURE PROJECTION
    # ========================================================

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.25
    )(x)


    # ========================================================
    # TEMPORAL ATTENTION
    # ========================================================

    x = TemporalAttention(
        name="temporal_attention"
    )(x)


    # ========================================================
    # MLP CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.40
    )(x)


    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.35
    )(x)


    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.Dropout(
        0.30
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
        name="mlp_temporal_attention"
    )


    return model