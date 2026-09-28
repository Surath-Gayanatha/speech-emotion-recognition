import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


# ============================================================
# PADDING-AWARE ATTENTION
# ============================================================

class MaskedTemporalAttention(layers.Layer):
    """
    Temporal attention that can ignore padded frames.

    inputs:
        x    -> (batch, time, features)
        mask -> (batch, time)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = layers.Dense(
            1,
            activation="tanh"
        )

    def call(self, inputs):
        x, mask = inputs

        # Attention scores
        scores = self.score_layer(x)

        # (batch, time, 1)
        mask = tf.cast(
            mask,
            scores.dtype
        )

        mask = tf.expand_dims(
            mask,
            axis=-1
        )

        # Give padded positions a very negative score
        scores = tf.where(
            mask > 0,
            scores,
            tf.ones_like(scores) * -1e9
        )

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        weighted = x * weights

        context = tf.reduce_sum(
            weighted,
            axis=1
        )

        return context


# ============================================================
# MODEL
# ============================================================

def build_mlp_attention_v3(
    input_shape,
    num_classes=6
):
    """
    MLP + Padding-Aware Temporal Attention V3

    Architecture:
        Dense projection
        Multi-head temporal attention
        Residual connection
        Feed-forward refinement
        Padding-aware attention pooling
        Mean pooling
        Fusion
        MLP classifier
    """

    inputs = layers.Input(
        shape=input_shape,
        name="input_features"
    )

    # ========================================================
    # CREATE PADDING MASK
    # ========================================================
    #
    # A frame is considered padded when all its features
    # are approximately zero.
    #
    # The actual StandardScaler can make this imperfect,
    # so we use a small threshold.
    # ========================================================

    frame_energy = layers.Lambda(
        lambda x: tf.reduce_sum(
            tf.abs(x),
            axis=-1
        ),
        name="frame_energy"
    )(inputs)

    mask = layers.Lambda(
        lambda x: tf.cast(
            x > 1e-6,
            tf.float32
        ),
        name="padding_mask"
    )(frame_energy)


    # ========================================================
    # FEATURE PROJECTION
    # ========================================================

    x = layers.Dense(
        160,
        activation="gelu",
        kernel_regularizer=regularizers.l2(
            1e-4
        ),
        name="feature_projection"
    )(inputs)

    x = layers.LayerNormalization(
        name="projection_norm"
    )(x)

    x = layers.Dropout(
        0.20,
        name="projection_dropout"
    )(x)


    # ========================================================
    # MULTI-HEAD TEMPORAL ATTENTION
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=40,
        dropout=0.15,
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
    # FEED-FORWARD REFINEMENT
    # ========================================================

    ff = layers.Dense(
        320,
        activation="gelu",
        kernel_regularizer=regularizers.l2(
            1e-4
        ),
        name="ff_dense_1"
    )(x)

    ff = layers.Dropout(
        0.20,
        name="ff_dropout"
    )(ff)

    ff = layers.Dense(
        160,
        activation=None,
        kernel_regularizer=regularizers.l2(
            1e-4
        ),
        name="ff_dense_2"
    )(ff)

    x = layers.Add(
        name="ff_residual"
    )(
        [x, ff]
    )

    x = layers.LayerNormalization(
        name="ff_norm"
    )(x)


    # ========================================================
    # PADDING-AWARE ATTENTION POOLING
    # ========================================================

    attention_context = MaskedTemporalAttention(
        name="masked_temporal_attention"
    )(
        [x, mask]
    )


    # ========================================================
    # MASKED MEAN POOLING
    # ========================================================

    mask_expanded = layers.Lambda(
        lambda m: tf.expand_dims(
            m,
            axis=-1
        ),
        name="expanded_mask"
    )(mask)


    masked_x = layers.Multiply(
        name="masked_features"
    )(
        [x, mask_expanded]
    )


    mean_context = layers.Lambda(
        lambda tensors: (
            tf.reduce_sum(
                tensors[0],
                axis=1
            )
            /
            (
                tf.reduce_sum(
                    tensors[1],
                    axis=1
                )
                + 1e-6
            )
        ),
        name="masked_mean_pooling"
    )(
        [
            masked_x,
            mask_expanded
        ]
    )


    # ========================================================
    # ATTENTION + MEAN FUSION
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
        kernel_regularizer=regularizers.l2(
            2e-4
        ),
        name="dense_256"
    )(x)

    x = layers.BatchNormalization(
        name="bn_256"
    )(x)

    x = layers.Dropout(
        0.45,
        name="dropout_256"
    )(x)


    x = layers.Dense(
        128,
        activation="gelu",
        kernel_regularizer=regularizers.l2(
            2e-4
        ),
        name="dense_128"
    )(x)

    x = layers.BatchNormalization(
        name="bn_128"
    )(x)

    x = layers.Dropout(
        0.40,
        name="dropout_128"
    )(x)


    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(
            2e-4
        ),
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.35,
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
        name="mlp_temporal_attention_v3"
    )

    return model