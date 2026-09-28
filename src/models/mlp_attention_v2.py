import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def build_mlp_attention_v2(
    input_shape,
    num_classes=6
):
    """
    MLP + Multi-Head Temporal Attention V2

    Input:
        (time_steps, 192)

    Architecture:
        Dense projection
        Multi-head temporal self-attention
        Attention pooling
        Mean pooling
        Feature fusion
        MLP classification head
    """

    inputs = layers.Input(
        shape=input_shape,
        name="input_features"
    )

    # ========================================================
    # FEATURE PROJECTION
    # ========================================================

    x = layers.Dense(
        160,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
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
        x,
        x
    )

    # Residual connection
    x = layers.Add(
        name="attention_residual"
    )(
        [x, attention_output]
    )

    x = layers.LayerNormalization(
        name="attention_norm"
    )(x)


    # ========================================================
    # ATTENTION POOLING
    # ========================================================

    attention_scores = layers.Dense(
        1,
        activation="tanh",
        name="temporal_attention_scores"
    )(x)

    attention_weights = layers.Softmax(
        axis=1,
        name="temporal_attention_weights"
    )(attention_scores)

    attention_context = layers.Multiply(
        name="attention_weighted_features"
    )(
        [x, attention_weights]
    )

    attention_context = layers.Lambda(
        lambda tensor: tf.reduce_sum(
            tensor,
            axis=1
        ),
        name="attention_pooling"
    )(
        attention_context
    )


    # ========================================================
    # MEAN POOLING
    # ========================================================

    mean_context = layers.GlobalAveragePooling1D(
        name="mean_pooling"
    )(x)


    # ========================================================
    # FUSE ATTENTION + MEAN INFORMATION
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
    # MLP CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256"
    )(x)

    x = layers.BatchNormalization(
        name="bn_256"
    )(x)

    x = layers.Dropout(
        0.40,
        name="dropout_256"
    )(x)


    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_128"
    )(x)

    x = layers.BatchNormalization(
        name="bn_128"
    )(x)

    x = layers.Dropout(
        0.35,
        name="dropout_128"
    )(x)


    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.30,
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
        name="mlp_temporal_attention_v2"
    )

    return model