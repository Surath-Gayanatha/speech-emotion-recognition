import tensorflow as tf
from tensorflow.keras import layers, Model


NUM_CLASSES = 6


def build_cnn_bilstm_mha_specaug(
    input_shape=(64, 174, 1),
    num_classes=NUM_CLASSES
):

    inputs = layers.Input(
        shape=input_shape,
        name="logmel_input"
    )

    # ========================================================
    # CNN Block 1
    # ========================================================

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.15)(x)


    # ========================================================
    # CNN Block 2
    # ========================================================

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.20)(x)


    # ========================================================
    # CNN Block 3
    # ========================================================

    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)


    # ========================================================
    # Convert CNN Feature Map to Sequence
    # ========================================================

    x = layers.Permute(
        (2, 1, 3)
    )(x)

    x = layers.Reshape(
        (21, 1024)
    )(x)


    # ========================================================
    # Projection
    # ========================================================

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.20)(x)


    # ========================================================
    # BiLSTM
    # ========================================================

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.20
        )
    )(x)


    # ========================================================
    # Multi-Head Self-Attention
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32,
        dropout=0.10
    )(
        x,
        x
    )


    # ========================================================
    # Residual Connection
    # ========================================================

    x = layers.Add()(
        [x, attention_output]
    )

    x = layers.LayerNormalization()(x)


    # ========================================================
    # Feed Forward Block
    # ========================================================

    ff = layers.Dense(
        256,
        activation="gelu"
    )(x)

    ff = layers.Dropout(0.20)(ff)

    ff = layers.Dense(
        256
    )(ff)

    x = layers.Add()(
        [x, ff]
    )

    x = layers.LayerNormalization()(x)


    # ========================================================
    # Global Pooling
    # ========================================================

    x = layers.GlobalAveragePooling1D()(x)


    # ========================================================
    # Classification Head
    # ========================================================

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)


    # ========================================================
    # Create Model
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiLSTM_MHA_SpecAug"
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-4
    )


    # ========================================================
    # Compile
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    return model