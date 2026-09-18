"""
Log-Mel CNN + BiLSTM + Self-Attention
=====================================

Input:
    Log-Mel Spectrogram
    Shape: (64, 174, 1)

Architecture:
    Conv2D(32)
        -> BatchNormalization
        -> ReLU
        -> MaxPooling
        -> Dropout

    Conv2D(64)
        -> BatchNormalization
        -> ReLU
        -> MaxPooling
        -> Dropout

    Conv2D(128)
        -> BatchNormalization
        -> ReLU
        -> MaxPooling
        -> Dropout

    Permute
    Reshape
    BiLSTM(128)
    Multi-Head Self-Attention
    Residual Connection
    Layer Normalization
    Global Average Pooling
    Dense(128)
    Dropout
    Softmax(6)

Compatibility:
    TensorFlow 2.20.x
    Keras 3.x

Important:
    SelfAttention is explicitly registered using
    tf.keras.utils.register_keras_serializable so that
    saved .keras checkpoints can be loaded correctly.
"""

import tensorflow as tf

from tensorflow.keras import layers, Model


# ============================================================
# 1. SELF-ATTENTION BLOCK
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="LogMelSER"
)
class SelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention block.

    This custom layer is registered with Keras so models
    containing SelfAttention can be saved and loaded using
    the Keras .keras format.
    """

    def __init__(
        self,
        num_heads=4,
        key_dim=32,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # ----------------------------------------------------
        # Store constructor arguments.
        # These are required for Keras serialization.
        # ----------------------------------------------------

        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.dropout_rate = float(dropout)

        # ----------------------------------------------------
        # Multi-Head Attention
        # ----------------------------------------------------

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name="multi_head_attention"
        )

        # ----------------------------------------------------
        # Dropout after attention
        # ----------------------------------------------------

        self.dropout_layer = layers.Dropout(
            self.dropout_rate,
            name="attention_dropout"
        )

        # ----------------------------------------------------
        # Layer Normalization
        # ----------------------------------------------------

        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="attention_layer_norm"
        )

    def call(
        self,
        inputs,
        training=None
    ):
        """
        Forward pass.

        Parameters
        ----------
        inputs:
            Sequence tensor from the BiLSTM.

        training:
            Boolean indicating training/inference mode.
        """

        # ----------------------------------------------------
        # Self-Attention
        #
        # Query = inputs
        # Key   = inputs
        # Value = inputs
        # ----------------------------------------------------

        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )

        # ----------------------------------------------------
        # Attention dropout
        # ----------------------------------------------------

        attention_output = self.dropout_layer(
            attention_output,
            training=training
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        output = inputs + attention_output

        # ----------------------------------------------------
        # Layer normalization
        # ----------------------------------------------------

        output = self.norm(output)

        return output

    def get_config(self):
        """
        Return configuration required for serialization.
        """

        config = super().get_config()

        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate
        })

        return config


# ============================================================
# 2. MODEL BUILDER
# ============================================================

def build_logmel_cnn_bilstm_attention(
    input_shape=(64, 174, 1),
    num_classes=6
):
    """
    Build the Log-Mel CNN + BiLSTM + Self-Attention model.

    Parameters
    ----------
    input_shape : tuple
        Input shape:
        (Mel bands, time frames, channels)

    num_classes : int
        Number of emotion classes.

    Returns
    -------
    tensorflow.keras.Model
        Uncompiled Keras model.
    """

    # ========================================================
    # INPUT
    # ========================================================

    inputs = layers.Input(
        shape=input_shape,
        name="logmel_input"
    )


    # ========================================================
    # CNN BLOCK 1
    # ========================================================

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv2d_1"
    )(inputs)

    x = layers.BatchNormalization(
        name="batch_norm_1"
    )(x)

    x = layers.Activation(
        "relu",
        name="relu_1"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pool_1"
    )(x)

    x = layers.Dropout(
        0.20,
        name="dropout_cnn_1"
    )(x)


    # ========================================================
    # CNN BLOCK 2
    # ========================================================

    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv2d_2"
    )(x)

    x = layers.BatchNormalization(
        name="batch_norm_2"
    )(x)

    x = layers.Activation(
        "relu",
        name="relu_2"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pool_2"
    )(x)

    x = layers.Dropout(
        0.25,
        name="dropout_cnn_2"
    )(x)


    # ========================================================
    # CNN BLOCK 3
    # ========================================================

    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv2d_3"
    )(x)

    x = layers.BatchNormalization(
        name="batch_norm_3"
    )(x)

    x = layers.Activation(
        "relu",
        name="relu_3"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pool_3"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_cnn_3"
    )(x)


    # ========================================================
    # CNN FEATURE MAP -> SEQUENCE
    # ========================================================

    """
    Input:
        (64, 174, 1)

    After Conv Block 1:
        (32, 87, 32)

    After Conv Block 2:
        (16, 43, 64)

    After Conv Block 3:
        (8, 21, 128)

    We use the time dimension (21) as the sequence axis.

    Before Permute:
        (8, 21, 128)

    After Permute:
        (21, 8, 128)

    After Reshape:
        (21, 1024)
    """

    # --------------------------------------------------------
    # Calculate dimensions after three 2x2 pooling layers.
    # --------------------------------------------------------

    freq_dim = input_shape[0] // 8
    time_dim = input_shape[1] // 8

    channels = 128


    # --------------------------------------------------------
    # Safety validation
    # --------------------------------------------------------

    if freq_dim <= 0 or time_dim <= 0:

        raise ValueError(
            f"Invalid input shape: {input_shape}"
        )


    # --------------------------------------------------------
    # Convert:
    #
    # (frequency, time, channels)
    #
    # ->
    #
    # (time, frequency, channels)
    # --------------------------------------------------------

    x = layers.Permute(
        (2, 1, 3),
        name="time_frequency_permute"
    )(x)


    # --------------------------------------------------------
    # Flatten frequency + channel dimensions
    # --------------------------------------------------------

    x = layers.Reshape(
        (
            time_dim,
            freq_dim * channels
        ),
        name="cnn_to_sequence"
    )(x)


    # ========================================================
    # BIDIRECTIONAL LSTM
    # ========================================================

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.0,
            name="bilstm"
        ),
        name="bidirectional_lstm"
    )(x)

    # BiLSTM output:
    #
    # (batch, 21, 256)


    # ========================================================
    # MULTI-HEAD SELF-ATTENTION
    # ========================================================

    x = SelfAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.10,
        name="self_attention"
    )(x)

    # Output:
    #
    # (batch, 21, 256)


    # ========================================================
    # SECOND NORMALIZATION
    # ========================================================

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name="attention_normalization"
    )(x)


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    x = layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    # Output:
    #
    # (batch, 256)


    # ========================================================
    # FULLY CONNECTED LAYER
    # ========================================================

    x = layers.Dense(
        128,
        activation="relu",
        name="dense_128"
    )(x)


    # ========================================================
    # FINAL DROPOUT
    # ========================================================

    x = layers.Dropout(
        0.40,
        name="final_dropout"
    )(x)


    # ========================================================
    # OUTPUT LAYER
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="LogMel_CNN_BiLSTM_Attention"
    )


    return model


# ============================================================
# 3. MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LOG-MEL CNN + BiLSTM + SELF-ATTENTION")
    print("=" * 70)

    print(
        f"\nTensorFlow version: "
        f"{tf.__version__}"
    )

    print(
        f"Keras version: "
        f"{tf.keras.__version__}"
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_logmel_cnn_bilstm_attention(
        input_shape=(64, 174, 1),
        num_classes=6
    )


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    model.summary()


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print(
        f"Input shape : {model.input_shape}"
    )

    print(
        f"Output shape: {model.output_shape}"
    )

    print(
        f"Parameters  : {model.count_params():,}"
    )


    # ========================================================
    # SELF-ATTENTION SERIALIZATION TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("SELF-ATTENTION SERIALIZATION TEST")
    print("=" * 70)


    attention_layer = model.get_layer(
        "self_attention"
    )


    config = attention_layer.get_config()


    print(
        f"num_heads : {config['num_heads']}"
    )

    print(
        f"key_dim   : {config['key_dim']}"
    )

    print(
        f"dropout   : {config['dropout']}"
    )


    # ========================================================
    # SERIALIZATION TEST
    # ========================================================

    serialized = tf.keras.utils.serialize_keras_object(
        attention_layer
    )


    restored_layer = (
        tf.keras.utils.deserialize_keras_object(
            serialized
        )
    )


    print(
        "\n✓ SelfAttention serialization test PASSED."
    )

    print(
        f"Restored layer type: "
        f"{type(restored_layer).__name__}"
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "\n✓ Model architecture created successfully."
    )

    print(
        "✓ SelfAttention is Keras-serializable."
    )

    print(
        "✓ Ready for checkpoint loading."
    )