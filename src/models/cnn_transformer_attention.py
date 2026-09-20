"""
CNN + Transformer Encoder + Multi-Head Attention
for Log-Mel Speech Emotion Recognition.

Dataset:
    CREMA-D

Input:
    64 x 174 Log-Mel Spectrogram

Output:
    6 emotion classes

Training:
    From scratch
"""

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================

class TransformerEncoder(layers.Layer):
    """
    Transformer Encoder consisting of:

    1. Multi-Head Self-Attention
    2. Residual Connection
    3. Layer Normalization
    4. Feed-Forward Network
    5. Residual Connection
    6. Layer Normalization
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.15,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name="self_attention"
        )

        self.ffn = keras.Sequential(
            [
                layers.Dense(
                    ff_dim,
                    activation="gelu",
                    name="ffn_dense_1"
                ),

                layers.Dropout(
                    dropout,
                    name="ffn_dropout"
                ),

                layers.Dense(
                    embed_dim,
                    name="ffn_dense_2"
                ),
            ],
            name="feed_forward_network"
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6,
            name="attention_layer_norm"
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6,
            name="ffn_layer_norm"
        )

        self.dropout1 = layers.Dropout(
            dropout,
            name="attention_dropout"
        )

        self.dropout2 = layers.Dropout(
            dropout,
            name="output_dropout"
        )

    def call(self, inputs, training=None):

        # ----------------------------------------------------
        # Multi-Head Self-Attention
        # ----------------------------------------------------

        attention_output = self.attention(
            inputs,
            inputs,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        # Residual connection + normalization
        x = self.norm1(
            inputs + attention_output
        )

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        ffn_output = self.ffn(
            x,
            training=training
        )

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        # Residual connection + normalization
        return self.norm2(
            x + ffn_output
        )


# ============================================================
# CNN + TRANSFORMER + ATTENTION MODEL
# ============================================================

def build_cnn_transformer_attention(
    input_shape=(64, 174, 1),
    num_classes=6
):
    """
    Build CNN + Transformer Encoder + Multi-Head Attention model.

    Parameters
    ----------
    input_shape : tuple
        Input Log-Mel spectrogram shape.

    num_classes : int
        Number of emotion classes.

    Returns
    -------
    keras.Model
        Compiled architecture is returned uncompiled.
    """

    inputs = keras.Input(
        shape=input_shape,
        name="logmel_input"
    )

    # ========================================================
    # CNN FEATURE EXTRACTION
    # ========================================================

    # ---- CNN Block 1 ----

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1"
    )(inputs)

    x = layers.BatchNormalization(
        name="bn1"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool1"
    )(x)

    x = layers.Dropout(
        0.15,
        name="dropout1"
    )(x)

    # ---- CNN Block 2 ----

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2"
    )(x)

    x = layers.BatchNormalization(
        name="bn2"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool2"
    )(x)

    x = layers.Dropout(
        0.20,
        name="dropout2"
    )(x)

    # ---- CNN Block 3 ----

    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv3"
    )(x)

    x = layers.BatchNormalization(
        name="bn3"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool3"
    )(x)

    x = layers.Dropout(
        0.25,
        name="dropout3"
    )(x)

    # ========================================================
    # CONVERT CNN FEATURE MAP TO SEQUENCE
    # ========================================================

    """
    After three MaxPooling2D layers:

    Frequency:
        64 -> 32 -> 16 -> 8

    Time:
        174 -> 87 -> 43 -> 21

    Therefore:

        (8, 21, 128)

    We treat the 21 time positions as sequence steps.
    """

    x = layers.Permute(
        (2, 1, 3),
        name="time_frequency_permutation"
    )(x)

    x = layers.Reshape(
        (21, 8 * 128),
        name="sequence_reshape"
    )(x)

    # ========================================================
    # TRANSFORMER EMBEDDING PROJECTION
    # ========================================================

    x = layers.Dense(
        256,
        activation=None,
        name="transformer_projection"
    )(x)

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name="projection_norm"
    )(x)

    # ========================================================
    # TRANSFORMER ENCODER BLOCK 1
    # ========================================================

    x = TransformerEncoder(
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        dropout=0.15,
        name="transformer_encoder_1"
    )(x)

    # ========================================================
    # TRANSFORMER ENCODER BLOCK 2
    # ========================================================

    x = TransformerEncoder(
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        dropout=0.15,
        name="transformer_encoder_2"
    )(x)

    # ========================================================
    # FINAL MULTI-HEAD ATTENTION
    # ========================================================

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32,
        dropout=0.10,
        name="final_multihead_attention"
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
        epsilon=1e-6,
        name="attention_norm"
    )(x)

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="dense128"
    )(x)

    x = layers.Dropout(
        0.40,
        name="classification_dropout"
    )(x)

    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_transformer_attention"
    )

    return model