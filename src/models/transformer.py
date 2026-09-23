import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


# ============================================================
# Squeeze-and-Excitation (SE) Channel Attention Module
# ============================================================

class SqueezeAndExcitation(layers.Layer):

    def __init__(self, channels, reduction=4):
        super().__init__()

        self.fc = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(
                channels // reduction,
                activation=tf.nn.gelu
            ),
            layers.Dense(
                channels,
                activation="sigmoid"
            ),
            layers.Reshape((1, channels))
        ])

    def call(self, x):
        weight = self.fc(x)
        return x * weight


# ============================================================
# Multi-Scale 1D Convolutional Local Feature Extractor
# ============================================================

class MultiScaleConvBlock(layers.Layer):

    def __init__(self, filters=128, dropout=0.15):
        super().__init__()

        branch_filters = filters // 3
        remainder = filters - (branch_filters * 3)

        self.conv3 = layers.Conv1D(
            branch_filters,
            kernel_size=3,
            padding="same"
        )

        self.conv5 = layers.Conv1D(
            branch_filters,
            kernel_size=5,
            padding="same"
        )

        self.conv7 = layers.Conv1D(
            branch_filters + remainder,
            kernel_size=7,
            padding="same"
        )

        self.bn = layers.BatchNormalization()

        self.act = layers.Activation(
            tf.nn.gelu
        )

        self.se = SqueezeAndExcitation(
            filters
        )

        self.dropout = layers.Dropout(
            dropout
        )

    def call(self, x, training=False):

        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        out = tf.concat(
            [c3, c5, c7],
            axis=-1
        )

        out = self.bn(
            out,
            training=training
        )

        out = self.act(out)

        out = self.se(out)

        return self.dropout(
            out,
            training=training
        )


# ============================================================
# Relative Convolutional Positional Encoding
# ============================================================

class DepthwiseConvPositionalEncoding(layers.Layer):

    def __init__(
        self,
        embed_dim,
        kernel_size=3
    ):
        super().__init__()

        self.conv = layers.Conv1D(
            filters=embed_dim,
            kernel_size=kernel_size,
            padding="same",
            groups=embed_dim
        )

    def call(self, x):

        return x + self.conv(x)


# ============================================================
# Conformer-Transformer Encoder Block
# ============================================================

class ConformerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        conv_kernel_size=5,
        dropout=0.15
    ):
        super().__init__()

        # ----------------------------------------------------
        # Self Attention
        # ----------------------------------------------------

        self.norm_attn = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(
                16,
                embed_dim // num_heads
            ),
            dropout=dropout
        )

        self.dropout_attn = layers.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Depthwise Convolution Module
        # ----------------------------------------------------

        self.norm_conv = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dw_conv = layers.Conv1D(
            filters=embed_dim,
            kernel_size=conv_kernel_size,
            padding="same",
            groups=embed_dim
        )

        self.bn_conv = layers.BatchNormalization()

        self.act_conv = layers.Activation(
            tf.nn.gelu
        )

        self.se_conv = SqueezeAndExcitation(
            embed_dim
        )

        self.proj_conv = layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            padding="same"
        )

        self.dropout_conv = layers.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        self.norm_ffn = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.ffn = keras.Sequential([
            layers.Dense(
                ff_dim,
                activation=tf.nn.gelu
            ),

            layers.Dropout(
                dropout
            ),

            layers.Dense(
                embed_dim
            ),

            layers.Dropout(
                dropout
            )
        ])

        # ----------------------------------------------------
        # Final Layer Normalization
        # ----------------------------------------------------

        self.norm_out = layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(
        self,
        x,
        training=False
    ):

        # ----------------------------------------------------
        # 1. Multi-Head Self Attention
        # ----------------------------------------------------

        norm_x = self.norm_attn(x)

        attn_out = self.attn(
            norm_x,
            norm_x,
            training=training
        )

        x = x + self.dropout_attn(
            attn_out,
            training=training
        )

        # ----------------------------------------------------
        # 2. Depthwise Convolution
        # ----------------------------------------------------

        conv_x = self.norm_conv(x)

        conv_x = self.dw_conv(
            conv_x
        )

        conv_x = self.bn_conv(
            conv_x,
            training=training
        )

        conv_x = self.act_conv(
            conv_x
        )

        conv_x = self.se_conv(
            conv_x
        )

        conv_x = self.proj_conv(
            conv_x
        )

        x = x + self.dropout_conv(
            conv_x,
            training=training
        )

        # ----------------------------------------------------
        # 3. Feed Forward Network
        # ----------------------------------------------------

        x = x + self.ffn(
            self.norm_ffn(x),
            training=training
        )

        return self.norm_out(x)


# ============================================================
# Multi-Head Attention Pooling
# ============================================================

class MultiHeadAttentionPooling(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads=4,
        dropout=0.10
    ):
        super().__init__()

        self.query = self.add_weight(
            name="query",
            shape=(1, 1, embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(
                16,
                embed_dim // num_heads
            ),
            dropout=dropout
        )

        self.norm = layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(
        self,
        x,
        training=False
    ):

        batch_size = tf.shape(x)[0]

        query = tf.broadcast_to(
            self.query,
            [
                batch_size,
                1,
                tf.shape(self.query)[-1]
            ]
        )

        pooled = self.attn(
            query,
            x,
            x,
            training=training
        )

        pooled = self.norm(
            pooled
        )

        return tf.squeeze(
            pooled,
            axis=1
        )


# ============================================================
# Temporal Statistics Pooling
# ============================================================

class TemporalStatsPooling(layers.Layer):

    def call(self, x):

        mean = tf.reduce_mean(
            x,
            axis=1
        )

        variance = tf.reduce_mean(
            tf.square(
                x - tf.expand_dims(
                    mean,
                    1
                )
            ),
            axis=1
        )

        std = tf.sqrt(
            tf.maximum(
                variance,
                1e-6
            )
        )

        return tf.concat(
            [mean, std],
            axis=-1
        )


# ============================================================
# Build Advanced Acoustic Transformer
# ============================================================

def build_transformer(
    input_shape,
    num_classes=NUM_CLASSES,
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    conv_kernel_size=5,
    dropout=0.15,
    head_dropout=0.25
):

    inputs = layers.Input(
        shape=input_shape,
        name="audio_features"
    )

    # --------------------------------------------------------
    # 1. Multi-Scale Convolution
    # --------------------------------------------------------

    x_conv = MultiScaleConvBlock(
        filters=embed_dim,
        dropout=dropout
    )(inputs)

    # Residual projection

    x_proj = layers.Dense(
        embed_dim
    )(inputs)

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(
        x_conv + x_proj
    )

    # --------------------------------------------------------
    # 2. Positional Encoding
    # --------------------------------------------------------

    x = DepthwiseConvPositionalEncoding(
        embed_dim=embed_dim,
        kernel_size=3
    )(x)

    x = layers.Dropout(
        dropout
    )(x)

    # --------------------------------------------------------
    # 3. Conformer Blocks
    # --------------------------------------------------------

    for i in range(num_layers):

        x = ConformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )(x)

    # --------------------------------------------------------
    # 4. Attention Pooling
    # --------------------------------------------------------

    attn_pooled = MultiHeadAttentionPooling(
        embed_dim=embed_dim,
        num_heads=4,
        dropout=dropout
    )(x)

    # --------------------------------------------------------
    # 5. Temporal Statistics
    # --------------------------------------------------------

    stats_pooled = TemporalStatsPooling()(x)

    pooled_features = layers.Concatenate(
        axis=-1
    )(
        [
            attn_pooled,
            stats_pooled
        ]
    )

    # --------------------------------------------------------
    # 6. Classification Head
    # --------------------------------------------------------

    h = layers.Dense(
        256
    )(pooled_features)

    h = layers.LayerNormalization(
        epsilon=1e-6
    )(h)

    h = layers.Activation(
        tf.nn.gelu
    )(h)

    h = layers.Dropout(
        head_dropout
    )(h)

    h = layers.Dense(
        128
    )(h)

    h = layers.LayerNormalization(
        epsilon=1e-6
    )(h)

    h = layers.Activation(
        tf.nn.gelu
    )(h)

    h = layers.Dropout(
        head_dropout
    )(h)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_probabilities"
    )(h)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="advanced_acoustic_transformer"
    )

    return model