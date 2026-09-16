import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


# ============================================================
# Positional Embedding
# ============================================================

class PositionalEmbedding(layers.Layer):

    def __init__(self, max_length, embed_dim):
        super().__init__()

        self.embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )

    def call(self, x):

        positions = tf.range(
            start=0,
            limit=tf.shape(x)[1]
        )

        return x + self.embedding(positions)


# ============================================================
# Local Block Attention
# ============================================================

class BlockSelfAttention(layers.Layer):
    """
    Local self-attention over fixed-size temporal blocks.

    Input:
        (batch, time_steps, embed_dim)

    Example:
        (batch, 120, 128)

    With block_size=10:
        120 frames -> 12 blocks
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        block_size=10,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout_rate = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        # Input shape should be (batch, time_steps, embed_dim)
        if input_shape[-1] is None:
            raise ValueError("The embedding dimension must be defined.")

        if input_shape[1] is None:
            raise ValueError("The time dimension must be defined.")

        self.time_steps = int(input_shape[1])
        self.input_dim = int(input_shape[-1])

        if self.time_steps % self.block_size != 0:
            raise ValueError(
                f"time_steps ({self.time_steps}) must be divisible "
                f"by block_size ({self.block_size})."
            )

        super().build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]

        num_blocks = self.time_steps // self.block_size

        # ---------------------------------------------------------
        # (B, T, D)
        # ->
        # (B, num_blocks, block_size, D)
        # ---------------------------------------------------------
        blocks = tf.reshape(
            x,
            (
                batch_size,
                num_blocks,
                self.block_size,
                self.input_dim
            )
        )

        # ---------------------------------------------------------
        # Process every block independently
        #
        # (B, num_blocks, block_size, D)
        # ->
        # (B * num_blocks, block_size, D)
        # ---------------------------------------------------------
        blocks_flat = tf.reshape(
            blocks,
            (
                batch_size * num_blocks,
                self.block_size,
                self.input_dim
            )
        )

        # Local self-attention
        attended = self.attention(
            blocks_flat,
            blocks_flat,
            training=training
        )

        attended = self.dropout(
            attended,
            training=training
        )

        # Residual connection + normalization
        attended = self.norm(
            blocks_flat + attended
        )

        # ---------------------------------------------------------
        # Restore original sequence structure
        #
        # (B * num_blocks, block_size, D)
        # ->
        # (B, T, D)
        # ---------------------------------------------------------
        output = tf.reshape(
            attended,
            (
                batch_size,
                self.time_steps,
                self.input_dim
            )
        )

        return output

    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "block_size": self.block_size,
            "dropout": self.dropout_rate
        })

        return config


# ============================================================
# Global Token Attention
# ============================================================

class GlobalTokenAttention(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.10
    ):
        super().__init__()

        self.norm = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):

        normalized = self.norm(x)

        attended = self.attention(
            normalized,
            normalized,
            training=training
        )

        attended = self.dropout(
            attended,
            training=training
        )

        return x + attended


# ============================================================
# Cross-Block Attention
# ============================================================

class CrossBlockAttention(layers.Layer):
    """
    Cross-block attention.

    The 120-frame sequence is divided into 12 blocks
    of 10 frames each.

    Each block is summarized into one block token.
    The block tokens then interact using self-attention.
    The resulting block information is expanded back
    to the original 120-frame sequence.
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        block_size=10,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout_rate = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                "The embedding dimension must be defined."
            )

        if input_shape[1] is None:
            raise ValueError(
                "The time dimension must be defined."
            )

        self.time_steps = int(input_shape[1])
        self.input_dim = int(input_shape[-1])

        if self.time_steps % self.block_size != 0:
            raise ValueError(
                f"time_steps ({self.time_steps}) must be divisible "
                f"by block_size ({self.block_size})."
            )

        super().build(input_shape)

    def call(self, x, training=None):

        batch_size = tf.shape(x)[0]

        num_blocks = self.time_steps // self.block_size

        # ---------------------------------------------------------
        # Split sequence into blocks
        #
        # (B, 120, 128)
        # ->
        # (B, 12, 10, 128)
        # ---------------------------------------------------------
        blocks = tf.reshape(
            x,
            (
                batch_size,
                num_blocks,
                self.block_size,
                self.input_dim
            )
        )

        # ---------------------------------------------------------
        # Create one token representing each block
        #
        # (B, 12, 10, 128)
        # ->
        # (B, 12, 128)
        # ---------------------------------------------------------
        block_tokens = tf.reduce_mean(
            blocks,
            axis=2
        )

        # ---------------------------------------------------------
        # Self-attention between blocks
        #
        # Each block can now attend to every other block.
        #
        # (B, 12, 128)
        # ->
        # (B, 12, 128)
        # ---------------------------------------------------------
        attended = self.attention(
            block_tokens,
            block_tokens,
            training=training
        )

        attended = self.dropout(
            attended,
            training=training
        )

        # Residual connection
        attended = self.norm(
            block_tokens + attended
        )

        # ---------------------------------------------------------
        # Expand each block token back to its 10 frames
        #
        # (B, 12, 128)
        # ->
        # (B, 12, 10, 128)
        # ---------------------------------------------------------
        expanded = tf.repeat(
            attended[:, :, tf.newaxis, :],
            repeats=self.block_size,
            axis=2
        )

        # ---------------------------------------------------------
        # Restore original sequence
        #
        # (B, 12, 10, 128)
        # ->
        # (B, 120, 128)
        # ---------------------------------------------------------
        output = tf.reshape(
            expanded,
            (
                batch_size,
                self.time_steps,
                self.input_dim
            )
        )

        # ---------------------------------------------------------
        # Residual connection with original input
        # ---------------------------------------------------------
        output = self.norm(
            x + output
        )

        return output

    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "block_size": self.block_size,
            "dropout": self.dropout_rate
        })

        return config


# ============================================================
# Feed Forward
# ============================================================

class FeedForward(layers.Layer):

    def __init__(
        self,
        embed_dim,
        ff_dim,
        dropout=0.10
    ):
        super().__init__()

        self.norm = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.network = keras.Sequential([
            layers.Dense(
                ff_dim,
                activation=tf.nn.gelu
            ),

            layers.Dropout(dropout),

            layers.Dense(
                embed_dim
            )
        ])

        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):

        normalized = self.norm(x)

        output = self.network(
            normalized,
            training=training
        )

        output = self.dropout(
            output,
            training=training
        )

        return x + output


# ============================================================
# Attention Pooling
# ============================================================

class AttentionPooling(layers.Layer):

    def __init__(self, embed_dim):
        super().__init__()

        self.score = layers.Dense(
            1,
            use_bias=False
        )

    def call(self, x):

        scores = self.score(x)

        scores = tf.nn.softmax(
            scores,
            axis=1
        )

        return tf.reduce_sum(
            x * scores,
            axis=1
        )


# ============================================================
# BAT Transformer
# ============================================================

def build_bat_transformer(
    input_shape,
    num_classes=NUM_CLASSES
):

    inputs = layers.Input(
        shape=input_shape
    )

    # --------------------------------------------------------
    # Feature projection
    # --------------------------------------------------------

    x = layers.Dense(
        128
    )(inputs)

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    # --------------------------------------------------------
    # Positional encoding
    # --------------------------------------------------------

    x = PositionalEmbedding(
        max_length=input_shape[0],
        embed_dim=128
    )(x)

    x = layers.Dropout(
        0.10
    )(x)

    # --------------------------------------------------------
    # Local block attention
    # --------------------------------------------------------

    x = BlockSelfAttention(
        embed_dim=128,
        num_heads=8,
        block_size=10,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Feed forward
    # --------------------------------------------------------

    x = FeedForward(
        embed_dim=128,
        ff_dim=256,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Global token attention
    # --------------------------------------------------------

    x = GlobalTokenAttention(
        embed_dim=128,
        num_heads=8,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Feed forward
    # --------------------------------------------------------

    x = FeedForward(
        embed_dim=128,
        ff_dim=256,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Cross-block attention
    # --------------------------------------------------------

    x = CrossBlockAttention(
        embed_dim=128,
        num_heads=8,
        block_size=10,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Final global attention
    # --------------------------------------------------------

    x = GlobalTokenAttention(
        embed_dim=128,
        num_heads=8,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Attention pooling
    # --------------------------------------------------------

    x = AttentionPooling(
        embed_dim=128
    )(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation=tf.nn.gelu
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    x = layers.Dense(
        64,
        activation=tf.nn.gelu
    )(x)

    x = layers.Dropout(
        0.20
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = keras.Model(
        inputs,
        outputs,
        name="bat_transformer"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model