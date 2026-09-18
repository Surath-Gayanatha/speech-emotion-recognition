import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


# ============================================================
# Learnable Positional Embedding
# ============================================================

class PositionalEmbedding(layers.Layer):

    def __init__(self, max_length, embed_dim):
        super().__init__()

        self.position_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )

    def call(self, x):

        positions = tf.range(
            start=0,
            limit=tf.shape(x)[1],
            delta=1
        )

        return x + self.position_embedding(positions)


# ============================================================
# GEGLU Feed Forward Network
# ============================================================

class GEGLU(layers.Layer):

    def __init__(self, embed_dim, ff_dim, dropout=0.10):
        super().__init__()

        self.projection = layers.Dense(ff_dim * 2)

        self.output_projection = layers.Dense(embed_dim)

        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):

        projected = self.projection(x)

        value, gate = tf.split(
            projected,
            num_or_size_splits=2,
            axis=-1
        )

        x = value * tf.nn.gelu(gate)

        x = self.dropout(
            x,
            training=training
        )

        return self.output_projection(x)


# ============================================================
# Pre-Norm Transformer Encoder
# ============================================================

class TransformerEncoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.10
    ):
        super().__init__()

        # Pre-normalization
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        # Gated feed-forward network
        self.ffn = GEGLU(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):

        # ----------------------------------------------------
        # Self Attention
        # ----------------------------------------------------

        normalized = self.norm1(x)

        attention_output = self.attention(
            normalized,
            normalized,
            training=training
        )

        x = x + self.dropout1(
            attention_output,
            training=training
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        normalized = self.norm2(x)

        ffn_output = self.ffn(
            normalized,
            training=training
        )

        x = x + self.dropout2(
            ffn_output,
            training=training
        )

        return x


# ============================================================
# Multi-Head Attention Pooling
# ============================================================

class AttentionPooling(layers.Layer):

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

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.norm = layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self, x, training=False):

        batch_size = tf.shape(x)[0]

        query = tf.broadcast_to(
            self.query,
            [batch_size, 1, tf.shape(self.query)[-1]]
        )

        pooled = self.attention(
            query,
            x,
            x,
            training=training
        )

        pooled = self.norm(pooled)

        return tf.squeeze(
            pooled,
            axis=1
        )


# ============================================================
# Build Transformer V2
# ============================================================

def build_transformer_v2(
    input_shape,
    num_classes=NUM_CLASSES
):

    inputs = layers.Input(
        shape=input_shape
    )

    # --------------------------------------------------------
    # Input projection
    # --------------------------------------------------------

    x = layers.Dense(
        128
    )(inputs)

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    # --------------------------------------------------------
    # Positional information
    # --------------------------------------------------------

    x = PositionalEmbedding(
        max_length=input_shape[0],
        embed_dim=128
    )(x)

    x = layers.Dropout(
        0.10
    )(x)

    # --------------------------------------------------------
    # Transformer Encoder 1
    # --------------------------------------------------------

    x = TransformerEncoder(
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Transformer Encoder 2
    # --------------------------------------------------------

    x = TransformerEncoder(
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Transformer Encoder 3
    # --------------------------------------------------------

    x = TransformerEncoder(
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Multi-head attention pooling
    # --------------------------------------------------------

    x = AttentionPooling(
        embed_dim=128,
        num_heads=4,
        dropout=0.10
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
        name="speech_transformer_v2"
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