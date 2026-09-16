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

        position_embeddings = self.position_embedding(
            positions
        )

        return x + position_embeddings


# ============================================================
# Transformer Encoder
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

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.ffn = keras.Sequential([
            layers.Dense(
                ff_dim,
                activation=tf.nn.gelu
            ),

            layers.Dropout(dropout),

            layers.Dense(
                embed_dim
            )
        ])

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):

        attention_output = self.attention(
            x,
            x,
            training=training
        )

        x = self.norm1(
            x + self.dropout1(
                attention_output,
                training=training
            )
        )

        ffn_output = self.ffn(
            x,
            training=training
        )

        x = self.norm2(
            x + self.dropout2(
                ffn_output,
                training=training
            )
        )

        return x


# ============================================================
# Attention Pooling
# ============================================================

class AttentionPooling(layers.Layer):

    def __init__(self, embed_dim):
        super().__init__()

        self.attention_score = layers.Dense(
            1,
            use_bias=False
        )

    def call(self, x):

        # x shape:
        # (batch, time, embed_dim)

        scores = self.attention_score(x)

        scores = tf.nn.softmax(
            scores,
            axis=1
        )

        weighted = x * scores

        return tf.reduce_sum(
            weighted,
            axis=1
        )


# ============================================================
# Build Transformer
# ============================================================

def build_transformer(
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
        64,
        activation=None
    )(inputs)

    x = layers.LayerNormalization()(x)

    # --------------------------------------------------------
    # Positional information
    # --------------------------------------------------------

    x = PositionalEmbedding(
        max_length=input_shape[0],
        embed_dim=64
    )(x)

    x = layers.Dropout(0.10)(x)

    # --------------------------------------------------------
    # Transformer Encoder 1
    # --------------------------------------------------------

    x = TransformerEncoder(
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Transformer Encoder 2
    # --------------------------------------------------------

    x = TransformerEncoder(
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.10
    )(x)

    # --------------------------------------------------------
    # Attention pooling
    # --------------------------------------------------------

    x = AttentionPooling(
        embed_dim=64
    )(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.Dense(
        64,
        activation=tf.nn.gelu
    )(x)

    x = layers.Dropout(0.25)(x)

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
        name="speech_transformer"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model