import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# FEATURE PROJECTION
# ============================================================

class FeatureProjection(layers.Layer):

    def __init__(
        self,
        embed_dim=128,
        dropout=0.15,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dropout_rate = dropout

        self.projection = layers.Dense(
            embed_dim
        )

        self.normalization = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout = layers.Dropout(
            dropout
        )

    def call(
        self,
        inputs,
        training=None
    ):

        x = self.projection(inputs)

        x = self.normalization(x)

        x = self.dropout(
            x,
            training=training
        )

        return x

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "dropout": self.dropout_rate
        })

        return config


# ============================================================
# LEARNABLE POSITIONAL EMBEDDING
# ============================================================

class PositionalEmbedding(layers.Layer):

    def __init__(
        self,
        sequence_length,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        self.position_embedding = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(self, inputs):

        positions = tf.range(
            start=0,
            limit=self.sequence_length,
            delta=1
        )

        position_embeddings = self.position_embedding(
            positions
        )

        return inputs + position_embeddings

    def get_config(self):

        config = super().get_config()

        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim
        })

        return config


# ============================================================
# TRANSFORMER ENCODER
# ============================================================

class MFTTransformerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        dropout=0.15,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.dropout1 = layers.Dropout(
            dropout
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.feed_forward = keras.Sequential([
            layers.Dense(
                ff_dim,
                activation=tf.nn.gelu
            ),

            layers.Dropout(
                dropout
            ),

            layers.Dense(
                embed_dim
            )
        ])

        self.dropout2 = layers.Dropout(
            dropout
        )

    def call(
        self,
        inputs,
        training=None
    ):

        # Pre-normalization
        x_norm = self.norm1(
            inputs
        )

        attention_output = self.attention(
            x_norm,
            x_norm,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = inputs + attention_output

        # Feed-forward network
        x_norm = self.norm2(
            x
        )

        ff_output = self.feed_forward(
            x_norm,
            training=training
        )

        ff_output = self.dropout2(
            ff_output,
            training=training
        )

        return x + ff_output

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })

        return config


# ============================================================
# ATTENTION POOLING
# ============================================================

class AttentionPooling(layers.Layer):

    def __init__(
        self,
        embed_dim=128,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim

        self.score = layers.Dense(
            1
        )

    def call(self, inputs):

        scores = self.score(
            inputs
        )

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        weighted = (
            inputs * weights
        )

        return tf.reduce_sum(
            weighted,
            axis=1
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim
        })

        return config


# ============================================================
# BUILD MFT TRANSFORMER
# ============================================================

def build_mft_transformer(
    input_shape,
    num_classes=6,
    embed_dim=128,
    num_heads=8,
    ff_dim=256,
    num_layers=4,
    dropout=0.15
):

    inputs = keras.Input(
        shape=input_shape,
        name="mft_input"
    )

    # --------------------------------------------------------
    # Feature projection
    # --------------------------------------------------------

    x = FeatureProjection(
        embed_dim=embed_dim,
        dropout=dropout
    )(inputs)

    # --------------------------------------------------------
    # Positional information
    # --------------------------------------------------------

    x = PositionalEmbedding(
        sequence_length=input_shape[0],
        embed_dim=embed_dim
    )(x)

    # --------------------------------------------------------
    # Transformer blocks
    # --------------------------------------------------------

    for _ in range(num_layers):

        x = MFTTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )(x)

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    # --------------------------------------------------------
    # Attention pooling
    # --------------------------------------------------------

    x = AttentionPooling(
        embed_dim=embed_dim
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
        activation="softmax",
        name="emotion_output"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MFT_Transformer"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    try:

        optimizer = keras.optimizers.AdamW(
            learning_rate=2e-4,
            weight_decay=1e-4
        )

    except AttributeError:

        optimizer = keras.optimizers.Adam(
            learning_rate=2e-4
        )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model