import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbedding(layers.Layer):
    """
    Converts a log-Mel spectrogram into a sequence of patches.

    Input:
        (batch, 128, 256, 1)

    Patch size:
        16 x 16

    Output:
        (batch, 128 patches, embed_dim)
    """

    def __init__(
        self,
        patch_size=16,
        embed_dim=128,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.projection = layers.Dense(embed_dim)

    def call(self, images):

        batch_size = tf.shape(images)[0]

        # Extract non-overlapping patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            strides=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        # Calculate number of patches
        patch_dims = (
            patches.shape[-1]
        )

        # Flatten patches into sequence
        patches = tf.reshape(
            patches,
            [batch_size, -1, patch_dims]
        )

        # Linear projection
        embeddings = self.projection(
            patches
        )

        return embeddings

    def get_config(self):

        config = super().get_config()

        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim
        })

        return config


# ============================================================
# CLASS TOKEN + POSITIONAL EMBEDDING
# ============================================================

class ClassTokenPositionEmbedding(layers.Layer):
    """
    Adds a learnable [CLS] token and positional embeddings.
    """

    def __init__(
        self,
        num_patches,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True
        )

        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, num_patches + 1, embed_dim),
            initializer=keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            trainable=True
        )

    def call(self, x):

        batch_size = tf.shape(x)[0]

        # Repeat CLS token for each sample
        cls_tokens = tf.broadcast_to(
            self.class_token,
            [batch_size, 1, self.embed_dim]
        )

        # Add CLS token
        x = tf.concat(
            [cls_tokens, x],
            axis=1
        )

        # Add positional embeddings
        x = x + self.position_embedding

        return x

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })

        return config


# ============================================================
# TRANSFORMER ENCODER
# ============================================================

class TransformerEncoder(layers.Layer):
    """
    Pre-Norm Transformer Encoder block.
    """

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

        self.dropout2 = layers.Dropout(
            dropout
        )

    def call(
        self,
        x,
        training=None
    ):

        # --------------------------------------------------------
        # Self-attention
        # --------------------------------------------------------

        normalized = self.norm1(x)

        attention_output = self.attention(
            normalized,
            normalized,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = x + attention_output

        # --------------------------------------------------------
        # Feed-forward network
        # --------------------------------------------------------

        normalized = self.norm2(x)

        ffn_output = self.ffn(
            normalized,
            training=training
        )

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        x = x + ffn_output

        return x

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
# VISION TRANSFORMER FOR SPEECH EMOTION RECOGNITION
# ============================================================

def build_vit_ser(
    input_shape=(128, 256, 1),
    num_classes=6,
    patch_size=16,
    embed_dim=128,
    num_heads=8,
    ff_dim=256,
    num_layers=4,
    dropout=0.15,
    learning_rate=2e-4,
    weight_decay=1e-4
):

    inputs = layers.Input(
        shape=input_shape,
        name="log_mel_input"
    )

    # --------------------------------------------------------
    # Normalize spectrogram
    # --------------------------------------------------------

    x = inputs

    # --------------------------------------------------------
    # Patch embedding
    # --------------------------------------------------------

    x = PatchEmbedding(
        patch_size=patch_size,
        embed_dim=embed_dim,
        name="patch_embedding"
    )(x)

    # 128 x 256 with 16 x 16 patches:
    #
    # 128 / 16 = 8
    # 256 / 16 = 16
    #
    # Total patches = 8 * 16 = 128
    #

    num_patches = (
        (input_shape[0] // patch_size)
        *
        (input_shape[1] // patch_size)
    )

    # --------------------------------------------------------
    # CLS token + positional embedding
    # --------------------------------------------------------

    x = ClassTokenPositionEmbedding(
        num_patches=num_patches,
        embed_dim=embed_dim,
        name="class_token_position"
    )(x)

    # --------------------------------------------------------
    # Transformer encoder stack
    # --------------------------------------------------------

    for i in range(num_layers):

        x = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_encoder_{i + 1}"
        )(x)

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    # --------------------------------------------------------
    # CLS token
    # --------------------------------------------------------

    x = x[:, 0, :]

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
        name="ViT_SER"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    try:

        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

    except AttributeError:

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model