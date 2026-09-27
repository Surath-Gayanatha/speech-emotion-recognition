import tensorflow as tf
from tensorflow.keras import layers, Model


class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        reduced = max(channels // reduction, 8)

        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(reduced, activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)

        return inputs * x


class AttentionPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.score = layers.Dense(1)
        super().build(input_shape)

    def call(self, inputs):
        scores = self.score(inputs)
        weights = tf.nn.softmax(scores, axis=1)

        return tf.reduce_sum(inputs * weights, axis=1)


def conv_block(x, filters, dropout_rate):
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = SEBlock(filters)(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(dropout_rate)(x)

    return x


def build_model(
    input_shape=(64, 174, 1),
    num_classes=6
):

    inputs = layers.Input(shape=input_shape)

    # --------------------------------------------------
    # CNN feature extraction
    # --------------------------------------------------

    x = conv_block(inputs, 32, 0.15)
    x = conv_block(x, 64, 0.20)
    x = conv_block(x, 128, 0.25)

    # --------------------------------------------------
    # Convert CNN feature map to sequence
    # --------------------------------------------------

    x = layers.Permute((2, 1, 3))(x)

    shape = x.shape

    x = layers.Reshape(
        (shape[1], shape[2] * shape[3])
    )(x)

    # --------------------------------------------------
    # Projection
    # --------------------------------------------------

    x = layers.Dense(256, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.20)(x)

    # --------------------------------------------------
    # BiLSTM
    # --------------------------------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.20
        )
    )(x)

    # --------------------------------------------------
    # Multi-Head Attention
    # --------------------------------------------------

    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32,
        dropout=0.10
    )(
        x, x
    )

    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # --------------------------------------------------
    # Feed Forward Network
    # --------------------------------------------------

    ffn = layers.Dense(
        256,
        activation="gelu"
    )(x)

    ffn = layers.Dropout(0.20)(ffn)

    ffn = layers.Dense(256)(ffn)

    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    # --------------------------------------------------
    # Attention Pooling
    # --------------------------------------------------

    x = AttentionPooling()(x)

    # --------------------------------------------------
    # Classification head
    # --------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="CNN_BiLSTM_MHA_V2"
    )

    return model