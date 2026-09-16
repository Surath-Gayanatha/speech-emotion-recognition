"""CNN + Multi-Head Self-Attention for Speech Emotion Recognition.

This architecture processes multi-modal spectral-temporal features 
(MFCCs, Log-Mel Spectrogram, Chromagram, and Prosodic Energy Features) 
using SpecAugment, Spatial Dropout, Multi-Head Attention, and Dual Pooling Fusion.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES

class SpecAugment(layers.Layer):
    """SpecAugment data augmentation layer (Frequency and Time Masking)."""
    def __init__(self, freq_mask_max=20, time_mask_max=24, num_freq_masks=2, num_time_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs

        x = inputs
        shape = tf.shape(x)
        B, T, F = shape[0], shape[1], shape[2]

        for _ in range(self.num_freq_masks):
            f_len = tf.random.uniform([], minval=1, maxval=self.freq_mask_max, dtype=tf.int32)
            f_start = tf.random.uniform([], minval=0, maxval=F - f_len, dtype=tf.int32)
            mask_left = tf.ones([B, T, f_start], dtype=inputs.dtype)
            mask_mid = tf.zeros([B, T, f_len], dtype=inputs.dtype)
            mask_right = tf.ones([B, T, F - f_start - f_len], dtype=inputs.dtype)
            mask = tf.concat([mask_left, mask_mid, mask_right], axis=-1)
            x = x * mask

        for _ in range(self.num_time_masks):
            t_len = tf.random.uniform([], minval=1, maxval=self.time_mask_max, dtype=tf.int32)
            t_start = tf.random.uniform([], minval=0, maxval=T - t_len, dtype=tf.int32)
            mask_left = tf.ones([B, t_start, F], dtype=inputs.dtype)
            mask_mid = tf.zeros([B, t_len, F], dtype=inputs.dtype)
            mask_right = tf.ones([B, T - t_start - t_len, F], dtype=inputs.dtype)
            mask = tf.concat([mask_left, mask_mid, mask_right], axis=1)
            x = x * mask

        return x

def build_enriched_cnn_attention(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Builds an enriched 1D CNN with Multi-Head Self-Attention and SpecAugment."""
    inputs = keras.Input(shape=input_shape)

    # SpecAugment Layer for Speaker Invariance
    x = SpecAugment(freq_mask_max=20, time_mask_max=24)(inputs)

    # Convolutional Block 1
    x = layers.Conv1D(128, kernel_size=5, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Convolutional Block 2
    x = layers.Conv1D(256, kernel_size=5, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.25)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Convolutional Block 3 + Attention
    x = layers.Conv1D(256, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Pooling Fusion (Average + Max)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Classification Head
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="enriched_cnn_attention")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"],
    )
    return model
