"""Training Script using 200D Enriched Features (MFCC + MelSpec + Chroma + RMS + ZCR + Spectral Features).

Applies:
1. Channel-wise StandardScaler fitted on X_train.
2. SpecAugment (Frequency & Time Masking).
3. Residual Multi-Head Attention & Feature Pooling Fusion.
4. Label Smoothing & AdamW Optimizer.
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    SPLITS_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    NUM_CLASSES,
)

EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

def load_actor_ids(filename):
    path = SPLITS_DIR / filename
    return set(path.read_text().splitlines())

def get_actor_id(filename):
    return filename.split("_")[0]

def create_split_mask(filenames, actor_ids):
    return np.array([get_actor_id(filename) in actor_ids for filename in filenames])

class SpecAugment(layers.Layer):
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

def build_enriched_model(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    x = SpecAugment(freq_mask_max=20, time_mask_max=24)(inputs)

    # Conv Block 1
    x = layers.Conv1D(128, kernel_size=5, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Conv Block 2
    x = layers.Conv1D(256, kernel_size=5, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.25)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Conv Block 3 with Attention
    x = layers.Conv1D(256, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 200D enriched features...")
    features = np.load(DATA_PROCESSED_DIR / "features_enriched.npy").astype(np.float32)
    labels = np.load(DATA_PROCESSED_DIR / "labels_enriched.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "filenames_enriched.npy")

    features = np.transpose(features, (0, 2, 1))  # (N, 174, 200)

    train_actors = load_actor_ids("train_actors.txt")
    val_actors = load_actor_ids("val_actors.txt")
    test_actors = load_actor_ids("test_actors.txt")

    train_mask = create_split_mask(filenames, train_actors)
    val_mask = create_split_mask(filenames, val_actors)
    test_mask = create_split_mask(filenames, test_actors)

    X_train_raw = features[train_mask]
    y_train_raw = labels[train_mask]

    X_val_raw = features[val_mask]
    y_val_raw = labels[val_mask]

    X_test_raw = features[test_mask]
    y_test_raw = labels[test_mask]

    print(f"Train: {X_train_raw.shape} | Val: {X_val_raw.shape} | Test: {X_test_raw.shape}")

    N_tr, T, F = X_train_raw.shape
    N_va = X_val_raw.shape[0]
    N_te = X_test_raw.shape[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(N_tr, T, F)
    X_val = scaler.transform(X_val_raw.reshape(-1, F)).reshape(N_va, T, F)
    X_test = scaler.transform(X_test_raw.reshape(-1, F)).reshape(N_te, T, F)

    y_train_cat = keras.utils.to_categorical(y_train_raw, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val_raw, NUM_CLASSES)

    classes = np.unique(y_train_raw)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_raw)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}

    model = build_enriched_model(input_shape=(T, F), num_classes=NUM_CLASSES)
    model.summary()

    checkpoint_path = MODELS_DIR / "enriched_cnn_best.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    print("\nStarting Training on Enriched 200D Acoustic Features...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = keras.models.load_model(
        checkpoint_path,
        custom_objects={"SpecAugment": SpecAugment}
    )

    y_prob = best_model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    test_acc = np.mean(y_pred == y_test_raw)
    print(f"\nEnriched Model Test Accuracy: {test_acc * 100:.2f}%")

    report = classification_report(y_test_raw, y_pred, target_names=EMOTION_NAMES, digits=4)
    cm = confusion_matrix(y_test_raw, y_pred)

    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    report_path = RESULTS_DIR / "enriched_cnn_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Enriched 200D Features CNN + SpecAugment Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    main()
