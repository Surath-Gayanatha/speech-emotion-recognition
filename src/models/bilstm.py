"""BiLSTM for Speech Emotion Recognition.

Architectural role vs. LSTM (do not treat these as near-duplicates in the
report -- justify the distinction explicitly): BiLSTM processes the MFCC
sequence in both forward and backward time directions and concatenates
both hidden states at each timestep. This lets the representation at any
point in the utterance draw on *both* what came before and what comes
after -- relevant for emotional speech where cues like a rising pitch or a
pause may only make sense in light of what follows. The trade-off to
discuss in Critical Analysis: roughly 2x the parameters/compute of the
unidirectional LSTM for (typically) a modest accuracy gain -- a genuine
complexity-vs-performance comparison point, not just "which number is
higher."
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES


def build_bilstm(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """input_shape: (time_steps, n_features)."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="bilstm")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
