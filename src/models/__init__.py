"""
Models package for Speech Emotion Recognition.
"""

from src.models.mlp import build_mlp
from src.models.cnn1d import build_cnn1d
from src.models.lstm import build_lstm
from src.models.bilstm import build_bilstm
from src.models.cnn_attention import build_cnn_attention
from src.models.cnn_bilstm_attention import build_cnn_bilstm_attention
from src.models.cnn_attention_enriched import build_enriched_cnn_attention

try:
    from src.models.wav2vec2_ser import SpeechEmotionTransformer
except ImportError:
    # Wav2Vec2 is an optional pipeline and must not block the TensorFlow models.
    SpeechEmotionTransformer = None

__all__ = [
    "build_mlp",
    "build_cnn1d",
    "build_lstm",
    "build_bilstm",
    "build_cnn_attention",
    "build_cnn_bilstm_attention",
    "build_enriched_cnn_attention",
    "SpeechEmotionTransformer",
]
