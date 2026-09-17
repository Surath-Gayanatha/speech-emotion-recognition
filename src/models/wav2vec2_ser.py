"""Pre-trained Transformer (Wav2Vec2 / WavLM) for Speech Emotion Recognition.

This module provides fine-tuning architectures for Speech Emotion Recognition
using Self-Supervised Learning (SSL) pre-trained speech representations.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2Model

class SpeechEmotionTransformer(nn.Module):
    """Speech Emotion Recognition model using a pre-trained SSL audio backbone."""
    def __init__(self, model_name: str = "superb/wav2vec2-base-superb-er", num_classes: int = 6):
        super().__init__()
        self.model_name = model_name
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, D)

        # Mean and Max Pooling Fusion across time steps
        mean_pool = torch.mean(hidden_states, dim=1)
        max_pool, _ = torch.max(hidden_states, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        logits = self.classifier(pooled)
        return logits

def load_feature_extractor(model_name: str = "superb/wav2vec2-base-superb-er"):
    """Loads feature extractor for waveform resampling and normalization."""
    return AutoFeatureExtractor.from_pretrained(model_name)
