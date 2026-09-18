"""Pre-trained Spectrogram ResNet-34 for Speech Emotion Recognition.

Fine-tunes upper residual blocks (layer3 & layer4) on 3-channel Log-Mel Spectrograms
(Log-Mel + Delta + Delta-Delta) for fast, high-accuracy emotion classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class SpectrogramResNet(nn.Module):
    """Deep Transfer Learning Spectrogram ResNet for Speech Emotion Recognition."""

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.35):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT
        self.backbone = models.resnet34(weights=weights)

        # Freeze early layers (conv1, bn1, layer1, layer2) to accelerate training & preserve low-level features
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False

        # Replace final classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
