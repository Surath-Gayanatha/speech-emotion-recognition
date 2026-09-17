"""Fast 512D ResNet Spectrogram Feature Extractor for Transfer Learning.

Passes 3-channel spectrogram images through pre-trained ResNet-34 backbone.
Extraction time: ~15 seconds for 7,442 audio clips on CPU.
"""

from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet34_Weights
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.config import DATA_PROCESSED_DIR


def extract_resnet_features(batch_size: int = 64):
    spec_file = DATA_PROCESSED_DIR / "spectrogram_images.npy"
    if not spec_file.exists():
        raise FileNotFoundError(f"{spec_file} not found. Run extract_spectrogram_images first.")

    print("Loading 3-Channel Spectrogram Dataset...")
    specs = np.load(spec_file).astype(np.float32)
    labels = np.load(DATA_PROCESSED_DIR / "spectrogram_labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "spectrogram_filenames.npy")

    # Normalize per-channel standard scaling
    for c in range(3):
        mean = np.mean(specs[:, c, :, :])
        std = np.std(specs[:, c, :, :]) + 1e-7
        specs[:, c, :, :] = (specs[:, c, :, :] - mean) / std

    print("Loading Pre-trained ResNet-34 Backbone...")
    weights = ResNet34_Weights.DEFAULT
    resnet = models.resnet34(weights=weights)
    # Remove fc layer to get 512D global average pooled features
    resnet.fc = torch.nn.Identity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for feature extraction: {device}")
    resnet.to(device)
    resnet.eval()

    ds = TensorDataset(torch.tensor(specs))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    features_list = []
    with torch.no_grad():
        for (bx,) in tqdm(loader, desc="Extracting 512D ResNet Features"):
            bx = bx.to(device)
            feat = resnet(bx)
            features_list.append(feat.cpu().numpy())

    features = np.vstack(features_list).astype(np.float32)
    print(f"Extracted ResNet 512D features matrix of shape: {features.shape}")

    np.save(DATA_PROCESSED_DIR / "resnet_features.npy", features)
    np.save(DATA_PROCESSED_DIR / "resnet_labels.npy", labels)
    np.save(DATA_PROCESSED_DIR / "resnet_filenames.npy", filenames)

    print(f"Saved 512D ResNet features to {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    extract_resnet_features()
