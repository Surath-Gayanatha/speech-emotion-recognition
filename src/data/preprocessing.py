"""Shared leakage-safe preprocessing for actor-level model training."""

import numpy as np

from src.config import DATA_PROCESSED_DIR


def _load_valid_lengths(features: np.ndarray) -> np.ndarray:
    """Load saved frame lengths, or infer them for legacy feature files."""
    lengths_path = DATA_PROCESSED_DIR / "lengths.npy"
    if lengths_path.exists():
        return np.load(lengths_path).astype(np.int32)

    valid_frames = np.any(features != 0, axis=1)
    return valid_frames.sum(axis=1).astype(np.int32)


def normalize_features(
    features: np.ndarray,
    filenames: np.ndarray,
    train_actors: set[str],
) -> np.ndarray:
    """Normalize MFCC channels using training actors only.

    Features are expected as ``(samples, channels, time)`` and returned as
    ``(samples, time, channels)``. Padding is reset to zero after scaling so
    its large raw MFCC values cannot affect the model.
    """
    lengths = _load_valid_lengths(features)
    if len(lengths) != len(features):
        raise ValueError("lengths.npy does not match features.npy")

    actor_ids = np.array([str(name).split("_")[0] for name in filenames])
    train_mask = np.isin(actor_ids, list(train_actors))
    features_time_major = np.transpose(features.astype(np.float32), (0, 2, 1))

    frame_indices = np.arange(features_time_major.shape[1])[None, :]
    valid_mask = frame_indices < lengths[:, None]
    train_valid_mask = valid_mask & train_mask[:, None]
    train_values = features_time_major[train_valid_mask]

    if train_values.size == 0:
        raise ValueError("No valid training frames found for normalization")

    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.maximum(std, 1e-6)

    normalized = (features_time_major - mean) / std
    normalized[~valid_mask] = 0.0

    np.save(DATA_PROCESSED_DIR / "feature_mean.npy", mean)
    np.save(DATA_PROCESSED_DIR / "feature_std.npy", std)
    return normalized.astype(np.float32)
