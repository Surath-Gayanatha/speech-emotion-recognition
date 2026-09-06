"""MFCC (+ delta / delta-delta) feature extraction for CREMA-D.

Rationale: plain MFCC-only is the most common recipe in public tutorials
for this exact dataset. Adding delta and delta-delta coefficients captures
the *rate of change* of the spectral envelope over time, which carries
useful prosodic/emotional information (e.g. how quickly pitch/energy
shifts) that static MFCC alone misses. This is a small but genuinely
justified deviation from the "MFCC-only" tutorial recipe -- document the
comparison (with vs without deltas) in the report's Preprocessing section
if time allows, rather than just asserting it.

Usage:
    python -m src.features.extract_features
"""

import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    SAMPLE_RATE,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    MAX_PAD_LEN,
    USE_DELTA,
    USE_DELTA_DELTA,
    EMOTION_LABELS,
)


def extract_mfcc(file_path: Path) -> tuple[np.ndarray, int]:
    """Return features and the number of valid frames before padding.

    The valid length is saved separately so training can ignore padded
    frames instead of treating them as speech.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    feature_stack = [mfcc]
    if USE_DELTA:
        feature_stack.append(librosa.feature.delta(mfcc, order=1))
    if USE_DELTA_DELTA:
        feature_stack.append(librosa.feature.delta(mfcc, order=2))

    features = np.vstack(feature_stack)  # (n_features, time)

    valid_length = min(features.shape[1], MAX_PAD_LEN)

    # Pad or truncate along the time axis to MAX_PAD_LEN.
    if features.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :MAX_PAD_LEN]

    return features, valid_length


def parse_label(file_path: Path) -> int:
    """CREMA-D filenames: '1001_DFA_ANG_XX.wav' -> emotion code 'ANG'."""
    emotion_code = file_path.stem.split("_")[2]
    return EMOTION_LABELS[emotion_code]


def main():
    wav_files = sorted(Path(DATA_RAW_DIR).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {DATA_RAW_DIR}. "
            "Download CREMA-D first (see data/README.md)."
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    features, lengths, labels, filenames = [], [], [], []
    for wav_path in tqdm(wav_files, desc="Extracting features"):
        try:
            feat, valid_length = extract_mfcc(wav_path)
            label = parse_label(wav_path)
        except (KeyError, IndexError):
            print(f"Skipping unparseable file: {wav_path.name}")
            continue
        features.append(feat)
        lengths.append(valid_length)
        labels.append(label)
        filenames.append(wav_path.stem)

    features = np.stack(features)   # (N, n_features, MAX_PAD_LEN)
    lengths = np.array(lengths, dtype=np.int32)
    labels = np.array(labels)

    np.save(DATA_PROCESSED_DIR / "features.npy", features)
    np.save(DATA_PROCESSED_DIR / "lengths.npy", lengths)
    np.save(DATA_PROCESSED_DIR / "labels.npy", labels)
    np.save(DATA_PROCESSED_DIR / "filenames.npy", np.array(filenames))

    print(f"Saved {features.shape[0]} feature arrays of shape {features.shape[1:]} to {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
