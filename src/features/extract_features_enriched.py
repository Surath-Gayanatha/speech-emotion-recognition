"""Enriched Feature Extraction Pipeline for Speech Emotion Recognition.

Extracts:
1. MFCC (40) + Delta (40) + Delta-Delta (40) = 120
2. Log-Mel Spectrogram (64)
3. Chroma STFT (12)
4. RMS Energy (1)
5. Zero-Crossing Rate (1)
6. Spectral Centroid (1)
7. Spectral Rolloff (1)

Total: 200 features per time frame.
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
    EMOTION_LABELS,
)

def extract_enriched_features(file_path: Path) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # 1. MFCC + Deltas (120)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # 2. Log-Mel Spectrogram (64)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Chroma STFT (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 4. RMS Energy (1)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    # 5. ZCR (1)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)

    # 6. Spectral Centroid (1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 7. Spectral Rolloff (1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Stack all features along axis 0
    features = np.vstack([
        mfcc,
        delta_mfcc,
        delta2_mfcc,
        log_mel,
        chroma,
        rms,
        zcr,
        centroid,
        rolloff
    ])  # shape: (200, time)

    # Pad or truncate to MAX_PAD_LEN
    if features.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :MAX_PAD_LEN]

    return features

def parse_label(file_path: Path) -> int:
    emotion_code = file_path.stem.split("_")[2]
    return EMOTION_LABELS[emotion_code]

def main():
    wav_files = sorted(Path(DATA_RAW_DIR).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in {DATA_RAW_DIR}")

    print(f"Found {len(wav_files)} audio files. Extracting enriched features...")

    features, labels, filenames = [], [], []
    for wav_path in tqdm(wav_files, desc="Extracting 200D features"):
        try:
            feat = extract_enriched_features(wav_path)
            label = parse_label(wav_path)
        except (KeyError, IndexError, Exception) as e:
            print(f"Skipping {wav_path.name}: {e}")
            continue
        features.append(feat)
        labels.append(label)
        filenames.append(wav_path.stem)

    features = np.stack(features)  # (N, 200, 174)
    labels = np.array(labels)

    out_dir = DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "features_enriched.npy", features)
    np.save(out_dir / "labels_enriched.npy", labels)
    np.save(out_dir / "filenames_enriched.npy", np.array(filenames))

    print(f"Saved enriched dataset of shape {features.shape} to {out_dir}")

if __name__ == "__main__":
    main()
