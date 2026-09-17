"""3-Channel Log-Mel Spectrogram Image Extractor for Deep Transfer Learning.

Extracts:
Channel 0: Log-Mel Spectrogram (64 bins)
Channel 1: Delta Log-Mel Spectrogram (64 bins)
Channel 2: Delta-Delta Log-Mel Spectrogram (64 bins)

Output shape: (N, 3, 64, 174)
"""

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    MAX_PAD_LEN,
    EMOTION_LABELS,
)

N_MELS = 64


def extract_spectrogram_images():
    wav_files = sorted(Path(DATA_RAW_DIR).glob("*.wav"))
    valid_files = [p for p in wav_files if p.stem.split("_")[2] in EMOTION_LABELS]

    if not valid_files:
        raise FileNotFoundError(f"No valid audio files found in {DATA_RAW_DIR}")

    print(f"Extracting 3-Channel Spectrogram Images for {len(valid_files)} audio clips...")

    specs, labels, filenames = [], [], []

    for path in tqdm(valid_files, desc="Extracting 3-Channel Spectrograms"):
        emotion_code = path.stem.split("_")[2]
        label = EMOTION_LABELS[emotion_code]

        y, sr = sf.read(str(path), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Log-Mel Spectrogram (Channel 0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Deltas (Channel 1 & 2)
        delta_mel = librosa.feature.delta(log_mel, order=1)
        delta2_mel = librosa.feature.delta(log_mel, order=2)

        # Stack into 3 RGB channels: (3, 64, time)
        img = np.stack([log_mel, delta_mel, delta2_mel], axis=0)

        # Pad or truncate time axis to MAX_PAD_LEN (174)
        if img.shape[2] < MAX_PAD_LEN:
            pad_w = MAX_PAD_LEN - img.shape[2]
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_w)), mode="constant")
        else:
            img = img[:, :, :MAX_PAD_LEN]

        specs.append(img.astype(np.float32))
        labels.append(label)
        filenames.append(path.stem)

    specs = np.stack(specs)
    labels = np.array(labels, dtype=np.int64)
    filenames = np.array(filenames)

    out_dir = DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "spectrogram_images.npy", specs)
    np.save(out_dir / "spectrogram_labels.npy", labels)
    np.save(out_dir / "spectrogram_filenames.npy", filenames)

    print(f"Successfully saved 3-channel spectrogram dataset of shape {specs.shape} to {out_dir}")


if __name__ == "__main__":
    extract_spectrogram_images()
