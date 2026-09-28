"""
MFT-TCN Attention V8 feature extraction.

Purpose:
    Create the same 192-D frame-level representation used by the team's
    BiLSTM + Attention experiment:

        64 Log-Mel + 64 Delta + 64 Delta-Delta

    This version is intentionally different from the older MFT extractor,
    which produced 201 features. The reduced representation makes the
    TCN experiment more directly comparable with the BiLSTM + Attention
    experiment while keeping the TCN + MHA architecture distinct.

Run from the repository root:
    python src/features/extract_mft_tcn_v8_features.py
"""

from pathlib import Path
import numpy as np
import librosa

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/mft_tcn_v8_processed")

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 640
HOP_LENGTH = 320
MAX_TIME_STEPS = 200
TRIM_DB = 30

EMOTION_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5,
}


def extract_features(audio_path: Path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size < int(SAMPLE_RATE * 0.2):
        return None

    audio, _ = librosa.effects.trim(audio, top_db=TRIM_DB)

    if audio.size < int(SAMPLE_RATE * 0.2):
        return None

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=sr // 2,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    width = min(
        9,
        log_mel.shape[1] if log_mel.shape[1] % 2 == 1
        else log_mel.shape[1] - 1,
    )

    if width < 3:
        return None

    delta = librosa.feature.delta(
        log_mel,
        width=width,
        order=1,
    )

    delta_delta = librosa.feature.delta(
        log_mel,
        width=width,
        order=2,
    )

    combined = np.vstack([
        log_mel,
        delta,
        delta_delta,
    ]).T.astype(np.float32)

    combined = combined[:MAX_TIME_STEPS]

    if combined.shape[0] < MAX_TIME_STEPS:
        padding = np.zeros(
            (
                MAX_TIME_STEPS - combined.shape[0],
                N_MELS * 3,
            ),
            dtype=np.float32,
        )
        combined = np.vstack([combined, padding])

    return combined


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(RAW_DIR.rglob("*.wav"))
    print(f"Found audio files: {len(audio_files)}")

    all_features = []
    all_labels = []
    all_actors = []
    all_names = []
    skipped = 0

    for index, audio_path in enumerate(audio_files, start=1):
        try:
            parts = audio_path.stem.split("_")

            if len(parts) < 3:
                skipped += 1
                continue

            actor_id = int(parts[0])
            emotion_code = parts[2]

            if emotion_code not in EMOTION_MAP:
                skipped += 1
                continue

            feature = extract_features(audio_path)

            if feature is None:
                skipped += 1
                continue

            all_features.append(feature)
            all_labels.append(EMOTION_MAP[emotion_code])
            all_actors.append(actor_id)
            all_names.append(audio_path.stem)

            if index % 500 == 0:
                print(f"Processed {index} / {len(audio_files)}")

        except Exception as exc:
            print(f"Error: {audio_path} -> {exc}")
            skipped += 1

    features = np.stack(all_features).astype(np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    actors = np.asarray(all_actors, dtype=np.int64)
    names = np.asarray(all_names)

    np.save(OUTPUT_DIR / "features.npy", features)
    np.save(OUTPUT_DIR / "labels.npy", labels)
    np.save(OUTPUT_DIR / "actors.npy", actors)
    np.save(OUTPUT_DIR / "names.npy", names)

    print("\nEXTRACTION COMPLETE")
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    print("Actors:", actors.shape)
    print("Skipped:", skipped)
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
