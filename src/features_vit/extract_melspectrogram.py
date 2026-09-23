import os
import glob
import numpy as np
import librosa


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA_DIR = "data/raw"

OUTPUT_DIR = "data/vit_processed"

SAMPLE_RATE = 16000

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256

MAX_TIME_STEPS = 256


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTION_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}


# ============================================================
# EXTRACT LOG-MEL SPECTROGRAM
# ============================================================

def extract_log_mel(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert power spectrogram to decibels
    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Pad / truncate time dimension
    # --------------------------------------------------------

    time_steps = log_mel.shape[1]

    if time_steps < MAX_TIME_STEPS:

        padding = MAX_TIME_STEPS - time_steps

        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, padding)),
            mode="constant",
            constant_values=-80
        )

    else:

        log_mel = log_mel[:, :MAX_TIME_STEPS]

    return log_mel.astype(np.float32)


# ============================================================
# GET EMOTION LABEL FROM CREMA-D FILENAME
# ============================================================

def get_emotion(filename):

    name = os.path.basename(filename)

    parts = name.split("_")

    if len(parts) < 3:
        return None

    emotion_code = parts[2]

    return EMOTION_MAP.get(emotion_code)


# ============================================================
# GET ACTOR ID
# ============================================================

def get_actor(filename):

    name = os.path.basename(filename)

    parts = name.split("_")

    if len(parts) < 1:
        return None

    return int(parts[0])


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    audio_files = glob.glob(
        os.path.join(
            RAW_DATA_DIR,
            "**",
            "*.wav"
        ),
        recursive=True
    )

    print(f"Found audio files: {len(audio_files)}")

    features = []
    labels = []
    actors = []

    skipped = 0

    for i, audio_path in enumerate(audio_files):

        emotion = get_emotion(audio_path)

        actor = get_actor(audio_path)

        if emotion is None or actor is None:

            skipped += 1
            continue

        try:

            feature = extract_log_mel(
                audio_path
            )

            features.append(feature)
            labels.append(emotion)
            actors.append(actor)

        except Exception as e:

            print(
                f"Error processing {audio_path}: {e}"
            )

            skipped += 1

        if (i + 1) % 500 == 0:

            print(
                f"Processed {i + 1}/{len(audio_files)}"
            )

    features = np.array(
        features,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    actors = np.array(
        actors,
        dtype=np.int64
    )

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Features shape : {features.shape}")
    print(f"Labels shape   : {labels.shape}")
    print(f"Actors shape   : {actors.shape}")
    print(f"Skipped        : {skipped}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "features.npy"
        ),
        features
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "labels.npy"
        ),
        labels
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "actors.npy"
        ),
        actors
    )

    print()
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()