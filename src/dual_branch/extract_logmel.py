import os
import numpy as np
import librosa
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "AudioWAV")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "logmel")

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

# Fixed number of time frames
MAX_LEN = 300

# CREMA-D emotion mapping
EMOTION_MAP = {
    "ANG": 0,   # Anger
    "DIS": 1,   # Disgust
    "FEA": 2,   # Fear
    "HAP": 3,   # Happy
    "NEU": 4,   # Neutral
    "SAD": 5    # Sad
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIXED LENGTH FUNCTION
# ============================================================

def fix_length(feature, max_len=MAX_LEN):
    """
    Make spectrogram have a fixed number of time frames.
    """

    current_len = feature.shape[1]

    if current_len < max_len:
        pad_width = max_len - current_len

        feature = np.pad(
            feature,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    elif current_len > max_len:
        feature = feature[:, :max_len]

    return feature


# ============================================================
# EXTRACT LOG-MEL
# ============================================================

def extract_logmel(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Convert to dB
    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Fixed size
    logmel = fix_length(logmel)

    return logmel.astype(np.float32)


# ============================================================
# GET EMOTION FROM CREMA-D FILENAME
# ============================================================

def get_emotion(filename):

    # Example:
    # 1001_DFA_ANG_XX.wav

    parts = filename.split("_")

    if len(parts) < 3:
        return None

    emotion_code = parts[2]

    return EMOTION_MAP.get(emotion_code)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREMA-D LOG-MEL SPECTROGRAM EXTRACTION")
    print("=" * 70)

    print(f"Audio directory : {AUDIO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(
            f"Audio directory not found:\n{AUDIO_DIR}"
        )

    wav_files = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ]

    wav_files.sort()

    print(f"\nTotal WAV files found: {len(wav_files)}")

    if len(wav_files) == 0:
        raise RuntimeError("No WAV files found!")

    features = []
    labels = []
    filenames = []

    failed = []

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    for filename in tqdm(
        wav_files,
        desc="Extracting Log-Mel"
    ):

        emotion = get_emotion(filename)

        if emotion is None:
            print(
                f"\nWARNING: Could not determine emotion: {filename}"
            )
            continue

        audio_path = os.path.join(
            AUDIO_DIR,
            filename
        )

        try:

            logmel = extract_logmel(audio_path)

            features.append(logmel)
            labels.append(emotion)
            filenames.append(filename)

        except Exception as e:

            failed.append(
                (filename, str(e))
            )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    features = np.array(
        features,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    filenames = np.array(
        filenames
    )

    # ========================================================
    # SAVE
    # ========================================================

    features_path = os.path.join(
        OUTPUT_DIR,
        "logmel_features.npy"
    )

    labels_path = os.path.join(
        OUTPUT_DIR,
        "labels.npy"
    )

    filenames_path = os.path.join(
        OUTPUT_DIR,
        "filenames.npy"
    )

    np.save(
        features_path,
        features
    )

    np.save(
        labels_path,
        labels
    )

    np.save(
        filenames_path,
        filenames
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETED")
    print("=" * 70)

    print(f"Features shape : {features.shape}")
    print(f"Labels shape   : {labels.shape}")
    print(f"Filenames shape: {filenames.shape}")

    print("\nEmotion distribution:")

    emotion_names = [
        "Anger",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad"
    ]

    for index, name in enumerate(emotion_names):

        count = np.sum(labels == index)

        print(
            f"{name:10s}: {count}"
        )

    print(f"\nFailed files: {len(failed)}")

    if failed:

        print("\nFirst failed files:")

        for filename, error in failed[:10]:

            print(
                f"{filename}: {error}"
            )

    print("\nSaved files:")

    print(features_path)
    print(labels_path)
    print(filenames_path)

    print("=" * 70)


if __name__ == "__main__":
    main()