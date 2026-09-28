import os
import random
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SAMPLE_RATE = 16000

N_MELS = 64

N_FFT = 640

HOP_LENGTH = 320

MAX_FRAMES = 200

TRIM_TOP_DB = 30

FEATURE_DIR = Path(
    "data/mlp_attention_processed"
)

RAW_DIR = Path(
    "data/raw/AudioWAV"
)


EMOTION_LABELS = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(audio):

    # --------------------------------------------------------
    # Log-Mel
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        log_mel
    )


    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(
        log_mel,
        order=2
    )


    # --------------------------------------------------------
    # Stack
    # --------------------------------------------------------

    features = np.concatenate(
        [
            log_mel,
            delta,
            delta_delta
        ],
        axis=0
    )


    # Expected:
    #
    # 64 Log-Mel
    # 64 Delta
    # 64 Delta-Delta
    #
    # Total = 192


    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    if features.shape[1] > MAX_FRAMES:

        features = features[
            :,
            :MAX_FRAMES
        ]


    # --------------------------------------------------------
    # Zero padding
    # --------------------------------------------------------

    elif features.shape[1] < MAX_FRAMES:

        padding = (
            MAX_FRAMES
            - features.shape[1]
        )

        features = np.pad(
            features,
            (
                (0, 0),
                (0, padding)
            ),
            mode="constant"
        )


    # --------------------------------------------------------
    # Convert:
    #
    # (192, 200)
    #
    # to:
    #
    # (200, 192)
    # --------------------------------------------------------

    features = features.T

    return features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("MLP + TEMPORAL ATTENTION FEATURE EXTRACTION")
print("=" * 70)

print("\nSearching for WAV files...")

wav_files = sorted(
    RAW_DIR.glob("*.wav")
)

print(
    f"Found {len(wav_files)} WAV files"
)


all_features = []
all_labels = []
all_actors = []
all_filenames = []

skipped = 0


# ============================================================
# PROCESS AUDIO
# ============================================================

for wav_path in tqdm(
    wav_files,
    desc="Extracting features"
):

    try:

        filename = wav_path.stem

        # CREMA-D format:
        #
        # ActorID_Sentence_Emotion_Level_...
        #
        # Example:
        # 1001_DFA_ANG_XX.wav

        parts = filename.split("_")

        if len(parts) < 3:

            skipped += 1

            continue


        actor_id = int(
            parts[0]
        )

        emotion_code = parts[2]


        if emotion_code not in EMOTION_LABELS:

            skipped += 1

            continue


        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        # ----------------------------------------------------
        # Silence trimming
        # ----------------------------------------------------

        audio, _ = librosa.effects.trim(
            audio,
            top_db=TRIM_TOP_DB
        )


        if len(audio) == 0:

            skipped += 1

            continue


        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        feature = extract_features(
            audio
        )


        # Safety check
        if feature.shape != (
            MAX_FRAMES,
            192
        ):

            skipped += 1

            continue


        all_features.append(
            feature
        )

        all_labels.append(
            EMOTION_LABELS[
                emotion_code
            ]
        )

        all_actors.append(
            actor_id
        )

        all_filenames.append(
            filename
        )


    except Exception as e:

        print(
            f"\nSkipped {wav_path.name}: {e}"
        )

        skipped += 1


# ============================================================
# CONVERT TO NUMPY
# ============================================================

features = np.asarray(
    all_features,
    dtype=np.float32
)

labels = np.asarray(
    all_labels,
    dtype=np.int32
)

actors = np.asarray(
    all_actors,
    dtype=np.int32
)

filenames = np.asarray(
    all_filenames
)


# ============================================================
# SAVE
# ============================================================

np.save(
    FEATURE_DIR / "features.npy",
    features
)

np.save(
    FEATURE_DIR / "labels.npy",
    labels
)

np.save(
    FEATURE_DIR / "actors.npy",
    actors
)

np.save(
    FEATURE_DIR / "filenames.npy",
    filenames
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print(
    "Features:",
    features.shape
)

print(
    "Labels:",
    labels.shape
)

print(
    "Actors:",
    actors.shape
)

print(
    "Skipped:",
    skipped
)

print(
    "\nSaved to:",
    FEATURE_DIR
)

print("\nExpected feature shape:")
print("(7442, 200, 192)")