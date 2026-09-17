"""
CREMA-D Log-Mel Spectrogram Feature Extraction
================================================

Purpose
-------
Extract Log-Mel Spectrogram features from the original
CREMA-D AudioWAV dataset for actor-independent
speech emotion classification.

Feature representation
----------------------
64 Mel frequency bands
x
174 fixed time frames

Methodology
-----------
1. Original CREMA-D WAV files are used.
2. Audio is converted to mono and resampled to 16 kHz.
3. Actor-independent train/validation/test splits are used.
4. Actor overlap is checked to prevent leakage.
5. Log-Mel Spectrogram features are extracted.
6. Features are truncated before normalization.
7. Training mean/std are calculated from TRAIN ONLY.
8. Validation and test use TRAIN normalization statistics.
9. Padding is applied AFTER normalization.
10. Metadata and split information are saved.

Usage
-----
python -m src.features.extract_logmel_features
"""

from pathlib import Path
import json
import random

import numpy as np
import librosa
from tqdm import tqdm

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    SPLITS_DIR,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    MAX_PAD_LEN,
    N_MELS,
    FMIN,
    FMAX,
    USE_LOG_MEL,
    EMOTION_LABELS,
    RANDOM_SEED,
    MONO_AUDIO,
    MIN_AUDIO_DURATION,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = RANDOM_SEED

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# PATH CONFIGURATION
# ============================================================

RAW_DIR = Path(DATA_RAW_DIR)
PROCESSED_DIR = Path(DATA_PROCESSED_DIR)
SPLIT_DIR = Path(SPLITS_DIR)

# IMPORTANT:
# Use a separate directory so existing MFCC files are untouched.
LOGMEL_DIR = PROCESSED_DIR / "logmel"

NORMALIZATION_DIR = LOGMEL_DIR / "normalization"


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_TO_EMOTION = {
    value: key
    for key, value in EMOTION_LABELS.items()
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    LOGMEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    NORMALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD ACTOR SPLITS
# ============================================================

def load_actor_ids(filename):
    """
    Load actor IDs from actor split file.
    """

    split_path = SPLIT_DIR / filename

    if not split_path.exists():
        raise FileNotFoundError(
            f"Actor split file not found: {split_path}"
        )

    with open(
        split_path,
        "r",
        encoding="utf-8"
    ) as file:

        actors = {
            line.strip()
            for line in file
            if line.strip()
        }

    if not actors:
        raise ValueError(
            f"Actor split file is empty: {split_path}"
        )

    return actors


# ============================================================
# VALIDATE ACTOR SPLITS
# ============================================================

def validate_actor_splits(
    train_actors,
    val_actors,
    test_actors
):

    train_val = train_actors.intersection(
        val_actors
    )

    train_test = train_actors.intersection(
        test_actors
    )

    val_test = val_actors.intersection(
        test_actors
    )

    if train_val:
        raise ValueError(
            "Actor leakage detected between "
            f"TRAIN and VALIDATION: {sorted(train_val)}"
        )

    if train_test:
        raise ValueError(
            "Actor leakage detected between "
            f"TRAIN and TEST: {sorted(train_test)}"
        )

    if val_test:
        raise ValueError(
            "Actor leakage detected between "
            f"VALIDATION and TEST: {sorted(val_test)}"
        )

    print()
    print("Actor split validation: PASSED")
    print(
        "No actor overlap between "
        "train, validation and test."
    )


# ============================================================
# PARSE ACTOR ID
# ============================================================

def parse_actor_id(file_path):

    parts = file_path.stem.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Invalid CREMA-D filename: "
            f"{file_path.name}"
        )

    return parts[0]


# ============================================================
# PARSE EMOTION LABEL
# ============================================================

def parse_label(file_path):

    parts = file_path.stem.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Invalid CREMA-D filename: "
            f"{file_path.name}"
        )

    emotion_code = parts[2]

    if emotion_code not in EMOTION_LABELS:
        raise ValueError(
            f"Unknown emotion code "
            f"'{emotion_code}' in "
            f"{file_path.name}"
        )

    return EMOTION_LABELS[emotion_code]


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=MONO_AUDIO
    )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "Audio signal is empty."
        )

    duration = len(audio) / sample_rate

    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio too short: "
            f"{duration:.4f} seconds"
        )

    return audio


# ============================================================
# EXTRACT LOG-MEL SPECTROGRAM
# ============================================================

def extract_logmel(audio):

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    if USE_LOG_MEL:

        log_mel = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max
        )

    else:

        log_mel = mel_spectrogram

    return log_mel.astype(
        np.float32
    )


# ============================================================
# TRUNCATE FEATURE
# ============================================================

def truncate_features(features):

    if features.shape[1] > MAX_PAD_LEN:

        features = features[
            :,
            :MAX_PAD_LEN
        ]

    return features


# ============================================================
# PAD FEATURE
# ============================================================

def pad_features(features):

    current_length = features.shape[1]

    if current_length >= MAX_PAD_LEN:

        return features[
            :,
            :MAX_PAD_LEN
        ]

    padding_length = (
        MAX_PAD_LEN - current_length
    )

    padded = np.pad(
        features,
        (
            (0, 0),
            (0, padding_length)
        ),
        mode="constant",
        constant_values=0
    )

    return padded


# ============================================================
# PROCESS ONE AUDIO FILE
# ============================================================

def process_audio_file(file_path):

    audio = load_audio(
        file_path
    )

    features = extract_logmel(
        audio
    )

    # IMPORTANT:
    # Truncate BEFORE normalization.
    features = truncate_features(
        features
    )

    valid_frames = features.shape[1]

    label = parse_label(
        file_path
    )

    actor_id = parse_actor_id(
        file_path
    )

    return (
        features.astype(np.float32),
        label,
        actor_id,
        file_path.stem,
        valid_frames
    )


# ============================================================
# GET FILES FOR ACTOR SPLIT
# ============================================================

def get_split_files(actor_ids):

    all_files = sorted(
        RAW_DIR.glob("*.wav")
    )

    selected_files = []

    for file_path in all_files:

        try:

            actor_id = parse_actor_id(
                file_path
            )

            if actor_id in actor_ids:

                selected_files.append(
                    file_path
                )

        except ValueError:

            continue

    return selected_files


# ============================================================
# EXTRACT ONE SPLIT
# ============================================================

def extract_split(
    actor_ids,
    split_name
):

    files = get_split_files(
        actor_ids
    )

    if not files:
        raise RuntimeError(
            f"No WAV files found for "
            f"{split_name} split."
        )

    print()
    print("=" * 70)
    print(
        f"{split_name.upper()} LOG-MEL EXTRACTION"
    )
    print("=" * 70)

    print(
        f"Actors      : {len(actor_ids)}"
    )

    print(
        f"Audio files : {len(files)}"
    )

    features = []
    labels = []
    actors = []
    filenames = []
    valid_lengths = []
    skipped_files = []

    for file_path in tqdm(
        files,
        desc=f"Processing {split_name}"
    ):

        try:

            (
                feat,
                label,
                actor_id,
                filename,
                valid_frames
            ) = process_audio_file(
                file_path
            )

            features.append(feat)
            labels.append(label)
            actors.append(actor_id)
            filenames.append(filename)
            valid_lengths.append(
                valid_frames
            )

        except (
            ValueError,
            KeyError,
            IndexError,
            RuntimeError
        ) as error:

            skipped_files.append(
                {
                    "file": file_path.name,
                    "reason": str(error)
                }
            )

    if not features:

        raise RuntimeError(
            f"No valid audio files processed "
            f"for {split_name}."
        )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    actors = np.asarray(
        actors
    )

    filenames = np.asarray(
        filenames
    )

    valid_lengths = np.asarray(
        valid_lengths,
        dtype=np.int32
    )

    print()
    print(
        f"{split_name} files processed: "
        f"{len(features)}"
    )

    print(
        f"{split_name} skipped files: "
        f"{len(skipped_files)}"
    )

    print(
        f"{split_name} min frames: "
        f"{valid_lengths.min()}"
    )

    print(
        f"{split_name} max frames: "
        f"{valid_lengths.max()}"
    )

    return (
        features,
        labels,
        actors,
        filenames,
        valid_lengths,
        skipped_files
    )


# ============================================================
# CALCULATE TRAINING STATISTICS
# ============================================================

def calculate_training_statistics(
    train_features
):

    if not train_features:

        raise ValueError(
            "Training feature list is empty."
        )

    n_mels = train_features[0].shape[0]

    channel_sum = np.zeros(
        n_mels,
        dtype=np.float64
    )

    channel_squared_sum = np.zeros(
        n_mels,
        dtype=np.float64
    )

    total_frames = 0

    for features in train_features:

        valid = features[
            :,
            :MAX_PAD_LEN
        ]

        channel_sum += np.sum(
            valid,
            axis=1,
            dtype=np.float64
        )

        channel_squared_sum += np.sum(
            np.square(valid),
            axis=1,
            dtype=np.float64
        )

        total_frames += valid.shape[1]

    if total_frames == 0:

        raise ValueError(
            "No valid training frames found."
        )

    mean = (
        channel_sum /
        total_frames
    )

    variance = (
        channel_squared_sum /
        total_frames
    ) - np.square(mean)

    variance = np.maximum(
        variance,
        0.0
    )

    std = np.sqrt(
        variance
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32)
    )


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(
    feature_list,
    mean,
    std
):

    normalized_features = []

    for features in feature_list:

        normalized = (
            features -
            mean[:, None]
        ) / std[:, None]

        normalized_features.append(
            normalized.astype(
                np.float32
            )
        )

    return normalized_features


# ============================================================
# PAD NORMALIZED FEATURES
# ============================================================

def pad_normalized_features(
    feature_list
):

    padded_features = []

    for features in feature_list:

        padded = pad_features(
            features
        )

        padded_features.append(
            padded.astype(
                np.float32
            )
        )

    return np.stack(
        padded_features
    )


# ============================================================
# SAVE NORMALIZATION
# ============================================================

def save_normalization_statistics(
    mean,
    std
):

    np.save(
        NORMALIZATION_DIR /
        "train_mean.npy",
        mean
    )

    np.save(
        NORMALIZATION_DIR /
        "train_std.npy",
        std
    )

    print()
    print(
        "Log-Mel training normalization "
        "statistics saved."
    )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(
    split_name,
    features,
    labels,
    actors,
    filenames,
    valid_lengths
):

    np.save(
        LOGMEL_DIR /
        f"{split_name}_features.npy",
        features
    )

    np.save(
        LOGMEL_DIR /
        f"{split_name}_labels.npy",
        labels
    )

    np.save(
        LOGMEL_DIR /
        f"{split_name}_actors.npy",
        actors
    )

    np.save(
        LOGMEL_DIR /
        f"{split_name}_filenames.npy",
        filenames
    )

    np.save(
        LOGMEL_DIR /
        f"{split_name}_valid_lengths.npy",
        valid_lengths
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def get_class_distribution(labels):

    distribution = {}

    for label_id in sorted(
        LABEL_TO_EMOTION.keys()
    ):

        emotion_code = (
            LABEL_TO_EMOTION[label_id]
        )

        count = int(
            np.sum(
                labels == label_id
            )
        )

        distribution[
            emotion_code
        ] = count

    return distribution


def save_class_distribution(
    train_labels,
    val_labels,
    test_labels
):

    distribution = {

        "train":
            get_class_distribution(
                train_labels
            ),

        "validation":
            get_class_distribution(
                val_labels
            ),

        "test":
            get_class_distribution(
                test_labels
            )
    }

    output_path = (
        LOGMEL_DIR /
        "class_distribution.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            distribution,
            file,
            indent=4
        )

    print()
    print("=" * 70)
    print("LOG-MEL CLASS DISTRIBUTION")
    print("=" * 70)

    for split, values in (
        distribution.items()
    ):

        print()
        print(split.upper())

        for emotion, count in (
            values.items()
        ):

            print(
                f"  {emotion}: {count}"
            )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    train_features,
    val_features,
    test_features,
    train_lengths,
    val_lengths,
    test_lengths,
    skipped_files
):

    metadata = {

        "dataset":
            "CREMA-D",

        "feature_type":
            "Log-Mel Spectrogram",

        "sample_rate":
            SAMPLE_RATE,

        "mono_audio":
            MONO_AUDIO,

        "n_mels":
            N_MELS,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "fmin":
            FMIN,

        "fmax":
            FMAX,

        "use_log_mel":
            USE_LOG_MEL,

        "max_pad_len":
            MAX_PAD_LEN,

        "feature_shape":
            [
                N_MELS,
                MAX_PAD_LEN
            ],

        "train_samples":
            len(train_features),

        "validation_samples":
            len(val_features),

        "test_samples":
            len(test_features),

        "train_min_valid_frames":
            int(train_lengths.min()),

        "train_max_valid_frames":
            int(train_lengths.max()),

        "validation_min_valid_frames":
            int(val_lengths.min()),

        "validation_max_valid_frames":
            int(val_lengths.max()),

        "test_min_valid_frames":
            int(test_lengths.min()),

        "test_max_valid_frames":
            int(test_lengths.max()),

        "split_strategy":
            "Actor-independent",

        "normalization":
            "Per-Mel-band mean/std",

        "normalization_source":
            "Training data only",

        "padding_excluded_from_statistics":
            True,

        "random_seed":
            SEED,

        "skipped_files":
            skipped_files
    }

    output_path = (
        LOGMEL_DIR /
        "metadata.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print()
    print(
        f"Metadata saved to: "
        f"{output_path}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_data(
    train_features,
    val_features,
    test_features,
    train_labels,
    val_labels,
    test_labels
):

    expected_shape = (
        N_MELS,
        MAX_PAD_LEN
    )

    if train_features.shape[1:] != expected_shape:

        raise ValueError(
            "Unexpected TRAIN feature shape: "
            f"{train_features.shape}"
        )

    if val_features.shape[1:] != expected_shape:

        raise ValueError(
            "Unexpected VALIDATION feature shape: "
            f"{val_features.shape}"
        )

    if test_features.shape[1:] != expected_shape:

        raise ValueError(
            "Unexpected TEST feature shape: "
            f"{test_features.shape}"
        )

    valid_labels = set(
        EMOTION_LABELS.values()
    )

    for labels, split_name in [
        (train_labels, "TRAIN"),
        (val_labels, "VALIDATION"),
        (test_labels, "TEST")
    ]:

        unique_labels = set(
            labels.tolist()
        )

        if not unique_labels.issubset(
            valid_labels
        ):

            raise ValueError(
                f"Invalid labels found in "
                f"{split_name}: "
                f"{unique_labels}"
            )

    for features, split_name in [
        (train_features, "TRAIN"),
        (val_features, "VALIDATION"),
        (test_features, "TEST")
    ]:

        if not np.isfinite(
            features
        ).all():

            raise ValueError(
                f"NaN or Inf values detected "
                f"in {split_name} features."
            )

    print()
    print(
        "Final Log-Mel data validation: PASSED"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "CREMA-D LOG-MEL FEATURE EXTRACTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check dataset
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Raw dataset directory not found: "
            f"{RAW_DIR}"
        )

    wav_files = sorted(
        RAW_DIR.glob("*.wav")
    )

    if not wav_files:

        raise FileNotFoundError(
            f"No WAV files found under: "
            f"{RAW_DIR}"
        )

    print()
    print(
        f"Total WAV files found: "
        f"{len(wav_files)}"
    )

    # --------------------------------------------------------
    # 2. Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # 3. Load actor splits
    # --------------------------------------------------------

    train_actors = load_actor_ids(
        "train_actors.txt"
    )

    val_actors = load_actor_ids(
        "val_actors.txt"
    )

    test_actors = load_actor_ids(
        "test_actors.txt"
    )

    print()
    print("=" * 70)
    print("ACTOR SPLITS")
    print("=" * 70)

    print(
        f"Train actors      : "
        f"{len(train_actors)}"
    )

    print(
        f"Validation actors : "
        f"{len(val_actors)}"
    )

    print(
        f"Test actors       : "
        f"{len(test_actors)}"
    )

    # --------------------------------------------------------
    # 4. Actor leakage check
    # --------------------------------------------------------

    validate_actor_splits(
        train_actors,
        val_actors,
        test_actors
    )

    # --------------------------------------------------------
    # 5. TRAIN
    # --------------------------------------------------------

    (
        train_raw,
        train_labels,
        train_actors_used,
        train_filenames,
        train_lengths,
        train_skipped
    ) = extract_split(
        train_actors,
        "train"
    )

    # --------------------------------------------------------
    # 6. VALIDATION
    # --------------------------------------------------------

    (
        val_raw,
        val_labels,
        val_actors_used,
        val_filenames,
        val_lengths,
        val_skipped
    ) = extract_split(
        val_actors,
        "validation"
    )

    # --------------------------------------------------------
    # 7. TEST
    # --------------------------------------------------------

    (
        test_raw,
        test_labels,
        test_actors_used,
        test_filenames,
        test_lengths,
        test_skipped
    ) = extract_split(
        test_actors,
        "test"
    )

    # --------------------------------------------------------
    # 8. TRAIN-ONLY NORMALIZATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAIN-ONLY LOG-MEL NORMALIZATION"
    )
    print("=" * 70)

    train_mean, train_std = (
        calculate_training_statistics(
            train_raw
        )
    )

    print(
        f"Mean shape: "
        f"{train_mean.shape}"
    )

    print(
        f"Std shape : "
        f"{train_std.shape}"
    )

    # --------------------------------------------------------
    # 9. Normalize
    # --------------------------------------------------------

    train_normalized = (
        normalize_features(
            train_raw,
            train_mean,
            train_std
        )
    )

    val_normalized = (
        normalize_features(
            val_raw,
            train_mean,
            train_std
        )
    )

    test_normalized = (
        normalize_features(
            test_raw,
            train_mean,
            train_std
        )
    )

    # --------------------------------------------------------
    # 10. PAD AFTER NORMALIZATION
    # --------------------------------------------------------

    train_features = (
        pad_normalized_features(
            train_normalized
        )
    )

    val_features = (
        pad_normalized_features(
            val_normalized
        )
    )

    test_features = (
        pad_normalized_features(
            test_normalized
        )
    )

    # --------------------------------------------------------
    # 11. Save normalization
    # --------------------------------------------------------

    save_normalization_statistics(
        train_mean,
        train_std
    )

    # --------------------------------------------------------
    # 12. Validate
    # --------------------------------------------------------

    validate_final_data(
        train_features,
        val_features,
        test_features,
        train_labels,
        val_labels,
        test_labels
    )

    # --------------------------------------------------------
    # 13. Save split files
    # --------------------------------------------------------

    save_split(
        "train",
        train_features,
        train_labels,
        train_actors_used,
        train_filenames,
        train_lengths
    )

    save_split(
        "validation",
        val_features,
        val_labels,
        val_actors_used,
        val_filenames,
        val_lengths
    )

    save_split(
        "test",
        test_features,
        test_labels,
        test_actors_used,
        test_filenames,
        test_lengths
    )

    # --------------------------------------------------------
    # 14. Class distribution
    # --------------------------------------------------------

    save_class_distribution(
        train_labels,
        val_labels,
        test_labels
    )

    # --------------------------------------------------------
    # 15. Metadata
    # --------------------------------------------------------

    all_skipped = (
        train_skipped
        + val_skipped
        + test_skipped
    )

    save_metadata(
        train_raw,
        val_raw,
        test_raw,
        train_lengths,
        val_lengths,
        test_lengths,
        all_skipped
    )

    # --------------------------------------------------------
    # 16. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOG-MEL FEATURE EXTRACTION COMPLETED"
    )
    print("=" * 70)

    print()
    print("FINAL DATA SHAPES")
    print("-" * 50)

    print(
        f"Train      : "
        f"{train_features.shape}"
    )

    print(
        f"Validation : "
        f"{val_features.shape}"
    )

    print(
        f"Test       : "
        f"{test_features.shape}"
    )

    print()
    print("FEATURE CONFIGURATION")
    print("-" * 50)

    print(
        f"Feature type : Log-Mel Spectrogram"
    )

    print(
        f"Mel bands    : {N_MELS}"
    )

    print(
        f"FFT size     : {N_FFT}"
    )

    print(
        f"Hop length   : {HOP_LENGTH}"
    )

    print(
        f"Time frames  : {MAX_PAD_LEN}"
    )

    print(
        f"Sample rate  : {SAMPLE_RATE}"
    )

    print()
    print("METHODOLOGY")
    print("-" * 50)

    print(
        "Split strategy   : Actor-independent"
    )

    print(
        "Normalization    : Training data only"
    )

    print(
        "Padding excluded : Yes"
    )

    print(
        f"Random seed      : {SEED}"
    )

    print()
    print(
        "Ready for Log-Mel model training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()