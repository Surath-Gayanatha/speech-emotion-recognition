"""
CREMA-D MFCC Feature Extraction Pipeline
=========================================

Purpose
-------
Extract acoustic features from the original CREMA-D AudioWAV dataset
for speaker/actor-independent speech emotion classification.

Feature representation
----------------------
40 MFCC
+ 40 Delta MFCC
+ 40 Delta-Delta MFCC
= 120 acoustic feature channels

Experimental methodology
------------------------
1. Original CREMA-D WAV files are used.
2. Audio is converted to mono and resampled to 16 kHz.
3. Actor-independent train/validation/test splits are enforced.
4. Actor overlap between splits is checked to prevent leakage.
5. MFCC, delta and delta-delta features are extracted.
6. Feature sequences are truncated/padded to a fixed temporal length.
7. Normalization statistics are calculated from TRAINING DATA ONLY.
8. Padding values are excluded from normalization statistics.
9. The training mean/std are applied to validation and test sets.
10. Class distributions and extraction metadata are saved.
11. A fixed random seed is used for reproducibility.

Important
---------
The saved combined features.npy file is provided only for compatibility
with older project scripts. For the final assignment experiments, models
should load the separate train/validation/test files from data/processed/mfcc.

Usage
-----
    python -m src.features.extract_features
"""


# ============================================================
# IMPORTS
# ============================================================

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
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    MAX_PAD_LEN,
    USE_DELTA,
    USE_DELTA_DELTA,
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

MFCC_DIR = PROCESSED_DIR / "mfcc"

NORMALIZATION_DIR = (
    PROCESSED_DIR / "normalization"
)


# ============================================================
# LABEL MAPPINGS
# ============================================================

LABEL_TO_EMOTION = {
    value: key
    for key, value in EMOTION_LABELS.items()
}


# ============================================================
# DIRECTORY CREATION
# ============================================================

def create_directories():
    """
    Create all required output directories.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MFCC_DIR.mkdir(
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
    Load actor IDs from an actor split file.

    Example file:

        1001
        1002
        1003
        ...

    Returns
    -------
    set
        Set of actor IDs represented as strings.
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
# ACTOR SPLIT LEAKAGE CHECK
# ============================================================

def validate_actor_splits(
    train_actors,
    val_actors,
    test_actors
):
    """
    Verify that no actor appears in more than one split.

    This is critical for speaker-independent evaluation.
    """

    train_val = (
        train_actors.intersection(
            val_actors
        )
    )

    train_test = (
        train_actors.intersection(
            test_actors
        )
    )

    val_test = (
        val_actors.intersection(
            test_actors
        )
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
    """
    Extract actor ID from a CREMA-D filename.

    Example:
        1001_DFA_ANG_XX.wav

    Actor ID:
        1001
    """

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
    """
    Extract emotion label from a CREMA-D filename.

    Example:
        1001_DFA_ANG_XX.wav

    Emotion code:
        ANG
    """

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

    return EMOTION_LABELS[
        emotion_code
    ]


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path):
    """
    Load an audio file as mono and resample it
    to the configured sampling rate.
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=MONO_AUDIO
    )

    if audio is None or len(audio) == 0:

        raise ValueError(
            "Audio signal is empty."
        )

    duration = (
        len(audio) / sample_rate
    )

    if duration < MIN_AUDIO_DURATION:

        raise ValueError(
            f"Audio too short: "
            f"{duration:.4f} seconds"
        )

    return audio


# ============================================================
# SAFE DELTA CALCULATION
# ============================================================

def calculate_delta(
    mfcc,
    order
):
    """
    Calculate delta or delta-delta features safely.

    Very short audio clips can contain too few frames for
    librosa's default delta window. This function adapts the
    width while preserving the requested derivative order.
    """

    n_frames = mfcc.shape[1]

    # Delta calculation requires an odd width.
    # Use the largest suitable odd window up to 9.

    if n_frames >= 9:

        width = 9

    elif n_frames >= 7:

        width = 7

    elif n_frames >= 5:

        width = 5

    elif n_frames >= 3:

        width = 3

    else:

        # Extremely short sequence.
        # Return zeros instead of crashing.
        return np.zeros_like(
            mfcc,
            dtype=np.float32
        )

    return librosa.feature.delta(
        mfcc,
        order=order,
        width=width
    )


# ============================================================
# EXTRACT MFCC FEATURES
# ============================================================

def extract_mfcc(audio):
    """
    Extract MFCC + delta + delta-delta features.

    Output
    ------
    np.ndarray

        Shape:
            (feature_channels, time_frames)

        Normally:
            (120, variable_time)
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    feature_stack = [
        mfcc
    ]

    # --------------------------------------------------------
    # First-order derivative
    # --------------------------------------------------------

    if USE_DELTA:

        delta = calculate_delta(
            mfcc,
            order=1
        )

        feature_stack.append(
            delta
        )

    # --------------------------------------------------------
    # Second-order derivative
    # --------------------------------------------------------

    if USE_DELTA_DELTA:

        delta_delta = calculate_delta(
            mfcc,
            order=2
        )

        feature_stack.append(
            delta_delta
        )

    features = np.vstack(
        feature_stack
    )

    return features.astype(
        np.float32
    )


# ============================================================
# TRUNCATE TO MAXIMUM LENGTH
# ============================================================

def truncate_features(features):
    """
    Truncate sequences longer than MAX_PAD_LEN.

    Padding is intentionally NOT performed here.

    This allows normalization statistics to be calculated
    without including artificial zero-padding values.
    """

    if features.shape[1] > MAX_PAD_LEN:

        features = features[
            :,
            :MAX_PAD_LEN
        ]

    return features


# ============================================================
# PAD TO FIXED LENGTH
# ============================================================

def pad_features(features):
    """
    Zero-pad a feature sequence to MAX_PAD_LEN.

    Input:
        (feature_channels, time)

    Output:
        (feature_channels, MAX_PAD_LEN)
    """

    current_length = (
        features.shape[1]
    )

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
    """
    Load audio and extract raw acoustic features.

    Returns
    -------
    features
    label
    actor_id
    filename
    valid_frames
    """

    audio = load_audio(
        file_path
    )

    features = extract_mfcc(
        audio
    )

    # Truncate BEFORE normalization.
    features = truncate_features(
        features
    )

    valid_frames = (
        features.shape[1]
    )

    label = parse_label(
        file_path
    )

    actor_id = parse_actor_id(
        file_path
    )

    return (
        features.astype(
            np.float32
        ),
        label,
        actor_id,
        file_path.stem,
        valid_frames
    )


# ============================================================
# GET FILES FOR ACTOR SPLIT
# ============================================================

def get_split_files(actor_ids):
    """
    Select WAV files belonging to the specified actors.
    """

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
    """
    Extract raw features for one actor split.

    Returns
    -------
    features : list[np.ndarray]
        Variable-length feature matrices.

    labels : np.ndarray

    actors : np.ndarray

    filenames : np.ndarray

    valid_lengths : np.ndarray

    skipped_files : list
    """

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
        f"{split_name.upper()} FEATURE EXTRACTION"
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

            features.append(
                feat
            )

            labels.append(
                label
            )

            actors.append(
                actor_id
            )

            filenames.append(
                filename
            )

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
        f"{split_name} labels shape: "
        f"{labels.shape}"
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
# CALCULATE TRAINING NORMALIZATION STATISTICS
# ============================================================

def calculate_training_statistics(
    train_features
):
    """
    Calculate per-channel mean and standard deviation
    using TRAINING DATA ONLY.

    IMPORTANT:
        Artificial zero-padding is excluded.

    Statistics are calculated across:
        - training samples
        - valid time frames

    Result:
        one mean/std value for each feature channel.
    """

    if not train_features:

        raise ValueError(
            "Training feature list is empty."
        )

    n_channels = (
        train_features[0].shape[0]
    )

    channel_sum = np.zeros(
        n_channels,
        dtype=np.float64
    )

    channel_squared_sum = np.zeros(
        n_channels,
        dtype=np.float64
    )

    total_frames = 0

    # --------------------------------------------------------
    # First pass: sum and squared sum
    # --------------------------------------------------------

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

        total_frames += (
            valid.shape[1]
        )

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

    # Numerical safety.
    variance = np.maximum(
        variance,
        0.0
    )

    std = np.sqrt(
        variance
    )

    # Avoid division by zero.
    std = np.where(
        std < 1e-8,
        1.0,
        std
    )

    mean = mean.astype(
        np.float32
    )

    std = std.astype(
        np.float32
    )

    return (
        mean,
        std
    )


# ============================================================
# NORMALIZE VARIABLE-LENGTH FEATURES
# ============================================================

def normalize_variable_features(
    feature_list,
    mean,
    std
):
    """
    Normalize variable-length feature sequences using
    training-set statistics.
    """

    normalized_features = []

    for features in feature_list:

        normalized = (
            features - mean[:, None]
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
    """
    Convert variable-length normalized sequences into
    fixed-size arrays.

    Output:
        (N, feature_channels, MAX_PAD_LEN)
    """

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
# SAVE NORMALIZATION STATISTICS
# ============================================================

def save_normalization_statistics(
    mean,
    std
):
    """
    Save training-only normalization statistics.
    """

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
        "Training normalization statistics saved."
    )


# ============================================================
# SAVE SPLIT DATA
# ============================================================

def save_split(
    split_name,
    features,
    labels,
    actors,
    filenames,
    valid_lengths
):
    """
    Save processed features and metadata for one split.
    """

    np.save(
        MFCC_DIR /
        f"{split_name}_features.npy",
        features
    )

    np.save(
        MFCC_DIR /
        f"{split_name}_labels.npy",
        labels
    )

    np.save(
        MFCC_DIR /
        f"{split_name}_actors.npy",
        actors
    )

    np.save(
        MFCC_DIR /
        f"{split_name}_filenames.npy",
        filenames
    )

    np.save(
        MFCC_DIR /
        f"{split_name}_valid_lengths.npy",
        valid_lengths
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def get_class_distribution(
    labels
):
    """
    Calculate the number of samples per emotion class.
    """

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


# ============================================================
# SAVE CLASS DISTRIBUTION
# ============================================================

def save_class_distribution(
    train_labels,
    val_labels,
    test_labels
):
    """
    Save class distribution for all splits.
    """

    distribution = {

        "train": get_class_distribution(
            train_labels
        ),

        "validation": get_class_distribution(
            val_labels
        ),

        "test": get_class_distribution(
            test_labels
        )
    }

    output_path = (
        PROCESSED_DIR /
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
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for split, values in (
        distribution.items()
    ):

        print()
        print(
            split.upper()
        )

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
    """
    Save feature extraction configuration and
    dataset processing metadata.
    """

    metadata = {

        "dataset": "CREMA-D",

        "sample_rate": SAMPLE_RATE,

        "mono_audio": MONO_AUDIO,

        "min_audio_duration": (
            MIN_AUDIO_DURATION
        ),

        "n_mfcc": N_MFCC,

        "use_delta": USE_DELTA,

        "use_delta_delta": (
            USE_DELTA_DELTA
        ),

        "n_fft": N_FFT,

        "hop_length": HOP_LENGTH,

        "max_pad_len": MAX_PAD_LEN,

        "feature_channels": (
            int(
                train_features[0].shape[0]
            )
        ),

        "train_samples": (
            len(train_features)
        ),

        "validation_samples": (
            len(val_features)
        ),

        "test_samples": (
            len(test_features)
        ),

        "train_min_valid_frames": (
            int(train_lengths.min())
        ),

        "train_max_valid_frames": (
            int(train_lengths.max())
        ),

        "validation_min_valid_frames": (
            int(val_lengths.min())
        ),

        "validation_max_valid_frames": (
            int(val_lengths.max())
        ),

        "test_min_valid_frames": (
            int(test_lengths.min())
        ),

        "test_max_valid_frames": (
            int(test_lengths.max())
        ),

        "split_strategy": (
            "Actor-independent"
        ),

        "normalization": (
            "Per-feature-channel mean/std"
        ),

        "normalization_source": (
            "Training data only"
        ),

        "padding_excluded_from_statistics": (
            True
        ),

        "random_seed": SEED,

        "skipped_files": skipped_files
    }

    output_path = (
        PROCESSED_DIR /
        "feature_extraction_metadata.json"
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
        f"Metadata saved to: {output_path}"
    )


# ============================================================
# CHECK FINAL DATA
# ============================================================

def validate_final_data(
    train_features,
    val_features,
    test_features,
    train_labels,
    val_labels,
    test_labels
):
    """
    Perform final sanity checks before saving.
    """

    # --------------------------------------------------------
    # Feature shapes
    # --------------------------------------------------------

    expected_channels = 0

    if USE_DELTA:

        expected_channels += N_MFCC

    if USE_DELTA_DELTA:

        expected_channels += N_MFCC

    expected_channels += N_MFCC

    expected_shape = (
        expected_channels,
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

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NaN / Inf check
    # --------------------------------------------------------

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
        "Final data validation: PASSED"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "CREMA-D MFCC FEATURE EXTRACTION PIPELINE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check raw dataset
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
    # 2. Create output directories
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
    # 4. Validate actor independence
    # --------------------------------------------------------

    validate_actor_splits(
        train_actors,
        val_actors,
        test_actors
    )

    # --------------------------------------------------------
    # 5. Extract TRAIN features
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
    # 6. Extract VALIDATION features
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
    # 7. Extract TEST features
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
    # 8. Calculate TRAIN-ONLY normalization
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAIN-ONLY NORMALIZATION"
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
    # 9. Normalize using TRAIN statistics
    # --------------------------------------------------------

    train_normalized = (
        normalize_variable_features(
            train_raw,
            train_mean,
            train_std
        )
    )

    val_normalized = (
        normalize_variable_features(
            val_raw,
            train_mean,
            train_std
        )
    )

    test_normalized = (
        normalize_variable_features(
            test_raw,
            train_mean,
            train_std
        )
    )

    # --------------------------------------------------------
    # 10. Pad AFTER normalization
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
    # 11. Save normalization statistics
    # --------------------------------------------------------

    save_normalization_statistics(
        train_mean,
        train_std
    )

    # --------------------------------------------------------
    # 12. Final validation
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
    # 13. Save individual splits
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
    # 14. Save class distributions
    # --------------------------------------------------------

    save_class_distribution(
        train_labels,
        val_labels,
        test_labels
    )

    # --------------------------------------------------------
    # 15. Save metadata
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
    # 16. Legacy combined files
    # --------------------------------------------------------
    #
    # These are retained for compatibility with older scripts.
    #
    # IMPORTANT:
    # Do NOT use these combined files for final experiments
    # if the script performs a new random split.
    #
    # Use:
    #   train_features.npy
    #   validation_features.npy
    #   test_features.npy
    #
    # instead.
    # --------------------------------------------------------

    all_features = np.concatenate(
        [
            train_features,
            val_features,
            test_features
        ],
        axis=0
    )

    all_labels = np.concatenate(
        [
            train_labels,
            val_labels,
            test_labels
        ],
        axis=0
    )

    all_filenames = np.concatenate(
        [
            train_filenames,
            val_filenames,
            test_filenames
        ],
        axis=0
    )

    np.save(
        PROCESSED_DIR /
        "features.npy",
        all_features
    )

    np.save(
        PROCESSED_DIR /
        "labels.npy",
        all_labels
    )

    np.save(
        PROCESSED_DIR /
        "filenames.npy",
        all_filenames
    )

    # --------------------------------------------------------
    # 17. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FEATURE EXTRACTION COMPLETED SUCCESSFULLY"
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
        f"MFCC             : {N_MFCC}"
    )

    print(
        f"Delta            : {USE_DELTA}"
    )

    print(
        f"Delta-Delta      : {USE_DELTA_DELTA}"
    )

    print(
        f"Total channels   : "
        f"{train_features.shape[1]}"
    )

    print(
        f"Time frames      : "
        f"{train_features.shape[2]}"
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
        "Random seed      : "
        f"{SEED}"
    )

    print()
    print(
        "Ready for model training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()