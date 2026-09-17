"""
Shared configuration for the CREMA-D Speech Emotion Recognition project.

This file keeps dataset paths, preprocessing settings, feature extraction
settings, reproducibility controls, and common training configuration in
one place so that all models use a consistent experimental setup.

Project:
    Comparative Analysis of Deep Learning Architectures for
    Speech Emotion Classification Using the CREMA-D Dataset

Main task:
    6-class speech emotion classification

Emotions:
    Anger, Disgust, Fear, Happy, Neutral, Sad

Evaluation strategy:
    Actor-independent train / validation / test split
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Raw CREMA-D WAV files
DATA_RAW_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
    / "AudioWAV"
)

# Processed features
DATA_PROCESSED_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)

# Actor-level split files
SPLITS_DIR = (
    ROOT_DIR
    / "data"
    / "splits"
)

# Model files
MODELS_DIR = (
    ROOT_DIR
    / "models"
)

# Experiment results
RESULTS_DIR = (
    ROOT_DIR
    / "results"
)

# Report figures
REPORT_FIGURES_DIR = (
    ROOT_DIR
    / "report"
    / "figures"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

# Global random seed.
# The same seed should be used wherever randomness is involved.
RANDOM_SEED = 42


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_NAME = "CREMA-D"


# ============================================================
# EMOTION LABELS
# ============================================================

# CREMA-D filename emotion codes -> numerical class labels
EMOTION_LABELS = {
    "ANG": 0,   # Anger
    "DIS": 1,   # Disgust
    "FEA": 2,   # Fear
    "HAP": 3,   # Happy
    "NEU": 4,   # Neutral
    "SAD": 5,   # Sad
}

# Number of emotion classes.
# Derived automatically from the label mapping.
NUM_CLASSES = len(EMOTION_LABELS)


# Reverse mapping for evaluation and visualization
ID_TO_EMOTION = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
}


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

# Target sampling frequency.
# All audio is loaded/resampled to this sampling rate.
SAMPLE_RATE = 16000

# Convert audio to mono.
MONO_AUDIO = True

# Ignore extremely short/corrupted audio files.
MIN_AUDIO_DURATION = 0.1


# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

# Number of static MFCC coefficients
N_MFCC = 40

# Include first-order temporal derivatives
USE_DELTA = True

# Include second-order temporal derivatives
USE_DELTA_DELTA = True

# FFT window size
N_FFT = 2048

# Hop size between consecutive frames
HOP_LENGTH = 512

# Fixed number of time frames.
# Short sequences are zero padded.
# Longer sequences are truncated.
MAX_PAD_LEN = 174


# ============================================================
# DERIVED FEATURE INFORMATION
# ============================================================

# Calculate the total number of MFCC-based feature channels.
#
# 40 MFCC
# + 40 Delta
# + 40 Delta-Delta
# = 120

MFCC_FEATURE_CHANNELS = N_MFCC

if USE_DELTA:
    MFCC_FEATURE_CHANNELS += N_MFCC

if USE_DELTA_DELTA:
    MFCC_FEATURE_CHANNELS += N_MFCC


# Expected input shape for sequence-based models:
#
# (time_steps, feature_channels)
#
# 174 time steps x 120 features

SEQUENCE_INPUT_SHAPE = (
    MAX_PAD_LEN,
    MFCC_FEATURE_CHANNELS
)


# ============================================================
# LOG-MEL SPECTROGRAM CONFIGURATION
# ============================================================

# Number of Mel frequency bands for 2D CNN experiments
N_MELS = 64

# Minimum frequency
FMIN = 20

# Maximum frequency.
# None means librosa will use Nyquist frequency.
FMAX = None

# Convert Mel power to decibels
USE_LOG_MEL = True


# Expected Log-Mel representation:
#
# (N_MELS, MAX_PAD_LEN)

LOG_MEL_INPUT_SHAPE = (
    N_MELS,
    MAX_PAD_LEN
)


# ============================================================
# NORMALIZATION
# ============================================================

# Normalization strategy:
#
# Statistics are calculated ONLY from training data.
# The same training statistics are then applied to:
#
#   Training
#   Validation
#   Test
#
# This prevents test-set information from influencing preprocessing.

NORMALIZATION_METHOD = "training_mean_std"

NORMALIZE_USING_TRAIN_ONLY = True


# ============================================================
# DATA AUGMENTATION
# ============================================================

# Augmentation must be applied ONLY to the training set.
#
# Validation and test data must remain untouched.

USE_AUGMENTATION = True

AUGMENTATION_TRAIN_ONLY = True


# Individual augmentation options.
# These can be enabled/disabled for controlled experiments.

USE_ADD_NOISE = True

USE_TIME_SHIFT = True

USE_GAIN_CHANGE = True

USE_PITCH_SHIFT = False

USE_TIME_STRETCH = False


# Augmentation probability.
#
# A value of 0.5 means an augmentation has a 50% chance
# of being applied when selected.

AUGMENTATION_PROBABILITY = 0.5


# Noise configuration
NOISE_MIN_DB = 15.0
NOISE_MAX_DB = 30.0


# Time shift range
MAX_TIME_SHIFT_SECONDS = 0.2


# Gain range in decibels
MIN_GAIN_DB = -6.0
MAX_GAIN_DB = 6.0


# Pitch shifting range
MIN_PITCH_SHIFT_STEPS = -2
MAX_PITCH_SHIFT_STEPS = 2


# Time stretching range
MIN_TIME_STRETCH_RATE = 0.9
MAX_TIME_STRETCH_RATE = 1.1


# ============================================================
# ACTOR-LEVEL DATA SPLIT
# ============================================================

# The split is performed at ACTOR level,
# not at individual audio-clip level.

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# Split file names
TRAIN_ACTORS_FILE = "train_actors.txt"
VAL_ACTORS_FILE = "val_actors.txt"
TEST_ACTORS_FILE = "test_actors.txt"


# Validate that the ratios add up to 1.0
SPLIT_RATIO_SUM = (
    TRAIN_RATIO
    + VAL_RATIO
    + TEST_RATIO
)

if abs(SPLIT_RATIO_SUM - 1.0) > 1e-6:
    raise ValueError(
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
    )


# ============================================================
# COMMON MODEL TRAINING CONFIGURATION
# ============================================================

# Maximum number of epochs
EPOCHS = 60

# Batch size
BATCH_SIZE = 32

# Initial learning rate
LEARNING_RATE = 0.001

# Optimizer
OPTIMIZER = "adam"

# Classification loss
LOSS_FUNCTION = "sparse_categorical_crossentropy"


# ============================================================
# TRAINING CALLBACK CONFIGURATION
# ============================================================

# Stop training when validation performance stops improving.
USE_EARLY_STOPPING = True

EARLY_STOPPING_MONITOR = "val_loss"

EARLY_STOPPING_PATIENCE = 8

RESTORE_BEST_WEIGHTS = True


# Reduce learning rate when validation loss stops improving.
USE_REDUCE_LR = True

REDUCE_LR_MONITOR = "val_loss"

REDUCE_LR_FACTOR = 0.5

REDUCE_LR_PATIENCE = 3

MIN_LEARNING_RATE = 1e-6


# ============================================================
# REGULARIZATION
# ============================================================

# Default dropout used by configurable models.
DEFAULT_DROPOUT = 0.3

# Default L2 regularization strength.
DEFAULT_L2 = 1e-4


# ============================================================
# MODEL ARCHITECTURE SETTINGS
# ============================================================

# ------------------------------------------------------------
# 1D CNN
# ------------------------------------------------------------

CNN1D_FILTERS = (
    64,
    128,
    256
)

CNN1D_KERNEL_SIZE = 3

CNN1D_POOL_SIZE = 2


# ------------------------------------------------------------
# 2D CNN
# ------------------------------------------------------------

CNN2D_FILTERS = (
    32,
    64,
    128
)

CNN2D_KERNEL_SIZE = 3

CNN2D_POOL_SIZE = 2


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

LSTM_UNITS = (
    128,
    64
)


# ------------------------------------------------------------
# BiLSTM
# ------------------------------------------------------------

BILSTM_UNITS = 64


# Dense layer used after sequence processing
DENSE_UNITS = 64


# ============================================================
# CLASSIFICATION
# ============================================================

ACTIVATION_HIDDEN = "relu"

ACTIVATION_OUTPUT = "softmax"


# ============================================================
# EVALUATION
# ============================================================

# Main evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]


# Primary comparison metric
PRIMARY_METRIC = "accuracy"


# Macro F1 gives equal importance to all emotion classes.
SECONDARY_METRIC = "macro_f1"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Main architectures used for the comparative study.
#
# These names describe the planned/core architecture groups.
# Individual training scripts may use more specialized variants.

MAIN_MODELS = [
    "CNN1D",
    "CNN2D",
    "LSTM",
    "CNN_BiLSTM",
]


# ------------------------------------------------------------
# Experiment 1:
# Architecture comparison
# ------------------------------------------------------------

RUN_ARCHITECTURE_COMPARISON = True


# ------------------------------------------------------------
# Experiment 2:
# Feature representation comparison
# ------------------------------------------------------------

RUN_FEATURE_COMPARISON = True


# MFCC feature variants
FEATURE_VARIANTS = [
    "MFCC",
    "MFCC_DELTA",
    "MFCC_DELTA_DELTA",
]


# ------------------------------------------------------------
# Experiment 3:
# Data augmentation comparison
# ------------------------------------------------------------

RUN_AUGMENTATION_COMPARISON = True


# ------------------------------------------------------------
# Experiment 4:
# Attention ablation
# ------------------------------------------------------------

RUN_ATTENTION_ABLATION = True


# ============================================================
# RESULTS
# ============================================================

# Save best model based on validation loss
SAVE_BEST_MODEL = True

MODEL_CHECKPOINT_MONITOR = "val_loss"

MODEL_CHECKPOINT_MODE = "min"


# Save training history
SAVE_TRAINING_HISTORY = True

# Save confusion matrices
SAVE_CONFUSION_MATRIX = True

# Save classification reports
SAVE_CLASSIFICATION_REPORT = True

# Save model comparison table
SAVE_MODEL_COMPARISON = True


# ============================================================
# COMPUTATIONAL EFFICIENCY
# ============================================================

# Record training time
RECORD_TRAINING_TIME = True

# Record inference time
RECORD_INFERENCE_TIME = True

# Record parameter count
RECORD_PARAMETER_COUNT = True


# ============================================================
# DATA LEAKAGE SAFETY
# ============================================================

# Actor-level split must be enforced.
ENFORCE_ACTOR_INDEPENDENT_SPLIT = True

# Test set must remain unseen during training/tuning.
KEEP_TEST_UNSEEN = True

# Normalization statistics must come from training data only.
ENFORCE_TRAIN_ONLY_NORMALIZATION = True

# Augmentation must not modify validation/test data.
ENFORCE_TRAIN_ONLY_AUGMENTATION = True


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_TITLE = (
    "Comparative Analysis of Deep Learning Architectures "
    "for Speech Emotion Classification Using the CREMA-D Dataset"
)