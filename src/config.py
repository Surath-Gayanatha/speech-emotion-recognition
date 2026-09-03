"""Shared constants and paths for the Speech Emotion Recognition pipeline.

Keeping these in one place means every model (MLP, 1D CNN, LSTM, BiLSTM)
extracts features and splits data identically -> fair comparison.
"""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw" / "AudioWAV"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = ROOT_DIR / "data" / "splits"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# --- Audio / feature extraction ---
SAMPLE_RATE = 16000          # Hz
N_MFCC = 40                  # number of MFCC coefficients
USE_DELTA = True             # include delta MFCC
USE_DELTA_DELTA = True       # include delta-delta MFCC
N_FFT = 2048
HOP_LENGTH = 512
MAX_PAD_LEN = 174            # frames, pad/truncate MFCC sequences to this length

# --- Labels ---
# CREMA-D emotion codes -> class index
EMOTION_LABELS = {
    "ANG": 0,  # Anger
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happy
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sad
}
NUM_CLASSES = len(EMOTION_LABELS)

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Split ratios (applied at ACTOR level, not clip level) ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
