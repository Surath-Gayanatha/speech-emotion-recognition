import os
import numpy as np
import librosa


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/mft_processed"

SAMPLE_RATE = 16000

N_MFCC = 40
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MAX_TIME_STEPS = 256


EMOTION_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND AUDIO FILES
# ============================================================

audio_files = []

for root, dirs, files in os.walk(RAW_DIR):

    for file in files:

        if file.lower().endswith(".wav"):

            audio_files.append(
                os.path.join(root, file)
            )


audio_files.sort()

print("Found audio files:", len(audio_files))


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )


    # --------------------------------------------------------
    # Log-Mel Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # RMS Energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # Match Time Dimensions
    # --------------------------------------------------------

    features = [
        mfcc,
        mfcc_delta,
        mfcc_delta2,
        log_mel,
        chroma,
        rms,
        zcr,
        centroid,
        bandwidth,
        rolloff
    ]

    min_frames = min(
        feature.shape[1]
        for feature in features
    )

    features = [
        feature[:, :min_frames]
        for feature in features
    ]


    # --------------------------------------------------------
    # Concatenate Features
    # --------------------------------------------------------

    combined = np.concatenate(
        features,
        axis=0
    )

    # Shape:
    # (feature_dimensions, time)


    # --------------------------------------------------------
    # Pad / Truncate Time
    # --------------------------------------------------------

    if combined.shape[1] < MAX_TIME_STEPS:

        padding = np.zeros(
            (
                combined.shape[0],
                MAX_TIME_STEPS - combined.shape[1]
            ),
            dtype=np.float32
        )

        combined = np.concatenate(
            [
                combined,
                padding
            ],
            axis=1
        )

    else:

        combined = combined[
            :,
            :MAX_TIME_STEPS
        ]


    # --------------------------------------------------------
    # Transpose
    # --------------------------------------------------------

    combined = combined.T

    return combined.astype(
        np.float32
    )


# ============================================================
# PROCESS DATASET
# ============================================================

all_features = []
all_labels = []
all_actors = []

skipped = 0


for index, audio_path in enumerate(audio_files):

    try:

        filename = os.path.basename(
            audio_path
        )

        parts = filename.split("_")

        if len(parts) < 3:
            skipped += 1
            continue


        actor_id = int(
            parts[0]
        )

        emotion_code = parts[2]

        if emotion_code not in EMOTION_MAP:
            skipped += 1
            continue


        feature = extract_features(
            audio_path
        )

        all_features.append(
            feature
        )

        all_labels.append(
            EMOTION_MAP[emotion_code]
        )

        all_actors.append(
            actor_id
        )


        if (index + 1) % 500 == 0:

            print(
                f"Processed {index + 1} / {len(audio_files)}"
            )


    except Exception as e:

        print(
            "Error:",
            audio_path,
            e
        )

        skipped += 1


# ============================================================
# SAVE
# ============================================================

features = np.stack(
    all_features
)

labels = np.array(
    all_labels,
    dtype=np.int64
)

actors = np.array(
    all_actors,
    dtype=np.int64
)


print("\nEXTRACTION COMPLETE")

print(
    "Features shape:",
    features.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Actors shape:",
    actors.shape
)

print(
    "Skipped:",
    skipped
)


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


print(
    "\nSaved to:",
    OUTPUT_DIR
)