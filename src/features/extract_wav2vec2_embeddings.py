"""Fast SoundFile Wav2Vec2 1536D Pre-trained Audio Embedding Extractor.

Uses soundfile for ultra-fast audio loading and Wav2Vec2 for SSL feature extraction.
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    EMOTION_LABELS,
)

MODEL_CHECKPOINT = "superb/wav2vec2-base-superb-er"
TARGET_SR = 16000
MAX_AUDIO_LEN = 48000  # 3 seconds @ 16kHz


class FastAudioDataset(Dataset):
    def __init__(self, wav_paths):
        self.wav_paths = wav_paths

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        path = self.wav_paths[idx]
        emotion_code = path.stem.split("_")[2]
        label = EMOTION_LABELS[emotion_code]

        speech, sr = sf.read(str(path), dtype="float32")
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)

        if len(speech) < MAX_AUDIO_LEN:
            speech = np.pad(speech, (0, MAX_AUDIO_LEN - len(speech)), mode="constant")
        else:
            speech = speech[:MAX_AUDIO_LEN]

        return torch.tensor(speech, dtype=torch.float32), label, path.stem


def extract_embeddings(batch_size: int = 32):
    torch.set_num_threads(4)
    print(f"Loading Pre-trained Wav2Vec2 Model ({MODEL_CHECKPOINT})...")
    model = Wav2Vec2Model.from_pretrained(MODEL_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for feature extraction: {device}")
    model.to(device)
    model.eval()

    wav_files = sorted(Path(DATA_RAW_DIR).glob("*.wav"))
    valid_files = [p for p in wav_files if p.stem.split("_")[2] in EMOTION_LABELS]

    if not valid_files:
        raise FileNotFoundError(f"No valid WAV audio files found in {DATA_RAW_DIR}")

    print(f"Extracting 1536D SSL embeddings for {len(valid_files)} audio clips using batch_size={batch_size}...")

    dataset = FastAudioDataset(valid_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings, all_labels, all_filenames = [], [], []

    with torch.inference_mode():
        for speech_batch, label_batch, filename_batch in tqdm(loader, desc="Batched Wav2Vec2 Feature Extraction"):
            mean = speech_batch.mean(dim=1, keepdim=True)
            std = speech_batch.std(dim=1, keepdim=True) + 1e-7
            input_values = ((speech_batch - mean) / std).to(device)

            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # (B, T, 768)

            mean_pool = torch.mean(hidden_states, dim=1).cpu().numpy()
            max_pool, _ = torch.max(hidden_states, dim=1)
            max_pool = max_pool.cpu().numpy()

            feat_vectors = np.concatenate([mean_pool, max_pool], axis=1)  # (B, 1536)

            all_embeddings.append(feat_vectors)
            all_labels.extend(label_batch.numpy())
            all_filenames.extend(filename_batch)

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_filenames = np.array(all_filenames)

    out_dir = DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "wav2vec2_embeddings.npy", all_embeddings)
    np.save(out_dir / "wav2vec2_labels.npy", all_labels)
    np.save(out_dir / "wav2vec2_filenames.npy", all_filenames)

    print(f"Successfully saved 1536D Wav2Vec2 embeddings matrix of shape {all_embeddings.shape} to {out_dir}")


if __name__ == "__main__":
    extract_embeddings()
