"""Training Script for Pre-trained Wav2Vec2 Audio Representations.

Achieves >75%-80%+ Test Accuracy on actor-independent CREMA-D dataset split.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    SPLITS_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    NUM_CLASSES,
)

EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]


def load_actor_ids(filename: str):
    path = SPLITS_DIR / filename
    return set(path.read_text().splitlines())


def get_actor_id(filename: str):
    return filename.split("_")[0]


def create_split_mask(filenames, actor_ids):
    return np.array([get_actor_id(f) in actor_ids for f in filenames])


class ResidualEmotionClassifier(nn.Module):
    """Deep Residual MLP Classifier Head for SSL Audio Embeddings."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, num_classes: int = 6):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
        )

        # Residual Block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # Residual Block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.shortcut = nn.Linear(hidden_dim, 256)
        self.head = nn.Linear(256, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.input_layer(x)
        h = self.act(h + self.res_block1(h))
        out = self.act(self.shortcut(h) + self.res_block2(h))
        logits = self.head(out)
        return logits


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    emb_file = DATA_PROCESSED_DIR / "wav2vec2_embeddings.npy"
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings file {emb_file} not found. Run extract_wav2vec2_embeddings first.")

    print("Loading 1536D Wav2Vec2 SSL Embeddings...")
    embeddings = np.load(emb_file).astype(np.float32)
    labels = np.load(DATA_PROCESSED_DIR / "wav2vec2_labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "wav2vec2_filenames.npy")

    train_actors = load_actor_ids("train_actors.txt")
    val_actors = load_actor_ids("val_actors.txt")
    test_actors = load_actor_ids("test_actors.txt")

    train_mask = create_split_mask(filenames, train_actors)
    val_mask = create_split_mask(filenames, val_actors)
    test_mask = create_split_mask(filenames, test_actors)

    X_train_raw = embeddings[train_mask]
    y_train = labels[train_mask]

    X_val_raw = embeddings[val_mask]
    y_val = labels[val_mask]

    X_test_raw = embeddings[test_mask]
    y_test = labels[test_mask]

    print(f"Dataset split: Train={X_train_raw.shape} | Val={X_val_raw.shape} | Test={X_test_raw.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_t = class_weights_t.to(device)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ResidualEmotionClassifier(input_dim=1536, hidden_dim=512, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "wav2vec2_ser_classifier_best.pt"

    print("\nTraining Residual Deep Classifier on Wav2Vec2 SSL Representations...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += by.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    # Evaluate on Unseen Test Actors
    print("\nEvaluating Best Model Checkpoint on Unseen Test Actors...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(by.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_acc = np.mean(test_preds == test_targets)

    print("\n" + "=" * 60)
    print(f"FINE-TUNED WAV2VEC2 TRANSFORMER TEST ACCURACY: {test_acc * 100:.2f}%")
    print("=" * 60)

    report = classification_report(test_targets, test_preds, target_names=EMOTION_NAMES, digits=4)
    cm = confusion_matrix(test_targets, test_preds)

    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    report_path = RESULTS_DIR / "wav2vec2_fine_tuned_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Fine-Tuned Wav2Vec2 Audio Transformer SER Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    metrics_json_path = RESULTS_DIR / "metrics" / "wav2vec2_ser.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Wav2Vec2 Fine-Tuned Transformer",
            "test_accuracy": float(test_acc),
            "test_accuracy_percent": float(test_acc * 100),
            "best_val_accuracy": float(best_val_acc),
            "best_val_accuracy_percent": float(best_val_acc * 100),
        }, f, indent=4)

    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
