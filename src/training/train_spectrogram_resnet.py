"""Fine-tuning Spectrogram ResNet for High-Accuracy Speech Emotion Recognition.

Applies Transfer Learning on 3-channel Spectrogram Images with SpecAugment, Mixup,
and Cosine Annealing.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
from src.models.spectrogram_resnet import SpectrogramResNet

EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]


def load_actor_ids(filename: str):
    path = SPLITS_DIR / filename
    return set(path.read_text().splitlines())


def get_actor_id(filename: str):
    return filename.split("_")[0]


def create_split_mask(filenames, actor_ids):
    return np.array([get_actor_id(f) in actor_ids for f in filenames])


class SpectrogramDataset(Dataset):
    def __init__(self, specs, labels, is_train=False):
        self.specs = specs
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx].copy()  # (3, 64, 174)
        label = self.labels[idx]

        # SpecAugment on Spectrogram Image
        if self.is_train:
            C, H, W = spec.shape
            if np.random.rand() < 0.5:
                f_len = np.random.randint(1, 10)
                f_start = np.random.randint(0, H - f_len)
                spec[:, f_start:f_start+f_len, :] = 0.0

            if np.random.rand() < 0.5:
                t_len = np.random.randint(1, 16)
                t_start = np.random.randint(0, W - t_len)
                spec[:, :, t_start:t_start+t_len] = 0.0

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    spec_file = DATA_PROCESSED_DIR / "spectrogram_images.npy"
    if not spec_file.exists():
        raise FileNotFoundError(f"{spec_file} not found. Run extract_spectrogram_images first.")

    print("Loading 3-Channel Spectrogram Dataset...", flush=True)
    specs = np.load(spec_file)
    labels = np.load(DATA_PROCESSED_DIR / "spectrogram_labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "spectrogram_filenames.npy")

    # Normalize per-channel standard scaling
    for c in range(3):
        mean = np.mean(specs[:, c, :, :])
        std = np.std(specs[:, c, :, :]) + 1e-7
        specs[:, c, :, :] = (specs[:, c, :, :] - mean) / std

    train_actors = load_actor_ids("train_actors.txt")
    val_actors = load_actor_ids("val_actors.txt")
    test_actors = load_actor_ids("test_actors.txt")

    train_mask = create_split_mask(filenames, train_actors)
    val_mask = create_split_mask(filenames, val_actors)
    test_mask = create_split_mask(filenames, test_actors)

    X_train, y_train = specs[train_mask], labels[train_mask]
    X_val, y_val = specs[val_mask], labels[val_mask]
    X_test, y_test = specs[test_mask], labels[test_mask]

    print(f"Dataset split: Train={X_train.shape} | Val={X_val.shape} | Test={X_test.shape}", flush=True)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_t = class_weights_t.to(device)

    train_ds = SpectrogramDataset(X_train, y_train, is_train=True)
    val_ds = SpectrogramDataset(X_val, y_val, is_train=False)
    test_ds = SpectrogramDataset(X_test, y_test, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SpectrogramResNet(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "spectrogram_resnet_best.pt"

    print("\nStarting Fast End-to-End Fine-Tuning of Spectrogram ResNet-34...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += by.size(0)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 3 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best Val: {best_val_acc*100:.2f}%)", flush=True)

    # Evaluate on Unseen Test Actors
    print("\nEvaluating Best ResNet Model Checkpoint on Unseen Test Actors...", flush=True)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            outputs = model(bx)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(by.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_acc = np.mean(test_preds == test_targets)

    print("\n" + "=" * 60, flush=True)
    print(f"FINE-TUNED SPECTROGRAM RESNET TEST ACCURACY: {test_acc * 100:.2f}%", flush=True)
    print("=" * 60, flush=True)

    report = classification_report(test_targets, test_preds, target_names=EMOTION_NAMES, digits=4)
    cm = confusion_matrix(test_targets, test_preds)

    print("\nClassification Report:\n", report, flush=True)
    print("Confusion Matrix:\n", cm, flush=True)

    report_path = RESULTS_DIR / "spectrogram_resnet_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Fine-Tuned Spectrogram ResNet-34 SER Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    metrics_json_path = RESULTS_DIR / "metrics" / "spectrogram_resnet.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Spectrogram ResNet-34 (Transfer Learning)",
            "test_accuracy": float(test_acc),
            "best_val_accuracy": float(best_val_acc)
        }, f, indent=4)

    print(f"\nSaved report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
