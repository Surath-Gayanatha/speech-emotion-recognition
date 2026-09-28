"""
Evaluation-only demo for the trained LSTM (SE4050, owner: Theekshana).

Loads an existing best-validation checkpoint from models/lstm/ and the saved
training-set normalisation statistics, rebuilds the exact same architecture
(build_lstm), and scores it on the committed actor split. NO training is
performed and NO files are written: results are only printed and compared
with the committed results/lstm/<tag>/metrics.json.

This re-scores models that were already selected on the validation set; it
is a reproduction/demo tool and must not be used to choose models.

Requires locally (all gitignored): the CREMA-D WAVs in data/raw/AudioWAV,
the feature cache in data/processed/ (rebuilt automatically if missing), and
models/lstm/best_lstm_<tag>.weights.h5 + models/lstm/norm_stats.npz produced
by src/training/train_lstm.py.

Usage (run from repo root)
    python -m src.evaluation.evaluate_lstm                           # seed 42, test split
    python -m src.evaluation.evaluate_lstm --tags attention attention_seed1 attention_seed7
    python -m src.evaluation.evaluate_lstm --split val               # validation split instead
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.training.train_bilstm import (
    EMOTION_NAMES,
    find_wavs,
    load_actor_split,
    build_features,
    fit_normaliser,
    to_padded,
    compute_metrics,
)
from src.training.train_lstm import build_lstm

METRICS = ["accuracy", "precision_macro", "recall_macro_UAR", "f1_macro", "f1_weighted", "roc_auc_macro_ovr"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved LSTM checkpoints without training.")
    p.add_argument("--tags", nargs="+", default=["attention"],
                   help="run tags, e.g. attention attention_seed1 attention_seed7")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--cache_dir", default="data/processed")
    p.add_argument("--model_dir", default="models/lstm")
    p.add_argument("--results_dir", default="results/lstm", help="committed results (read only)")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def load_split(args):
    """Features of the requested split, standardised with the saved training statistics."""
    items = find_wavs(args.data_dir)
    feats, labels, actors, _ = build_features(items, args.cache_dir, quick=False)
    train_a, val_a, test_a = load_actor_split(args.splits_dir, set(actors), seed=42)
    idx = {s: np.where(np.isin(actors, list(a)))[0] for s, a in
           [("train", train_a), ("val", val_a), ("test", test_a)]}

    stats_path = Path(args.model_dir) / "norm_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found - run src/training/train_lstm.py first.")
    stats = np.load(stats_path)
    mean, std = stats["mean"], stats["std"]

    # Sanity check: the saved statistics must be the training-actor statistics.
    ref_mean, ref_std = fit_normaliser([feats[i] for i in idx["train"]])
    if not (np.allclose(mean, ref_mean, atol=1e-5) and np.allclose(std, ref_std, atol=1e-5)):
        raise ValueError(f"{stats_path} does not match the training-split statistics.")
    print(f"[norm] loaded {stats_path} (matches training-actor statistics)")

    s = args.split
    X, _ = to_padded([feats[i] for i in idx[s]], mean, std)
    y = labels[idx[s]]
    print(f"[data] {s} split: {len(y)} clips from {len(set(actors[idx[s]]))} actors")
    return X, y


def evaluate_tag(tag, X, y, args):
    cfg_path = Path(args.results_dir) / tag / "run_config.json"
    ckpt = Path(args.model_dir) / f"best_lstm_{tag}.weights.h5"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found - train this run first.")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"pooling": "attention"}

    model = build_lstm(cfg["pooling"], cfg.get("lr", 1e-3))
    model.load_weights(str(ckpt))

    t0 = time.time()
    probs = model.predict(X, batch_size=args.batch_size, verbose=0)
    infer_ms = (time.time() - t0) / len(y) * 1000
    m = compute_metrics(y, probs)

    print(f"\n==== {tag}  (seed {cfg.get('seed', '?')}, {cfg['pooling']} pooling, "
          f"{model.count_params():,} params) - {args.split} ====")
    for k in METRICS:
        print(f"  {k:<18} {m[k]:.4f}")
    print(f"  inference          {infer_ms:.2f} ms/clip")
    print(classification_report(y, probs.argmax(1), labels=range(6),
                                target_names=EMOTION_NAMES, digits=4, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y, probs.argmax(1), labels=range(6)))

    committed = Path(args.results_dir) / tag / "metrics.json"
    if committed.exists():
        ref = json.loads(committed.read_text())["metrics"][args.split]
        diff = max(abs(m[k] - ref[k]) for k in METRICS)
        print(f"[check] vs committed {committed}: max |difference| = {diff:.6f} "
              f"-> {'reproduced' if diff < 1e-3 else 'DIFFERENT'}")
    return m


def main():
    args = parse_args()
    X, y = load_split(args)
    results = {tag: evaluate_tag(tag, X, y, args) for tag in args.tags}

    if len(results) > 1:
        print(f"\n==== {len(results)} runs, {args.split} split: mean +/- sample std (ddof=1) ====")
        for k in METRICS:
            v = [r[k] for r in results.values()]
            print(f"  {k:<18} {np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")
    print("\nNo training was run and no files were written.")


if __name__ == "__main__":
    main()
