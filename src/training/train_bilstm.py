"""
BiLSTM for Speech Emotion Recognition on CREMA-D (SE4050).

Owner: Surath (BiLSTM model)

What makes this a *pure recurrent* model (distinct from the team's
CNN+BiLSTM+Attention hybrids): there is NO convolutional front-end. The
frame-level acoustic features go straight into stacked bidirectional LSTMs,
so any performance difference vs. the CNN models can be attributed to
recurrent vs. convolutional feature learning.

Pipeline
    wav -> silence trim -> log-mel (64) + delta + delta-delta (192-D / frame)
        -> per-feature standardisation (fit on TRAIN actors only)
        -> pad/truncate to fixed length, padded frames MASKED
        -> BiLSTM(128) -> BiLSTM(64) -> temporal pooling -> Dense -> softmax(6)

Leakage safeguards
    * Uses the team's actor-level split from data/splits/ (same as every model)
    * Normalisation statistics computed from training actors only
    * Hyper-parameter / early-stopping decisions use VALIDATION only
    * Test set is evaluated exactly once, at the end, with the best weights

Usage (run from repo root)
    python src/training/train_bilstm.py --quick           # 2-minute smoke test
    python src/training/train_bilstm.py                    # full run (attention pooling)
    python src/training/train_bilstm.py --pooling last     # ablation: last hidden state
    python src/training/train_bilstm.py --rnn_dropout 0.3 --weight_decay 1e-4   # stronger regularisation
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}

SR = 16000
N_MELS = 64
N_FFT = 640          # 40 ms window
HOP = 320            # 20 ms hop  -> 50 frames / second
MAX_FRAMES = 200     # 4.0 s after silence trimming (covers almost all clips)
TRIM_DB = 30
FEAT_DIM = N_MELS * 3

# Regularisation settings. These defaults reproduce the original runs; main()
# overwrites them from the command line before building the model and dataset.
TIME_MASK = 20       # SpecAugment: max frames masked
FREQ_MASK = 8        # SpecAugment: max mel bands masked (same bands in deltas)
RNN_DROPOUT = 0.2    # input dropout inside both LSTM layers
WEIGHT_DECAY = 0.0   # > 0 switches the optimiser from Adam to AdamW
REG_DEFAULTS = {"rnn_dropout": RNN_DROPOUT, "weight_decay": WEIGHT_DECAY,
                "time_mask": TIME_MASK, "freq_mask": FREQ_MASK}
REG_TAGS = {"rnn_dropout": "rd", "weight_decay": "wd", "time_mask": "tm", "freq_mask": "fm"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--cache_dir", default="data/processed")
    p.add_argument("--out_dir", default="results/bilstm")
    p.add_argument("--model_dir", default="models/bilstm")
    p.add_argument("--pooling", choices=["attention", "mean", "last"], default="attention")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny subset, 2 epochs - checks setup")
    p.add_argument("--rnn_dropout", type=float, default=RNN_DROPOUT, help="dropout inside both LSTM layers")
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="> 0 uses AdamW with this decay")
    p.add_argument("--time_mask", type=int, default=TIME_MASK, help="SpecAugment: max frames masked")
    p.add_argument("--freq_mask", type=int, default=FREQ_MASK, help="SpecAugment: max mel bands masked")
    return p.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# ----------------------------------------------------------------------------
# Data: file discovery + actor split
# ----------------------------------------------------------------------------
def find_wavs(data_dir):
    files = sorted(Path(data_dir).rglob("*.wav"))
    items = []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[2] in LABEL_MAP:
            items.append((f, parts[0], LABEL_MAP[parts[2]]))
    if not items:
        raise FileNotFoundError(
            f"No CREMA-D .wav files found under '{data_dir}'. "
            "Download the dataset first (see data/README.md)."
        )
    return items


def read_ids(path):
    """Accepts actor IDs ('1001') or clip names ('1001_DFA_ANG_XX.wav') per line."""
    ids = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip().replace(",", " ").split()[0] if line.strip() else ""
        if line:
            ids.add(Path(line).stem.split("_")[0])
    return ids


def load_actor_split(splits_dir, all_actors, seed):
    """Use the TEAM'S split so every model is evaluated on the same actors."""
    sd = Path(splits_dir)
    tr, va, te = sd / "train_actors.txt", sd / "val_actors.txt", sd / "test_actors.txt"

    if tr.exists() and va.exists():
        train, val = read_ids(tr), read_ids(va)
        test = read_ids(te) if te.exists() else set(all_actors) - train - val
        print(f"[split] Using team split from {sd}" + ("" if te.exists() else
              "  (test = all remaining actors; no test_actors.txt found)"))
    else:
        print("[split] WARNING: no split files found - generating a 70/15/15 actor split "
              f"with seed={seed}. Make sure the rest of the team uses the SAME split!")
        actors = sorted(all_actors)
        random.Random(seed).shuffle(actors)
        n = len(actors)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        train, val, test = set(actors[:n_tr]), set(actors[n_tr:n_tr + n_va]), set(actors[n_tr + n_va:])
        sd.mkdir(parents=True, exist_ok=True)
        for name, s in [("train", train), ("val", val), ("test", test)]:
            (sd / f"{name}_actors.txt").write_text("\n".join(sorted(s)))

    assert not (train & val) and not (train & test) and not (val & test), \
        "Actor overlap between splits -> data leakage!"
    return train, val, test


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def extract_one(path):
    """Returns (T, 192) float32 array, or None for broken / empty clips."""
    try:
        y, _ = librosa.load(path, sr=SR)
        if y.size < SR * 0.2:
            return None
        y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
        if y.size < SR * 0.2:
            return None
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                             n_mels=N_MELS, fmin=20, fmax=SR // 2)
        logmel = librosa.power_to_db(mel, ref=np.max)
        width = min(9, logmel.shape[1] if logmel.shape[1] % 2 == 1 else logmel.shape[1] - 1)
        if width < 3:
            return None
        d1 = librosa.feature.delta(logmel, width=width, order=1)
        d2 = librosa.feature.delta(logmel, width=width, order=2)
        return np.vstack([logmel, d1, d2]).T.astype(np.float32)[:MAX_FRAMES]
    except Exception:
        return None


def build_features(items, cache_dir, quick):
    cache = Path(cache_dir) / f"bilstm_logmel{N_MELS}_d_dd_h{HOP}_T{MAX_FRAMES}{'_quick' if quick else ''}.npz"
    if cache.exists():
        print(f"[features] Loading cache {cache}")
        z = np.load(cache, allow_pickle=True)
        return list(z["feats"]), z["labels"], z["actors"], z["names"]

    print(f"[features] Extracting from {len(items)} files (first run only, then cached)...")
    feats = Parallel(n_jobs=-1)(delayed(extract_one)(p) for p, _, _ in tqdm(items))
    keep = [i for i, f in enumerate(feats) if f is not None]
    dropped = len(items) - len(keep)
    if dropped:
        print(f"[features] Skipped {dropped} unreadable/empty/too-short clips (report as a data-quality issue)")
    feats = [feats[i] for i in keep]
    labels = np.array([items[i][2] for i in keep])
    actors = np.array([items[i][1] for i in keep])
    names = np.array([items[i][0].stem for i in keep])
    cache.parent.mkdir(parents=True, exist_ok=True)
    obj = np.empty(len(feats), dtype=object)
    obj[:] = feats
    np.savez(cache, feats=obj, labels=labels, actors=actors, names=names)
    return feats, labels, actors, names


def fit_normaliser(train_feats):
    stacked = np.concatenate(train_feats, axis=0)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def to_padded(feats, mean, std):
    """Standardise, then zero-pad. Zero rows are masked by the model."""
    X = np.zeros((len(feats), MAX_FRAMES, FEAT_DIM), dtype=np.float32)
    lengths = np.zeros(len(feats), dtype=np.int32)
    for i, f in enumerate(feats):
        f = (f - mean) / std
        X[i, :len(f)] = f
        lengths[i] = len(f)
    return X, lengths


# ----------------------------------------------------------------------------
# SpecAugment (training data only)
# ----------------------------------------------------------------------------
def spec_augment(x, y):
    f = tf.random.uniform([], 0, FREQ_MASK + 1, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, N_MELS - f + 1, dtype=tf.int32)
    band = tf.logical_and(tf.range(N_MELS) >= f0, tf.range(N_MELS) < f0 + f)
    band = tf.tile(band, [3])                       # same bands in log-mel, delta, delta-delta
    x = tf.where(band[None, :], tf.zeros_like(x), x)

    t = tf.random.uniform([], 0, TIME_MASK + 1, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, MAX_FRAMES - t + 1, dtype=tf.int32)
    rng = tf.range(MAX_FRAMES)
    tmask = tf.logical_and(rng >= t0, rng < t0 + t)
    x = tf.where(tmask[:, None], tf.zeros_like(x), x)
    return x, y


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RightPaddingMask(layers.Layer):
    """Builds a mask that is True up to the LAST non-zero frame and False after.

    Why not keras.layers.Masking? SpecAugment zeroes whole frames in the middle
    of a clip. A plain Masking layer would mark those as padding, giving masks
    with 'holes' - which cuDNN's fast LSTM kernel rejects, and which would also
    silently skip frames instead of showing the model a masked (mean-valued)
    frame as SpecAugment intends. Only the genuine right-padding is masked."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        ops = keras.ops
        nonzero = ops.cast(ops.any(ops.not_equal(inputs, 0.0), axis=-1), "int32")
        from_end = ops.flip(ops.cumsum(ops.flip(nonzero, axis=1), axis=1), axis=1)
        return ops.greater(from_end, 0)

    def call(self, inputs):
        return inputs


class MaskedAttentionPooling(layers.Layer):
    """Additive attention over time; padded frames get zero weight.
    Returns the weighted sum of BiLSTM outputs (a learned 'which frames
    carry the emotion' summary instead of just the final hidden state)."""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(d, self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.units, 1), initializer="glorot_uniform")

    def call(self, h, mask=None):
        score = tf.squeeze(tf.matmul(tf.tanh(tf.matmul(h, self.W) + self.b), self.v), -1)  # (B, T)
        if mask is not None:
            score = tf.where(mask, score, tf.fill(tf.shape(score), -1e9))
        alpha = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(h * alpha[..., None], axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def build_bilstm(pooling="attention", lr=1e-3):
    inp = layers.Input(shape=(MAX_FRAMES, FEAT_DIM), name="logmel_deltas")
    x = RightPaddingMask(name="right_padding_mask")(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=RNN_DROPOUT), name="bilstm_1")(x)
    last = pooling == "last"
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=not last, dropout=RNN_DROPOUT), name="bilstm_2")(x)
    if pooling == "attention":
        x = MaskedAttentionPooling(64, name="attention_pool")(x)
    elif pooling == "mean":
        x = layers.GlobalAveragePooling1D(name="mean_pool")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(EMOTIONS), activation="softmax", name="emotion")(x)

    model = keras.Model(inp, out, name=f"bilstm_{pooling}")
    if WEIGHT_DECAY > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def compute_metrics(y_true, probs):
    y_pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(np.eye(len(EMOTIONS))[y_true], probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro_UAR": float(r),
        "f1_macro": float(f),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_macro_ovr": float(auc),
    }


def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, a in [("loss", ax[0]), ("accuracy", ax[1])]:
        a.plot(hist[k], label="train")
        a.plot(hist[f"val_{k}"], label="validation")
        a.set_title(f"BiLSTM {k}")
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + "\n(colour = row-normalised recall, numbers = counts)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global RNN_DROPOUT, WEIGHT_DECAY, TIME_MASK, FREQ_MASK
    args = parse_args()
    set_seeds(args.seed)
    RNN_DROPOUT, WEIGHT_DECAY = args.rnn_dropout, args.weight_decay
    TIME_MASK, FREQ_MASK = args.time_mask, args.freq_mask
    regularisation = {k: getattr(args, k) for k in REG_DEFAULTS}
    changed = [f"{REG_TAGS[k]}{v:g}" for k, v in regularisation.items() if v != REG_DEFAULTS[k]]
    reg_tag = ("_reg-" + "-".join(changed)) if changed else ""
    tag = (args.pooling + ("" if args.seed == 42 else f"_seed{args.seed}") + reg_tag
           + ("_quick" if args.quick else ""))
    out_dir = Path(args.out_dir) / tag
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    items = find_wavs(args.data_dir)
    if args.quick:
        random.Random(args.seed).shuffle(items)
        items = items[:400]
        args.epochs = 2
    print(f"[data] {len(items)} clips, {len({a for _, a, _ in items})} actors")

    feats, labels, actors, names = build_features(items, args.cache_dir, args.quick)
    train_a, val_a, test_a = load_actor_split(args.splits_dir, set(actors), args.seed)

    idx = {s: np.where(np.isin(actors, list(a)))[0] for s, a in
           [("train", train_a), ("val", val_a), ("test", test_a)]}
    for s in idx:
        if len(idx[s]) == 0:
            raise ValueError(f"The '{s}' split matched 0 clips - check the IDs in {args.splits_dir}")
    for s in idx:
        print(f"[split] {s:5s}: {len(idx[s]):5d} clips from "
              f"{len(set(actors[idx[s]])):2d} actors | class counts {np.bincount(labels[idx[s]], minlength=6).tolist()}")

    mean, std = fit_normaliser([feats[i] for i in idx["train"]])
    np.savez(model_dir / "norm_stats.npz", mean=mean, std=std)

    X, lens = {}, {}
    y = {s: labels[idx[s]] for s in idx}
    for s in idx:
        X[s], lens[s] = to_padded([feats[i] for i in idx[s]], mean, std)
    truncated = int(sum(len(f) >= MAX_FRAMES for f in feats))
    print(f"[features] frame dim={FEAT_DIM}, max_frames={MAX_FRAMES}, "
          f"median length={int(np.median(np.concatenate(list(lens.values()))))} frames, "
          f"{truncated} clips truncated")

    onehot = lambda v: np.eye(len(EMOTIONS), dtype=np.float32)[v]
    train_ds = tf.data.Dataset.from_tensor_slices((X["train"], onehot(y["train"])))
    train_ds = train_ds.shuffle(len(y["train"]), seed=args.seed, reshuffle_each_iteration=True)
    if not args.no_augment:
        train_ds = train_ds.map(spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X["val"], onehot(y["val"]))).batch(args.batch_size)

    cw = compute_class_weight("balanced", classes=np.arange(6), y=y["train"])
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model = build_bilstm(args.pooling, args.lr)
    model.summary()
    ckpt = model_dir / f"best_bilstm_{tag}.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy", mode="max",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     class_weight=class_weight, callbacks=callbacks, verbose=2)
    train_time = time.time() - t0
    model.load_weights(str(ckpt))
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    plot_curves(history, out_dir / "learning_curves.png")

    # ---- evaluation (test set touched exactly once, here) ----
    results = {}
    for s in ["train", "val", "test"]:
        t1 = time.time()
        probs = model.predict(X[s], batch_size=args.batch_size, verbose=0)
        infer_ms = (time.time() - t1) / len(y[s]) * 1000
        results[s] = compute_metrics(y[s], probs)
        if s == "test":
            results["test"]["inference_ms_per_clip"] = round(infer_ms, 3)
            cm = confusion_matrix(y[s], probs.argmax(1), labels=range(6))
            np.save(out_dir / "test_confusion_matrix.npy", cm)
            np.save(out_dir / "test_probabilities.npy", probs)
            np.save(out_dir / "test_labels.npy", y[s])
            np.save(out_dir / "test_filenames.npy", names[idx[s]])
            plot_confusion(cm, out_dir / "test_confusion_matrix.png", f"BiLSTM ({args.pooling}) - test set")
            report = classification_report(y[s], probs.argmax(1), labels=range(6),
                                           target_names=EMOTION_NAMES, digits=4, zero_division=0)
            (out_dir / "test_classification_report.txt").write_text(report)
            print("\n" + report)

    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    summary = {
        "model": f"BiLSTM ({args.pooling} pooling)",
        "features": f"log-mel {N_MELS} + delta + delta-delta ({FEAT_DIM}-D), trimmed, train-only standardisation",
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 1),
        "sec_per_epoch": round(train_time / len(history["loss"]), 1),
        "generalisation_gap_train_minus_test_acc": round(results["train"]["accuracy"] - results["test"]["accuracy"], 4),
        "regularisation": regularisation,
        "metrics": results,
    }
    config = {**vars(args), "sr": SR, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP,
              "max_frames": MAX_FRAMES, "trim_db": TRIM_DB, "time_mask": TIME_MASK,
              "freq_mask": FREQ_MASK, "label_smoothing": 0.1, "class_weight": class_weight,
              "n_clips": {s: int(len(y[s])) for s in y},
              "n_actors": {"train": len(train_a), "val": len(val_a), "test": len(test_a)}}

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    if not args.quick:
        mdir = Path("results/metrics")
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / f"bilstm_{tag}.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    for s in ["train", "val", "test"]:
        m = results[s]
        print(f"{s:5s}  acc={m['accuracy']:.4f}  macroF1={m['f1_macro']:.4f}  "
              f"UAR={m['recall_macro_UAR']:.4f}  AUC={m['roc_auc_macro_ovr']:.4f}")
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
