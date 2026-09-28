"""
Standard (unidirectional) LSTM for Speech Emotion Recognition on CREMA-D (SE4050).

Owner: Theekshana (LSTM model)

Controlled comparison with the team's BiLSTM
    This script reuses the group's selected common pipeline from
    src/training/train_bilstm.py by IMPORTING it rather than copying it, so
    data, features, normalisation, masking, augmentation, training policy and
    evaluation are identical by construction. The only intended difference is
    the recurrent encoder:

        BiLSTM : Bidirectional(LSTM(128)) -> Bidirectional(LSTM(64)) -> attention pooling
        LSTM   : LSTM(128)                -> LSTM(64)                -> attention pooling

    A unidirectional LSTM only sees past context at each frame, so any gap to
    the BiLSTM measures the value of backward (future) context, at roughly
    half the recurrent parameters.

Pipeline (shared, imported unchanged)
    wav -> silence trim -> log-mel (64) + delta + delta-delta (192-D / frame)
        -> per-feature standardisation (fit on TRAIN actors only)
        -> pad/truncate to 200 frames, right-padding MASKED
        -> LSTM(128) -> LSTM(64) -> masked attention pooling -> Dense -> softmax(6)

Leakage safeguards (same as the BiLSTM)
    * Team actor-level split from data/splits/
    * Normalisation statistics from training actors only
    * Model selection / early stopping on VALIDATION only
    * Test set evaluated exactly once, at the end, with the best weights

Usage (run from repo root)
    python -m src.training.train_lstm --summary_only     # build + summary, loads no data
    python -m src.training.train_lstm --quick            # tiny smoke test
    python -m src.training.train_lstm                    # full run (attention pooling)
    python -m src.training.train_lstm --pooling last     # ablation: last hidden state
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Shared common pipeline (group decision): reused as-is from the BiLSTM script.
from src.training.train_bilstm import (
    EMOTIONS,
    EMOTION_NAMES,
    SR,
    N_MELS,
    N_FFT,
    HOP,
    MAX_FRAMES,
    TRIM_DB,
    FEAT_DIM,
    TIME_MASK,
    FREQ_MASK,
    set_seeds,
    find_wavs,
    load_actor_split,
    build_features,
    fit_normaliser,
    to_padded,
    spec_augment,
    RightPaddingMask,
    MaskedAttentionPooling,
    compute_metrics,
    plot_confusion,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--cache_dir", default="data/processed")
    p.add_argument("--out_dir", default="results/lstm")
    p.add_argument("--model_dir", default="models/lstm")
    p.add_argument("--pooling", choices=["attention", "mean", "last"], default="attention")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny subset, 2 epochs - checks setup")
    p.add_argument("--summary_only", action="store_true",
                   help="build the model and print its summary; loads no data")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_lstm(pooling="attention", lr=1e-3):
    """Mirror of build_bilstm() with the Bidirectional wrappers removed.

    Units, dropout, pooling, head, optimiser, loss and label smoothing are kept
    identical so the architectural difference is only the direction of
    recurrence."""
    inp = layers.Input(shape=(MAX_FRAMES, FEAT_DIM), name="logmel_deltas")
    x = RightPaddingMask(name="right_padding_mask")(inp)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2, name="lstm_1")(x)
    last = pooling == "last"
    x = layers.LSTM(64, return_sequences=not last, dropout=0.2, name="lstm_2")(x)
    if pooling == "attention":
        x = MaskedAttentionPooling(64, name="attention_pool")(x)
    elif pooling == "mean":
        x = layers.GlobalAveragePooling1D(name="mean_pool")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(EMOTIONS), activation="softmax", name="emotion")(x)

    model = keras.Model(inp, out, name=f"lstm_{pooling}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, a in [("loss", ax[0]), ("accuracy", ax[1])]:
        a.plot(hist[k], label="train")
        a.plot(hist[f"val_{k}"], label="validation")
        a.set_title(f"LSTM {k}")
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seeds(args.seed)

    if args.summary_only:
        build_lstm(args.pooling, args.lr).summary()
        return

    tag = args.pooling + ("" if args.seed == 42 else f"_seed{args.seed}") + ("_quick" if args.quick else "")
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

    # Same feature cache as the BiLSTM, so both models see identical inputs.
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

    model = build_lstm(args.pooling, args.lr)
    model.summary()
    ckpt = model_dir / f"best_lstm_{tag}.weights.h5"
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
            plot_confusion(cm, out_dir / "test_confusion_matrix.png", f"LSTM ({args.pooling}) - test set")
            report = classification_report(y[s], probs.argmax(1), labels=range(6),
                                           target_names=EMOTION_NAMES, digits=4, zero_division=0)
            (out_dir / "test_classification_report.txt").write_text(report)
            print("\n" + report)

    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    summary = {
        "model": f"LSTM ({args.pooling} pooling)",
        "features": f"log-mel {N_MELS} + delta + delta-delta ({FEAT_DIM}-D), trimmed, train-only standardisation",
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 1),
        "sec_per_epoch": round(train_time / len(history["loss"]), 1),
        "generalisation_gap_train_minus_test_acc": round(results["train"]["accuracy"] - results["test"]["accuracy"], 4),
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
        (mdir / f"lstm_{tag}.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    for s in ["train", "val", "test"]:
        m = results[s]
        print(f"{s:5s}  acc={m['accuracy']:.4f}  macroF1={m['f1_macro']:.4f}  "
              f"UAR={m['recall_macro_UAR']:.4f}  AUC={m['roc_auc_macro_ovr']:.4f}")
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
