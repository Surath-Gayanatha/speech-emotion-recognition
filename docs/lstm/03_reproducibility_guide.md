# LSTM — Reproducibility Guide

How to set up the environment, obtain the data, re-run the three LSTM
experiments, and inspect the committed results **without retraining**.
All commands are run from the repository root on Windows (PowerShell); the
Python code itself is platform-independent.

---

## 1. Environment

The results were produced with:

| Component | Version |
|---|---|
| Python | 3.12.13 |
| TensorFlow | 2.16.2 (CPU; native Windows TensorFlow has no GPU support) |
| Keras | 3.15.1 |
| librosa | 0.11.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.9.1 |

Setup (using [uv](https://docs.astral.sh/uv/); plain `python -m venv` + `pip`
also works with Python 3.9–3.12):

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -c "import tensorflow as tf, keras, librosa, numpy; print(tf.__version__, keras.__version__, librosa.__version__, numpy.__version__)"
```

`.venv/` is gitignored.

## 2. Dataset

CREMA-D from Kaggle (`ejlok1/cremad`, ≈ 451 MB download). Either download the
zip from the Kaggle website, or:

```powershell
python -c "import kagglehub; print(kagglehub.dataset_download('ejlok1/cremad'))"
```

Copy the `AudioWAV` folder so that the WAV files sit directly in:

```
data/raw/AudioWAV/1001_DFA_ANG_XX.wav
data/raw/AudioWAV/... (7,442 files)
```

`data/raw/` and `data/processed/` are gitignored and must never be committed.

Expected dataset check: 7,442 WAV files, 91 actors (1001–1091), 1,271 clips
each for ANG/DIS/FEA/HAP/SAD and 1,087 for NEU, all 16 kHz mono.

The actor split is **committed** in `data/splits/` (63 / 13 / 15 actors).
Do not regenerate it (`src/data/split.py` would overwrite it).

## 3. Commands

| Purpose | Command | Time (CPU) |
|---|---|---|
| Show the architecture only (loads no data) | `python -m src.training.train_lstm --summary_only` | seconds |
| Seed 42 (headline config) | `python -m src.training.train_lstm` | ≈ 40 min |
| Seed 1 | `python -m src.training.train_lstm --seed 1` | ≈ 50 min |
| Seed 7 | `python -m src.training.train_lstm --seed 7` | ≈ 62 min |

- All defaults (`--pooling attention`, 60 epochs, batch 64, lr 1e-3,
  SpecAugment on) are the configuration used for the reported results. Only
  `--seed` changes between the three runs.
- The **first** run extracts features for all 7,442 clips (≈ 7 min) into
  `data/processed/bilstm_logmel64_d_dd_h320_T200.npz`; later runs (and the
  BiLSTM) reuse this cache.
- `--quick` is a 2-epoch setup check on 400 clips. **It still evaluates on the
  test clips at the end**, so its numbers must not be reported.
- Each full run evaluates the test set once, at the end, after early stopping
  has restored the best-validation weights.

Retraining will reproduce the procedure; because of floating-point
nondeterminism (thread scheduling, oneDNN), numbers may differ slightly in the
last decimal places. The committed artifacts are the reference results.

## 4. Where the results are

| Path | Contents | In Git? |
|---|---|---|
| `src/training/train_lstm.py` | LSTM model + training script | yes |
| `src/evaluation/evaluate_lstm.py` | evaluation-only demo (no training) | yes |
| `results/lstm/attention/` | seed 42 run | yes |
| `results/lstm/attention_seed1/` | seed 1 run | yes |
| `results/lstm/attention_seed7/` | seed 7 run | yes |
| `results/lstm/seed_summary.json` | three-seed mean ± std (headline) | yes |
| `results/metrics/lstm_attention*.json` | copies of each run's `metrics.json` | yes |
| `docs/lstm/` | analysis and documentation | yes |
| `models/lstm/best_lstm_<tag>.weights.h5` | best-validation weights per seed | no (gitignored, local) |
| `models/lstm/norm_stats.npz` | training-set mean/std (identical for all seeds) | no (gitignored, local) |
| `results/lstm/<tag>/test_*.npy` | test probabilities, labels, filenames, confusion matrix | no (gitignored, local) |
| `data/processed/*.npz` | shared feature cache | no (gitignored, local) |

Each run folder contains:

| File | What it is |
|---|---|
| `run_config.json` | every setting used (seed, features, class weights, clip counts) |
| `metrics.json` | train / val / test metrics, parameters, epochs, best epoch, timing |
| `history.json`, `training_log.csv` | per-epoch loss, accuracy and learning rate |
| `learning_curves.png` | train vs validation loss and accuracy |
| `test_classification_report.txt` | per-class precision / recall / F1 |
| `test_confusion_matrix.png` | row-normalised confusion matrix with counts |

## 5. Inspecting results without retraining

Nothing below trains a model or touches the audio.

Headline (three-seed mean ± std):

```powershell
python -c "import json; s=json.load(open('results/lstm/seed_summary.json')); [print(k.ljust(18), round(s[k]['mean'],4), '+/-', round(s[k]['std'],4)) for k in ['accuracy','f1_macro','recall_macro_UAR','precision_macro','f1_weighted','roc_auc_macro_ovr','val_accuracy']]"
```

Per-seed test metrics:

```powershell
python -c "import json; [print(t, {k: round(v,4) for k,v in json.load(open(f'results/lstm/{t}/metrics.json'))['metrics']['test'].items()}) for t in ['attention','attention_seed1','attention_seed7']]"
```

Per-class report and configuration of one run:

```powershell
Get-Content results\lstm\attention\test_classification_report.txt
Get-Content results\lstm\attention\run_config.json
```

Open `results/lstm/<tag>/learning_curves.png` and
`test_confusion_matrix.png` in any image viewer (or VS Code).

### Recomputing the three-seed summary

`seed_summary.json` uses the **sample** standard deviation (`ddof=1`), the
same convention as `results/bilstm/seed_summary.json`. It can be checked with:

```python
import json, numpy as np
tags = ["attention", "attention_seed1", "attention_seed7"]
test = [json.load(open(f"results/lstm/{t}/metrics.json"))["metrics"]["test"] for t in tags]
for k in ["accuracy", "f1_macro", "recall_macro_UAR", "precision_macro", "f1_weighted", "roc_auc_macro_ovr"]:
    v = [m[k] for m in test]
    print(f"{k:<18} {np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")
```

Expected output: accuracy 0.5693 ± 0.0012, macro F1 0.5658 ± 0.0023,
UAR 0.5713 ± 0.0007, macro precision 0.5836 ± 0.0053,
weighted F1 0.5637 ± 0.0023, ROC-AUC 0.8614 ± 0.0005.

### Evaluation-only demo (saved checkpoints, no training)

`src/evaluation/evaluate_lstm.py` rebuilds the LSTM, loads
`models/lstm/best_lstm_<tag>.weights.h5` and `models/lstm/norm_stats.npz`
(checked against the training-actor statistics), scores the committed split,
prints metrics, the classification report and the confusion matrix, and
compares them with the committed `metrics.json`. It trains nothing and writes
no files. It needs the local (gitignored) checkpoints, dataset and feature
cache.

```powershell
python -m src.evaluation.evaluate_lstm                                                  # seed 42, test split
python -m src.evaluation.evaluate_lstm --tags attention attention_seed1 attention_seed7  # all seeds + mean/std
python -m src.evaluation.evaluate_lstm --split val                                      # validation split
```

Takes ≈ 1 minute for all three seeds on CPU; each run should report
`max |difference| = 0.000000 -> reproduced`.

### Verifying the runs differ only by seed

```python
import json
base = json.load(open("results/lstm/attention/run_config.json"))
for t in ["attention_seed1", "attention_seed7"]:
    c = json.load(open(f"results/lstm/{t}/run_config.json"))
    print(t, sorted(k for k in base if base[k] != c[k]))   # -> ['seed']
```
