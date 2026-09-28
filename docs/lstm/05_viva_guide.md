# LSTM — Viva and Evaluation Guide

A concise revision sheet for the LSTM part of the SE4050 viva. Details are in
`01_performance_analysis.md`, `02_architecture_and_methodology.md`,
`03_reproducibility_guide.md` and `04_lstm_vs_bilstm.md`.

## Headline (test set, 3 seeds, mean ± sample std)

| Metric | Result |
|---|---|
| **Accuracy** | **56.93% ± 0.12%** |
| **Macro F1** | **56.58% ± 0.23%** |
| **UAR (macro recall)** | **57.13% ± 0.07%** |
| **ROC-AUC (macro, one-vs-rest)** | **86.14% ± 0.05%** |
| **Parameters** | **222,534** |

Test set = 1,229 clips from 15 actors never seen in training or validation.
Chance for 6 classes ≈ 17%.

---

## 1. Architecture (one sentence per block)

`Input (200 × 192) → RightPaddingMask → LSTM(128) → LSTM(64) → MaskedAttentionPooling → Dropout 0.4 → Dense(64, ReLU) → Dropout 0.3 → Dense(6, softmax)`

- **Input:** up to 200 frames (4 s), each a 192-number vector.
- **LSTM(128) → LSTM(64):** read the frames forwards in time and keep a
  memory of what came before; both return one vector per frame.
- **Attention pooling:** learns which frames matter and averages them into one
  64-number summary.
- **Head:** a small dense layer with dropout, then 6 outputs.
- **Parameters:** 164,352 + 49,408 (LSTMs) + 4,224 (attention) + 4,160 +
  390 (dense) = 222,534.

## 2. Preprocessing

- 16 kHz mono → trim silence (30 dB) → 64-band log-mel spectrogram
  (40 ms window, 20 ms hop) → add delta and delta-delta → 192 features/frame.
- **Log-mel:** how energy is spread over frequencies, on a perceptual (mel)
  and decibel scale.
- **Delta / delta-delta:** how that spectrum changes over time (speed and
  acceleration), capturing rising/falling pitch and energy.
- **Standardisation:** each feature scaled with the **training actors'**
  mean and std only (no information from validation/test speakers).
- **Actor-independent split:** 63 / 13 / 15 actors, so the model is tested on
  new voices, not on memorised speakers.

## 3. Masking

- Clips have different lengths, so each is **zero-padded at the end** to 200
  frames.
- `RightPaddingMask` marks every frame up to the last real one as valid; the
  LSTMs skip padded steps and attention gives them weight 0.
- It masks only **trailing** padding because SpecAugment also zeroes frames
  *inside* clips; those must stay visible (and holes in a mask break cuDNN).

## 4. Attention

- Score each frame: *s_t = vᵀ tanh(W h_t + b)*; padded frames get −10⁹.
- Softmax over time → weights α_t ≥ 0 that sum to 1.
- Output Σ α_t h_t: a weighted average focusing on emotionally informative
  frames instead of only the last frame.

## 5. Softmax

The last layer outputs 6 scores; softmax converts them into 6 probabilities
that sum to 1. The highest probability is the prediction; the full
probability vector is used for ROC-AUC. Training uses cross-entropy with
**label smoothing 0.1** (targets 0.9167 / 0.0167) to reduce over-confidence.

## 6. Training and the three seeds

- Adam (lr 1e-3, gradient clipping 1.0), batch 64, up to 60 epochs, balanced
  class weights, SpecAugment on training data only.
- **Three seeds (42, 1, 7)** change only random initialisation and data
  order; `run_config.json` files differ only in `seed`. Reporting mean ± std
  shows the result is not a lucky seed.
- Test accuracy per seed: 57.04%, 56.96%, 56.79%.

## 7. Early stopping

- Monitors **validation accuracy**; stops after 12 epochs without improvement
  and **restores the best weights**.
- Runs stopped at epochs 30, 38 and 41 (best epochs 18, 26, 29).
- ReduceLROnPlateau halved the learning rate when validation loss stalled for
  4 epochs.
- The test set was evaluated **once**, after this validation-based selection,
  and never used for any decision.

## 8. Overfitting

- Train ≈ 85% vs validation ≈ 61% vs test ≈ 57% accuracy → a train–test gap
  of ≈ 28 points.
- Validation loss reached its minimum early (epochs 18, 16, 24) and then rose
  while training loss kept falling.
- Cause in context: only 63 training speakers and an unseen-speaker test.
  Regularisation used: dropout, SpecAugment, label smoothing, early stopping,
  learning-rate reduction.
- Validation is ≈ 4 points above test: the model is selected on only 13
  validation actors.

## 9. Metrics — what each means

| Metric | Meaning |
|---|---|
| Accuracy | fraction of clips classified correctly |
| Macro precision | per class: of the clips predicted as that class, how many were right; averaged over classes |
| UAR (macro recall) | per class: how many of its clips were found; averaged equally over classes (standard in speech emotion recognition) |
| Macro F1 | per-class harmonic mean of precision and recall, averaged equally |
| ROC-AUC | how well the probabilities rank the true class above others, 0.5 = random, 1.0 = perfect |

Per class: **Neutral** best (F1 0.652), **Fear** worst (0.463); **Anger** is
over-predicted (recall 0.794, precision 0.525); **Sad** under-predicted
(precision 0.709, recall 0.436). Biggest confusions: Happy→Anger,
Disgust→Anger, Fear→Happy.

## 10. LSTM vs BiLSTM

| Test (3 seeds) | LSTM | BiLSTM |
|---|---|---|
| Accuracy | 56.93% ± 0.12% | 58.10% ± 0.51% |
| Macro F1 | 56.58% ± 0.23% | 57.84% ± 0.63% |
| UAR | 57.13% ± 0.07% | 58.19% ± 0.49% |
| ROC-AUC | 86.14% ± 0.05% | 87.10% ± 0.19% |
| Parameters | 222,534 | 510,022 |

- Same data, features, training procedure and seeds; the only design
  difference is one vs two reading directions.
- BiLSTM ≈ 1 point better on every metric, with 2.29 × the parameters
  (the LSTM has 56.4% fewer).
- **Careful claim:** bidirectionality also doubles capacity, so the experiment
  **cannot** say whether the gain comes from future context or from extra
  parameters. No significance test was run (3 seeds each).

## 11. Commands for the viva

Show the architecture (no data, no training):

```powershell
python -m src.training.train_lstm --summary_only
```

Show the committed headline (no model needed):

```powershell
python -c "import json; s=json.load(open('results/lstm/seed_summary.json')); [print(k.ljust(18), round(s[k]['mean'],4), '+/-', round(s[k]['std'],4)) for k in ['accuracy','f1_macro','recall_macro_UAR','precision_macro','f1_weighted','roc_auc_macro_ovr','val_accuracy']]"
```

Show one run's per-class report and settings:

```powershell
Get-Content results\lstm\attention\test_classification_report.txt
Get-Content results\lstm\attention\run_config.json
```

Figures: `results/lstm/<tag>/learning_curves.png` and
`test_confusion_matrix.png` for `attention`, `attention_seed1`,
`attention_seed7`.

**Evaluation-only demo** (loads saved weights, no training, writes nothing;
needs the local checkpoints and dataset):

```powershell
python -m src.evaluation.evaluate_lstm --tags attention attention_seed1 attention_seed7
```

Expected: each seed prints `max |difference| = 0.000000 -> reproduced`, and
the mean ± std matches the headline above.
