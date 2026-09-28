# LSTM vs BiLSTM — Comparison

**Sources (committed files only):**
- LSTM (Theekshana): `results/lstm/{attention, attention_seed1, attention_seed7}/`,
  `results/lstm/seed_summary.json`
- BiLSTM (Surath): `results/bilstm/{attention, attention_seed1, attention_seed7}/`,
  `results/bilstm/seed_summary.json` (added in commit `d417592`)

No model was retrained for this comparison, and no BiLSTM file was modified.

---

## 1. What is shared, and what differs

Both models are trained by the same pipeline code. `train_lstm.py` imports the
data, feature, normalisation, masking, augmentation, pooling and metric
functions from `train_bilstm.py`. Comparing the `run_config.json` files, the
only differing keys are the output/model folders, the data path (the BiLSTM
was run on Kaggle from `/kaggle/input/cremad`, the LSTM locally from
`data/raw`) and the LSTM-only `summary_only` flag.

| Aspect | LSTM | BiLSTM |
|---|---|---|
| Data, actor split, 5,147 / 1,066 / 1,229 clips | same | same |
| 16 kHz, silence trim 30 dB, 64 log-mel + Δ + ΔΔ (200 × 192) | same | same |
| Train-only standardisation, right padding, `RightPaddingMask` | same | same |
| SpecAugment (time 20, freq 8), balanced class weights | same | same |
| Adam 1e-3, clipnorm 1.0, label smoothing 0.1, batch 64, ≤ 60 epochs | same | same |
| Early stopping / LR schedule / best-validation-accuracy selection | same | same |
| `MaskedAttentionPooling(64)` + Dense(64) head + softmax(6) | same | same |
| Seeds | 42, 1, 7 | 42, 1, 7 |
| **Recurrent encoder** | `LSTM(128) → LSTM(64)` (forward only) | `Bidirectional(LSTM(128)) → Bidirectional(LSTM(64))` |
| **Trainable parameters** | **222,534** | **510,022** |
| Hardware | local CPU (≈ 79–91 s/epoch) | Kaggle (≈ 4.3–4.5 s/epoch) |

Because the hardware differs, training and inference times are **not
comparable** and are not used in this comparison.

### Where the parameter difference comes from

| Layer | LSTM | BiLSTM |
|---|---|---|
| Recurrent layer 1 | 164,352 | 328,704 (two directions) |
| Recurrent layer 2 | 49,408 (input 128) | 164,352 (two directions, input 256) |
| Attention pooling | 4,224 (input 64) | 8,320 (input 128) |
| Dense(64) | 4,160 | 8,256 |
| Softmax(6) | 390 | 390 |
| **Total** | **222,534** | **510,022** |

Making the encoder bidirectional doubles each recurrent layer **and** doubles
the width of everything after it. The two models therefore differ in both
direction of processing and capacity (2.29 × parameters).

## 2. Test results (three seeds, mean ± sample std)

| Test metric | LSTM | BiLSTM | LSTM − BiLSTM | Relative |
|---|---|---|---|---|
| Accuracy | 0.5693 ± 0.0012 | 0.5810 ± 0.0051 | −0.0117 (−1.17 pts) | −2.0% |
| Macro F1 | 0.5658 ± 0.0023 | 0.5784 ± 0.0063 | −0.0126 (−1.26 pts) | −2.2% |
| UAR (macro recall) | 0.5713 ± 0.0007 | 0.5819 ± 0.0049 | −0.0106 (−1.06 pts) | −1.8% |
| Macro precision | 0.5836 ± 0.0053 | 0.5914 ± 0.0055 | −0.0078 (−0.78 pts) | −1.3% |
| Weighted F1 | 0.5637 ± 0.0023 | 0.5769 ± 0.0066 | −0.0132 (−1.32 pts) | −2.3% |
| Macro ROC-AUC | 0.8614 ± 0.0005 | 0.8710 ± 0.0019 | −0.0096 (−0.96 pts) | −1.1% |
| **Parameters** | **222,534** | **510,022** | **−287,488** | **−56.4%** |

Per seed, every BiLSTM run is above every LSTM run on the four headline
metrics (e.g. test accuracy: LSTM 0.5679–0.5704, BiLSTM 0.5753–0.5850).
The LSTM keeps ≈ 98% of the BiLSTM's test accuracy and macro F1 with 43.6% of
its parameters.

## 3. Generalisation

| Accuracy (mean of 3 seeds) | LSTM | BiLSTM |
|---|---|---|
| Train | 0.8532 ± 0.0383 | 0.9378 ± 0.0182 |
| Validation | 0.6119 ± 0.0141 | 0.6304 ± 0.0108 |
| Test | 0.5693 ± 0.0012 | 0.5810 ± 0.0051 |
| Train − test gap | 0.2839 ± 0.0392 | 0.3568 ± 0.0219 |
| Epochs run / best epoch | 30/18, 38/26, 41/29 | 40/28, 59/47, 36/24 |

Both models overfit to the training speakers. The BiLSTM fits the training
set more closely (≈ 94% vs 85%) and has the larger train–test gap, while also
scoring slightly higher on validation and test. The LSTM's test scores vary
less across seeds (accuracy std 0.0012 vs 0.0051).

## 4. Per-class F1 (test, mean of 3 seeds)

| Class | LSTM P / R / F1 | BiLSTM P / R / F1 | ΔF1 (LSTM − BiLSTM) |
|---|---|---|---|
| Anger | 0.525 / 0.794 / 0.632 | 0.553 / 0.792 / 0.651 | −0.019 |
| Disgust | 0.540 / 0.608 / 0.572 | 0.566 / 0.621 / 0.591 | −0.020 |
| Fear | 0.508 / 0.425 / 0.463 | 0.484 / 0.459 / 0.470 | −0.008 |
| Happy | 0.567 / 0.513 / 0.537 | 0.609 / 0.541 / 0.572 | −0.035 |
| Neutral | 0.653 / 0.652 / 0.652 | 0.664 / 0.618 / 0.640 | **+0.012** |
| Sad | 0.709 / 0.436 / 0.540 | 0.672 / 0.460 / 0.546 | −0.005 |

The two models share the same pattern: Neutral and Anger are the easiest,
Fear the hardest, Anger has high recall/low precision and Sad high
precision/low recall. The largest gap is on Happy; the LSTM is slightly
better on Neutral.

## 5. What can and cannot be concluded

**Supported by these results:**
- Under an identical pipeline and the same three seeds, the BiLSTM scores
  about 1 point higher than the LSTM on test accuracy, macro F1, UAR and
  ROC-AUC, consistently across seeds.
- The LSTM achieves this with 56.4% fewer parameters and is more stable
  across seeds on the test set.

**Not established by this experiment:**
- That **bidirectionality alone** causes the difference. The BiLSTM also has
  2.29 × the parameters and wider attention/dense layers, so direction and
  capacity are confounded. Separating them would need, for example, a
  parameter-matched LSTM (e.g. wider layers) — not run here.
- Statistical significance. Three seeds per model is a small sample and no
  significance test was performed; the standard deviations are indicative.
- Any speed/efficiency conclusion, because the two models ran on different
  hardware.

A fair one-sentence summary: *with the same data, features, training
procedure and seeds, the bidirectional model was about one percentage point
more accurate, at 2.3 times the parameter count; this experiment does not
isolate whether the gain comes from backward context or from extra capacity.*
