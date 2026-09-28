# LSTM — Final Performance Analysis

**Model:** standard (unidirectional) LSTM with masked attention pooling
**Owner:** Theekshana · **Branch:** `feature/lstm-theekshana`
**Source of every number below:** the committed artifacts in `results/lstm/`
(`attention/`, `attention_seed1/`, `attention_seed7/`, `seed_summary.json`).
No result in this document was re-run or edited.

---

## 1. Headline result (three seeds)

The headline LSTM result is the **mean ± sample standard deviation (ddof = 1)
over seeds 42, 1 and 7** — no single seed is reported as "the" result.
The same convention is used in `results/bilstm/seed_summary.json`.

| Test metric (n = 1,229 clips, 15 unseen actors) | Mean ± std |
|---|---|
| **Accuracy** | **0.5693 ± 0.0012** (56.93% ± 0.12%) |
| **Macro F1** | **0.5658 ± 0.0023** (56.58% ± 0.23%) |
| **Macro recall (UAR)** | **0.5713 ± 0.0007** (57.13% ± 0.07%) |
| Macro precision | 0.5836 ± 0.0053 |
| Weighted F1 | 0.5637 ± 0.0023 |
| **Macro ROC-AUC (one-vs-rest)** | **0.8614 ± 0.0005** (86.14% ± 0.05%) |
| Validation accuracy (13 actors) | 0.6119 ± 0.0141 |
| Trainable parameters | 222,534 |

Chance level for six balanced-ish classes is ≈ 0.17, so the model is well
above chance. The ROC-AUC of 0.86 shows that the predicted probabilities rank
the correct emotion highly even when the top-1 prediction is wrong.

## 2. Individual seeds

| Seed | Epochs run | Best epoch | Train acc | Val acc | Test acc | Macro P | UAR | Macro F1 | Weighted F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 30 | 18 | 0.8090 | 0.6107 | 0.5704 | 0.5861 | 0.5717 | 0.5684 | 0.5662 | 0.8620 | 2,357 s |
| 1 | 38 | 26 | 0.8743 | 0.6266 | 0.5696 | 0.5872 | 0.5718 | 0.5653 | 0.5631 | 0.8611 | 3,021 s |
| 7 | 41 | 29 | 0.8762 | 0.5985 | 0.5679 | 0.5776 | 0.5705 | 0.5638 | 0.5617 | 0.8611 | 3,747 s |

Training ran on CPU (≈ 79–91 s per epoch). The only difference between the
three `run_config.json` files is the `seed` value.

## 3. Interpretation of the metrics

- **Accuracy vs macro F1 vs UAR.** The three are within 0.6 points of each
  other. CREMA-D is nearly balanced (Neutral has 179 test clips, every other
  class 210) and balanced class weights were used, so accuracy is not being
  inflated by a majority class. UAR (unweighted average recall) is the
  standard speech-emotion metric; 0.571 means that on average 57% of each
  emotion's clips are recognised.
- **Precision above recall (0.584 vs 0.571 macro).** Driven mainly by Sad,
  which is predicted rarely but correctly when it is predicted (see §5).
- **ROC-AUC 0.861.** Considerably higher than accuracy: the correct class is
  often the second choice. This is typical when classes overlap acoustically
  (e.g. high-arousal Anger / Happy / Fear).

## 4. Generalisation and overfitting

| Split (mean of 3 seeds) | Accuracy |
|---|---|
| Train | 0.8532 ± 0.0383 |
| Validation | 0.6119 ± 0.0141 |
| Test | 0.5693 ± 0.0012 |

Observations, all from `history.json` / `metrics.json`:

1. **Clear overfitting.** The train–test accuracy gap is 0.239, 0.305 and
   0.308 (mean 0.284 ± 0.039). The model memorises training speakers much
   better than it generalises to new speakers — expected with 63 training
   actors and an actor-independent split.
2. **Validation loss bottoms out before validation accuracy peaks.**
   Minimum validation loss occurred at epochs 18, 16 and 24; the restored
   (best validation-accuracy) epochs were 18, 26 and 29. By the last epoch
   validation loss had risen again (1.230 → 1.273, 1.202 → 1.268,
   1.250 → 1.287) while training loss kept falling. The model became more
   confident on its errors even while its top-1 accuracy crept up.
3. **Regularisation did its job partially.** Early stopping (patience 12 on
   validation accuracy) ended every run well before 60 epochs, and
   ReduceLROnPlateau halved the learning rate 3–6 times per run (seed 42:
   from epochs 14/23/27; seed 1: from 21/25/29/33/37; seed 7: from
   15/23/29/33/37/41). Dropout, SpecAugment and label smoothing were active
   throughout but did not close the speaker gap.
4. **Validation is optimistic.** Validation accuracy exceeds test accuracy by
   ≈ 4 points on average. The model is selected on the validation actors, and
   13 actors is a small sample, so some selection bias is expected. The test
   set was evaluated only once per seed, after selection, and never used for
   any decision.
5. **Seed stability.** Test metrics are very stable (accuracy range
   0.5679–0.5704, std 0.0012). Validation accuracy varies more
   (0.5985–0.6266, std 0.0141), which again reflects the small validation set.
   Three seeds is a small sample, so the standard deviations are indicative,
   not precise.

## 5. Per-class performance

Averaged over the three seeds (from the committed
`test_classification_report.txt` files):

| Class | Precision | Recall | F1 | F1 range over seeds |
|---|---|---|---|---|
| Anger | 0.525 | **0.794** | 0.632 | 0.628 – 0.637 |
| Disgust | 0.540 | 0.608 | 0.572 | 0.563 – 0.582 |
| Fear | 0.508 | 0.425 | **0.463** (lowest) | 0.433 – 0.487 |
| Happy | 0.567 | 0.513 | 0.537 | 0.523 – 0.550 |
| Neutral | 0.653 | 0.652 | **0.652** (highest) | 0.649 – 0.655 |
| Sad | **0.709** | 0.436 | 0.540 | 0.512 – 0.574 |

- **Neutral** is the most reliable class, with balanced precision and recall.
- **Anger** has the highest recall but low precision: it is over-predicted.
- **Fear** is the hardest class in every seed.
- **Sad** is under-predicted: high precision, low recall.

## 6. Confusion patterns

Test confusion matrices summed over the three seeds (rows = true class,
columns = predicted; 3 × 1,229 clips). Source: the per-run
`test_confusion_matrix.npy` files saved by the training script (the committed
`test_confusion_matrix.png` files show the same matrices per seed).

| true \ predicted | Anger | Disgust | Fear | Happy | Neutral | Sad |
|---|---|---|---|---|---|---|
| **Anger** (630) | **500** | 74 | 15 | 31 | 8 | 2 |
| **Disgust** (630) | 116 | **383** | 65 | 23 | 17 | 26 |
| **Fear** (630) | 102 | 82 | **268** | 106 | 24 | 48 |
| **Happy** (630) | 155 | 40 | 61 | **323** | 47 | 4 |
| **Neutral** (537) | 53 | 32 | 17 | 52 | **350** | 33 |
| **Sad** (630) | 26 | 99 | 101 | 38 | 91 | **275** |

Largest confusions:

| True → Predicted | Count (3 seeds) |
|---|---|
| Happy → Anger | 155 |
| Disgust → Anger | 116 |
| Fear → Happy | 106 |
| Fear → Anger | 102 |
| Sad → Fear | 101 |
| Sad → Disgust | 99 |

- **Anger acts as an attractor.** It is predicted 952 times for 630 true
  Anger clips; Happy, Disgust and Fear are all pulled towards it.
- **High-arousal emotions are confused with each other** (Happy ↔ Anger,
  Fear ↔ Happy/Anger).
- **Sad spreads into low-arousal classes** (Fear, Disgust, Neutral) and is
  predicted only 388 times for 630 true clips, explaining its low recall.

These are descriptive observations of the confusion matrix; the experiment
does not identify the acoustic cause of each confusion.

## 7. Summary

The standard LSTM reaches **56.93% ± 0.12% test accuracy, 56.58% ± 0.23%
macro F1, 57.13% ± 0.07% UAR and 0.8614 ± 0.0005 ROC-AUC** on unseen
speakers, with 222,534 parameters. Results are very stable across seeds, but
there is a substantial train–test gap (≈ 28 points), the validation set is
mildly optimistic, and Fear and Sad remain the hardest emotions, while Anger
is over-predicted.
