# LSTM — Architecture and Training Methodology

**Code:** `src/training/train_lstm.py` (model: `build_lstm()`)
**Shared pipeline (imported, not copied):** `src/training/train_bilstm.py`
**Owner:** Theekshana · **Branch:** `feature/lstm-theekshana`

The LSTM is trained on the group's selected common pipeline, the same one used
by the BiLSTM. `train_lstm.py` imports the data loading, feature extraction,
normalisation, masking, SpecAugment, attention pooling and metric functions
from `train_bilstm.py`, so these parts are identical by construction.

---

## 1. Data

| Item | Value |
|---|---|
| Dataset | CREMA-D, audio only (7,442 WAV clips, 91 actors) |
| Classes (6) | Anger, Disgust, Fear, Happy, Neutral, Sad (from the filename's emotion code) |
| Split | Actor-independent, from the committed `data/splits/*_actors.txt` |
| Train | 63 actors, 5,147 clips |
| Validation | 13 actors, 1,066 clips |
| Test | 15 actors, 1,229 clips |

No actor appears in more than one split (`load_actor_split()` asserts this),
so every evaluation is on **unseen speakers**.

## 2. Feature extraction: log-mel + delta + delta-delta

Implemented in `extract_one()` and cached once by `build_features()` in
`data/processed/bilstm_logmel64_d_dd_h320_T200.npz` (shared with the BiLSTM).

1. **Load** at 16 kHz, mono (`librosa.load`).
2. **Trim leading/trailing silence** with `librosa.effects.trim(top_db=30)`,
   so the sequence starts and ends with speech. Clips shorter than 0.2 s
   (before or after trimming) would be dropped; none of the 7,442 were.
3. **Log-mel spectrogram**: 64 mel bands, 20 Hz – 8 kHz, `n_fft = 640`
   (40 ms window), `hop = 320` (20 ms step → 50 frames per second),
   converted to decibels with `power_to_db(ref=max)`, i.e. relative to the
   loudest point of that clip.
4. **Delta and delta-delta**: the first and second time derivatives of the
   log-mel (window of up to 9 frames). Log-mel describes *what* the spectrum
   looks like at each moment; deltas describe *how it is changing* (e.g. how
   fast pitch and energy rise or fall), which carries emotional information.
5. **Stack** → 64 + 64 + 64 = **192 features per frame**.
6. **Cap at 200 frames** (4.0 s of trimmed speech). Clip lengths range from 37
   to 200 frames, median 119; 72 of 7,442 clips reach the cap and are
   truncated.

Each clip therefore becomes a sequence of up to 200 vectors of 192 numbers.

## 3. Normalisation and padding

- **Per-feature standardisation** (`fit_normaliser()`): the mean and standard
  deviation of each of the 192 features are computed **only from training-
  actor frames** (padding is not included, because it is added afterwards).
  The same statistics are applied to validation and test data. This prevents
  information from validation/test speakers leaking into preprocessing.
- **Padding** (`to_padded()`): after standardisation, each sequence is placed
  at the start of a 200 × 192 array and the remaining frames are filled with
  zeros (**right padding**).

## 4. Architecture

```
Input (200, 192)                       log-mel + Δ + ΔΔ, standardised, right-padded
 └─ RightPaddingMask                   marks real frames vs trailing padding
     └─ LSTM(128, return_sequences=True, dropout=0.2)   (200, 128)
         └─ LSTM(64, return_sequences=True, dropout=0.2) (200, 64)
             └─ MaskedAttentionPooling(64)               (64,)
                 └─ Dropout(0.4)
                     └─ Dense(64, ReLU)
                         └─ Dropout(0.3)
                             └─ Dense(6, softmax)          emotion probabilities
```

| Layer | Output shape | Trainable parameters |
|---|---|---|
| Input `logmel_deltas` | (200, 192) | 0 |
| `RightPaddingMask` | (200, 192) + mask | 0 |
| `LSTM(128)` `lstm_1` | (200, 128) | 164,352 |
| `LSTM(64)` `lstm_2` | (200, 64) | 49,408 |
| `MaskedAttentionPooling(64)` | (64,) | 4,224 |
| `Dropout(0.4)` | (64,) | 0 |
| `Dense(64, ReLU)` | (64,) | 4,160 |
| `Dropout(0.3)` | (64,) | 0 |
| `Dense(6, softmax)` `emotion` | (6,) | 390 |
| **Total** | | **222,534** |

The LSTM counts follow 4 × (units × (input + units) + units):
4 × (128 × (192 + 128) + 128) = 164,352 and 4 × (64 × (128 + 64) + 64) = 49,408.

### 4.1 RightPaddingMask (masking)

Zero-padding is needed to batch clips of different lengths, but padded frames
contain no speech. `RightPaddingMask` builds a boolean mask that is **True from
the first frame up to the last non-zero frame** and False for the trailing
padding. Keras passes this mask to the LSTM layers, which then do not update
their state on padded steps, and to the attention layer, which gives padded
frames zero weight.

Why not Keras' built-in `Masking` layer? SpecAugment (§5) zeroes whole frames
*inside* a clip. A plain `Masking(0.0)` would treat those as padding, creating
"holes" in the mask (which cuDNN's fast LSTM kernel does not support) and
skipping frames that SpecAugment meant to show as blanked. Only genuine
trailing padding is masked.

### 4.2 Unidirectional LSTM layers

Each LSTM reads the sequence **forwards in time**; its output at frame *t*
depends only on frames 1…*t*. Stacking two layers (128 then 64 units) gives a
hierarchy: the first layer learns short-term acoustic patterns, the second
combines them into longer-range patterns. Both return the full sequence
(one vector per frame) so that attention can choose which frames matter.
`dropout=0.2` randomly drops 20% of the inputs to each LSTM during training.

This is the only architectural difference from the team BiLSTM, which wraps
the same two layers in `Bidirectional(...)` so each frame also sees future
context.

### 4.3 MaskedAttentionPooling

Emotion is often expressed in a few parts of an utterance (a stressed word, a
rising end), not evenly. Instead of using only the last hidden state, the
model learns a weighted average over time:

1. score each frame: *s_t = vᵀ · tanh(W · h_t + b)*
2. set the score of padded frames to −10⁹ (so they get zero weight)
3. convert scores to weights with a softmax over time: *α_t = softmax(s)_t*,
   so the weights are positive and sum to 1
4. output the weighted sum *Σ α_t h_t* (a 64-dimensional summary)

### 4.4 Classification head and softmax

`Dropout(0.4) → Dense(64, ReLU) → Dropout(0.3) → Dense(6, softmax)`.
The final softmax turns the six output scores into **probabilities that sum to
1**; the predicted emotion is the one with the highest probability, and the
full probability vector is used for ROC-AUC.

## 5. Training methodology

| Setting | Value |
|---|---|
| Loss | Categorical cross-entropy with **label smoothing 0.1** (one-hot targets 1/0 become 0.9167/0.0167, which discourages over-confident predictions) |
| Optimiser | Adam, learning rate 1e-3, gradient clipping `clipnorm = 1.0` (protects the LSTM against exploding gradients) |
| Batch size / max epochs | 64 / 60 |
| Class weights | Balanced, computed from training labels: Neutral 1.1407, all others 0.9759 |
| Augmentation | **SpecAugment on training batches only**: one band of up to 8 mel bins (the same bins in log-mel, Δ and ΔΔ) and one block of up to 20 frames are set to zero per clip. Validation and test data are never augmented. |
| Early stopping | Monitor validation accuracy, patience 12, restore the best weights |
| Learning-rate schedule | ReduceLROnPlateau on validation loss: × 0.5 after 4 epochs without improvement, minimum 1e-5 |
| Model selection | Checkpoint with the best **validation accuracy** is reloaded before evaluation |
| Test evaluation | Once, at the end, with the selected weights. Never used for tuning. |
| Seeds | 42, 1, 7 (identical configuration otherwise); results reported as mean ± std |

## 6. Outputs of a run

For each seed tag (`attention`, `attention_seed1`, `attention_seed7`):

- `results/lstm/<tag>/`: `history.json`, `training_log.csv`,
  `learning_curves.png`, `metrics.json`, `run_config.json`,
  `test_classification_report.txt`, `test_confusion_matrix.png`
- `results/metrics/lstm_<tag>.json`: copy of `metrics.json`
- `models/lstm/best_lstm_<tag>.weights.h5` and `models/lstm/norm_stats.npz`
  (local only, gitignored)
