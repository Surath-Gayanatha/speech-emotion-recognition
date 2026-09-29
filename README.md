# Speech Emotion Recognition for Supporting Emotional Well-Being of Adults Living Independently at Home

**SE4050 Deep Learning · BSc (Hons) in Information Technology · SLIIT · 2026**

Repository: https://github.com/Surath-Gayanatha/speech-emotion-recognition

| Member | Student ID | Model |
|---|---|---|
| Gayanatha D.S (group leader) | IT23233058 | BiLSTM with Attention |
| Theekshana | [ID] | LSTM with Attention Pooling |
| Sandani A.W.A | IT23228108 | CNN-BiLSTM-MHA with SpecAugment |
| Sarangi K.P.E | IT23136960 | MFT-TCN with Attention |
| Additional models | | CNN-BiGRU Dual Attention, CNN-SE-BiLSTM-MHA |

---

## 1. Project overview

Adults living alone can go through changes in mood that occasional questionnaires and interviews miss. This project studies **speech emotion recognition (SER)** as a possible building block for continuous, non-intrusive well-being monitoring.

The task is **supervised six-class classification**: a short speech clip is classified as Anger, Disgust, Fear, Happy, Neutral or Sad. We compare six deep learning models on the CREMA-D dataset and evaluate them on **speakers never seen in training** (actor-level split).

CREMA-D contains *acted* emotional speech, not natural speech from older adults at home. The report discusses this gap in its limitations; results here should not be read as ready for deployment.

## 2. Models and results

All six models use the same actor-level split (63 / 13 / 15 actors) and the same 1,229-clip test set. Test results:

| Model | Owner | Input features | Parameters | Test accuracy | Macro-F1 |
|---|---|---|---|---|---|
| BiLSTM with Attention | Gayanatha | log-mel + Δ + ΔΔ, 192 × 200 | 510,022 | 58.50% (seed 42) · **58.1 ± 0.5%** (3 seeds) | 58.30% |
| LSTM with Attention Pooling | Theekshana | log-mel + Δ + ΔΔ, 192 × 200 | 222,534 | 57.04% (seed 42) · **56.9 ± 0.1%** (3 seeds) | 56.84% |
| CNN-BiLSTM-MHA + SpecAugment | Sandani | log-mel, 64 × 174 | 1,180,166 | 59.89% | 59.73% |
| MFT-TCN with Attention (V7) | Sarangi | 201 multi-feature × 256 | 1,080,935 | 54.03% | 53.51% |
| CNN-BiGRU Dual Attention (V3) | additional | log-mel, 64 × 174 | 1,016,103 | 58.99% | 58.96% |
| CNN-SE-BiLSTM-MHA | additional | log-mel, 64 × 174 | 1,186,054 | 59.15% | 59.31% |

**Read this table carefully.** The split and test set are identical, but the models were not trained under identical settings: feature representations, learning rates, batch sizes and augmentation differ (see each model's `config.json` / `run_config.json`). Only BiLSTM vs LSTM is a strictly controlled comparison (LSTM imports the BiLSTM pipeline unchanged). Only BiLSTM and LSTM were run with three seeds; the other four are single runs (seed 42), so gaps of about 1 to 1.5 points between them are within run-to-run noise. Reference points: chance is 16.7%, and human listeners scored 40.9% from audio alone (Cao et al., 2014).

BiLSTM experiments beyond the main run (all in `results/bilstm/`):

| Experiment | Test accuracy | Finding |
|---|---|---|
| Last-hidden-state pooling instead of attention (`last`) | 53.0% | Attention adds about 5 points for 1.7% more parameters |
| Dropout 0.3 + weight decay 1e-4 | 58.0% | Train–validation gap 20.2 → 12.8 points, test unchanged |
| Above + stronger SpecAugment (30 / 12) | 57.5% | Gap → 8.6 points, test unchanged |

## 3. Dataset

**CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset): 7,442 clips from 91 actors (48 male, 43 female, aged 20 to 74), 12 fixed sentences, six emotions, four intensity levels. We use the **audio only**.

| Split | Actors | Clips |
|---|---|---|
| Train | 63 | 5,147 |
| Validation | 13 | 1,066 |
| Test | 15 | 1,229 |

The split is defined by actor ID in `data/splits/` (committed; do not regenerate). No actor appears in more than one split, so a model cannot succeed by recognising a speaker's voice.

- Original source: https://github.com/CheyneyComputerScience/CREMA-D
- Copy used: https://www.kaggle.com/datasets/ejlok1/cremad
- Licence: Open Database License (see the original repository for the full terms)

> Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390. doi:10.1109/TAFFC.2014.2336244

## 4. Setup

Use **Python 3.12**. TensorFlow does not yet support Python 3.14, so `pip install tensorflow` fails there.

```bash
git clone https://github.com/Surath-Gayanatha/speech-emotion-recognition.git
cd speech-emotion-recognition
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install tensorflow librosa soundfile scikit-learn matplotlib seaborn pandas tqdm joblib
```

`requirements.txt` is the original scaffold and pins older versions; the line above is what the BiLSTM and LSTM runs used.

**Download CREMA-D** (about 2 GB, not stored in the repo). Get it from the Kaggle page above and place the audio at:

```
data/raw/AudioWAV/*.wav
```

If Windows blocks SciPy ("An Application Control policy has blocked this file"), run the same commands on Google Colab with a T4 GPU (the BiLSTM was trained this way). Run every command below **from the repository root**.

## 5. How to run each model

Features are computed on the first run and cached (`data/processed/`, `data/mft_processed/`; not committed). Add `--quick` to the BiLSTM/LSTM scripts for a two-minute check.

| Model | Feature step | Train | Evaluate | Results folder |
|---|---|---|---|---|
| BiLSTM with Attention | automatic | `python -m src.training.train_bilstm` | included | `results/bilstm/` |
| LSTM with Attention Pooling | automatic (shared with BiLSTM) | `python -m src.training.train_lstm` | included | `results/lstm/` |
| CNN-BiLSTM-MHA + SpecAugment | `python -m src.features.extract_logmel_features` | `python -m src.training.train_cnn_bilstm_mha_specaug_BASELINE` | `python -m src.evaluation.evaluate_cnn_bilstm_mha_specaug` | `results/cnn_bilstm_mha_specaug/` |
| MFT-TCN with Attention (V7) | `python src/features_mft/extract_mft_features.py` | `python -m src.training.train_mft_tcn_attention_v7` | included | `results/mft_tcn_attention/` |
| CNN-BiGRU Dual Attention (V3) | `python -m src.features.extract_logmel_features` | `python -m src.training.train_cnn_bigru_dual_attention_v3` | `python -m src.training.evaluate_cnn_bigru_dual_attention_v3` | `results/cnn_bigru_dual_attention_v3/` |
| CNN-SE-BiLSTM-MHA | `python -m src.features.extract_logmel_features` | `python -m src.training.train_cnn_se_bilstm_attention` | `python -m src.evaluation.evaluate_cnn_se_bilstm_attention` | `results/cnn_se_bilstm_attention/` |

Extra options for the BiLSTM and LSTM scripts:

```bash
python -m src.training.train_bilstm --seed 1                       # other seeds (reported: 42, 1, 7)
python -m src.training.train_bilstm --pooling last                 # ablation without attention
python -m src.training.train_bilstm --rnn_dropout 0.3 --weight_decay 1e-4   # regularisation experiment
python -m src.training.train_lstm --summary_only                   # print the architecture, load no data
```

Other variants of the same models are kept for the record and write to separate folders: `train_cnn_bilstm_mha_specaug` (weaker SpecAugment, 4 / 10) and `train_mft_tcn_attention_v8` (192-D input, run with `python src/features/extract_mft_tcn_v8_features.py` first).

## 6. Where to find results

Every run folder holds the evidence for its numbers. The BiLSTM and LSTM runs contain, for example:

| File | Contents |
|---|---|
| `metrics.json` | parameters, epochs, best epoch, timing, and train / validation / test metrics |
| `test_classification_report.txt` | per-emotion precision, recall and F1 |
| `test_confusion_matrix.png` | confusion matrix |
| `learning_curves.png`, `history.json`, `training_log.csv` | per-epoch loss and accuracy |
| `run_config.json` | every setting used for that run |

Runs are named by configuration: `attention` (seed 42), `attention_seed1`, `attention_seed7`, `last`, `attention_reg-…`. `results/bilstm/seed_summary.json` holds the mean ± SD over seeds. The other models store the same information in their own `config.json`, `history.json` / `*_History.csv`, `test_results.json` / `*_Results.txt` and confusion-matrix files.

## 7. Repository structure

```
├── data/
│   ├── splits/            actor-level train / val / test lists (committed)
│   └── raw/               CREMA-D audio (you download it; not committed)
├── src/
│   ├── features/          log-mel feature extraction
│   ├── features_mft/      multi-feature extraction for MFT-TCN
│   ├── models/            model definitions
│   ├── training/          training scripts (train_bilstm.py holds the shared pipeline)
│   └── evaluation/        test-set evaluation and comparison scripts
├── results/               one folder per model / run: metrics, reports, figures
├── models/                saved weights (not committed)
├── configs/               original scaffold only; actual settings are saved with each run
└── report/                final report
```

`models/`, cached features and raw audio are excluded by `.gitignore` because of their size.

## 8. Reproducibility and leakage control

- **Actor-independent split**, fixed in `data/splits/` and identical for every model.
- **Normalisation statistics come from the training actors only**; SpecAugment is applied to training data only.
- **Validation data drives all selection** (early stopping, learning-rate schedule, which version to report). The test set is evaluated once per run.
- **Seeds:** 42 for all models; BiLSTM and LSTM also 1 and 7.
- Training and validation accuracy in the report should be compared like for like: the BiLSTM and LSTM scripts also print train accuracy on clean data after training (dropout and augmentation off), which is higher than the accuracy Keras reports during training.

## 9. Other experiments in this repository

Earlier and extended experiments are kept under `results/` and are **not** part of the six-model comparison: MFCC baselines (MLP 29.78%, 1D CNN 47.76%, an earlier LSTM and BiLSTM at 39.71% and 42.88%), CNN + attention variants, CNN-Transformer, ViT, wav2vec 2.0, an ensemble, dual-branch and other MFT-TCN versions. `results/all_models_summary.json` lists several of them. They used different pipelines and are not directly comparable with the table above.

## 10. Team contributions

| Member | Contribution |
|---|---|
| Gayanatha D.S | Shared feature pipeline and actor split, BiLSTM with Attention (3 seeds, pooling ablation, regularisation experiments), repository and report coordination |
| Theekshana | LSTM with Attention Pooling on the shared pipeline (3 seeds) |
| Sandani A.W.A | CNN-BiLSTM-MHA with SpecAugment, log-mel feature extraction |
| Sarangi K.P.E | MFT-TCN with Attention (multi-feature extraction, versions V1 to V8) |
| Group | CNN-BiGRU Dual Attention, CNN-SE-BiLSTM-MHA, report, video, viva |

Commit history shows each member's individual work.

## 11. Acknowledgements

Dataset: CREMA-D (Cao et al., 2014). Libraries: TensorFlow / Keras, NumPy, pandas, librosa, scikit-learn, Matplotlib. AI-assisted tools were used for scaffolding, debugging, explanation and language refinement; the team ran all experiments and reviewed and takes responsibility for the results and text (see the declaration in the report).
