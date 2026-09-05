# Speech Emotion Recognition for Supporting Emotional Well-Being of Adults Living Independently at Home

**SE4050 – Deep Learning | BSc (Hons) in Information Technology | SLIIT | 2026**

Group members: *[Name 1 – Reg No.]* (Leader), *[Name 2 – Reg No.]*, *[Name 3 – Reg No.]*, *[Name 4 – Reg No.]*

---

## 1. Project Overview

Adults living independently at home may experience changes in emotional well-being that are hard to catch through occasional check-ins. This project explores **Speech Emotion Recognition (SER)** using deep learning as a potential building block for continuous, non-intrusive emotional well-being monitoring.

We frame this as a **supervised deep learning classification task**: given a short speech clip, predict one of six emotion categories. We implement and critically compare **four distinct architectures**:

| Model | Role |
|---|---|
| **MLP** | Baseline — flattened MFCC features through dense layers |
| **1D CNN** | Learns local spectral-temporal patterns via convolution over the MFCC time axis |
| **LSTM** | Models sequential/temporal evolution of emotional cues across an utterance |
| **BiLSTM** | Extends LSTM with backward-direction context for richer temporal representation |

The comparison covers not just accuracy, but generalization (actor-level split), training stability, computational cost, and per-class error behaviour — see [`notebooks/06_model_comparison.ipynb`](notebooks/06_model_comparison.ipynb) and Section 8 of the report.

**Important framing note:** CREMA-D consists of *actor-performed* emotional speech, not naturalistic speech from elderly individuals living independently. Our discussion explicitly addresses this domain gap rather than treating lab results as directly deployable — see the Critical Analysis section of the report.

---

## 2. Dataset

**CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)

- Source: [Kaggle mirror](https://www.kaggle.com/datasets/ejlok1/cremad?resource=download)
- Original dataset: Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*. IEEE Transactions on Affective Computing.
- 7,442 audio-visual clips from 91 actors, 6 emotion categories (Anger, Disgust, Fear, Happy, Neutral, Sad), varying intensity levels.
- We use the **audio-only** portion of the dataset.

Full description (class balance, actor demographics, duration distributions, known data-quality issues) is in `notebooks/01_eda.ipynb` and Section 3 of the report.

**Note:** Raw audio files are not committed to this repository (see [`data/README.md`](data/README.md) for download instructions).

### Citation
```
Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014).
CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.
IEEE Transactions on Affective Computing, 5(4), 377–390.
```

---

## 3. Repository Structure

```
speech-emotion-recognition/
├── data/            # dataset (raw not committed) + actor-level split definitions
├── notebooks/        # EDA, preprocessing, per-model training, cross-model comparison
├── src/               # reusable pipeline code (config, features, models, training, evaluation)
├── configs/           # per-model hyperparameter + seed YAML files (for reproducibility)
├── models/            # saved checkpoints (not committed — see .gitignore)
├── results/           # metrics, confusion matrices, learning curves, comparison figures
└── report/            # final report source + figures
```

Design choices worth noting:
- `data/splits/` is **actor-level**, not clip-level or random — this prevents the same actor's voice appearing in both train and test, which would leak speaker identity cues and inflate accuracy. See `src/data/split.py`.
- `configs/*.yaml` store the exact hyperparameters and random seed used for each model's reported results, so any member (or the marker) can reproduce them.
- `results/metrics/` and `results/confusion_matrices/` are small files and **are** committed, so results are visible in the repo history even without re-running training.

---

## 4. Setup Instructions

```bash
# 1. Clone the repository
git clone <repo-url>
cd speech-emotion-recognition

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download CREMA-D (see data/README.md for details) and place audio files under:
#    data/raw/AudioWAV/

# 5. Run preprocessing to generate features
python -m src.features.extract_features --config configs/preprocessing.yaml
```

---

## 5. How to Run Each Model

Each model has a dedicated notebook (for exploration/report figures) and a corresponding script (for reproducible CLI training). Config files fix hyperparameters and the random seed for reproducibility.

| Model | Notebook | Script |
|---|---|---|
| MLP | `notebooks/03_mlp.ipynb` | `python -m src.training.train --model mlp --config configs/mlp.yaml` |
| 1D CNN | `notebooks/04_cnn1d.ipynb` | `python -m src.training.train --model cnn1d --config configs/cnn1d.yaml` |
| LSTM | `notebooks/05_lstm_bilstm.ipynb` | `python -m src.training.train --model lstm --config configs/lstm.yaml` |
| BiLSTM | `notebooks/05_lstm_bilstm.ipynb` | `python -m src.training.train --model bilstm --config configs/bilstm.yaml` |

Cross-model comparison and evaluation plots:
```bash
python -m src.evaluation.compare_models --results_dir results/metrics/
```
or open `notebooks/06_model_comparison.ipynb`.

Each training run saves:
- checkpoint → `models/<model_name>/`
- logs/learning curves → `results/logs/<model_name>/`
- final test metrics → `results/metrics/<model_name>.json`
- confusion matrix → `results/confusion_matrices/<model_name>.png`

---

## 6. Team Member Contributions

| Member | Reg No. | Primary Responsibility |
|---|---|---|
| [Name 1] (Leader) | [Reg No.] | Data pipeline, MFCC/feature preprocessing, actor-level split, GitHub/README maintenance, report coordination & integration |
| [Name 2] | [Reg No.] | MLP baseline — architecture, training, evaluation |
| [Name 3] | [Reg No.] | 1D CNN — architecture, training, evaluation |
| [Name 4] | [Reg No.] | LSTM & BiLSTM — architecture, training, evaluation |

Cross-model comparison, the Critical Analysis & Discussion section, and viva preparation are **joint work** across all four members — see individual commit history for detailed, traceable contributions.

See `Members.txt` for full names, student numbers, and emails.

---

## 7. Reproducibility

- Random seeds are fixed and recorded per model in `configs/*.yaml`.
- Train/val/test splits are actor-level and stored explicitly in `data/splits/` (not regenerated randomly on each run).
- Exact package versions are pinned in `requirements.txt`.

---

## 8. Acknowledgements

- Dataset: CREMA-D (Cao et al., 2014) — see citation above.
- Any AI-assistance used in code scaffolding, debugging, or report drafting is disclosed in the report's Acknowledgements/References section per the assignment's academic integrity requirements.
