# Contributing & Commit Conventions

This project is assessed partly on **GitHub repository and group contributions** (Implementation criterion — the rubric explicitly checks for regular, meaningful, traceable commits from *every* member, and penalizes large batches of trivial or last-minute commits). Please follow these conventions so everyone's work is clearly visible.

## 1. Commit frequency

- **Minimum:** each member commits and pushes their own work **at least once per week**, every week of the project, not just in the final days.
- Small, frequent commits are better than one giant commit. If you worked on preprocessing for three days, that's ideally 3+ commits (e.g. "add MFCC extraction function", "add delta/delta-delta features", "fix normalization bug"), not one commit at the end.

## 2. Commit message format

Use a short prefix + clear, specific description:

```
<type>: <short description>

[optional longer explanation]
```

**Types:**
- `feat` – new functionality (e.g. a new model, a new preprocessing step)
- `fix` – bug fix
- `refactor` – restructuring code without changing behaviour
- `docs` – README, comments, report text
- `exp` – experiment/training run (include config or seed used)
- `eval` – evaluation scripts, metrics, plots
- `chore` – housekeeping (deps, gitignore, folder structure)

**Good examples:**
```
feat: implement 1D CNN architecture with 3 conv blocks
fix: correct actor-level split leak in data/split.py
exp: train LSTM with seed=42, lr=1e-3 (see configs/lstm.yaml)
eval: add per-class confusion matrix plot for BiLSTM
docs: add dataset citation and EDA summary to README
```

**Avoid:**
```
update
fix stuff
final version
asdf
```

## 3. Branching

- Each member works on their own branch for their model: `feature/mlp`, `feature/cnn1d`, `feature/lstm-bilstm`, `feature/preprocessing`.
- Open a **Pull Request** into `main` when a piece of work is ready. PRs don't need to be huge — merge often.
- Use GitHub Issues to track outstanding tasks (e.g. "Add ROC-AUC plot for CNN", "Fix actor overlap in split").

## 4. What NOT to do

- Don't dump all your work in a single commit the night before the deadline — this reads as low/no traceable participation, even if you did the work earlier, and is explicitly penalized.
- Don't commit large binary files (raw audio, model checkpoints) — see `.gitignore`. Use `data/README.md` instructions instead so everyone downloads the dataset locally.
- Don't commit directly to `main` for anything non-trivial — use a branch + PR so changes are reviewable and attributable.

## 5. Reproducibility expectations

- If you run an experiment that produces a result used in the report, commit the config file (with the seed) that produced it, and the resulting metrics file under `results/metrics/`.
- Name experiment branches/commits so it's clear which config/result they correspond to.

## 6. Weekly check-in

At the end of each week, each member should have at least one commit reflecting that week's progress. The leader will do a quick weekly scan of the commit graph (not to police anyone, but so nobody discovers contribution gaps only at submission time).
