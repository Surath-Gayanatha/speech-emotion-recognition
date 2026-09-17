"""Evaluate all trained models on the fixed actor-level test split and display test accuracy comparison."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, NUM_CLASSES, RESULTS_DIR
from src.training.train import (
    create_split_mask,
    load_actor_ids,
    normalize_feature_splits,
)


def load_test_data():
    features = np.transpose(
        np.load(DATA_PROCESSED_DIR / "features.npy"),
        (0, 2, 1),
    )
    labels = np.load(DATA_PROCESSED_DIR / "labels.npy")
    filenames = np.load(DATA_PROCESSED_DIR / "filenames.npy")
    train_mask = create_split_mask(filenames, load_actor_ids("train_actors.txt"))
    test_mask = create_split_mask(filenames, load_actor_ids("test_actors.txt"))
    normalized_splits = normalize_feature_splits([
        (features[train_mask], labels[train_mask]),
        (features[test_mask], labels[test_mask]),
    ])
    return normalized_splits[1]


def extract_acc_from_report(report_path: Path):
    if not report_path.exists():
        return None
    content = report_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Test Accuracy:\s*([0-9\.]+)%", content)
    if match:
        return float(match.group(1))
    match_dec = re.search(r"Test Accuracy:\s*([0-9\.]+)", content)
    if match_dec:
        val = float(match_dec.group(1))
        return val * 100 if val <= 1.0 else val
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1. Baseline models (MLP, 1D CNN, LSTM, BiLSTM)
    try:
        X_test, y_test = load_test_data()
        has_test_data = True
    except Exception as e:
        has_test_data = False

    baseline_models = ("mlp", "cnn1d", "lstm", "bilstm")
    for model_name in baseline_models:
        checkpoint = MODELS_DIR / model_name / f"{model_name}_best.keras"
        if not checkpoint.exists():
            checkpoint = MODELS_DIR / f"{model_name}_best.keras"
        if model_name == "cnn1d" and not checkpoint.exists():
            checkpoint = MODELS_DIR / "cnn1d_standardized_best.keras"

        if checkpoint.exists() and has_test_data:
            try:
                model = tf.keras.models.load_model(checkpoint)
                model_input = X_test.reshape(len(X_test), -1) if model_name == "mlp" else X_test
                probabilities = model.predict(model_input, verbose=0)
                predictions = np.argmax(probabilities, axis=1)
                acc = float(accuracy_score(y_test, predictions)) * 100
                summary[model_name.upper()] = f"{acc:.2f}%"
                continue
            except Exception:
                pass

        rep = extract_acc_from_report(RESULTS_DIR / f"{model_name}_classification_report.txt")
        if rep is not None:
            summary[model_name.upper()] = f"{rep:.2f}%"
        else:
            summary[model_name.upper()] = "N/A (Not Trained Yet)"

    # 2. Advanced / Attention / Transfer / Ensemble Models
    other_models = [
        ("CNN + Attention", RESULTS_DIR / "cnn_attention_v3_classification_report.txt"),
        ("CNN + BiLSTM + Attention", RESULTS_DIR / "cnn_bilstm_attention_classification_report.txt"),
        ("Enriched 200D CNN + Attention", RESULTS_DIR / "enriched_cnn_report.txt"),
        ("Spectrogram ResNet-34", RESULTS_DIR / "spectrogram_resnet_report.txt"),
        ("Wav2Vec2 SSL Transformer", RESULTS_DIR / "wav2vec2_fine_tuned_report.txt"),
        ("Ensemble Stacking Model", RESULTS_DIR / "ensemble_model_report.txt"),
    ]

    for display_name, report_file in other_models:
        acc = extract_acc_from_report(report_file)
        if acc is not None:
            summary[display_name] = f"{acc:.2f}%"
        else:
            summary[display_name] = "N/A (Not Trained Yet)"

    (output_dir / "all_models_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("ALL MODELS TEST ACCURACY SUMMARY")
    print("=" * 60)
    for model_name, acc_str in summary.items():
        print(f"{model_name:<32} : {acc_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()