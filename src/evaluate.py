"""Evaluate saved or freshly trained models on a reproducible data split."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any, Mapping, cast

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data_loader import load_interactions, make_synthetic_interactions
from .train import train_models


def evaluate_models(result: Mapping[str, Any], output_dir: str | Path = "reports") -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_test = result["test_frame"].drop(columns=["user_id"])
    y_test = result["y_test"]
    y_train = result["y_train"]
    majority = str(y_train.mode().iloc[0])
    baseline_predictions = [majority] * len(y_test)
    rows = [
        {
            "model": "majority_baseline",
            "accuracy": accuracy_score(y_test, baseline_predictions),
            "precision": precision_score(y_test, baseline_predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, baseline_predictions, average="weighted", zero_division=0),
            "macro_f1": f1_score(y_test, baseline_predictions, average="macro", zero_division=0),
        }
    ]
    per_class: dict[str, dict[str, dict[str, float]]] = {}
    labels = sorted(set(y_test))
    for name, model in result["models"].items():
        predictions = model.predict(x_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
            }
        )
        labels = sorted(set(y_test) | set(predictions))
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        class_report = cast(
            dict[str, dict[str, float]],
            classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        )
        per_class[name] = {
            label: {
                metric: float(class_report.get(label, {}).get(metric, 0.0))
                for metric in ("precision", "recall", "f1-score")
            }
            for label in labels
        }
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Actual", title=f"{name} confusion matrix")
        for row in range(len(labels)):
            for col in range(len(labels)):
                axis.text(col, row, matrix[row, col], ha="center", va="center")
        figure.tight_layout()
        figure.savefig(output / f"{name}_confusion_matrix.png", dpi=140)
        plt.close(figure)
    report = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    report.to_csv(output / "model_comparison.csv", index=False)
    metadata = dict(result.get("metadata", {}))
    metadata["evaluation"] = {
        "test_rows": int(len(y_test)),
        "classes": labels,
        "per_class": per_class,
        "models_evaluated": list(report["model"]),
    }
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.synthetic == bool(args.data_dir):
        parser.error("Choose exactly one of --synthetic or --data-dir.")
    data = make_synthetic_interactions() if args.synthetic else load_interactions(args.data_dir)
    result = train_models(data, args.model_dir)
    print(evaluate_models(result, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
