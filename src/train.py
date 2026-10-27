"""Train and persist the learning-path models."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from .data_loader import load_interactions, make_synthetic_interactions
from .feature_engineering import build_features
from .preprocessing import make_preprocessor, split_by_student
from .target_creation import create_target


class _XGBoostStringClassifier(ClassifierMixin, BaseEstimator):
    """Adapt XGBoost's numeric-label requirement to the project's string target."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        from xgboost import XGBClassifier

        self.estimator = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.seed,
            n_jobs=2,
        )

    def fit(self, x: Any, y: pd.Series) -> "_XGBoostStringClassifier":
        self.classes_ = np.array(sorted(y.astype(str).unique()))
        encoded = pd.Series(y.astype(str)).map({label: index for index, label in enumerate(self.classes_)})
        self.estimator.fit(x, encoded.to_numpy())
        return self

    def predict(self, x: Any) -> np.ndarray:
        return self.classes_[self.estimator.predict(x).astype(int)]

    def predict_proba(self, x: Any) -> np.ndarray:
        return self.estimator.predict_proba(x)


def _models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=150, random_state=seed, n_jobs=-1, class_weight="balanced"
        ),
    }


def _optional_xgboost(seed: int) -> Any | None:
    if importlib.util.find_spec("xgboost") is None:
        return None
    return _XGBoostStringClassifier(seed)


def prepare_training_data(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered = interactions.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    target = create_target(ordered)
    features = build_features(ordered)
    feature_rows = features.assign(user_id=ordered["user_id"].values)
    train_frame, test_frame, y_train, y_test = split_by_student(feature_rows, target)
    return train_frame, test_frame, y_train, y_test


def train_models(
    interactions: pd.DataFrame,
    output_dir: str | Path = "models",
    seed: int = 42,
    data_mode: str | None = None,
) -> dict[str, object]:
    """Train available models and save pipelines plus a JSON training summary."""
    train_frame, test_frame, y_train, y_test = prepare_training_data(interactions)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train = train_frame.drop(columns=["user_id"])
    fitted: dict[str, object] = {}
    model_names = _models(seed)
    xgb = _optional_xgboost(seed)
    if xgb is not None:
        model_names["xgboost"] = xgb
    for name, estimator in model_names.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, output / f"{name}.joblib")
        fitted[name] = pipeline
    resolved_mode = data_mode or (
        "synthetic" if str(interactions["user_id"].iloc[0]).startswith("synthetic_") else "ednet"
    )
    metadata = {
        "data_mode": resolved_mode,
        "n_rows": int(len(interactions)),
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "classes": sorted(y_train.unique().tolist()),
        "models": sorted(fitted),
        "feature_names": list(x_train.columns),
        "seed": seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"models": fitted, "train_frame": train_frame, "test_frame": test_frame, "y_train": y_train, "y_test": y_test, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-users", type=int, default=50)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.synthetic == bool(args.data_dir):
        parser.error("Choose exactly one of --synthetic or --data-dir.")
    data = make_synthetic_interactions() if args.synthetic else load_interactions(args.data_dir, args.max_users, args.max_rows)
    result = train_models(data, args.output_dir, seed=args.seed)
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
