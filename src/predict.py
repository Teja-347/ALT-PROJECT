"""Prediction helpers with strict input validation."""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .feature_engineering import HISTORY_FEATURES


def validate_prediction_input(values: dict[str, Any]) -> pd.DataFrame:
    missing = set(HISTORY_FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing prediction features: {sorted(missing)}")
    frame = pd.DataFrame([values])[HISTORY_FEATURES]
    for column in HISTORY_FEATURES:
        value = values[column]
        if column in {"source", "platform"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{column} must be a non-empty string.")
        elif not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise ValueError(f"{column} must be numeric.")
    if any(float(values[column]) < 0 for column in HISTORY_FEATURES if column.startswith("prior_")):
        raise ValueError("Prior history features cannot be negative.")
    return frame


def predict_from_values(model_path: str | Path, values: dict[str, Any]) -> dict[str, Any]:
    frame = validate_prediction_input(values)
    model = joblib.load(model_path)
    prediction = str(model.predict(frame)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(frame)[0].tolist()))
    return {"prediction": prediction, "probabilities": probabilities}
