"""SHAP explanations with a safe model-importance fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def feature_importance(model: Any, x: pd.DataFrame) -> pd.DataFrame:
    """Return global importance, using SHAP when available and compatible."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(x)
    try:
        import shap
        from shap.utils._exceptions import ExplainerError
    except ImportError:
        return _fallback_importance(preprocessor, estimator)
    try:
        explainer = shap.Explainer(estimator, transformed)
        values: Any = explainer(transformed)
        importance = np.abs(values.values).mean(axis=0)
        names = preprocessor.get_feature_names_out()
        return pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False)
    except (AttributeError, TypeError, ValueError, RuntimeError, ExplainerError):
        return _fallback_importance(preprocessor, estimator)


def _fallback_importance(preprocessor: Any, estimator: Any) -> pd.DataFrame:
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(
        {"feature": preprocessor.get_feature_names_out(), "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)


def explain_prediction(model: Any, x: pd.DataFrame) -> str:
    """Produce a student-readable, non-causal explanation from top features."""
    importance = feature_importance(model, x)
    if importance.empty:
        return "A feature-level explanation is unavailable for this model."
    top = ", ".join(importance.head(3)["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False))
    return f"The prediction is most associated with these observed inputs: {top}. These are associations, not causes."
