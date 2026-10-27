"""Leakage-aware target construction."""

from __future__ import annotations

import pandas as pd

from .data_loader import ACTION_TYPES


def create_target(frame: pd.DataFrame) -> pd.Series:
    """Return the observed current KT2 action type as the prediction target."""
    if "action_type" not in frame.columns:
        raise ValueError("Expected an action_type column.")
    target = frame["action_type"].astype(str)
    invalid = sorted(set(target) - set(ACTION_TYPES))
    if invalid:
        raise ValueError(f"Unsupported action types: {invalid}")
    return target.rename("target_action")
