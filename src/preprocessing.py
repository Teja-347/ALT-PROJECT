"""Model preprocessing and chronological student-aware splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import HISTORY_FEATURES

NUMERIC_FEATURES = [column for column in HISTORY_FEATURES if column not in {"source", "platform"}]
CATEGORICAL_FEATURES = ["source", "platform"]


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def split_by_student(
    frame: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split whole students, preserving chronological order within each split."""
    if "user_id" not in frame.columns:
        raise ValueError("Student-aware splitting requires user_id.")
    users = sorted(frame["user_id"].astype(str).unique())
    if len(users) < 2:
        raise ValueError("At least two students are required for a student-aware split.")
    test_count = max(1, int(round(len(users) * test_size)))
    test_users = set(users[-test_count:])
    test_mask = frame["user_id"].astype(str).isin(test_users)
    train_frame = frame.loc[~test_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    return train_frame, test_frame, target.loc[train_frame.index], target.loc[test_frame.index]
