"""Chronological student-history feature engineering."""

from __future__ import annotations

import pandas as pd

HISTORY_FEATURES = [
    "prior_actions",
    "prior_enters",
    "prior_responds",
    "prior_submits",
    "prior_unique_items",
    "prior_answered",
    "prior_mean_elapsed_ms",
    "current_item_is_question",
    "current_item_is_bundle",
    "source",
    "platform",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features using only rows before each event within each student."""
    required = {"user_id", "timestamp", "action_type", "item_id", "source", "platform"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")
    ordered = frame.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    records: list[dict[str, object]] = []
    for _, history in ordered.groupby("user_id", sort=False):
        counts = {"enter": 0, "respond": 0, "submit": 0}
        prior_items: set[str] = set()
        elapsed: list[float] = []
        for row in history.itertuples(index=False):
            item_id = str(row.item_id)
            records.append(
                {
                    "prior_actions": sum(counts.values()),
                    "prior_enters": counts["enter"],
                    "prior_responds": counts["respond"],
                    "prior_submits": counts["submit"],
                    "prior_unique_items": len(prior_items),
                    "prior_answered": sum(1 for value in elapsed if value >= 0),
                    "prior_mean_elapsed_ms": sum(elapsed) / len(elapsed) if elapsed else 0.0,
                    "current_item_is_question": int(item_id.startswith("q")),
                    "current_item_is_bundle": int(item_id.startswith("b")),
                    "source": str(row.source) or "unknown",
                    "platform": str(row.platform) or "unknown",
                }
            )
            counts[str(row.action_type)] = counts.get(str(row.action_type), 0) + 1
            prior_items.add(item_id)
            value = getattr(row, "elapsed_time", None)
            if value is not None and pd.notna(value):
                elapsed.append(float(value))
    return pd.DataFrame(records, columns=HISTORY_FEATURES)
