"""Loading and validation helpers for EdNet KT2-compatible data."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import zipfile

import numpy as np
import pandas as pd

KT2_COLUMNS = ["timestamp", "action_type", "item_id", "source", "user_answer", "platform"]
ACTION_TYPES = ("enter", "respond", "submit")


@dataclass(frozen=True)
class DataQualityReport:
    """Bounded-load diagnostics for a dataset selection."""

    files_discovered: int
    files_processed: int
    files_skipped: int
    rows_read: int
    rows_returned: int
    invalid_timestamps: int
    unsupported_actions: int
    duplicate_rows_removed: int
    missing_values: dict[str, int]
    errors: tuple[str, ...] = ()


def _read_csv(path: Path, user_id: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "user_id" not in frame.columns:
        frame["user_id"] = user_id or path.stem
    return frame


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KT2_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required KT2 columns: {', '.join(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce")
    for column in ["action_type", "item_id", "source", "user_answer", "platform", "user_id"]:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    result = result.dropna(subset=["timestamp"])
    result = result[result["action_type"].isin(ACTION_TYPES)]
    result["timestamp"] = result["timestamp"].astype("int64")
    result = result.drop_duplicates(subset=["user_id", "timestamp", "action_type", "item_id"])
    return result.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def _normalise_with_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows = len(frame)
    if "timestamp" in frame.columns:
        numeric_timestamp = pd.to_numeric(frame["timestamp"], errors="coerce")
        invalid_timestamps = int(numeric_timestamp.isna().sum())
    else:
        invalid_timestamps = original_rows
    unsupported_actions = (
        int((~frame["action_type"].astype(str).isin(ACTION_TYPES)).sum())
        if "action_type" in frame.columns
        else original_rows
    )
    result = _normalise(frame)
    return result, {
        "rows_read": original_rows,
        "invalid_timestamps": invalid_timestamps,
        "unsupported_actions": unsupported_actions,
        "duplicate_rows_removed": max(0, original_rows - len(result)),
    }


def load_interactions_with_report(
    path: str | Path,
    max_users: int | None = 50,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load bounded KT2 records and return data-quality diagnostics.

    Files with an invalid schema are skipped individually so one bad student file
    does not prevent a bounded exploratory load of the remaining dataset.
    """
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be positive or None for unlimited loading.")
    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"EdNet path does not exist: {location}")
    if location.is_file() and location.suffix.lower() == ".zip":
        members = [name for name in zipfile.ZipFile(location).namelist() if name.lower().endswith(".csv")]
        sources = members[:max_users] if max_users is not None else members
        discovered = len(members)
    elif location.is_file():
        sources = [location]
        discovered = 1
    elif location.is_dir():
        files = sorted(location.rglob("*.csv"))
        sources = files[:max_users] if max_users is not None else files
        discovered = len(files)
    else:
        raise ValueError(f"Unsupported EdNet path: {location}")
    if not sources:
        raise ValueError(f"No CSV records found at {location}")

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    rows_read = invalid_timestamps = unsupported_actions = duplicates = 0
    skipped = 0
    missing_values: dict[str, int] = {}
    archive = zipfile.ZipFile(location) if location.is_file() and location.suffix.lower() == ".zip" else None
    try:
        for source in sources:
            try:
                if archive is not None:
                    with archive.open(str(source)) as stream:
                        frame = pd.read_csv(stream)
                    user_id = Path(str(source)).stem
                else:
                    source_path = Path(source)
                    frame = _read_csv(source_path)
                    user_id = source_path.stem
                if "user_id" not in frame.columns:
                    frame["user_id"] = user_id
                missing = [column for column in KT2_COLUMNS if column not in frame.columns]
                if missing:
                    raise ValueError(f"missing columns: {', '.join(missing)}")
                for column in KT2_COLUMNS:
                    missing_values[column] = missing_values.get(column, 0) + int(frame[column].isna().sum())
                normalised, stats = _normalise_with_report(frame)
                rows_read += stats["rows_read"]
                invalid_timestamps += stats["invalid_timestamps"]
                unsupported_actions += stats["unsupported_actions"]
                duplicates += stats["duplicate_rows_removed"]
                frames.append(normalised)
            except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
                skipped += 1
                errors.append(f"{Path(str(source)).name}: {exc}")
    finally:
        if archive is not None:
            archive.close()
    if not frames:
        raise ValueError(f"No valid KT2 CSV records found at {location}. Errors: {'; '.join(errors[:3])}")
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    if max_rows is not None:
        result = result.head(max_rows).copy()
    report = DataQualityReport(
        files_discovered=discovered,
        files_processed=len(frames),
        files_skipped=skipped,
        rows_read=rows_read,
        rows_returned=len(result),
        invalid_timestamps=invalid_timestamps,
        unsupported_actions=unsupported_actions,
        duplicate_rows_removed=duplicates,
        missing_values=missing_values,
        errors=tuple(errors[:20]),
    )
    return result, report


def load_interactions(
    path: str | Path,
    max_users: int | None = 50,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load bounded KT2 records from a directory, CSV, or ZIP archive."""
    result, _ = load_interactions_with_report(path, max_users, max_rows)
    return result


def make_synthetic_interactions(
    n_students: int = 12,
    events_per_student: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create clearly synthetic KT2-shaped records for tests and demonstrations."""
    if n_students < 2 or events_per_student < 6:
        raise ValueError("Synthetic data needs at least 2 students and 6 events per student.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    actions = np.array(ACTION_TYPES)
    sources = np.array(["sprint", "tutor", "in_review"])
    platforms = np.array(["mobile", "web"])
    for student in range(n_students):
        timestamp = 1_600_000_000_000 + student * 10_000_000
        for event in range(events_per_student):
            # A deterministic but non-trivial sequence gives every class examples.
            action = actions[(event + student) % len(actions)]
            rows.append(
                {
                    "user_id": f"synthetic_{student:03d}",
                    "timestamp": timestamp + event * int(rng.integers(1_000, 20_000)),
                    "action_type": action,
                    "item_id": f"{'b' if action != 'respond' else 'q'}{(student * 97 + event) % 6000}",
                    "source": sources[(student + event // 4) % len(sources)],
                    "user_answer": str(rng.choice(["", "a", "b", "c", "d"])) if action == "respond" else "",
                    "platform": platforms[(student + event) % len(platforms)],
                }
            )
    return _normalise(pd.DataFrame(rows))
