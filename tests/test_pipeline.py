import pandas as pd
import pytest

from src.data_loader import (
    load_interactions,
    load_interactions_with_report,
    make_synthetic_interactions,
)
from src.evaluate import evaluate_models
from src.feature_engineering import build_features
from src.predict import validate_prediction_input
from src.target_creation import create_target
from src.train import train_models


def test_synthetic_loader_has_kt2_schema():
    data = make_synthetic_interactions(n_students=4, events_per_student=8)
    assert {"user_id", "timestamp", "action_type", "item_id", "source", "platform"} <= set(data.columns)
    assert data["action_type"].isin(["enter", "respond", "submit"]).all()


def test_loader_drops_invalid_duplicates(tmp_path):
    path = tmp_path / "student.csv"
    pd.DataFrame(
        [
            {"timestamp": "2", "action_type": "enter", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
            {"timestamp": "2", "action_type": "enter", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
            {"timestamp": "1", "action_type": "unknown", "item_id": "q1", "source": "sprint", "user_answer": "", "platform": "web"},
        ]
    ).to_csv(path, index=False)
    loaded = load_interactions(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["timestamp"] == 2


def test_loader_reports_missing_path():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_interactions("data/raw/path-that-does-not-exist")


def test_loader_uses_default_limit_for_directory(tmp_path):
    for index in range(60):
        pd.DataFrame(
            [{
                "timestamp": str(index),
                "action_type": "enter",
                "item_id": f"b{index}",
                "source": "sprint",
                "user_answer": "",
                "platform": "web",
            }]
        ).to_csv(tmp_path / f"student_{index:02d}.csv", index=False)
    loaded, report = load_interactions_with_report(tmp_path)
    assert report.files_discovered == 60
    assert report.files_processed == 50
    assert len(loaded) == 50
    assert load_interactions_with_report(tmp_path, max_users=None)[1].files_processed == 60


def test_loader_reports_quality_and_skips_bad_files(tmp_path):
    valid = tmp_path / "u1.csv"
    pd.DataFrame(
        [
            {"timestamp": "3", "action_type": "enter", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
            {"timestamp": "4", "action_type": "respond", "item_id": "q1", "source": "sprint", "user_answer": "a", "platform": "web"},
        ]
    ).to_csv(valid, index=False)
    pd.DataFrame([{"timestamp": 1, "action_type": "enter"}]).to_csv(tmp_path / "bad.csv", index=False)
    loaded, report = load_interactions_with_report(tmp_path)
    assert len(loaded) == 2
    assert report.files_discovered == 2
    assert report.files_processed == 1
    assert report.files_skipped == 1


def test_features_use_prior_history_only():
    data = make_synthetic_interactions(n_students=2, events_per_student=8)
    features = build_features(data)
    assert features.iloc[0]["prior_actions"] == 0
    assert features.iloc[1]["prior_actions"] == 1


def test_target_creation_rejects_unknown_labels():
    with pytest.raises(ValueError):
        create_target(pd.DataFrame({"action_type": ["enter", "future_label"]}))


def test_training_evaluation_and_prediction(tmp_path):
    data = make_synthetic_interactions(n_students=8, events_per_student=12)
    result = train_models(data, tmp_path / "models")
    report = evaluate_models(result, tmp_path / "reports")
    assert {"accuracy", "precision", "recall", "macro_f1"} <= set(report.columns)
    assert "majority_baseline" in set(report["model"])
    assert (tmp_path / "reports" / "evaluation_metadata.json").exists()
    assert len(report) >= 3


def test_invalid_prediction_input_is_rejected():
    with pytest.raises(ValueError, match="Missing prediction features"):
        validate_prediction_input({})
