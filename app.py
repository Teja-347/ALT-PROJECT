"""Streamlit dashboard for personalized learning-path prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_loader import (
    DataQualityReport,
    load_interactions_with_report,
    make_synthetic_interactions,
)
from src.evaluate import evaluate_models
from src.explainability import explain_prediction, feature_importance
from src.feature_engineering import build_features
from src.predict import predict_from_values
from src.train import train_models

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EDNET_PATH = PROJECT_ROOT / "data" / "raw" / "ednet-kt2"
MODEL_ROOT = PROJECT_ROOT / "models"
REPORT_ROOT = PROJECT_ROOT / "reports"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --navy: #081426; --panel: #101f36; --indigo: #5b61d6; --cyan: #45d6e8; --muted: #9eacc2; }
        .stApp { background: linear-gradient(135deg, #081426 0%, #0c1930 52%, #111c3a 100%); color: #eef4ff; }
        [data-testid="stSidebar"] { background: #071121; border-right: 1px solid #203354; }
        [data-testid="stSidebar"] .block-container { padding: 2rem 1.25rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        h1 { font-size: 2.25rem !important; margin-bottom: .15rem !important; }
        h2 { font-size: 1.45rem !important; }
        h3 { font-size: 1.05rem !important; color: #dce8ff; }
        .hero { border: 1px solid #263d65; border-radius: 16px; padding: 1.4rem 1.6rem; background: linear-gradient(110deg, #12294b, #18234d); margin: .35rem 0 1.25rem; }
        .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
        .subtitle, .muted { color: var(--muted); }
        .status { display: inline-block; padding: .35rem .7rem; border-radius: 999px; background: #18395b; color: #9debf2; font-size: .78rem; font-weight: 700; }
        .metric-card { border: 1px solid #263d65; border-radius: 12px; background: #101f36; padding: .95rem 1rem; min-height: 92px; }
        .metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; }
        .metric-value { color: #f5f8ff; font-size: 1.55rem; font-weight: 700; margin-top: .35rem; }
        div[data-testid="stTabs"] button { color: #b7c8e5; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color: #75e4ef; border-bottom-color: #45d6e8; }
        .workflow-step { border-left: 2px solid #45d6e8; padding: .2rem .8rem; margin: .5rem 0; color: #cfdbef; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_synthetic() -> pd.DataFrame:
    return make_synthetic_interactions()


@st.cache_data(show_spinner=False)
def _load_ednet(path: str) -> tuple[pd.DataFrame, DataQualityReport]:
    data, report = load_interactions_with_report(path, max_users=50, max_rows=100_000)
    return data, report


@st.cache_resource(show_spinner="Training models on the selected dataset…")
def _prepare_models(data: pd.DataFrame, mode: str) -> dict[str, Any]:
    mode_key = "synthetic" if mode == "Synthetic demonstration" else "ednet"
    model_dir = MODEL_ROOT / mode_key
    report_dir = REPORT_ROOT / mode_key
    result = train_models(data, model_dir, data_mode=mode_key)
    evaluate_models(result, report_dir)
    result["model_dir"] = model_dir
    result["report_dir"] = report_dir
    result["mode"] = mode
    return result


def _load_selected_data(
    mode: str, path: str
) -> tuple[pd.DataFrame | None, DataQualityReport | None, str | None]:
    if mode == "Synthetic demonstration":
        data = _load_synthetic()
        return data, None, None
    location = Path(path).expanduser()
    if not location.exists():
        return None, None, (
            f"EdNet data was not found at `{location}`. Download EdNet KT2 from "
            "the official Riiid repository, extract the student CSV files, and "
            "set the sidebar path to that folder."
        )
    try:
        data, report = _load_ednet(str(location))
        return data, report, None
    except (OSError, ValueError, UnicodeError) as exc:
        return None, None, f"EdNet could not be loaded from `{location}`: {exc}"


def _render_home(mode: str, result: dict[str, Any]) -> None:
    st.subheader("A clearer next step for every learner")
    st.write(
        "This dashboard models the next EdNet KT2 interaction type from a student's "
        "chronological interaction history. It is designed to support exploration "
        "of learning behavior, not to make causal or clinical claims."
    )
    st.markdown("#### Project objectives")
    objectives = st.columns(3)
    objectives[0].write("**Understand**\n\nSummarize engagement and activity patterns.")
    objectives[1].write("**Predict**\n\nCompare models for `enter`, `respond`, and `submit` actions.")
    objectives[2].write("**Explain**\n\nShow feature associations in student-readable language.")
    st.markdown("#### Workflow")
    for step in [
        "Load and validate KT2-compatible interactions",
        "Build features from each student's prior events only",
        "Train and compare available models",
        "Inspect predictions, metrics, and feature importance",
    ]:
        st.markdown(f'<div class="workflow-step">{step}</div>', unsafe_allow_html=True)
    st.caption(f"Current mode: {mode} · {len(result['models'])} trained model(s)")


def _render_analysis(data: pd.DataFrame) -> None:
    st.subheader("Student interaction analysis")
    st.write("Explore the observed records supplied to this dashboard.")
    quality = st.session_state.get("quality")
    if quality is not None:
        st.caption(
            f"Loaded {quality.rows_returned:,} rows from {quality.files_processed:,} files "
            f"({quality.files_skipped:,} skipped; {quality.duplicate_rows_removed:,} duplicates removed)."
        )
    first, second, third, fourth = st.columns(4)
    with first:
        _metric("Students", f"{data['user_id'].nunique():,}")
    with second:
        _metric("Interactions", f"{len(data):,}")
    with third:
        _metric("Unique items", f"{data['item_id'].nunique():,}")
    with fourth:
        _metric("Activities", f"{data['action_type'].nunique():,}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Interaction mix")
        st.bar_chart(data["action_type"].value_counts(), color="#45d6e8")
    with right:
        st.markdown("#### Learning sources")
        st.bar_chart(data["source"].replace("", "unknown").value_counts(), color="#7778e8")
    st.markdown("#### Recent records")
    st.dataframe(data.head(25), width="stretch", hide_index=True)


def _render_prediction(data: pd.DataFrame, result: dict[str, Any]) -> None:
    st.subheader("Learning path prediction")
    st.write("Adjust the observed history inputs to generate a model prediction.")
    features = build_features(data)
    model_name = st.selectbox("Model", sorted(result["models"]), key="prediction_model")
    defaults = features.iloc[-1].to_dict()
    values: dict[str, Any] = {}
    numeric = st.columns(2)
    for index, column in enumerate(features.columns):
        target = numeric[index % 2] if column not in {"source", "platform"} else st
        if column in {"source", "platform"}:
            values[column] = st.text_input(column.replace("_", " ").title(), str(defaults[column]))
        else:
            values[column] = target.number_input(
                column.replace("_", " ").title(),
                value=float(defaults[column]),
                min_value=0.0,
            )
    if st.button("Generate prediction", type="primary"):
        try:
            output = predict_from_values(
                Path(result["model_dir"]) / f"{model_name}.joblib", values
            )
            st.success(f"Predicted interaction: {output['prediction']}")
            if output["probabilities"]:
                st.markdown("#### Class probabilities")
                st.bar_chart(pd.Series(output["probabilities"], name="probability"), color="#45d6e8")
            st.info(
                "This is an association learned from interaction data. It is not proof "
                "that a behavior causes the predicted activity."
            )
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Prediction could not be generated: {exc}")


def _render_evaluation(result: dict[str, Any]) -> None:
    st.subheader("Model evaluation")
    report_dir = Path(result["report_dir"])
    report_path = report_dir / "model_comparison.csv"
    if not report_path.exists():
        st.info("Evaluation metrics are not available yet for this dataset.")
        return
    report = pd.read_csv(report_path)
    if report.empty:
        st.info("The evaluation report contains no model results.")
        return
    best = report.iloc[0]
    first, second, third = st.columns(3)
    with first:
        _metric("Best model", str(best["model"]).replace("_", " ").title())
    with second:
        _metric("Accuracy", f"{best['accuracy']:.1%}")
    with third:
        _metric("Macro F1", f"{best['macro_f1']:.1%}")
    st.dataframe(report.style.format({column: "{:.3f}" for column in report.columns if column != "model"}), width="stretch", hide_index=True)
    st.markdown("#### Confusion matrices")
    columns = st.columns(min(3, len(report)))
    for column, model_name in zip(columns, report["model"]):
        image = report_dir / f"{model_name}_confusion_matrix.png"
        with column:
            st.caption(model_name.replace("_", " ").title())
            if image.exists():
                st.image(str(image), width="stretch")
            else:
                st.info("Matrix unavailable.")


def _render_explainability(result: dict[str, Any]) -> None:
    st.subheader("Explainability")
    model_name = st.selectbox("Explanation model", sorted(result["models"]), key="explanation_model")
    test_frame = result["test_frame"].drop(columns=["user_id"])
    importance = feature_importance(result["models"][model_name], test_frame)
    if importance.empty:
        st.info("Feature-level explanation is unavailable for this model.")
        return
    st.write("Global importance shows which observed inputs the model used most. These associations are not causal explanations.")
    st.bar_chart(importance.head(15).set_index("feature"), color="#45d6e8")
    st.info(explain_prediction(result["models"][model_name], test_frame))


def _dashboard() -> None:
    st.set_page_config(page_title="Learning Path Analytics", page_icon="📘", layout="wide", initial_sidebar_state="expanded")
    _inject_styles()
    with st.sidebar:
        st.markdown('<div class="eyebrow">ALT · Applied ML</div>', unsafe_allow_html=True)
        st.markdown("## Learning Path Analytics")
        st.caption("Personalized interaction modeling")
        st.divider()
        st.markdown("#### Dataset controls")
        mode = st.radio("Data source", ["Synthetic demonstration", "Local EdNet KT2"], index=0)
        path_text = str(DEFAULT_EDNET_PATH)
        if mode == "Local EdNet KT2":
            path_text = st.text_input("EdNet folder or CSV", path_text)
        load_clicked = st.button("Load data", type="primary", width="stretch")
        if load_clicked or "data" not in st.session_state:
            data, quality, error = _load_selected_data(mode, path_text)
            if error:
                st.session_state.pop("data", None)
                st.session_state.pop("result", None)
                st.session_state["load_error"] = error
            else:
                st.session_state["data"] = data
                st.session_state["quality"] = quality
                st.session_state["mode"] = mode
                st.session_state.pop("load_error", None)
        st.divider()
        st.caption("EdNet KT2 target: enter · respond · submit")

    st.markdown(
        '<div class="hero"><div class="eyebrow">Personalized Learning Path Prediction</div>'
        '<h1>Understand the next learning interaction</h1>'
        '<div class="subtitle">A focused analytics workspace for student-system behavior.</div></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("load_error"):
        st.warning(st.session_state["load_error"])
        st.info("Synthetic demonstration remains available: select it in the sidebar and click Load data.")
        return
    data = st.session_state.get("data")
    if data is None:
        st.info("Select Synthetic demonstration in the sidebar and click Load data to begin.")
        return
    mode = st.session_state.get("mode", "Synthetic demonstration")
    status = "SYNTHETIC DATA" if mode == "Synthetic demonstration" else "EDNET KT2 DATA"
    st.markdown(f'<span class="status">{status}</span>', unsafe_allow_html=True)
    result = _prepare_models(data, mode)
    tabs = st.tabs(["Home", "Interaction analysis", "Learning path", "Model evaluation", "Explainability"])
    with tabs[0]:
        _render_home(mode, result)
    with tabs[1]:
        _render_analysis(data)
    with tabs[2]:
        _render_prediction(data, result)
    with tabs[3]:
        _render_evaluation(result)
    with tabs[4]:
        _render_explainability(result)


if __name__ == "__main__":
    _dashboard()
