"""
Training Dashboard page.
Displays live training metrics from MLflow: loss curves, reward, Sharpe, CVaR.
No async/await — polling via st.rerun().
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Pure helper functions (no Streamlit imports) ──────────────────────────────

DEFAULT_TRACKING_URI = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "mlruns",
)

WATCHED_METRICS = [
    "train/reward",
    "eval/mean_reward",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "eval/sharpe",
    "risk/cvar",
    "risk/violation_rate",
]


def get_mlflow_experiments(tracking_uri: str) -> List[Dict[str, Any]]:
    """Return list of active MLflow experiments as dicts.

    Returns [] on any error (mlflow not installed, no experiments, etc.).
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        exps = client.search_experiments()
        return [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in exps
            if e.lifecycle_stage == "active"
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch MLflow experiments: %s", exc)
        return []


def get_available_runs(
    tracking_uri: str, experiment_id: str
) -> List[Dict[str, Any]]:
    """Return runs for *experiment_id* as dicts, newest first."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name or r.info.run_id[:8],
                "status": r.info.status,
                "start_time": r.info.start_time,
            }
            for r in runs
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch runs for experiment %s: %s", experiment_id, exc)
        return []


def get_run_metrics(
    tracking_uri: str,
    run_id: str,
    metric_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame with columns [step, metric, value].

    Fetches history for each key in *metric_keys* (or WATCHED_METRICS).
    Returns an empty DataFrame on any error.
    """
    if metric_keys is None:
        metric_keys = WATCHED_METRICS

    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        rows: List[Dict[str, Any]] = []
        for key in metric_keys:
            try:
                history = client.get_metric_history(run_id, key)
                for m in history:
                    rows.append({"step": m.step, "metric": key, "value": m.value})
            except Exception:  # noqa: BLE001
                pass  # metric simply not logged
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["step", "metric", "value"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch metrics for run %s: %s", run_id, exc)
        return pd.DataFrame(columns=["step", "metric", "value"])


def parse_run_params(run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract display-friendly params dict from a run_info dict (as returned by
    get_available_runs).  Handles missing keys gracefully."""
    return {
        "run_id": run_info.get("run_id", ""),
        "run_name": run_info.get("run_name", "unknown"),
        "status": run_info.get("status", "unknown"),
        "start_time": run_info.get("start_time"),
    }


def build_metric_chart(
    df: pd.DataFrame,
    metric_name: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Build a Plotly line chart for a single metric over training steps.

    Args:
        df: Tidy DataFrame with columns [step, metric, value].
        metric_name: Which metric to filter on (e.g. 'train/reward').
        title: Chart title (defaults to metric_name).

    Returns:
        A Plotly Figure (empty trace if data is absent).
    """
    fig = go.Figure()
    filtered = df[df["metric"] == metric_name] if not df.empty else pd.DataFrame()

    if not filtered.empty:
        filtered = filtered.sort_values("step")
        fig.add_trace(
            go.Scatter(
                x=filtered["step"],
                y=filtered["value"],
                mode="lines",
                name=metric_name,
                line={"width": 2},
            )
        )

    fig.update_layout(
        title=title or metric_name,
        xaxis_title="Step",
        yaxis_title="Value",
        height=300,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def build_multi_metric_chart(
    df: pd.DataFrame,
    metric_names: List[str],
    title: str = "Training Metrics",
) -> go.Figure:
    """Build an overlay chart for multiple metrics (normalised to [-1,1] if scales differ)."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=title, height=350)
        return fig

    for name in metric_names:
        sub = df[df["metric"] == name].sort_values("step")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(x=sub["step"], y=sub["value"], mode="lines", name=name)
        )

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_training_dashboard() -> None:
    """Render the Training Dashboard Streamlit page (synchronous)."""
    import streamlit as st  # local import so module is mockable in tests

    st.title("Training Dashboard")

    # ── Sidebar controls ──────────────────────────────────────────────────
    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        value=st.session_state.get("mlflow_tracking_uri", DEFAULT_TRACKING_URI),
    )
    st.session_state["mlflow_tracking_uri"] = tracking_uri

    auto_refresh = st.sidebar.checkbox("Auto-refresh (10 s)", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (s)", min_value=5, max_value=60, value=10, step=5
    )

    # ── Experiment / run selection ────────────────────────────────────────
    experiments = get_mlflow_experiments(tracking_uri)

    if not experiments:
        st.info("No MLflow experiments found. Start training to populate this dashboard.")
        if auto_refresh:
            import time
            time.sleep(refresh_interval)
            st.rerun()
        return

    exp_names = [e["name"] for e in experiments]
    selected_exp_name = st.selectbox("Experiment", exp_names)
    selected_exp = next(e for e in experiments if e["name"] == selected_exp_name)

    runs = get_available_runs(tracking_uri, selected_exp["experiment_id"])
    if not runs:
        st.info("No runs found for this experiment.")
        return

    run_labels = [f"{r['run_name']} ({r['status']})" for r in runs]
    selected_run_label = st.selectbox("Run", run_labels)
    selected_run = runs[run_labels.index(selected_run_label)]

    # ── Metric display ────────────────────────────────────────────────────
    with st.spinner("Loading metrics…"):
        df = get_run_metrics(tracking_uri, selected_run["run_id"])

    if df.empty:
        st.warning("No metrics logged for this run yet.")
    else:
        available_metrics = df["metric"].unique().tolist()

        # Row 1: reward metrics
        reward_metrics = [m for m in available_metrics if "reward" in m]
        if reward_metrics:
            st.subheader("Reward")
            col_a, col_b = st.columns(2)
            for i, m in enumerate(reward_metrics[:2]):
                (col_a if i == 0 else col_b).plotly_chart(
                    build_metric_chart(df, m), use_container_width=True
                )

        # Row 2: loss metrics
        loss_metrics = [m for m in available_metrics if "loss" in m]
        if loss_metrics:
            st.subheader("Losses")
            cols = st.columns(min(3, len(loss_metrics)))
            for i, m in enumerate(loss_metrics[:3]):
                cols[i].plotly_chart(
                    build_metric_chart(df, m), use_container_width=True
                )

        # Row 3: risk metrics
        risk_metrics = [m for m in available_metrics if "risk" in m or "sharpe" in m]
        if risk_metrics:
            st.subheader("Risk / Sharpe")
            st.plotly_chart(
                build_multi_metric_chart(df, risk_metrics, "Risk Metrics"),
                use_container_width=True,
            )

        # Summary table: latest value per metric
        latest = (
            df.sort_values("step").groupby("metric")["value"].last().reset_index()
        )
        latest.columns = ["Metric", "Latest Value"]
        st.subheader("Latest Values")
        st.dataframe(latest, use_container_width=True, hide_index=True)

    # Run info
    params = parse_run_params(selected_run)
    with st.expander("Run info"):
        st.json(params)

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()
