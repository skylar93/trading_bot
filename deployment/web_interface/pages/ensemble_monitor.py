"""
Ensemble Monitor page.
Shows agent weights over time, regime detection visualisation, per-agent performance.
No async/await.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Pure helper functions ─────────────────────────────────────────────────────

REGIME_LABELS = {0: "Low-vol / Trending", 1: "Medium-vol / Ranging", 2: "High-vol / Crisis"}
REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "checkpoints",
)


def load_ensemble_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load ensemble summary from a JSON checkpoint file.

    Returns a dict with keys:
        weights: {agent_name: float}
        metrics: {agent_name: {sharpe, max_dd, total_return}}
        regime: int (0/1/2) or None
        step: int

    Returns an empty dict if the file is missing or malformed.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("Ensemble checkpoint not found: %s", checkpoint_path)
        return {}

    try:
        with open(path) as f:
            data = json.load(f)
        # Validate minimal structure
        if not isinstance(data, dict):
            logger.warning("Ensemble checkpoint is not a dict: %s", checkpoint_path)
            return {}
        # Fill defaults for optional fields
        data.setdefault("weights", {})
        data.setdefault("metrics", {})
        data.setdefault("regime", None)
        data.setdefault("step", 0)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load ensemble checkpoint %s: %s", checkpoint_path, exc)
        return {}


def build_weights_chart(weights: Dict[str, float]) -> go.Figure:
    """Build a pie chart of ensemble agent weights.

    Args:
        weights: {agent_name: weight_float}.  Weights need not sum to 1.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()
    if not weights:
        fig.update_layout(title="Agent Weights (no data)", height=350)
        return fig

    names = list(weights.keys())
    values = [max(float(v), 0.0) for v in weights.values()]

    fig.add_trace(
        go.Pie(
            labels=names,
            values=values,
            hole=0.35,
            textinfo="label+percent",
        )
    )
    fig.update_layout(title="Ensemble Agent Weights", height=350)
    return fig


def build_regime_timeline(
    regime_history: List[Dict[str, Any]],
) -> go.Figure:
    """Build a colour-coded bar chart of regime over time.

    Args:
        regime_history: list of {"step": int, "regime": int (0/1/2)}.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()
    if not regime_history:
        fig.update_layout(title="Regime History (no data)", height=250)
        return fig

    df = pd.DataFrame(regime_history).sort_values("step")

    for regime_id, label in REGIME_LABELS.items():
        sub = df[df["regime"] == regime_id]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["step"],
                y=[1] * len(sub),
                name=label,
                marker_color=REGIME_COLORS[regime_id],
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Market Regime Timeline",
        barmode="stack",
        xaxis_title="Step",
        yaxis={"visible": False},
        height=250,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.3},
    )
    return fig


def build_agent_performance_table(
    metrics: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Convert per-agent metrics dict into a display DataFrame.

    Args:
        metrics: {agent_name: {sharpe, max_dd, total_return, ...}}.

    Returns:
        DataFrame with columns [Agent, Sharpe, Max DD, Total Return, ...].
        Empty DataFrame if metrics is empty.
    """
    if not metrics:
        return pd.DataFrame(columns=["Agent", "Sharpe", "Max DD (%)", "Total Return (%)"])

    rows = []
    for agent_name, m in metrics.items():
        rows.append(
            {
                "Agent": agent_name,
                "Sharpe": round(float(m.get("sharpe", 0.0)), 3),
                "Max DD (%)": round(float(m.get("max_dd", 0.0)) * 100, 2),
                "Total Return (%)": round(float(m.get("total_return", 0.0)) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return weights normalised to sum to 1 (no-op if sum is 0)."""
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n > 0 else {}
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def get_ensemble_checkpoints(checkpoint_dir: str) -> List[str]:
    """Return sorted list of ensemble JSON checkpoint files."""
    base = Path(checkpoint_dir)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("ensemble_*.json"))


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_ensemble_monitor() -> None:
    """Render the Ensemble Monitor Streamlit page (synchronous)."""
    import streamlit as st  # local import so module is mockable in tests

    st.title("Ensemble Monitor")

    # ── Sidebar ───────────────────────────────────────────────────────────
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint directory",
        value=st.session_state.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR),
    )
    st.session_state["checkpoint_dir"] = checkpoint_dir

    checkpoints = get_ensemble_checkpoints(checkpoint_dir)

    if not checkpoints:
        st.info(
            "No ensemble checkpoints found. "
            f"Train an ensemble agent and save snapshots to `{checkpoint_dir}`."
        )
        return

    selected_ckpt = st.sidebar.selectbox(
        "Checkpoint snapshot",
        options=checkpoints,
        format_func=lambda p: Path(p).name,
    )

    # ── Load data ─────────────────────────────────────────────────────────
    data = load_ensemble_checkpoint(selected_ckpt)
    if not data:
        st.error(f"Could not load checkpoint: {selected_ckpt}")
        return

    weights = data.get("weights", {})
    metrics = data.get("metrics", {})
    regime = data.get("regime")
    step = data.get("step", 0)

    # ── Current regime badge ──────────────────────────────────────────────
    col_info1, col_info2 = st.columns(2)
    col_info1.metric("Training Step", f"{step:,}")
    if regime is not None:
        col_info2.metric("Current Regime", REGIME_LABELS.get(int(regime), "Unknown"))

    st.markdown("---")

    # ── Weights chart + performance table ────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent Weights")
        st.plotly_chart(build_weights_chart(weights), use_container_width=True)

    with col2:
        st.subheader("Per-Agent Performance")
        perf_df = build_agent_performance_table(metrics)
        if not perf_df.empty:
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
        else:
            st.info("No per-agent metrics in this checkpoint.")

    # ── Regime timeline (if history is stored) ────────────────────────────
    regime_history = data.get("regime_history", [])
    if regime_history:
        st.subheader("Regime History")
        st.plotly_chart(build_regime_timeline(regime_history), use_container_width=True)

    # ── Raw data expander ─────────────────────────────────────────────────
    with st.expander("Raw checkpoint data"):
        st.json(data)
