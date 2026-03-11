"""
Paper Trading page.
Connects to the RL environment, loads a trained SB3 model, and runs a simulation.
No async/await — single-step advance on button press or auto-play via st.rerun().
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "checkpoints",
)
DEFAULT_DATA_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "test_data.csv",
)

# ── Pure helper functions ─────────────────────────────────────────────────────


def get_available_checkpoints(checkpoint_dir: str) -> List[str]:
    """Return sorted list of SB3 model checkpoint files (*.zip).

    Includes only .zip files (SB3 save format), sorted by name descending
    so latest checkpoints appear first.
    """
    base = Path(checkpoint_dir)
    if not base.exists():
        return []
    files = sorted(base.glob("**/*.zip"), reverse=True)
    return [str(f) for f in files]


def format_action(action: Any) -> str:
    """Convert a continuous action value to a human-readable string.

    SB3 continuous actions are typically in [-1, 1].
    Convention used by SingleAssetRLTradingEnv:
        > +0.1  → BUY  (fraction of action)
        < -0.1  → SELL (fraction of |action|)
        else    → HOLD
    """
    try:
        val = float(np.squeeze(action))
    except (TypeError, ValueError):
        return "HOLD"
    if val > 0.1:
        return f"BUY  ({val:.3f})"
    if val < -0.1:
        return f"SELL ({val:.3f})"
    return "HOLD"


def compute_portfolio_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics from a list of step records.

    Args:
        history: list of {"step": int, "portfolio_value": float,
                          "action": float, "reward": float}.

    Returns:
        dict with keys: total_return, max_drawdown, sharpe, win_rate,
        n_trades, best_step_return, worst_step_return.
    """
    if not history:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "best_step_return": 0.0,
            "worst_step_return": 0.0,
        }

    values = np.array([h["portfolio_value"] for h in history], dtype=float)
    rewards = np.array([h.get("reward", 0.0) for h in history], dtype=float)
    actions = np.array([h.get("action", 0.0) for h in history], dtype=float)

    initial = values[0] if values[0] > 0 else 1.0
    total_return = (values[-1] - initial) / initial

    # Max drawdown
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / np.maximum(peak, 1e-9)
    max_drawdown = float(drawdowns.max())

    # Sharpe (annualised assuming daily steps)
    if len(rewards) > 1:
        r_mean = rewards.mean()
        r_std = rewards.std(ddof=1)
        sharpe = float(r_mean / (r_std + 1e-8) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Trade count and win rate (non-hold actions)
    trade_mask = np.abs(actions) > 0.1
    n_trades = int(trade_mask.sum())
    step_returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
    win_rate = float((step_returns > 0).mean()) if len(step_returns) > 0 else 0.0

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "best_step_return": float(step_returns.max()) if len(step_returns) > 0 else 0.0,
        "worst_step_return": float(step_returns.min()) if len(step_returns) > 0 else 0.0,
    }


def build_paper_trading_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """Build a combined portfolio-value + reward chart from simulation history.

    Returns an empty figure if history is empty.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Portfolio Value", "Step Reward"),
    )

    if not history:
        fig.update_layout(title="Paper Trading Simulation", height=450)
        return fig

    df = pd.DataFrame(history).sort_values("step")

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["portfolio_value"],
            mode="lines",
            name="Portfolio Value",
            line={"color": "#2980b9", "width": 2},
        ),
        row=1,
        col=1,
    )

    # Reward
    if "reward" in df.columns:
        colors = ["#27ae60" if r >= 0 else "#e74c3c" for r in df["reward"]]
        fig.add_trace(
            go.Bar(
                x=df["step"],
                y=df["reward"],
                name="Reward",
                marker_color=colors,
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Paper Trading Simulation",
        height=450,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
    )
    return fig


def _load_data(data_path: str) -> Optional[pd.DataFrame]:
    """Load OHLCV CSV. Returns None on failure."""
    try:
        df = pd.read_csv(data_path)
        required = {"$open", "$high", "$low", "$close", "$volume"}
        if not required.issubset(df.columns):
            logger.warning("Data missing required columns: %s", required - set(df.columns))
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load data from %s: %s", data_path, exc)
        return None


def _load_sb3_model(model_path: str, env: Any) -> Optional[Any]:
    """Load an SB3 model from *model_path*. Returns None on failure."""
    try:
        from stable_baselines3 import PPO, SAC, TD3, A2C  # noqa: F401
        import zipfile

        # Detect algo type from filename heuristic
        name_lower = Path(model_path).stem.lower()
        if "sac" in name_lower:
            cls = SAC
        elif "td3" in name_lower:
            cls = TD3
        elif "a2c" in name_lower:
            cls = A2C
        else:
            cls = PPO

        model = cls.load(model_path, env=env)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load model from %s: %s", model_path, exc)
        return None


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_paper_trading() -> None:
    """Render the Paper Trading Streamlit page (synchronous)."""
    import streamlit as st  # local import so module is mockable in tests

    st.title("Paper Trading")

    # ── Sidebar ───────────────────────────────────────────────────────────
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint directory",
        value=st.session_state.get("pt_checkpoint_dir", DEFAULT_CHECKPOINT_DIR),
    )
    st.session_state["pt_checkpoint_dir"] = checkpoint_dir

    data_path = st.sidebar.text_input(
        "Market data CSV path",
        value=st.session_state.get("pt_data_path", DEFAULT_DATA_PATH),
    )
    st.session_state["pt_data_path"] = data_path

    initial_capital = st.sidebar.number_input(
        "Initial capital ($)", min_value=1000.0, value=10000.0, step=1000.0
    )
    window_size = st.sidebar.slider("Window size", min_value=5, max_value=60, value=20)
    auto_play = st.sidebar.checkbox("Auto-play (step every rerun)", value=False)

    # ── Model selection ───────────────────────────────────────────────────
    checkpoints = get_available_checkpoints(checkpoint_dir)
    use_random = st.sidebar.checkbox("Use random policy (no model)", value=not bool(checkpoints))

    selected_model_path: Optional[str] = None
    if not use_random:
        if not checkpoints:
            st.sidebar.warning("No .zip checkpoints found in the directory.")
        else:
            selected_model_path = st.sidebar.selectbox(
                "Model checkpoint",
                options=checkpoints,
                format_func=lambda p: Path(p).name,
            )

    # ── Controls ──────────────────────────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    start_btn = col_btn1.button("Start / Reset", type="primary")
    step_btn = col_btn2.button("Step →")
    stop_btn = col_btn3.button("Stop")

    # ── State init ────────────────────────────────────────────────────────
    if "pt_history" not in st.session_state:
        st.session_state["pt_history"] = []
    if "pt_env" not in st.session_state:
        st.session_state["pt_env"] = None
    if "pt_obs" not in st.session_state:
        st.session_state["pt_obs"] = None
    if "pt_model" not in st.session_state:
        st.session_state["pt_model"] = None
    if "pt_running" not in st.session_state:
        st.session_state["pt_running"] = False
    if "pt_done" not in st.session_state:
        st.session_state["pt_done"] = False
    if "pt_step" not in st.session_state:
        st.session_state["pt_step"] = 0

    # ── Start / reset ─────────────────────────────────────────────────────
    if start_btn:
        df = _load_data(data_path)
        if df is None:
            st.error(f"Failed to load market data from: {data_path}")
        else:
            try:
                from envs.single_asset_rl_env import SingleAssetRLTradingEnv

                env = SingleAssetRLTradingEnv(
                    data=df,
                    initial_capital=initial_capital,
                    window_size=window_size,
                )
                obs, _ = env.reset()
                model = None
                if selected_model_path:
                    model = _load_sb3_model(selected_model_path, env)
                    if model is None:
                        st.warning("Could not load model — using random policy.")

                st.session_state["pt_env"] = env
                st.session_state["pt_obs"] = obs
                st.session_state["pt_model"] = model
                st.session_state["pt_history"] = [
                    {
                        "step": 0,
                        "portfolio_value": initial_capital,
                        "action": 0.0,
                        "reward": 0.0,
                    }
                ]
                st.session_state["pt_running"] = True
                st.session_state["pt_done"] = False
                st.session_state["pt_step"] = 0
                st.success("Paper trading initialised.")
            except Exception as exc:
                st.error(f"Failed to initialise environment: {exc}")

    if stop_btn:
        st.session_state["pt_running"] = False

    # ── Step logic ────────────────────────────────────────────────────────
    def _do_step() -> None:
        env = st.session_state.get("pt_env")
        obs = st.session_state.get("pt_obs")
        model = st.session_state.get("pt_model")
        if env is None or obs is None:
            return
        if st.session_state.get("pt_done", False):
            return

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs_new, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_num = st.session_state["pt_step"] + 1

        portfolio_value = float(info.get("portfolio_value", initial_capital))
        st.session_state["pt_history"].append(
            {
                "step": step_num,
                "portfolio_value": portfolio_value,
                "action": float(np.squeeze(action)),
                "reward": float(reward),
            }
        )
        st.session_state["pt_obs"] = obs_new
        st.session_state["pt_step"] = step_num
        st.session_state["pt_done"] = done

    if step_btn and st.session_state.get("pt_running"):
        _do_step()

    if auto_play and st.session_state.get("pt_running") and not st.session_state.get("pt_done"):
        _do_step()
        import time
        time.sleep(0.1)
        st.rerun()

    # ── Display ───────────────────────────────────────────────────────────
    history = st.session_state.get("pt_history", [])

    if not history:
        st.info("Press **Start / Reset** to begin a paper trading simulation.")
        return

    # Summary metrics
    mets = compute_portfolio_metrics(history)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{mets['total_return']:.2%}")
    col2.metric("Sharpe", f"{mets['sharpe']:.3f}")
    col3.metric("Max Drawdown", f"{mets['max_drawdown']:.2%}")
    col4.metric("Win Rate", f"{mets['win_rate']:.2%}")

    # Chart
    st.plotly_chart(build_paper_trading_chart(history), use_container_width=True)

    # Last action
    if history:
        last = history[-1]
        st.caption(
            f"Step {last['step']} | "
            f"Action: {format_action(last['action'])} | "
            f"Reward: {last['reward']:.4f}"
        )

    if st.session_state.get("pt_done"):
        st.success("Episode finished.")

    # Trade history table (last 20 rows)
    st.subheader("Recent Steps")
    df_hist = pd.DataFrame(history[-20:])
    df_hist["action_label"] = df_hist["action"].apply(format_action)
    st.dataframe(
        df_hist[["step", "portfolio_value", "action_label", "reward"]],
        use_container_width=True,
        hide_index=True,
    )
