"""
Scheduled Retraining Pipeline — Week 80 (H11-H12).

Uses Prefect 2 local worker.  Prefect is the recommended orchestrator for
solo-dev setups (Airflow is overkill at this scale).

Flow steps
----------
1. fetch_latest_data   — load OHLCV DataFrame from configured source
2. compute_features    — apply feature engineering to raw OHLCV
3. train_model         — RL training, save checkpoint to disk
4. walkforward_eval    — purged K-fold gate check (H10)
5. register_staging    — write to ModelRegistry as "staging" stage

Automatic promotion is intentionally disabled: G5 requires manual
``scripts/promote_model.py`` invocation after human review.

Integration with RetrainingTrigger (Week 67 / S58)
---------------------------------------------------
    from deployment.monitoring.retraining_trigger import RetrainingTrigger
    from training.pipelines.retrain_flow import make_retrain_callback

    trigger = RetrainingTrigger(
        config={"drawdown_trigger_pct": 0.15},
        on_trigger=make_retrain_callback(config),
    )
    # When trigger fires inside the trading loop → Prefect flow runs in background.

Running manually
----------------
    python -m training.pipelines.retrain_flow           # uses default config
    prefect run -p training/pipelines/retrain_flow.py   # via Prefect CLI
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Prefect import (optional — graceful degradation when not installed)
# ---------------------------------------------------------------------------
try:
    from prefect import flow, task, get_run_logger
    from prefect.states import Completed, Failed
    HAS_PREFECT = True
except ImportError:  # pragma: no cover
    HAS_PREFECT = False

    def task(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when Prefect is not installed."""
        def _dec(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _dec

    def flow(*args, **kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _dec

    def get_run_logger():  # type: ignore[misc]
        return logging.getLogger("retrain_flow")


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "source": "csv",
        "path": "test_data.csv",
        "symbol": "BTC/USDT",
    },
    "features": {
        "window": 20,
    },
    "training": {
        "agent_type": "ppo",
        "total_timesteps": 10_000,
        "checkpoint_dir": "checkpoints/retrain",
    },
    "walkforward": {
        "n_splits": 6,
        "embargo_bars": 20,
        "total_timesteps": 5_000,
    },
    "registry": {
        "registry_dir": "model_registry",
    },
}


# ---------------------------------------------------------------------------
# Task: fetch_latest_data
# ---------------------------------------------------------------------------

@task(name="fetch-latest-data", retries=2, retry_delay_seconds=30)
def fetch_latest_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data from the configured source.

    Supports:
    - ``source: "csv"``   — local CSV file (default; for offline/testing)
    - ``source: "ccxt"``  — live exchange via CCXTLiveDataSource (Track F)
    - ``source: "yfinance"`` — YFinance fallback

    Returns a DataFrame with lowercase columns: open, high, low, close, volume.
    """
    log = get_run_logger() if HAS_PREFECT else logger
    data_cfg = config.get("data", {})
    source = data_cfg.get("source", "csv")

    if source == "csv":
        path = data_cfg.get("path", "test_data.csv")
        full_path = Path(path) if Path(path).is_absolute() else _ROOT / path
        log.info("Loading CSV data from %s", full_path)
        df = pd.read_csv(full_path)

    elif source == "ccxt":
        try:
            from data.sources.ccxt_live import CCXTLiveDataSource
            src = CCXTLiveDataSource(config=data_cfg)
            df = src.fetch_ohlcv(
                symbol=data_cfg.get("symbol", "BTC/USDT"),
                limit=data_cfg.get("limit", 500),
            )
            log.info("Fetched %d OHLCV rows via CCXT", len(df))
        except Exception as exc:
            log.warning("CCXT fetch failed (%s); falling back to CSV", exc)
            path = data_cfg.get("fallback_csv", "test_data.csv")
            df = pd.read_csv(_ROOT / path)

    elif source == "yfinance":
        import yfinance as yf
        ticker = data_cfg.get("ticker", "BTC-USD")
        period = data_cfg.get("period", "6mo")
        df = yf.download(ticker, period=period, auto_adjust=True)
        df.columns = [c.lower() for c in df.columns]
        log.info("Fetched %d rows from yfinance (%s)", len(df), ticker)

    else:
        raise ValueError(f"Unknown data source: {source!r}")

    # Normalise column names
    df.columns = [c.lower().lstrip("$") for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")

    log.info("Data loaded: %d rows, columns=%s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Task: compute_features
# ---------------------------------------------------------------------------

@task(name="compute-features")
def compute_features(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply feature engineering to raw OHLCV data.

    Uses the project's FeatureRegistry (H9) to compute and record features.
    Falls back to a minimal RSI/returns set if the full pipeline isn't available.
    """
    log = get_run_logger() if HAS_PREFECT else logger
    feat_cfg = config.get("features", {})

    try:
        from training.data.feature_engineering import compute_features as _compute
        df = _compute(data, config=feat_cfg)
        log.info("Features computed via feature_engineering: %d columns", len(df.columns))
    except Exception as exc:
        log.warning("Full feature pipeline failed (%s); using minimal feature set", exc)
        df = _minimal_features(data, window=feat_cfg.get("window", 20))
        log.info("Minimal features computed: %d columns", len(df.columns))

    df = df.dropna().reset_index(drop=True)

    try:
        from training.features.registry import FeatureRegistry
        reg = FeatureRegistry()
        for col in df.columns:
            if col not in {"open", "high", "low", "close", "volume"}:
                reg.register(name=col, compute_fn=None, input_schema=list(data.columns))
    except Exception:
        pass  # registry is best-effort

    return df


def _minimal_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Fallback: compute returns and RSI without external dependencies."""
    import numpy as np
    out = df.copy()
    out["returns"] = out["close"].pct_change()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["sma"] = out["close"].rolling(window).mean()
    out["vol_ratio"] = out["volume"] / (out["volume"].rolling(window).mean() + 1e-8)
    return out


# ---------------------------------------------------------------------------
# Task: train_model
# ---------------------------------------------------------------------------

@task(name="train-model", timeout_seconds=3600)
def train_model(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train an RL agent and save a checkpoint.

    Returns
    -------
    dict with keys:
        checkpoint_path : str   — path to saved model file
        metrics         : dict  — training metrics (sharpe, etc.)
        version_tag     : str   — timestamp-based tag
    """
    log = get_run_logger() if HAS_PREFECT else logger
    train_cfg = config.get("training", {})
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/retrain"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    version_tag = f"retrain_{int(time.time())}"
    checkpoint_path = checkpoint_dir / f"{version_tag}.zip"

    log.info("Starting training — timesteps=%d", train_cfg.get("total_timesteps", 10_000))

    try:
        from training.env_factory import create_env
        from agents.strategies.agent_factory import create_agent

        split = int(len(data) * 0.8)
        train_df = data.iloc[:split].reset_index(drop=True)
        eval_df = data.iloc[split:].reset_index(drop=True)

        env = create_env(config, train_df)
        agent_type = train_cfg.get("agent_type", "ppo")
        agent = create_agent(
            agent_type=agent_type,
            config=config.get("agent", {}),
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        agent.learn(total_timesteps=train_cfg.get("total_timesteps", 10_000))
        agent.save(str(checkpoint_path))

        # Quick eval
        eval_env = create_env(config, eval_df)
        obs, _ = eval_env.reset()
        rewards = []
        for _ in range(min(len(eval_df), 200)):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        import numpy as np
        mean_r = float(np.mean(rewards)) if rewards else 0.0
        std_r = float(np.std(rewards)) if rewards else 1.0
        sharpe = mean_r / (std_r + 1e-8) * (252 ** 0.5)
        metrics = {"train_sharpe": round(sharpe, 4), "mean_reward": round(mean_r, 6)}

    except Exception as exc:
        log.warning("Full training failed (%s); writing placeholder checkpoint", exc)
        checkpoint_path.write_text(f"placeholder:{version_tag}")
        metrics = {"train_sharpe": 0.0, "mean_reward": 0.0, "error": str(exc)}

    log.info("Training complete — checkpoint=%s metrics=%s", checkpoint_path, metrics)
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "version_tag": version_tag,
    }


# ---------------------------------------------------------------------------
# Task: walkforward_eval
# ---------------------------------------------------------------------------

@task(name="walkforward-eval")
def walkforward_eval(
    checkpoint_info: Dict[str, Any],
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run purged K-fold walk-forward evaluation (H10).

    Returns
    -------
    dict with keys:
        passes  : bool
        report  : WalkForwardReport (serialised to dict)
        summary : str
    """
    log = get_run_logger() if HAS_PREFECT else logger
    wf_cfg = config.get("walkforward", {})

    log.info("Starting walk-forward evaluation — n_splits=%d", wf_cfg.get("n_splits", 6))

    try:
        from training.evaluation.walkforward import evaluate_for_promotion
        from training.env_factory import create_env
        from agents.strategies.agent_factory import create_agent

        train_cfg = config.get("training", {})

        def agent_factory():
            env = create_env(config, data.iloc[:10])
            return create_agent(
                agent_type=train_cfg.get("agent_type", "ppo"),
                config=config.get("agent", {}),
                observation_space=env.observation_space,
                action_space=env.action_space,
            )

        def env_factory(df):
            return create_env(config, df)

        report_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/retrain"))
        report_path = report_dir / f"{checkpoint_info['version_tag']}_wf_report.json"

        report = evaluate_for_promotion(
            agent_factory=agent_factory,
            env_factory=env_factory,
            data=data,
            n_splits=wf_cfg.get("n_splits", 6),
            gap_bars=wf_cfg.get("embargo_bars", 20),
            total_timesteps=wf_cfg.get("total_timesteps", 5_000),
            report_path=str(report_path),
        )

        passes = report.passes_staging_gate()
        summary = report.summary_line()
        log.info("Walk-forward result: %s", summary)

        return {
            "passes": passes,
            "report": report.to_dict(),
            "summary": summary,
            "report_path": str(report_path),
        }

    except Exception as exc:
        log.warning("Walk-forward eval failed (%s); marking as not passing", exc)
        return {
            "passes": False,
            "report": {},
            "summary": f"EVAL_ERROR: {exc}",
            "report_path": None,
        }


# ---------------------------------------------------------------------------
# Task: register_staging
# ---------------------------------------------------------------------------

@task(name="register-staging")
def register_staging(
    checkpoint_info: Dict[str, Any],
    eval_result: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[int]:
    """Register the trained model as a staging candidate in ModelRegistry.

    Only registers if the walk-forward gate passes.  Automatic promotion
    to canary/prod is intentionally forbidden here (G5 manual gate).

    Returns the VersionID if registered, None otherwise.
    """
    log = get_run_logger() if HAS_PREFECT else logger

    if not eval_result.get("passes", False):
        log.warning(
            "Walk-forward gate FAILED (%s) — model NOT registered",
            eval_result.get("summary", ""),
        )
        return None

    try:
        from training.registry.model_registry import ModelRegistry

        reg_cfg = config.get("registry", {})
        registry_dir = reg_cfg.get("registry_dir", "model_registry")
        registry = ModelRegistry(registry_dir=registry_dir)

        metrics = {
            **checkpoint_info.get("metrics", {}),
            "walkforward_summary": eval_result.get("summary", ""),
            "walkforward_passes": True,
        }

        version_id = registry.register(
            model_path=checkpoint_info["checkpoint_path"],
            name=f"retrain_{checkpoint_info['version_tag']}",
            metrics=metrics,
            tags={"pipeline": "retrain_flow", "source": "scheduled"},
        )

        registry.promote(
            version=version_id,
            to_stage="staging",
            actor="retrain_flow",
            reason=eval_result.get("summary", "automated retrain"),
        )

        log.info(
            "Model registered as staging — version=%s checkpoint=%s",
            version_id,
            checkpoint_info["checkpoint_path"],
        )
        return int(version_id)

    except Exception as exc:
        log.error("Registry staging failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Flow: retrain_flow
# ---------------------------------------------------------------------------

@flow(name="retrain-flow", log_prints=True)
def retrain_flow(
    config: Optional[Dict[str, Any]] = None,
    trigger_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end retraining pipeline.

    Parameters
    ----------
    config : dict, optional
        Full pipeline config.  Defaults to ``_DEFAULT_CONFIG``.
    trigger_event : dict, optional
        Serialised ``RetrainingEvent`` that initiated this run (for audit).

    Returns
    -------
    dict with keys:
        status          : "registered" | "gate_failed" | "error"
        version_id      : int or None
        eval_summary    : str
        trigger_event   : dict or None
    """
    log = get_run_logger() if HAS_PREFECT else logger
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    log.info(
        "retrain_flow started — trigger=%s",
        trigger_event.get("condition", "manual") if trigger_event else "manual",
    )

    try:
        data = fetch_latest_data(cfg)
        featured_data = compute_features(data, cfg)
        checkpoint_info = train_model(featured_data, cfg)
        eval_result = walkforward_eval(checkpoint_info, featured_data, cfg)
        version_id = register_staging(checkpoint_info, eval_result, cfg)

        status = "registered" if version_id is not None else "gate_failed"
        log.info("retrain_flow complete — status=%s version=%s", status, version_id)

        return {
            "status": status,
            "version_id": version_id,
            "eval_summary": eval_result.get("summary", ""),
            "trigger_event": trigger_event,
        }

    except Exception as exc:
        log.error("retrain_flow failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "version_id": None,
            "eval_summary": str(exc),
            "trigger_event": trigger_event,
        }


# ---------------------------------------------------------------------------
# Callback factory for RetrainingTrigger integration
# ---------------------------------------------------------------------------

def make_retrain_callback(
    config: Optional[Dict[str, Any]] = None,
) -> "Callable[[Any], None]":
    """Return an ``on_trigger`` callback suitable for :class:`RetrainingTrigger`.

    The callback launches ``retrain_flow`` in a daemon thread so it does not
    block the trading loop.  Overlapping runs are guarded by a lock — if a
    retrain is already in progress, the new trigger is logged and dropped.

    Parameters
    ----------
    config : dict, optional
        Pipeline config forwarded to ``retrain_flow``.

    Returns
    -------
    Callable[[RetrainingEvent], None]
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    _lock = threading.Lock()
    _running = [False]

    def _callback(event: Any) -> None:
        if not _lock.acquire(blocking=False):
            logger.warning(
                "retrain_flow already running; skipping trigger %s", event
            )
            return

        def _run():
            try:
                _running[0] = True
                retrain_flow(
                    config=cfg,
                    trigger_event=event.to_dict() if hasattr(event, "to_dict") else {},
                )
            finally:
                _running[0] = False
                _lock.release()

        t = threading.Thread(target=_run, name="retrain_flow", daemon=True)
        t.start()
        logger.info("retrain_flow launched in background (trigger=%s)", event)

    return _callback


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Run retrain_flow manually")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (optional)",
    )
    args = parser.parse_args()

    run_config = None
    if args.config:
        run_config = _json.loads(Path(args.config).read_text())

    result = retrain_flow(config=run_config)
    print(_json.dumps(result, indent=2, default=str))
