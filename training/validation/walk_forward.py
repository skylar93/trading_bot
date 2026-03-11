"""
Walk-Forward Validator for RL trading agents.

Implements expanding/rolling walk-forward validation to detect overfitting
and measure out-of-sample (OOS) generalization.

Design
------
Each fold consists of three non-overlapping windows:
  1. train_window  — agent trains on this data
  2. val_window    — model selection (best checkpoint by val Sharpe)
  3. test_window   — OOS evaluation (never seen during training)

The validator steps forward by `step_size` rows after each fold.

Key metric: Stability Ratio = OOS Sharpe / IS (validation) Sharpe
  > 0.7  → strong generalization
  > 0.5  → decent generalization
  < 0.3  → likely overfitting
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""
    fold_idx: int

    # Window sizes (in rows)
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    # In-sample (validation window) metrics
    val_sharpe: float = 0.0
    val_max_drawdown: float = 0.0
    val_total_return: float = 0.0
    val_n_steps: int = 0

    # Out-of-sample (test window) metrics
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0
    test_total_return: float = 0.0
    test_n_steps: int = 0

    # Stability ratio for this fold
    stability_ratio: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated results across all folds."""
    folds: List[FoldResult] = field(default_factory=list)

    # Aggregate OOS metrics
    oos_sharpe_mean: float = 0.0
    oos_sharpe_std: float = 0.0
    oos_max_drawdown_mean: float = 0.0
    oos_total_return_mean: float = 0.0

    # Aggregate IS (val) metrics
    is_sharpe_mean: float = 0.0
    is_sharpe_std: float = 0.0

    # Stability
    stability_ratio: float = 0.0       # OOS Sharpe / IS Sharpe
    stability_rating: str = "unknown"  # "strong" | "decent" | "overfitting" | "unknown"

    n_folds: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "oos_sharpe_mean": self.oos_sharpe_mean,
            "oos_sharpe_std": self.oos_sharpe_std,
            "oos_max_drawdown_mean": self.oos_max_drawdown_mean,
            "oos_total_return_mean": self.oos_total_return_mean,
            "is_sharpe_mean": self.is_sharpe_mean,
            "is_sharpe_std": self.is_sharpe_std,
            "stability_ratio": self.stability_ratio,
            "stability_rating": self.stability_rating,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_sharpe(portfolio_values: np.ndarray, annualise: bool = True) -> float:
    """Annualised Sharpe ratio from a portfolio-value series (log returns)."""
    if len(portfolio_values) < 2:
        return 0.0
    log_rets = np.diff(np.log(np.maximum(portfolio_values, 1e-10)))
    std = log_rets.std(ddof=1)
    if std < 1e-12:
        return 0.0
    sharpe = log_rets.mean() / std
    if annualise:
        sharpe *= np.sqrt(252)
    return float(sharpe)


def _compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction [0, 1]."""
    if len(portfolio_values) < 2:
        return 0.0
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / np.maximum(peak, 1e-10)
    return float(drawdowns.max())


def _compute_total_return(portfolio_values: np.ndarray) -> float:
    """Total return as a fraction."""
    if len(portfolio_values) < 2:
        return 0.0
    return float((portfolio_values[-1] / portfolio_values[0]) - 1.0)


def _rollout_episode(env, agent, max_steps: Optional[int] = None) -> np.ndarray:
    """
    Run one deterministic episode and return portfolio-value history.

    Supports:
    - SB3AgentWrapper  (predict method)
    - Any object with get_action(obs) -> action
    - Any callable agent(obs) -> action
    """
    obs, _ = env.reset()
    portfolio_values = []
    steps = 0

    while True:
        # Get action
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        elif hasattr(agent, "get_action"):
            action = agent.get_action(obs)
        else:
            action = agent(obs)

        obs, _reward, done, truncated, info = env.step(action)
        pv = info.get("portfolio_value", info.get("net_worth", None))
        if pv is not None:
            portfolio_values.append(float(pv))
        steps += 1
        if done or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    return np.array(portfolio_values) if portfolio_values else np.array([1.0])


# ──────────────────────────────────────────────────────────────────────────────
# WalkForwardValidator
# ──────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Walk-forward validation for RL trading agents.

    Parameters
    ----------
    train_window : int
        Number of rows in each training slice (default: 252 — one trading year).
    val_window : int
        Validation rows immediately following train_window (default: 63 — one quarter).
    test_window : int
        Test rows immediately following val_window (default: 21 — one month).
    step_size : int
        Rows to advance between folds (default: 21).
    total_timesteps_per_fold : int
        SB3 timesteps to train per fold.
    env_factory : callable
        ``env_factory(df_slice) -> gym.Env`` — creates a fresh env from a DataFrame slice.
    agent_factory : callable
        ``agent_factory(env) -> agent`` — creates a fresh, untrained agent.
    mlflow_manager : optional
        If provided, fold metrics are logged to MLflow.
    """

    def __init__(
        self,
        train_window: int = 252,
        val_window: int = 63,
        test_window: int = 21,
        step_size: int = 21,
        total_timesteps_per_fold: int = 10_000,
        env_factory: Optional[Callable] = None,
        agent_factory: Optional[Callable] = None,
        mlflow_manager=None,
    ):
        if train_window < 10:
            raise ValueError("train_window must be at least 10")
        if val_window < 2:
            raise ValueError("val_window must be at least 2")
        if test_window < 2:
            raise ValueError("test_window must be at least 2")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")

        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step_size = step_size
        self.total_timesteps_per_fold = total_timesteps_per_fold
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.mlflow_manager = mlflow_manager

    # ── public API ─────────────────────────────────────────────────────────────

    def split(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return a list of fold window dicts without running any training.

        Each dict has keys: train_start, train_end, val_start, val_end,
        test_start, test_end (all integer row indices, end-exclusive).
        """
        return list(self._iter_windows(len(df)))

    def validate(
        self,
        df: pd.DataFrame,
        env_factory: Optional[Callable] = None,
        agent_factory: Optional[Callable] = None,
    ) -> WalkForwardResult:
        """
        Run full walk-forward validation.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV dataset (rows = timesteps).
        env_factory : callable, optional
            Override ``self.env_factory``.
        agent_factory : callable, optional
            Override ``self.agent_factory``.

        Returns
        -------
        WalkForwardResult
        """
        ef = env_factory or self.env_factory
        af = agent_factory or self.agent_factory

        if ef is None:
            raise ValueError("env_factory must be provided (either at init or call time)")
        if af is None:
            raise ValueError("agent_factory must be provided (either at init or call time)")

        windows = self.split(df)
        if not windows:
            raise ValueError(
                f"DataFrame too short ({len(df)} rows) for windows "
                f"train={self.train_window} + val={self.val_window} + test={self.test_window}"
            )

        fold_results: List[FoldResult] = []

        for fold_idx, win in enumerate(windows):
            logger.info(
                "Fold %d/%d  train[%d:%d]  val[%d:%d]  test[%d:%d]",
                fold_idx + 1, len(windows),
                win["train_start"], win["train_end"],
                win["val_start"],   win["val_end"],
                win["test_start"],  win["test_end"],
            )

            fold_result = self._run_fold(fold_idx, df, win, ef, af)
            fold_results.append(fold_result)

            if self.mlflow_manager is not None:
                self._log_fold_to_mlflow(fold_result)

        result = self._aggregate(fold_results)

        if self.mlflow_manager is not None:
            self._log_aggregate_to_mlflow(result)

        return result

    # ── internal ───────────────────────────────────────────────────────────────

    def _iter_windows(self, n_rows: int):
        """Yield fold window dicts."""
        fold_size = self.train_window + self.val_window + self.test_window
        if n_rows < fold_size:
            return

        start = 0
        fold_idx = 0
        while start + fold_size <= n_rows:
            train_start = start
            train_end   = start + self.train_window
            val_start   = train_end
            val_end     = val_start + self.val_window
            test_start  = val_end
            test_end    = test_start + self.test_window
            yield {
                "fold_idx":    fold_idx,
                "train_start": train_start,
                "train_end":   train_end,
                "val_start":   val_start,
                "val_end":     val_end,
                "test_start":  test_start,
                "test_end":    test_end,
            }
            start += self.step_size
            fold_idx += 1

    def _run_fold(
        self,
        fold_idx: int,
        df: pd.DataFrame,
        win: Dict[str, Any],
        env_factory: Callable,
        agent_factory: Callable,
    ) -> FoldResult:
        """Train on train slice, evaluate on val and test slices."""
        df_train = df.iloc[win["train_start"]:win["train_end"]].reset_index(drop=True)
        df_val   = df.iloc[win["val_start"]:win["val_end"]].reset_index(drop=True)
        df_test  = df.iloc[win["test_start"]:win["test_end"]].reset_index(drop=True)

        # --- Training ---
        train_env = env_factory(df_train)
        agent = agent_factory(train_env)

        try:
            self._train_agent(agent, train_env)
        finally:
            try:
                train_env.close()
            except Exception:
                pass

        # --- Validation (IS) ---
        val_env = env_factory(df_val)
        try:
            val_pv = _rollout_episode(val_env, agent)
        finally:
            try:
                val_env.close()
            except Exception:
                pass

        val_sharpe = _compute_sharpe(val_pv)
        val_mdd    = _compute_max_drawdown(val_pv)
        val_ret    = _compute_total_return(val_pv)

        # --- Test (OOS) ---
        test_env = env_factory(df_test)
        try:
            test_pv = _rollout_episode(test_env, agent)
        finally:
            try:
                test_env.close()
            except Exception:
                pass

        test_sharpe = _compute_sharpe(test_pv)
        test_mdd    = _compute_max_drawdown(test_pv)
        test_ret    = _compute_total_return(test_pv)

        # Stability ratio (per fold)
        if abs(val_sharpe) > 1e-8:
            fold_stability = test_sharpe / val_sharpe
        else:
            fold_stability = 0.0

        logger.info(
            "  Fold %d  val_sharpe=%.3f  test_sharpe=%.3f  stability=%.3f",
            fold_idx, val_sharpe, test_sharpe, fold_stability,
        )

        return FoldResult(
            fold_idx=fold_idx,
            train_start=win["train_start"],
            train_end=win["train_end"],
            val_start=win["val_start"],
            val_end=win["val_end"],
            test_start=win["test_start"],
            test_end=win["test_end"],
            val_sharpe=val_sharpe,
            val_max_drawdown=val_mdd,
            val_total_return=val_ret,
            val_n_steps=len(val_pv),
            test_sharpe=test_sharpe,
            test_max_drawdown=test_mdd,
            test_total_return=test_ret,
            test_n_steps=len(test_pv),
            stability_ratio=fold_stability,
        )

    def _train_agent(self, agent, env) -> None:
        """Delegate to agent's train method (SB3 or custom)."""
        from agents.sb3.sb3_agent_wrapper import SB3AgentWrapper

        if isinstance(agent, SB3AgentWrapper):
            agent.train(env, total_timesteps=self.total_timesteps_per_fold)
        elif hasattr(agent, "train"):
            agent.train(env, total_timesteps=self.total_timesteps_per_fold)
        else:
            raise TypeError(
                f"Agent {type(agent).__name__} has no train() method; "
                "provide an SB3AgentWrapper or compatible agent"
            )

    @staticmethod
    def _aggregate(folds: List[FoldResult]) -> WalkForwardResult:
        if not folds:
            return WalkForwardResult()

        oos_sharpes  = [f.test_sharpe        for f in folds]
        is_sharpes   = [f.val_sharpe         for f in folds]
        oos_mdds     = [f.test_max_drawdown  for f in folds]
        oos_returns  = [f.test_total_return  for f in folds]

        oos_sharpe_mean = float(np.mean(oos_sharpes))
        oos_sharpe_std  = float(np.std(oos_sharpes, ddof=1) if len(oos_sharpes) > 1 else 0.0)
        is_sharpe_mean  = float(np.mean(is_sharpes))
        is_sharpe_std   = float(np.std(is_sharpes, ddof=1) if len(is_sharpes) > 1 else 0.0)

        if abs(is_sharpe_mean) > 1e-8:
            stability_ratio = oos_sharpe_mean / is_sharpe_mean
        else:
            stability_ratio = 0.0

        if stability_ratio >= 0.7:
            rating = "strong"
        elif stability_ratio >= 0.5:
            rating = "decent"
        elif stability_ratio >= 0.3:
            rating = "marginal"
        else:
            rating = "overfitting"

        return WalkForwardResult(
            folds=folds,
            n_folds=len(folds),
            oos_sharpe_mean=oos_sharpe_mean,
            oos_sharpe_std=oos_sharpe_std,
            oos_max_drawdown_mean=float(np.mean(oos_mdds)),
            oos_total_return_mean=float(np.mean(oos_returns)),
            is_sharpe_mean=is_sharpe_mean,
            is_sharpe_std=is_sharpe_std,
            stability_ratio=stability_ratio,
            stability_rating=rating,
        )

    def _log_fold_to_mlflow(self, fold: FoldResult) -> None:
        step = fold.fold_idx
        try:
            self.mlflow_manager.log_metrics(
                {
                    "wf/val_sharpe":       fold.val_sharpe,
                    "wf/val_mdd":          fold.val_max_drawdown,
                    "wf/test_sharpe":      fold.test_sharpe,
                    "wf/test_mdd":         fold.test_max_drawdown,
                    "wf/test_return":      fold.test_total_return,
                    "wf/stability_ratio":  fold.stability_ratio,
                },
                step=step,
            )
        except Exception as exc:
            logger.warning("MLflow fold logging failed: %s", exc)

    def _log_aggregate_to_mlflow(self, result: WalkForwardResult) -> None:
        try:
            self.mlflow_manager.log_metrics(result.as_dict())
        except Exception as exc:
            logger.warning("MLflow aggregate logging failed: %s", exc)
