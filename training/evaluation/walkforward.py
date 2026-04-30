"""
Walk-forward CV with Purged K-Fold — Week 79 (H10).

Wraps the existing ``training.validation.walk_forward.WalkForwardValidator``
and adds:
  - :class:`PurgedKFoldSplitter`  — strict embargo between train and test
  - :class:`WalkForwardReport`    — structured JSON-serialisable report
  - :func:`evaluate_for_promotion` — one-call integration for staging gate

All promotion candidates **must** pass ``evaluate_for_promotion()`` before
the ``staging → canary`` transition in :class:`~training.registry.ModelRegistry`.

Usage::

    from training.evaluation.walkforward import evaluate_for_promotion

    report = evaluate_for_promotion(
        agent_factory=...,
        env_factory=...,
        data=df,
        n_splits=8,
        gap_bars=20,
    )
    if report.passes_staging_gate():
        registry.promote(version, "canary", actor="ci", reason=report.summary_line())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from training.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardResult,
    FoldResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Staging gate thresholds (mirrors PROMOTION_CRITERIA in model_registry.py)
# ---------------------------------------------------------------------------

STAGING_GATE: Dict[str, Any] = {
    "min_oos_sharpe":        0.3,   # mean OOS Sharpe across folds
    "min_stability_ratio":   0.4,   # OOS / IS Sharpe ratio
    "max_mean_drawdown":     0.35,  # mean max drawdown across folds
    "min_n_folds":           4,     # at least N valid folds required
}


# ---------------------------------------------------------------------------
# PurgedKFoldSplitter
# ---------------------------------------------------------------------------

class PurgedKFoldSplitter:
    """
    Generates train/test splits with a **purge gap** between them.

    Unlike a standard K-fold, every fold:
    1. Respects temporal order (no future data leaks into training).
    2. Drops ``embargo_bars`` rows on each side of the train/test boundary
       to prevent feature-level leakage (e.g., rolling windows crossing
       the boundary).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    embargo_bars : int
        Rows to drop immediately before and after the train/test boundary.
        Set to ``max(window_size, lag)`` for your feature set.
    min_train_frac : float
        Minimum fraction of the dataset allocated to training in the first fold.
    test_frac : float
        Fraction of the dataset used for each test window (approximate).
    """

    def __init__(
        self,
        n_splits: int = 6,
        embargo_bars: int = 20,
        min_train_frac: float = 0.5,
        test_frac: float = 0.1,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_frac = min_train_frac
        self.test_frac = test_frac

    def split(
        self, data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Yield (train_df, test_df) pairs with purge gap applied.

        Raises
        ------
        ValueError
            If the dataset is too small to form the requested splits.
        """
        n = len(data)
        test_size = max(int(n * self.test_frac), 1)
        min_train = max(int(n * self.min_train_frac), 1)
        gap = self.embargo_bars

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

        for k in range(self.n_splits):
            # Expanding training window
            train_end = min_train + k * test_size
            test_start = train_end + gap
            test_end = test_start + test_size

            if test_end > n:
                logger.info("Stopping after %d folds (data exhausted)", k)
                break

            # Purge: drop the last `embargo_bars` rows from train
            purged_train_end = max(0, train_end - gap)
            if purged_train_end < 2:
                continue

            train_df = data.iloc[:purged_train_end].copy()
            test_df = data.iloc[test_start:test_end].copy()

            if len(train_df) < 2 or len(test_df) < 1:
                continue

            splits.append((train_df, test_df))

        if not splits:
            raise ValueError(
                f"PurgedKFoldSplitter: could not create any valid splits "
                f"from {n} rows (n_splits={self.n_splits}, "
                f"embargo={self.embargo_bars}, test_frac={self.test_frac})"
            )

        logger.info(
            "PurgedKFold: %d splits, embargo=%d bars",
            len(splits), self.embargo_bars,
        )
        return splits


# ---------------------------------------------------------------------------
# WalkForwardReport
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardReport:
    """
    Structured, JSON-serialisable walk-forward evaluation report.

    Produced by :class:`WalkForwardEvaluator` and consumed by the
    staging promotion gate.
    """

    model_version: Optional[int]
    n_folds: int
    oos_sharpe_mean: float
    oos_sharpe_std: float
    is_sharpe_mean: float
    stability_ratio: float
    mean_max_drawdown: float
    fold_results: List[Dict[str, float]] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    gate_config: Dict[str, Any] = field(default_factory=lambda: dict(STAGING_GATE))
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def passes_staging_gate(self) -> bool:
        """Return True if all staging promotion conditions are met."""
        g = self.gate_config
        return (
            self.oos_sharpe_mean >= g.get("min_oos_sharpe", 0.3)
            and self.stability_ratio >= g.get("min_stability_ratio", 0.4)
            and self.mean_max_drawdown <= g.get("max_mean_drawdown", 0.35)
            and self.n_folds >= g.get("min_n_folds", 4)
        )

    def gate_failures(self) -> List[str]:
        """Return human-readable list of failed gate conditions."""
        g = self.gate_config
        failures: List[str] = []
        if self.oos_sharpe_mean < g.get("min_oos_sharpe", 0.3):
            failures.append(
                f"oos_sharpe_mean={self.oos_sharpe_mean:.3f} "
                f"< {g['min_oos_sharpe']}"
            )
        if self.stability_ratio < g.get("min_stability_ratio", 0.4):
            failures.append(
                f"stability_ratio={self.stability_ratio:.3f} "
                f"< {g['min_stability_ratio']}"
            )
        if self.mean_max_drawdown > g.get("max_mean_drawdown", 0.35):
            failures.append(
                f"mean_max_drawdown={self.mean_max_drawdown:.3f} "
                f"> {g['max_mean_drawdown']}"
            )
        if self.n_folds < g.get("min_n_folds", 4):
            failures.append(
                f"n_folds={self.n_folds} < {g['min_n_folds']}"
            )
        return failures

    def summary_line(self) -> str:
        status = "PASS" if self.passes_staging_gate() else "FAIL"
        return (
            f"walkforward [{status}] "
            f"OOS_Sharpe={self.oos_sharpe_mean:.3f}±{self.oos_sharpe_std:.3f} "
            f"stability={self.stability_ratio:.3f} "
            f"mean_dd={self.mean_max_drawdown:.3f} "
            f"folds={self.n_folds}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("WalkForwardReport saved to %s", p)

    @classmethod
    def load(cls, path: str | Path) -> "WalkForwardReport":
        data = json.loads(Path(path).read_text())
        fold_results = data.pop("fold_results", [])
        report = cls(**{k: v for k, v in data.items() if k != "fold_results"})
        report.fold_results = fold_results
        return report


# ---------------------------------------------------------------------------
# WalkForwardEvaluator
# ---------------------------------------------------------------------------

class WalkForwardEvaluator:
    """
    Runs purged-K-fold walk-forward evaluation and produces a
    :class:`WalkForwardReport`.

    This class bridges :class:`PurgedKFoldSplitter` and the existing
    :class:`~training.validation.walk_forward.WalkForwardValidator`.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    embargo_bars : int
        Purge gap between train and test.
    total_timesteps : int
        Training budget per fold.
    eval_episodes : int
        Evaluation episodes per fold.
    """

    def __init__(
        self,
        n_splits: int = 6,
        embargo_bars: int = 20,
        total_timesteps: int = 10_000,
        eval_episodes: int = 5,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes
        self._splitter = PurgedKFoldSplitter(
            n_splits=n_splits,
            embargo_bars=embargo_bars,
        )
        self._validator = WalkForwardValidator(
            n_splits=n_splits,
            gap_days=embargo_bars,
        )

    def evaluate(
        self,
        agent_factory: Callable[[], Any],
        env_factory: Callable[[pd.DataFrame], Any],
        data: pd.DataFrame,
        model_version: Optional[int] = None,
    ) -> WalkForwardReport:
        """
        Run purged walk-forward evaluation.

        Returns
        -------
        WalkForwardReport
            Structured report with gate pass/fail status.
        """
        splits = self._splitter.split(data)

        fold_results: List[Dict[str, float]] = []
        all_is_sharpes: List[float] = []
        all_oos_sharpes: List[float] = []
        all_drawdowns: List[float] = []

        for i, (train_df, test_df) in enumerate(splits):
            logger.info(
                "Fold %d/%d — train=%d rows, test=%d rows",
                i + 1, len(splits), len(train_df), len(test_df),
            )

            # Reuse WalkForwardValidator helpers via inline call
            agent = agent_factory()
            train_env = env_factory(train_df)
            is_returns = WalkForwardValidator._train_and_collect_returns(
                agent, train_env, self.total_timesteps, self.eval_episodes
            )
            is_sharpe = WalkForwardValidator._compute_sharpe(is_returns)

            test_env = env_factory(test_df)
            oos_returns, _ = WalkForwardValidator._evaluate(
                agent, test_env, self.eval_episodes
            )
            oos_sharpe = WalkForwardValidator._compute_sharpe(oos_returns)
            oos_dd = WalkForwardValidator._max_drawdown(oos_returns)
            oos_total = float(np.sum(oos_returns)) if len(oos_returns) > 0 else 0.0

            all_is_sharpes.append(is_sharpe)
            all_oos_sharpes.append(oos_sharpe)
            all_drawdowns.append(oos_dd)

            fold_results.append(
                {
                    "fold": i,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe,
                    "oos_max_drawdown": oos_dd,
                    "oos_total_return": oos_total,
                }
            )

            logger.info(
                "Fold %d — IS=%.3f  OOS=%.3f  DD=%.3f",
                i + 1, is_sharpe, oos_sharpe, oos_dd,
            )

        oos_mean = float(np.mean(all_oos_sharpes)) if all_oos_sharpes else 0.0
        oos_std = float(np.std(all_oos_sharpes)) if all_oos_sharpes else 0.0
        is_mean = float(np.mean(all_is_sharpes)) if all_is_sharpes else 0.0
        stability = oos_mean / is_mean if abs(is_mean) > 1e-8 else 0.0
        mean_dd = float(np.mean(all_drawdowns)) if all_drawdowns else 0.0

        report = WalkForwardReport(
            model_version=model_version,
            n_folds=len(splits),
            oos_sharpe_mean=oos_mean,
            oos_sharpe_std=oos_std,
            is_sharpe_mean=is_mean,
            stability_ratio=stability,
            mean_max_drawdown=mean_dd,
            fold_results=fold_results,
        )

        logger.info(report.summary_line())
        return report


# ---------------------------------------------------------------------------
# Convenience function for staging gate
# ---------------------------------------------------------------------------

def evaluate_for_promotion(
    agent_factory: Callable[[], Any],
    env_factory: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    *,
    n_splits: int = 6,
    gap_bars: int = 20,
    total_timesteps: int = 10_000,
    eval_episodes: int = 5,
    model_version: Optional[int] = None,
    report_path: Optional[str | Path] = None,
) -> WalkForwardReport:
    """
    Single-call walk-forward evaluation for the staging promotion gate.

    Parameters
    ----------
    agent_factory : callable
        Returns a fresh untrained agent.
    env_factory : callable
        Given a DataFrame, returns a Gymnasium env.
    data : pd.DataFrame
        Full dataset (split temporally inside).
    n_splits : int
        Number of folds.
    gap_bars : int
        Purge embargo between train and test.
    total_timesteps : int
        Training budget per fold.
    eval_episodes : int
        OOS evaluation episodes per fold.
    model_version : int, optional
        Registry version number — embedded in the report.
    report_path : path, optional
        If given, save the report JSON here.

    Returns
    -------
    WalkForwardReport
        Call ``.passes_staging_gate()`` to check promotion readiness.
    """
    evaluator = WalkForwardEvaluator(
        n_splits=n_splits,
        embargo_bars=gap_bars,
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes,
    )
    report = evaluator.evaluate(
        agent_factory=agent_factory,
        env_factory=env_factory,
        data=data,
        model_version=model_version,
    )

    if report_path:
        report.save(report_path)

    return report
