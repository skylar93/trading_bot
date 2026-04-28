"""Walk-Forward Validation for time-series RL agents.

Implements expanding-window walk-forward validation with gap periods
to prevent look-ahead bias. This is the gold standard for evaluating
trading strategies on non-stationary data.

Usage
-----
    validator = WalkForwardValidator(n_splits=12)
    result = validator.validate(
        agent_factory=lambda: create_agent("ppo", ...),
        env_factory=lambda data: SingleAssetRLTradingEnv(data=data, ...),
        data=full_df,
    )
    print(result.oos_sharpe, result.stability_ratio)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_idx: int
    train_size: int
    test_size: int
    is_sharpe: float       # in-sample Sharpe
    oos_sharpe: float      # out-of-sample Sharpe
    oos_returns: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    oos_max_drawdown: float = 0.0
    oos_total_return: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    folds: List[FoldResult]

    @property
    def oos_sharpe(self) -> float:
        """Mean OOS Sharpe across all folds."""
        sharpes = [f.oos_sharpe for f in self.folds]
        return float(np.mean(sharpes)) if sharpes else 0.0

    @property
    def is_sharpe(self) -> float:
        """Mean IS Sharpe across all folds."""
        return float(np.mean([f.is_sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def stability_ratio(self) -> float:
        """OOS Sharpe / IS Sharpe. Values > 0.5 are acceptable, > 0.7 is good."""
        is_s = self.is_sharpe
        if abs(is_s) < 1e-8:
            return 0.0
        return self.oos_sharpe / is_s

    @property
    def oos_sharpe_std(self) -> float:
        return float(np.std([f.oos_sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def mean_max_drawdown(self) -> float:
        return float(np.mean([f.oos_max_drawdown for f in self.folds])) if self.folds else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "oos_sharpe_mean": self.oos_sharpe,
            "oos_sharpe_std": self.oos_sharpe_std,
            "is_sharpe_mean": self.is_sharpe,
            "stability_ratio": self.stability_ratio,
            "mean_max_drawdown": self.mean_max_drawdown,
            "n_folds": len(self.folds),
        }


class WalkForwardValidator:
    """Time-series walk-forward cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of train/test splits.
    train_ratio : float
        Minimum fraction of data used for training in the first fold.
        Subsequent folds use expanding windows.
    gap_days : int
        Number of rows to skip between train and test to prevent
        look-ahead bias from overlapping features (e.g., rolling windows).
    min_test_size : int
        Minimum number of rows in the test set.
    mode : str
        'expanding' (default) — train set grows each fold.
        'sliding' — fixed-size train window slides forward.
    """

    def __init__(
        self,
        n_splits: int = 12,
        train_ratio: float = 0.5,
        gap_days: int = 5,
        min_test_size: int = 20,
        mode: str = "expanding",
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_days = gap_days
        self.min_test_size = min_test_size
        self.mode = mode

    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits respecting temporal order.

        Returns
        -------
        List of (train_df, test_df) tuples.
        """
        n = len(data)
        min_train = max(int(n * self.train_ratio), 1)

        # Calculate test size per fold
        remaining = n - min_train - self.gap_days
        if remaining <= 0:
            raise ValueError(
                f"Not enough data ({n} rows) for train_ratio={self.train_ratio} "
                f"and gap_days={self.gap_days}"
            )

        test_size = max(remaining // self.n_splits, self.min_test_size)
        splits = []

        for i in range(self.n_splits):
            if self.mode == "expanding":
                train_end = min_train + i * test_size
            else:  # sliding
                train_end = min_train + i * test_size
                train_start = max(0, train_end - min_train)
                data_slice_start = train_start

            test_start = train_end + self.gap_days
            test_end = test_start + test_size

            if test_end > n:
                break

            if self.mode == "expanding":
                train_df = data.iloc[:train_end].copy()
            else:
                train_df = data.iloc[data_slice_start:train_end].copy()

            test_df = data.iloc[test_start:test_end].copy()
            splits.append((train_df, test_df))

        if not splits:
            raise ValueError(
                f"Could not create any valid splits from {n} rows with "
                f"n_splits={self.n_splits}"
            )

        logger.info(
            "Created %d walk-forward splits (mode=%s, gap=%d)",
            len(splits), self.mode, self.gap_days,
        )
        return splits

    def validate(
        self,
        agent_factory: Callable[[], Any],
        env_factory: Callable[[pd.DataFrame], Any],
        data: pd.DataFrame,
        total_timesteps: int = 10000,
        eval_episodes: int = 5,
    ) -> WalkForwardResult:
        """Run full walk-forward validation.

        Parameters
        ----------
        agent_factory : callable
            Returns a fresh (untrained) agent instance.
        env_factory : callable
            Given a DataFrame, returns a Gymnasium env.
        data : pd.DataFrame
            Full dataset (will be split temporally).
        total_timesteps : int
            Training budget per fold.
        eval_episodes : int
            Evaluation episodes per fold for Sharpe estimation.
        """
        splits = self.split(data)
        folds: List[FoldResult] = []

        for i, (train_df, test_df) in enumerate(splits):
            logger.info(
                "Fold %d/%d — train=%d rows, test=%d rows",
                i + 1, len(splits), len(train_df), len(test_df),
            )

            # Train
            agent = agent_factory()
            train_env = env_factory(train_df)
            is_returns = self._train_and_collect_returns(
                agent, train_env, total_timesteps, eval_episodes
            )
            is_sharpe = self._compute_sharpe(is_returns)

            # Test (out-of-sample)
            test_env = env_factory(test_df)
            oos_returns = self._evaluate(agent, test_env, eval_episodes)
            oos_sharpe = self._compute_sharpe(oos_returns)
            oos_dd = self._max_drawdown(oos_returns)
            oos_total = float(np.sum(oos_returns)) if len(oos_returns) > 0 else 0.0

            fold = FoldResult(
                fold_idx=i,
                train_size=len(train_df),
                test_size=len(test_df),
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                oos_returns=oos_returns,
                oos_max_drawdown=oos_dd,
                oos_total_return=oos_total,
            )
            folds.append(fold)

            logger.info(
                "Fold %d result — IS Sharpe=%.3f, OOS Sharpe=%.3f, OOS DD=%.3f",
                i + 1, is_sharpe, oos_sharpe, oos_dd,
            )

            if is_sharpe > 0 and oos_sharpe > 0:
                ratio = is_sharpe / oos_sharpe
                if ratio > 2.0:
                    logger.warning(
                        "Overfitting suspected: IS/OOS Sharpe ratio = %.2f "
                        "(fold %d, IS=%.3f, OOS=%.3f)",
                        ratio, i + 1, is_sharpe, oos_sharpe,
                    )

        result = WalkForwardResult(folds=folds)
        logger.info(
            "Walk-forward complete — OOS Sharpe=%.3f (std=%.3f), Stability=%.3f",
            result.oos_sharpe, result.oos_sharpe_std, result.stability_ratio,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_action(agent, obs, deterministic: bool = False):
        """Get an action from either a custom-wrapper agent (get_action)
        or a raw SB3 model (predict). SB3's predict returns (action, state)."""
        if hasattr(agent, "get_action"):
            return agent.get_action(obs, deterministic=deterministic) if deterministic \
                else agent.get_action(obs)
        # SB3 BaseAlgorithm path
        action, _ = agent.predict(obs, deterministic=deterministic)
        return action

    @staticmethod
    def _train_and_collect_returns(
        agent, env, total_timesteps: int, eval_episodes: int
    ) -> np.ndarray:
        """Train agent and return IS episode returns.

        Two supported agent APIs:
          * custom-wrapper:  agent.get_action(obs) + agent.train_step(...)
                             (manual step-by-step gradient updates per env step)
          * SB3 BaseAlgorithm: agent.learn(total_timesteps) + agent.predict(obs)
                             (SB3 owns the rollout/optimisation loop)
        SB3 agents do their own rollout, so we call learn() once for training
        and then run a separate evaluation loop to collect IS episode returns.
        """
        is_sb3 = not hasattr(agent, "train_step")

        if is_sb3:
            # SB3 owns the env it was constructed with; rebind to our train_env
            # so it actually trains on the fold's data.
            try:
                agent.set_env(env)
            except Exception:  # noqa: BLE001
                pass
            agent.learn(total_timesteps=total_timesteps, progress_bar=False)
            # After training, roll out a few episodes to score IS Sharpe.
            return WalkForwardValidator._evaluate(agent, env, max(1, eval_episodes))

        # Legacy custom-wrapper path
        episode_returns = []
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0

        while steps < total_timesteps:
            action = agent.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.train_step(obs, action, reward, next_obs, done or truncated)
            ep_reward += reward
            obs = next_obs
            steps += 1

            if done or truncated:
                episode_returns.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        # Final incomplete episode
        if ep_reward != 0.0:
            episode_returns.append(ep_reward)

        return np.array(episode_returns, dtype=np.float64)

    @staticmethod
    def _evaluate(agent, env, n_episodes: int) -> np.ndarray:
        """Evaluate agent without training. Returns per-episode returns."""
        returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = WalkForwardValidator._agent_action(agent, obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                if truncated:
                    break
            returns.append(ep_reward)
        return np.array(returns, dtype=np.float64)

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Annualised Sharpe ratio (assumes daily returns for episodes)."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free
        std = np.std(excess, ddof=1)
        if std < 1e-8:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Maximum drawdown from cumulative returns."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
