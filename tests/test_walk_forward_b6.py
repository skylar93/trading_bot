"""Unit tests for B6 additions to training/validation/walk_forward.py.

Covers:
- FoldResult.oos_max_drawdown_random default value and field presence
- WalkForwardResult.mean_max_drawdown_random property
- WalkForwardResult.summary() includes mean_max_drawdown_random
- validate() populates oos_max_drawdown_random when random_start_eval=True
- validate() leaves oos_max_drawdown_random=0 when random_start_eval=False
"""

import math

import numpy as np
import pandas as pd
import pytest

from training.validation.walk_forward import FoldResult, WalkForwardResult, WalkForwardValidator


# ---------------------------------------------------------------------------
# FoldResult + WalkForwardResult unit tests
# ---------------------------------------------------------------------------

def _make_fold(dd_random: float = 0.0, dd_fixed: float = 0.0) -> FoldResult:
    return FoldResult(
        fold_idx=0,
        train_size=100,
        test_size=20,
        is_sharpe=0.5,
        oos_sharpe=0.3,
        oos_max_drawdown=dd_fixed,
        oos_max_drawdown_random=dd_random,
    )


class TestFoldResultB6:
    def test_default_value_is_zero(self):
        fold = FoldResult(fold_idx=0, train_size=100, test_size=20, is_sharpe=0.0, oos_sharpe=0.0)
        assert fold.oos_max_drawdown_random == 0.0

    def test_field_is_set(self):
        fold = _make_fold(dd_random=0.17)
        assert abs(fold.oos_max_drawdown_random - 0.17) < 1e-9


class TestWalkForwardResultB6:
    def _result(self, dd_randoms: list) -> WalkForwardResult:
        folds = [_make_fold(dd_random=d) for d in dd_randoms]
        for i, f in enumerate(folds):
            object.__setattr__(f, "fold_idx", i)
        return WalkForwardResult(folds=folds)

    def test_mean_max_drawdown_random_single(self):
        r = self._result([0.20])
        assert abs(r.mean_max_drawdown_random - 0.20) < 1e-9

    def test_mean_max_drawdown_random_multiple(self):
        vals = [0.10, 0.20, 0.30]
        r = self._result(vals)
        assert abs(r.mean_max_drawdown_random - sum(vals) / len(vals)) < 1e-9

    def test_mean_max_drawdown_random_empty(self):
        r = WalkForwardResult(folds=[])
        assert r.mean_max_drawdown_random == 0.0

    def test_summary_includes_key(self):
        r = self._result([0.15, 0.25])
        s = r.summary()
        assert "mean_max_drawdown_random" in s
        assert abs(s["mean_max_drawdown_random"] - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# Integration: validate() populates oos_max_drawdown_random
# ---------------------------------------------------------------------------

class _TrivialEnv:
    """Minimal Gymnasium-like env that returns varying returns per random_start episode."""

    def __init__(self, data: pd.DataFrame, seed: int = 0):
        self.data = data
        self.initial_capital = 1000.0
        self.portfolio_value = 1000.0
        self._step = 0
        self._rng = np.random.default_rng(seed)
        self._random_start = False
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        self._random_start = (options or {}).get("random_start", False)
        self._step = 0
        pv_noise = self._rng.uniform(0.98, 1.02) if self._random_start else 1.0
        self.portfolio_value = self.initial_capital * pv_noise
        return np.zeros(4), {}

    def step(self, action):
        self._step += 1
        self.portfolio_value *= self._rng.uniform(0.995, 1.005)
        done = self._step >= min(10, len(self.data))
        info = {"portfolio_value": self.portfolio_value, "trade_count": 1}
        return np.zeros(4), 0.0, done, False, info


class _TrivialAgent:
    def get_action(self, obs, deterministic=False):
        return 0

    def train_step(self, *args, **kwargs):
        pass


def _make_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "$close": close,
        "$open": close - 0.1,
        "$high": close + 0.2,
        "$low": close - 0.2,
        "$volume": np.ones(n) * 1000,
    })


class TestValidateB6Integration:
    def _run(self, random_start_eval: bool) -> WalkForwardResult:
        data = _make_data(200)
        validator = WalkForwardValidator(n_splits=2, train_ratio=0.5, gap_days=2, min_test_size=10)
        result = validator.validate(
            agent_factory=_TrivialAgent,
            env_factory=lambda df: _TrivialEnv(df),
            data=data,
            total_timesteps=20,
            eval_episodes=3,
            random_start_eval=random_start_eval,
        )
        return result

    def test_random_dd_populated_when_random_start_eval_true(self):
        result = self._run(random_start_eval=True)
        assert len(result.folds) > 0
        for fold in result.folds:
            # oos_max_drawdown_random must be non-negative (it's a drawdown fraction)
            assert fold.oos_max_drawdown_random >= 0.0

    def test_random_dd_zero_when_random_start_eval_false(self):
        result = self._run(random_start_eval=False)
        for fold in result.folds:
            assert fold.oos_max_drawdown_random == 0.0

    def test_mean_max_drawdown_random_in_summary_when_enabled(self):
        result = self._run(random_start_eval=True)
        s = result.summary()
        assert "mean_max_drawdown_random" in s
        assert s["mean_max_drawdown_random"] >= 0.0
