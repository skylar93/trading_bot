"""
Week 7 tests — Walk-Forward Validation.

Coverage:
  - WalkForwardValidator: constructor validation, split(), validate()
  - FoldResult / WalkForwardResult dataclasses
  - _compute_sharpe, _compute_max_drawdown, _compute_total_return helpers
  - _rollout_episode with SB3AgentWrapper-style predict(), get_action() and callable
  - Stability ratio + rating thresholds
  - run_walk_forward_validation pipeline function
  - Config YAML has validation section
  - Edge cases: empty folds, data too short, single fold, zero-std returns

All tests use only CPU (no GPU dependency).
"""

import numpy as np
import pandas as pd
import pytest
import gymnasium as gym
from gymnasium import spaces

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame compatible with SingleAssetRLTradingEnv."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {
            "$open":   close * (1 + rng.uniform(-0.002, 0.002, n)),
            "$high":   close * (1 + rng.uniform(0.000, 0.005, n)),
            "$low":    close * (1 - rng.uniform(0.000, 0.005, n)),
            "$close":  close,
            "$volume": rng.uniform(1e4, 1e6, n),
        }
    )


def _make_prices(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=n)
    return 100.0 * np.cumprod(1.0 + returns)


class _DummyEnv(gym.Env):
    """Minimal env for testing: steps through price series, returns portfolio value."""

    def __init__(self, prices: np.ndarray, window: int = 5):
        super().__init__()
        self._prices = prices
        self._window = window
        self._idx = window
        n_obs = window
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._portfolio = 1000.0
        self._position = 0.0

    def reset(self, **kwargs):
        self._idx = self._window
        self._portfolio = 1000.0
        self._position = 0.0
        return self._obs(), {}

    def step(self, action):
        target = float(np.clip(action, 0.0, 1.0).item() if hasattr(action, "item") else action)
        prev_price = self._prices[self._idx - 1]
        curr_price = self._prices[self._idx]
        ret = (curr_price - prev_price) / (prev_price + 1e-10)
        self._position = target
        self._portfolio *= 1.0 + self._position * ret
        self._idx += 1
        done = self._idx >= len(self._prices)
        info = {"portfolio_value": self._portfolio}
        return self._obs(), float(ret * target), done, False, info

    def _obs(self):
        i = self._idx
        w = self._window
        prices = self._prices[max(0, i - w):i]
        if len(prices) < w:
            prices = np.pad(prices, (w - len(prices), 0))
        log_rets = np.diff(np.log(np.maximum(prices, 1e-10)))
        return np.pad(log_rets, (w - len(log_rets), 0)).astype(np.float32)


class _HoldAgent:
    """Always hold 100% (deterministic)."""
    def predict(self, obs, deterministic=True):
        return np.array([1.0]), None

    def get_action(self, obs):
        return np.array([1.0])

    def train(self, env, total_timesteps=1000):
        pass  # no-op for testing


class _RandomAgent:
    """Random action agent — no learning."""
    def __init__(self):
        self._rng = np.random.default_rng(42)

    def predict(self, obs, deterministic=False):
        return self._rng.uniform(0.0, 1.0, size=(1,)), None

    def get_action(self, obs):
        return self._rng.uniform(0.0, 1.0, size=(1,))

    def train(self, env, total_timesteps=1000):
        pass


def _make_env_factory(window: int = 5):
    def factory(df: pd.DataFrame):
        prices = df["$close"].values.astype(np.float64)
        return _DummyEnv(prices, window=window)
    return factory


def _make_agent_factory():
    def factory(env):
        return _HoldAgent()
    return factory


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricHelpers:
    def test_sharpe_flat(self):
        from training.validation.walk_forward import _compute_sharpe
        pv = np.ones(100) * 1000.0
        assert _compute_sharpe(pv) == 0.0

    def test_sharpe_positive_trend(self):
        from training.validation.walk_forward import _compute_sharpe
        # Steadily increasing portfolio → positive Sharpe
        pv = np.linspace(1000, 1200, 100)
        assert _compute_sharpe(pv) > 0.0

    def test_sharpe_negative_trend(self):
        from training.validation.walk_forward import _compute_sharpe
        pv = np.linspace(1200, 800, 100)
        assert _compute_sharpe(pv) < 0.0

    def test_sharpe_single_value(self):
        from training.validation.walk_forward import _compute_sharpe
        assert _compute_sharpe(np.array([1000.0])) == 0.0

    def test_sharpe_empty(self):
        from training.validation.walk_forward import _compute_sharpe
        assert _compute_sharpe(np.array([])) == 0.0

    def test_max_drawdown_zero(self):
        from training.validation.walk_forward import _compute_max_drawdown
        pv = np.linspace(1000, 1200, 50)
        assert _compute_max_drawdown(pv) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_full_loss(self):
        from training.validation.walk_forward import _compute_max_drawdown
        # Peak = 100, then drops to 50 → 50% drawdown
        pv = np.array([100.0, 120.0, 80.0, 50.0])
        dd = _compute_max_drawdown(pv)
        assert dd == pytest.approx((120 - 50) / 120, rel=1e-6)

    def test_max_drawdown_range(self):
        from training.validation.walk_forward import _compute_max_drawdown
        rng = np.random.default_rng(0)
        pv = 1000 * np.cumprod(1 + rng.normal(0, 0.01, 200))
        dd = _compute_max_drawdown(pv)
        assert 0.0 <= dd <= 1.0

    def test_total_return_positive(self):
        from training.validation.walk_forward import _compute_total_return
        pv = np.array([1000.0, 1100.0])
        assert _compute_total_return(pv) == pytest.approx(0.1, rel=1e-6)

    def test_total_return_negative(self):
        from training.validation.walk_forward import _compute_total_return
        pv = np.array([1000.0, 900.0])
        assert _compute_total_return(pv) == pytest.approx(-0.1, rel=1e-6)

    def test_total_return_single(self):
        from training.validation.walk_forward import _compute_total_return
        assert _compute_total_return(np.array([1000.0])) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# _rollout_episode
# ──────────────────────────────────────────────────────────────────────────────

class TestRolloutEpisode:
    def _make_env(self):
        prices = _make_prices(50)
        return _DummyEnv(prices, window=5)

    def test_rollout_with_predict(self):
        from training.validation.walk_forward import _rollout_episode
        env = self._make_env()
        agent = _HoldAgent()
        pv = _rollout_episode(env, agent)
        assert isinstance(pv, np.ndarray)
        assert len(pv) > 0

    def test_rollout_with_get_action(self):
        from training.validation.walk_forward import _rollout_episode

        class GetActionAgent:
            def get_action(self, obs):
                return np.array([0.5])

        env = self._make_env()
        pv = _rollout_episode(env, GetActionAgent())
        assert len(pv) > 0

    def test_rollout_with_callable(self):
        from training.validation.walk_forward import _rollout_episode
        env = self._make_env()
        pv = _rollout_episode(env, lambda obs: np.array([0.3]))
        assert len(pv) > 0

    def test_rollout_max_steps(self):
        from training.validation.walk_forward import _rollout_episode
        prices = _make_prices(200)
        env = _DummyEnv(prices, window=5)
        pv = _rollout_episode(env, _HoldAgent(), max_steps=10)
        assert len(pv) <= 10

    def test_rollout_returns_array(self):
        from training.validation.walk_forward import _rollout_episode
        env = self._make_env()
        pv = _rollout_episode(env, _HoldAgent())
        assert pv.dtype in (np.float32, np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# WalkForwardValidator constructor validation
# ──────────────────────────────────────────────────────────────────────────────

class TestWalkForwardValidatorInit:
    def test_default_construction(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator()
        assert v.train_window == 252
        assert v.val_window == 63
        assert v.test_window == 21
        assert v.step_size == 21

    def test_custom_windows(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_window=50, val_window=10, test_window=5, step_size=5)
        assert v.train_window == 50
        assert v.val_window == 10
        assert v.test_window == 5

    def test_invalid_train_window(self):
        from training.validation.walk_forward import WalkForwardValidator
        with pytest.raises(ValueError, match="train_window"):
            WalkForwardValidator(train_window=5)

    def test_invalid_val_window(self):
        from training.validation.walk_forward import WalkForwardValidator
        with pytest.raises(ValueError, match="val_window"):
            WalkForwardValidator(val_window=1)

    def test_invalid_test_window(self):
        from training.validation.walk_forward import WalkForwardValidator
        with pytest.raises(ValueError, match="test_window"):
            WalkForwardValidator(test_window=1)

    def test_invalid_step_size(self):
        from training.validation.walk_forward import WalkForwardValidator
        with pytest.raises(ValueError, match="step_size"):
            WalkForwardValidator(step_size=0)


# ──────────────────────────────────────────────────────────────────────────────
# split()
# ──────────────────────────────────────────────────────────────────────────────

class TestSplit:
    def _validator(self, train=50, val=10, test=10, step=10):
        from training.validation.walk_forward import WalkForwardValidator
        return WalkForwardValidator(
            train_window=train, val_window=val, test_window=test, step_size=step
        )

    def test_empty_when_too_short(self):
        v = self._validator(train=50, val=10, test=10, step=10)
        df = _make_ohlcv(60)  # 60 < 70 needed
        assert v.split(df) == []

    def test_single_fold(self):
        v = self._validator(train=50, val=10, test=10, step=10)
        df = _make_ohlcv(70)
        folds = v.split(df)
        assert len(folds) == 1
        f = folds[0]
        assert f["train_start"] == 0
        assert f["train_end"] == 50
        assert f["val_start"] == 50
        assert f["val_end"] == 60
        assert f["test_start"] == 60
        assert f["test_end"] == 70

    def test_multiple_folds(self):
        v = self._validator(train=50, val=10, test=10, step=10)
        df = _make_ohlcv(200)
        folds = v.split(df)
        assert len(folds) >= 2

    def test_windows_non_overlapping(self):
        v = self._validator(train=50, val=10, test=10, step=10)
        df = _make_ohlcv(200)
        folds = v.split(df)
        for f in folds:
            assert f["train_end"] == f["val_start"]
            assert f["val_end"] == f["test_start"]
            assert f["test_end"] == f["test_start"] + 10

    def test_fold_indices_in_bounds(self):
        v = self._validator(train=50, val=10, test=10, step=10)
        df = _make_ohlcv(200)
        folds = v.split(df)
        n = len(df)
        for f in folds:
            assert f["test_end"] <= n

    def test_step_advances_correctly(self):
        v = self._validator(train=50, val=10, test=10, step=10)
        df = _make_ohlcv(200)
        folds = v.split(df)
        for i in range(1, len(folds)):
            assert folds[i]["train_start"] == folds[i - 1]["train_start"] + 10

    def test_fold_idx_sequential(self):
        v = self._validator(train=30, val=10, test=10, step=10)
        df = _make_ohlcv(200)
        folds = v.split(df)
        for i, f in enumerate(folds):
            assert f["fold_idx"] == i


# ──────────────────────────────────────────────────────────────────────────────
# validate()
# ──────────────────────────────────────────────────────────────────────────────

class TestValidate:
    def _validator(self, n_fold_ts=500):
        from training.validation.walk_forward import WalkForwardValidator
        return WalkForwardValidator(
            train_window=30,
            val_window=10,
            test_window=10,
            step_size=10,
            total_timesteps_per_fold=n_fold_ts,
        )

    def test_raises_without_env_factory(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_window=30, val_window=10, test_window=10)
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match="env_factory"):
            v.validate(df, agent_factory=_make_agent_factory())

    def test_raises_without_agent_factory(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_window=30, val_window=10, test_window=10)
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match="agent_factory"):
            v.validate(df, env_factory=_make_env_factory())

    def test_raises_when_too_short(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_window=100, val_window=30, test_window=20)
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match="too short"):
            v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())

    def test_returns_walk_forward_result(self):
        from training.validation.walk_forward import WalkForwardValidator, WalkForwardResult
        v = self._validator()
        df = _make_ohlcv(150)
        result = v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())
        assert isinstance(result, WalkForwardResult)

    def test_n_folds_correct(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = self._validator()
        df = _make_ohlcv(150)
        result = v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())
        expected = len(v.split(df))
        assert result.n_folds == expected

    def test_fold_results_populated(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = self._validator()
        df = _make_ohlcv(150)
        result = v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())
        assert len(result.folds) == result.n_folds

    def test_oos_sharpe_is_float(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = self._validator()
        df = _make_ohlcv(150)
        result = v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())
        assert isinstance(result.oos_sharpe_mean, float)

    def test_max_drawdown_non_negative(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = self._validator()
        df = _make_ohlcv(150)
        result = v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())
        assert result.oos_max_drawdown_mean >= 0.0

    def test_stability_rating_is_string(self):
        from training.validation.walk_forward import WalkForwardValidator
        v = self._validator()
        df = _make_ohlcv(150)
        result = v.validate(df, env_factory=_make_env_factory(), agent_factory=_make_agent_factory())
        assert result.stability_rating in ("strong", "decent", "marginal", "overfitting", "unknown")

    def test_factory_args_override_init(self):
        from training.validation.walk_forward import WalkForwardValidator
        # init with None factories, pass at validate() time — should work
        v = WalkForwardValidator(train_window=30, val_window=10, test_window=10, step_size=10)
        df = _make_ohlcv(120)
        result = v.validate(
            df,
            env_factory=_make_env_factory(),
            agent_factory=_make_agent_factory(),
        )
        assert result.n_folds >= 1


# ──────────────────────────────────────────────────────────────────────────────
# FoldResult
# ──────────────────────────────────────────────────────────────────────────────

class TestFoldResult:
    def _make_fold(self, val_sharpe=1.0, test_sharpe=0.6):
        from training.validation.walk_forward import FoldResult
        return FoldResult(
            fold_idx=0,
            train_start=0, train_end=50,
            val_start=50, val_end=60,
            test_start=60, test_end=70,
            val_sharpe=val_sharpe,
            test_sharpe=test_sharpe,
        )

    def test_fields_accessible(self):
        f = self._make_fold()
        assert f.fold_idx == 0
        assert f.val_sharpe == 1.0
        assert f.test_sharpe == 0.6

    def test_stability_ratio_stored(self):
        from training.validation.walk_forward import FoldResult
        f = FoldResult(
            fold_idx=0,
            train_start=0, train_end=50,
            val_start=50, val_end=60,
            test_start=60, test_end=70,
            val_sharpe=1.0,
            test_sharpe=0.7,
            stability_ratio=0.7,
        )
        assert f.stability_ratio == 0.7


# ──────────────────────────────────────────────────────────────────────────────
# WalkForwardResult
# ──────────────────────────────────────────────────────────────────────────────

class TestWalkForwardResult:
    def test_as_dict_keys(self):
        from training.validation.walk_forward import WalkForwardResult
        r = WalkForwardResult(
            n_folds=3,
            oos_sharpe_mean=0.8,
            oos_sharpe_std=0.1,
            oos_max_drawdown_mean=0.05,
            oos_total_return_mean=0.12,
            is_sharpe_mean=1.1,
            is_sharpe_std=0.15,
            stability_ratio=0.73,
            stability_rating="strong",
        )
        d = r.as_dict()
        for key in [
            "n_folds", "oos_sharpe_mean", "oos_sharpe_std",
            "oos_max_drawdown_mean", "oos_total_return_mean",
            "is_sharpe_mean", "is_sharpe_std",
            "stability_ratio", "stability_rating",
        ]:
            assert key in d

    def test_empty_result(self):
        from training.validation.walk_forward import WalkForwardResult
        r = WalkForwardResult()
        assert r.n_folds == 0
        assert r.stability_rating == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Stability ratio thresholds
# ──────────────────────────────────────────────────────────────────────────────

class TestStabilityRating:
    def _aggregate(self, oos, is_):
        from training.validation.walk_forward import WalkForwardValidator, FoldResult
        v = WalkForwardValidator.__new__(WalkForwardValidator)
        folds = [
            FoldResult(
                fold_idx=0,
                train_start=0, train_end=50,
                val_start=50, val_end=60,
                test_start=60, test_end=70,
                val_sharpe=is_,
                test_sharpe=oos,
                val_total_return=0.0,
                val_max_drawdown=0.0,
                test_total_return=0.0,
                test_max_drawdown=0.0,
            )
        ]
        return WalkForwardValidator._aggregate(folds)

    def test_strong(self):
        r = self._aggregate(oos=0.7, is_=1.0)
        assert r.stability_rating == "strong"

    def test_decent(self):
        r = self._aggregate(oos=0.5, is_=1.0)
        assert r.stability_rating == "decent"

    def test_marginal(self):
        r = self._aggregate(oos=0.3, is_=1.0)
        assert r.stability_rating == "marginal"

    def test_overfitting(self):
        r = self._aggregate(oos=0.1, is_=1.0)
        assert r.stability_rating == "overfitting"

    def test_zero_is_sharpe(self):
        r = self._aggregate(oos=0.5, is_=0.0)
        assert r.stability_ratio == 0.0

    def test_negative_is_sharpe(self):
        r = self._aggregate(oos=-0.5, is_=-1.0)
        assert r.stability_ratio == pytest.approx(0.5, abs=1e-6)

    def test_aggregate_multi_fold(self):
        from training.validation.walk_forward import WalkForwardValidator, FoldResult
        folds = [
            FoldResult(
                fold_idx=i,
                train_start=i * 10, train_end=i * 10 + 50,
                val_start=i * 10 + 50, val_end=i * 10 + 60,
                test_start=i * 10 + 60, test_end=i * 10 + 70,
                val_sharpe=1.0,
                test_sharpe=0.6 + i * 0.1,
                val_total_return=0.05,
                val_max_drawdown=0.02,
                test_total_return=0.04,
                test_max_drawdown=0.03,
            )
            for i in range(3)
        ]
        result = WalkForwardValidator._aggregate(folds)
        assert result.n_folds == 3
        assert result.oos_sharpe_mean == pytest.approx(np.mean([0.6, 0.7, 0.8]), rel=1e-6)
        assert result.is_sharpe_mean == pytest.approx(1.0, rel=1e-6)

    def test_aggregate_empty(self):
        from training.validation.walk_forward import WalkForwardValidator, WalkForwardResult
        result = WalkForwardValidator._aggregate([])
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds == 0


# ──────────────────────────────────────────────────────────────────────────────
# run_walk_forward_validation (pipeline function)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunWalkForwardValidation:
    def _config(self):
        return {
            "validation": {
                "train_window": 30,
                "val_window": 10,
                "test_window": 10,
                "step_size": 10,
                "total_timesteps_per_fold": 500,
            }
        }

    def test_basic_run(self):
        from training.train_pipeline import run_walk_forward_validation
        df = _make_ohlcv(150)
        result = run_walk_forward_validation(
            df,
            env_factory=_make_env_factory(),
            agent_factory=_make_agent_factory(),
            config=self._config(),
        )
        assert result.n_folds >= 1

    def test_returns_walk_forward_result(self):
        from training.train_pipeline import run_walk_forward_validation
        from training.validation.walk_forward import WalkForwardResult
        df = _make_ohlcv(150)
        result = run_walk_forward_validation(
            df,
            env_factory=_make_env_factory(),
            agent_factory=_make_agent_factory(),
            config=self._config(),
        )
        assert isinstance(result, WalkForwardResult)

    def test_without_config(self):
        # Should use defaults (large windows → 0 folds on small df, but no crash)
        from training.train_pipeline import run_walk_forward_validation
        df = _make_ohlcv(400)
        # Default windows sum to 252+63+21=336 rows, so 400 rows → at least 1 fold
        result = run_walk_forward_validation(
            df,
            env_factory=_make_env_factory(),
            agent_factory=_make_agent_factory(),
        )
        assert result.n_folds >= 1

    def test_oos_std_present(self):
        from training.train_pipeline import run_walk_forward_validation
        df = _make_ohlcv(200)
        result = run_walk_forward_validation(
            df,
            env_factory=_make_env_factory(),
            agent_factory=_make_agent_factory(),
            config=self._config(),
        )
        assert isinstance(result.oos_sharpe_std, float)

    def test_fold_count_matches_split(self):
        from training.train_pipeline import run_walk_forward_validation
        from training.validation.walk_forward import WalkForwardValidator
        cfg = self._config()
        df = _make_ohlcv(200)
        val_cfg = cfg["validation"]
        v = WalkForwardValidator(
            train_window=val_cfg["train_window"],
            val_window=val_cfg["val_window"],
            test_window=val_cfg["test_window"],
            step_size=val_cfg["step_size"],
        )
        expected = len(v.split(df))
        result = run_walk_forward_validation(
            df,
            env_factory=_make_env_factory(),
            agent_factory=_make_agent_factory(),
            config=cfg,
        )
        assert result.n_folds == expected


# ──────────────────────────────────────────────────────────────────────────────
# Config YAML
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigYAML:
    def test_validation_section_exists(self):
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "validation" in cfg, "config/training_config.yaml must have 'validation' key"

    def test_validation_fields(self):
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        v = cfg["validation"]
        for key in ["train_window", "val_window", "test_window", "step_size"]:
            assert key in v, f"validation.{key} missing from config"

    def test_window_values_positive(self):
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        v = cfg["validation"]
        assert v["train_window"] > 0
        assert v["val_window"] > 0
        assert v["test_window"] > 0
        assert v["step_size"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# Module imports
# ──────────────────────────────────────────────────────────────────────────────

class TestImports:
    def test_validation_package(self):
        from training.validation import WalkForwardValidator, WalkForwardResult, FoldResult
        assert WalkForwardValidator is not None
        assert WalkForwardResult is not None
        assert FoldResult is not None

    def test_walk_forward_module(self):
        import training.validation.walk_forward as wf
        assert hasattr(wf, "WalkForwardValidator")
        assert hasattr(wf, "WalkForwardResult")
        assert hasattr(wf, "FoldResult")
        assert hasattr(wf, "_compute_sharpe")
        assert hasattr(wf, "_compute_max_drawdown")
        assert hasattr(wf, "_compute_total_return")
        assert hasattr(wf, "_rollout_episode")

    def test_run_wf_in_train_pipeline(self):
        from training.train_pipeline import run_walk_forward_validation
        assert callable(run_walk_forward_validation)
