"""
Week 2 tests:
  - MultiComponentReward (envs/rewards.py)
  - Log-return observations (envs/single_asset_rl_env.py)
  - SB3CompatWrapper + make_sb3_env (envs/wrap_env.py)
"""

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from envs.rewards import MultiComponentReward, RewardConfig
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from envs.wrap_env import SB3CompatWrapper, make_sb3_env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 100) -> pd.DataFrame:
    """Synthetic OHLCV data with a gentle uptrend."""
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    high   = close * (1 + rng.uniform(0, 0.005, n))
    low    = close * (1 - rng.uniform(0, 0.005, n))
    open_  = close * (1 + rng.normal(0, 0.003, n))
    vol    = rng.uniform(1_000, 100_000, n)
    return pd.DataFrame({
        "$open": open_, "$high": high, "$low": low,
        "$close": close, "$volume": vol,
    })


@pytest.fixture
def df():
    return _make_df(100)


@pytest.fixture
def env(df):
    return SingleAssetRLTradingEnv(data=df, window_size=10, min_episode_steps=5)


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

class TestRewardConfig:
    def test_default_weights_sum_to_one(self):
        cfg = RewardConfig()
        total = cfg.pnl_weight + cfg.sharpe_weight + cfg.drawdown_weight + cfg.cost_weight
        assert abs(total - 1.0) < 1e-9

    def test_validate_raises_on_bad_weights(self):
        cfg = RewardConfig(pnl_weight=0.9)   # sum = 1.5
        with pytest.raises(ValueError, match="sum to 1.0"):
            cfg.validate()

    def test_validate_raises_on_negative_weight(self):
        cfg = RewardConfig(pnl_weight=-0.1, sharpe_weight=0.5, drawdown_weight=0.4, cost_weight=0.2)
        with pytest.raises(ValueError):
            cfg.validate()


# ---------------------------------------------------------------------------
# MultiComponentReward
# ---------------------------------------------------------------------------

class TestMultiComponentReward:
    def test_output_range(self):
        """Reward must be in (-1, 1)."""
        fn = MultiComponentReward()
        fn.reset()
        for _ in range(50):
            pv = float(np.random.uniform(5_000, 20_000))
            ppv = float(np.random.uniform(5_000, 20_000))
            pkv = max(pv, ppv) * 1.1
            cost = float(np.random.uniform(0, 10))
            r, comps = fn.compute(pv, ppv, pkv, cost)
            assert -1.0 <= r <= 1.0, f"reward {r} out of bounds"

    def test_components_in_range(self):
        """Each named component must be in (-1, 1)."""
        fn = MultiComponentReward()
        fn.reset()
        r, comps = fn.compute(10_100, 10_000, 10_100, 5.0)
        for key in ("pnl", "sharpe", "drawdown", "cost"):
            assert -1.0 <= comps[key] <= 1.0, f"component '{key}' out of bounds: {comps[key]}"

    def test_positive_return_positive_pnl(self):
        fn = MultiComponentReward()
        fn.reset()
        _, comps = fn.compute(11_000, 10_000, 11_000, 0.0)
        assert comps["pnl"] > 0, "positive return should give positive PnL component"

    def test_negative_return_negative_pnl(self):
        fn = MultiComponentReward()
        fn.reset()
        _, comps = fn.compute(9_000, 10_000, 10_000, 0.0)
        assert comps["pnl"] < 0, "negative return should give negative PnL component"

    def test_drawdown_penalizes(self):
        """High drawdown → negative drawdown component."""
        fn = MultiComponentReward()
        fn.reset()
        # portfolio dropped 50% from peak
        _, comps = fn.compute(5_000, 5_000, 10_000, 0.0)
        assert comps["drawdown"] < 0, "large drawdown should penalize"

    def test_cost_penalizes(self):
        """Non-zero cost → negative cost component."""
        fn = MultiComponentReward()
        fn.reset()
        _, comps_no_cost  = fn.compute(10_000, 10_000, 10_000, 0.0)
        fn.reset()
        _, comps_with_cost = fn.compute(10_000, 10_000, 10_000, 100.0)
        assert comps_with_cost["cost"] < comps_no_cost["cost"]

    def test_sharpe_zero_at_start(self):
        """Sharpe component is 0 when buffer has < 5 entries."""
        fn = MultiComponentReward()
        fn.reset()
        _, comps = fn.compute(10_100, 10_000, 10_100, 0.0)
        assert comps["sharpe"] == 0.0, "not enough data yet for Sharpe"

    def test_sharpe_nonzero_after_warmup(self):
        fn = MultiComponentReward(RewardConfig(sharpe_lookback=5))
        fn.reset()
        pv = 10_000.0
        for _ in range(6):
            pv *= 1.001
            fn.compute(pv, pv / 1.001, pv, 0.0)
        _, comps = fn.compute(pv * 1.001, pv, pv * 1.001, 0.0)
        assert comps["sharpe"] != 0.0, "Sharpe should be nonzero after warm-up"

    def test_reset_clears_buffer(self):
        fn = MultiComponentReward()
        fn.reset()
        for _ in range(10):
            fn.compute(10_100, 10_000, 10_100, 0.0)
        fn.reset()
        assert fn.get_sharpe_ratio() == 0.0

    def test_get_sharpe_ratio(self):
        fn = MultiComponentReward(RewardConfig(sharpe_lookback=10))
        fn.reset()
        pv = 10_000.0
        for _ in range(12):
            pv *= 1.001
            fn.compute(pv, pv / 1.001, pv, 0.0)
        sr = fn.get_sharpe_ratio()
        assert np.isfinite(sr)


# ---------------------------------------------------------------------------
# Observation space (log-return based)
# ---------------------------------------------------------------------------

class TestLogReturnObservation:
    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (10, 5), f"Expected (10, 5), got {obs.shape}"

    def test_obs_bounds(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= -10.0), "obs below -10"
        assert np.all(obs <= 10.0), "obs above 10"

    def test_obs_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_not_raw_prices(self, df):
        """Log-return obs should be near 0, not in the hundreds (raw price range)."""
        env = SingleAssetRLTradingEnv(data=df, window_size=10)
        obs, _ = env.reset()
        # Close log-returns of synthetic data are ~±0.02; raw close is ~100+
        assert np.abs(obs[:, 3]).max() < 1.0, (
            "close column looks like raw prices, not log-returns"
        )

    def test_obs_finite(self, env):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), "NaN/Inf in initial observation"

    def test_obs_changes_over_steps(self, env):
        obs0, _ = env.reset()
        obs1, _, _, _, _ = env.step(env.action_space.sample())
        assert not np.allclose(obs0, obs1), "observation did not change after step"

    def test_observation_space_bounds_match(self, env):
        assert env.observation_space.low.min() == pytest.approx(-10.0)
        assert env.observation_space.high.max() == pytest.approx(10.0)

    def test_multi_step_obs_in_bounds(self, env):
        env.reset()
        for _ in range(20):
            obs, _, done, _, _ = env.step(env.action_space.sample())
            assert np.all(obs >= -10.0) and np.all(obs <= 10.0), "obs out of bounds mid-episode"
            if done:
                env.reset()


# ---------------------------------------------------------------------------
# Reward range in full episode
# ---------------------------------------------------------------------------

class TestRewardRange:
    def test_reward_in_minus_one_to_one(self, env):
        env.reset()
        for _ in range(30):
            _, reward, done, _, _ = env.step(env.action_space.sample())
            assert -1.0 <= reward <= 1.0, f"reward {reward} out of (-1, 1)"
            if done:
                break

    def test_reward_components_in_info(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "reward_components" in info
        for key in ("pnl", "sharpe", "drawdown", "cost", "total"):
            assert key in info["reward_components"]

    def test_check_env(self, df):
        """SB3 check_env must pass without errors."""
        env = SB3CompatWrapper(
            SingleAssetRLTradingEnv(data=df, window_size=10, min_episode_steps=5)
        )
        check_env(env, warn=True)  # raises on hard failures


# ---------------------------------------------------------------------------
# SB3CompatWrapper
# ---------------------------------------------------------------------------

class TestSB3CompatWrapper:
    def test_episode_info_present_on_termination(self, df):
        env = SB3CompatWrapper(
            SingleAssetRLTradingEnv(data=df, window_size=10, min_episode_steps=5)
        )
        env.reset()
        done = False
        info = {}
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        assert "episode" in info
        assert "r" in info["episode"]
        assert "l" in info["episode"]

    def test_action_clipping(self, df):
        """Wrapper clips actions that exceed action space."""
        env = SB3CompatWrapper(
            SingleAssetRLTradingEnv(data=df, window_size=10)
        )
        env.reset()
        # pass an out-of-bounds action — should not raise
        obs, reward, term, trunc, info = env.step(np.array([99.0]))
        assert np.isfinite(reward)

    def test_episode_return_accumulates(self, df):
        env = SB3CompatWrapper(
            SingleAssetRLTradingEnv(data=df, window_size=10, min_episode_steps=5)
        )
        env.reset()
        total = 0.0
        done = False
        info = {}
        while not done:
            _, r, terminated, truncated, info = env.step(env.action_space.sample())
            total += r
            done = terminated or truncated
        assert pytest.approx(info["episode"]["r"], abs=1e-5) == total


# ---------------------------------------------------------------------------
# make_sb3_env
# ---------------------------------------------------------------------------

class TestMakeSB3Env:
    def test_returns_vec_env(self, df):
        from stable_baselines3.common.vec_env import VecNormalize
        vec_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True,
                               window_size=10, min_episode_steps=5)
        assert isinstance(vec_env, VecNormalize)

    def test_no_vec_normalize(self, df):
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = make_sb3_env(df, n_envs=1, use_vec_normalize=False,
                               window_size=10, min_episode_steps=5)
        assert isinstance(vec_env, DummyVecEnv)

    def test_reset_and_step(self, df):
        vec_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True,
                               window_size=10, min_episode_steps=5)
        obs = vec_env.reset()
        assert obs.shape == (1, 10, 5)
        action = np.array([[0.5]])
        obs, rewards, dones, infos = vec_env.step(action)
        assert obs.shape == (1, 10, 5)
        assert np.isfinite(rewards).all()

    def test_n_envs(self, df):
        vec_env = make_sb3_env(df, n_envs=3, use_vec_normalize=False,
                               window_size=10, min_episode_steps=5)
        obs = vec_env.reset()
        assert obs.shape[0] == 3

    def test_ppo_trains_short(self, df):
        """PPO can complete 512 steps without crashing."""
        from stable_baselines3 import PPO
        vec_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True,
                               window_size=10, min_episode_steps=5)
        model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, verbose=0)
        model.learn(total_timesteps=256)   # just checks it doesn't explode
