"""Test risk enforcement in SingleAssetRLTradingEnv."""
import numpy as np
import pandas as pd
import pytest

def _make_data(prices):
    n = len(prices)
    return pd.DataFrame({
        "open": prices, "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices], "close": prices,
        "volume": [1000.0] * n,
    })


class TestCapitalFloor:
    def test_episode_ends_at_half_capital(self):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv
        prices = [100.0] * 100
        data = _make_data(prices)
        env = SingleAssetRLTradingEnv(
            data=data, initial_capital=10000.0, window_size=20,
            min_episode_steps=0,
        )
        obs, _ = env.reset()
        # Manually reduce capital
        env.current_capital = 4999.0
        obs, r, done, trunc, info = env.step(np.array([0.0]))
        assert done, "Episode should end when capital < initial * 0.5"


class TestSharpeAnnualization:
    def test_default_daily(self):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv
        prices = [100.0] * 100
        data = _make_data(prices)
        env = SingleAssetRLTradingEnv(data=data, initial_capital=10000.0, window_size=20)
        assert abs(env._annualize_factor - np.sqrt(252)) < 0.01

    def test_hourly(self):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv
        prices = [100.0] * 100
        data = _make_data(prices)
        env = SingleAssetRLTradingEnv(
            data=data, initial_capital=10000.0, window_size=20,
            data_frequency="hourly",
        )
        assert abs(env._annualize_factor - np.sqrt(252 * 6.5)) < 0.1
