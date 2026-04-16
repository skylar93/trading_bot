"""
E16 — Numerical Canary Tests

CI guard: runs 100 random seeds × 100 steps across both env types.
Any NaN or Inf in observation or reward is a hard failure.

Rules:
- No new tests added here beyond what the plan specifies.
- Seeds fixed per test class so runs are deterministic.
- Covers SingleAssetRLTradingEnv (normal + stress) and
  MultiAgentMultiAssetEnv (shared capital + independent capital).
"""

import numpy as np
import pandas as pd
import pytest

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SEEDS = 100
N_STEPS = 100
WINDOW_SIZE = 10
DATA_LEN = N_STEPS + WINDOW_SIZE + 10  # plenty of data


def _make_price_series(rng: np.random.Generator, n: int, base: float = 100.0) -> np.ndarray:
    """GBM-like log-normal price series."""
    returns = rng.normal(0, 0.01, size=n)
    prices = base * np.exp(np.cumsum(returns))
    return np.clip(prices, 1.0, None)  # keep positive


def _make_single_asset_df(rng: np.random.Generator) -> pd.DataFrame:
    prices = _make_price_series(rng, DATA_LEN)
    idx = pd.date_range("2023-01-01", periods=DATA_LEN, freq="1h")
    volume = rng.uniform(1e5, 1e7, size=DATA_LEN)
    return pd.DataFrame(
        {
            "$open": prices,
            "$high": prices * rng.uniform(1.0, 1.02, size=DATA_LEN),
            "$low": prices * rng.uniform(0.98, 1.0, size=DATA_LEN),
            "$close": prices,
            "$volume": volume,
        },
        index=idx,
    )


def _make_single_env(df: pd.DataFrame) -> SingleAssetRLTradingEnv:
    return SingleAssetRLTradingEnv(
        data=df,
        window_size=WINDOW_SIZE,
        initial_capital=10_000.0,
        trading_fee=0.001,
        apply_slippage=False,
    )


def _assert_finite(arr, label: str, seed: int, step: int):
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise AssertionError(
            f"[seed={seed} step={step}] {label}: {bad} non-finite values"
        )


# ---------------------------------------------------------------------------
# SingleAssetRLTradingEnv canary
# ---------------------------------------------------------------------------

class TestSingleAssetNaNCanary:
    """100 seeds × 100 steps — zero NaN/Inf tolerance."""

    def _run_seed(self, seed: int):
        rng = np.random.default_rng(seed)
        df = _make_single_asset_df(rng)
        env = _make_single_env(df)
        obs, _ = env.reset(seed=seed)
        _assert_finite(obs, "obs@reset", seed, 0)

        for step in range(N_STEPS):
            action = np.array([rng.uniform(-1.0, 1.0)])
            obs, reward, done, _, _ = env.step(action)
            _assert_finite(obs, "obs", seed, step)
            assert np.isfinite(reward), (
                f"[seed={seed} step={step}] reward is {reward}"
            )
            if done:
                break

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_single_asset_no_nan(self, seed: int):
        self._run_seed(seed)


class TestSingleAssetStressNaNCanary:
    """Stress: extreme actions (full buy/sell every step)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_single_asset_stress_no_nan(self, seed: int):
        rng = np.random.default_rng(seed + 1000)
        df = _make_single_asset_df(rng)
        env = _make_single_env(df)
        obs, _ = env.reset(seed=seed)
        _assert_finite(obs, "obs@reset", seed, 0)

        for step in range(N_STEPS):
            # Alternate between max buy and max sell
            action = np.array([1.0 if step % 2 == 0 else -1.0])
            obs, reward, done, _, _ = env.step(action)
            _assert_finite(obs, "obs", seed, step)
            assert np.isfinite(reward), (
                f"[seed={seed} step={step}] reward is {reward}"
            )
            if done:
                break


# ---------------------------------------------------------------------------
# MultiAgentMultiAssetEnv canary
# ---------------------------------------------------------------------------

ASSETS = ["BTC", "ETH", "SPY", "GOLD"]

AGENT_CONFIGS_SHARED = [
    {
        "id": "agent_0",
        "initial_balance": 5_000.0,
        "assigned_assets": ["BTC", "ETH"],
        "priority": 1,
    },
    {
        "id": "agent_1",
        "initial_balance": 5_000.0,
        "assigned_assets": ["SPY", "GOLD"],
        "priority": 1,
    },
]

AGENT_CONFIGS_INDEP = [
    {
        "id": "agent_0",
        "initial_balance": 10_000.0,
        "assigned_assets": ["BTC", "ETH"],
        "priority": 1,
    },
    {
        "id": "agent_1",
        "initial_balance": 10_000.0,
        "assigned_assets": ["SPY", "GOLD"],
        "priority": 1,
    },
]


def _make_multi_asset_data(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    data = {}
    for asset in ASSETS:
        base = rng.uniform(50.0, 500.0)
        prices = _make_price_series(rng, DATA_LEN, base=base)
        idx = pd.date_range("2023-01-01", periods=DATA_LEN, freq="1h")
        volume = rng.uniform(1e5, 1e7, size=DATA_LEN)
        data[asset] = pd.DataFrame(
            {
                "$open": prices,
                "$high": prices * rng.uniform(1.0, 1.02, size=DATA_LEN),
                "$low": prices * rng.uniform(0.98, 1.0, size=DATA_LEN),
                "$close": prices,
                "$volume": volume,
            },
            index=idx,
        )
    return data


def _run_multi_agent_seed(seed: int, shared_capital: bool, agent_configs: list):
    rng = np.random.default_rng(seed)
    data = _make_multi_asset_data(rng)
    env = MultiAgentMultiAssetEnv(
        data=data,
        agent_configs=agent_configs,
        window_size=WINDOW_SIZE,
        trading_fee=0.001,
        action_type="portfolio_weights",
        shared_capital=shared_capital,
    )
    observations, _ = env.reset(seed=seed)
    for agent_id, obs in observations.items():
        _assert_finite(obs, f"obs@reset[{agent_id}]", seed, 0)

    for step in range(N_STEPS):
        actions = {}
        for agent_id in env.agents:
            n_assets = len(env.agent_assets[agent_id])
            raw = rng.uniform(0.0, 1.0, size=n_assets)
            actions[agent_id] = raw

        observations, rewards, dones, _, _ = env.step(actions)

        for agent_id, obs in observations.items():
            _assert_finite(obs, f"obs[{agent_id}]", seed, step)
        for agent_id, rew in rewards.items():
            assert np.isfinite(rew), (
                f"[seed={seed} step={step}] reward[{agent_id}] is {rew}"
            )

        if all(dones.values()):
            break


class TestMultiAgentSharedCapitalNaNCanary:
    """Shared capital mode: 100 seeds × 100 steps — zero NaN/Inf tolerance."""

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_multi_agent_shared_no_nan(self, seed: int):
        _run_multi_agent_seed(seed, shared_capital=True, agent_configs=AGENT_CONFIGS_SHARED)


class TestMultiAgentIndependentCapitalNaNCanary:
    """Independent capital mode: 100 seeds × 100 steps."""

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_multi_agent_indep_no_nan(self, seed: int):
        _run_multi_agent_seed(seed, shared_capital=False, agent_configs=AGENT_CONFIGS_INDEP)
