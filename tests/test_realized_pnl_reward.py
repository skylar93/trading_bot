"""Phase 8-Gamma G2: realized-PnL reward function tests."""
import numpy as np
import pandas as pd
import pytest

from envs.single_asset_rl_env import SingleAssetRLTradingEnv


def _make_data(n=200, price_drift=0.0):
    """Deterministic linear-drift price for predictable PnL."""
    base = np.linspace(100.0, 100.0 + price_drift * n, n)
    return pd.DataFrame(
        {
            "$open": base,
            "$high": base * 1.001,
            "$low": base * 0.999,
            "$close": base,
            "$volume": np.ones(n) * 1000.0,
        }
    )


def _make_env(data, **kw):
    defaults = dict(
        initial_capital=100_000.0,
        window_size=20,
        min_episode_steps=30,
        partial_fills=False,
        apply_slippage=False,
        max_position_size=1.0,
        reward_function="realized_pnl",
        risk_adjusted_reward=False,
        sharpe_weight=0.0,
    )
    defaults.update(kw)
    return SingleAssetRLTradingEnv(data=data, **defaults)


# ---- Test 1: hold gives zero reward ----
def test_hold_gives_zero_reward():
    """When agent holds (no position change), reward must be 0."""
    env = _make_env(_make_data(100, price_drift=1.0))
    env.reset(seed=0)
    rewards = []
    for _ in range(20):
        obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        rewards.append(r)
        if term or trunc:
            break
    assert all(abs(r) < 1e-9 for r in rewards), (
        f"Hold-only episode should give zero reward; got {rewards}"
    )


# ---- Test 2: position open gives zero reward ----
def test_open_position_gives_zero_reward():
    """Opening a position is not a realization — reward = 0."""
    env = _make_env(_make_data(100))
    env.reset(seed=0)
    obs, r, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert abs(r) < 1e-9, f"Opening position should give zero reward, got {r}"
    assert env.current_position > 0


# ---- Test 3: profitable close gives positive reward ----
def test_profitable_long_close_gives_positive_reward():
    """Open long, hold while price rises, close → positive realized PnL reward."""
    env = _make_env(_make_data(100, price_drift=1.0))
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))  # open long at ~100
    for _ in range(10):
        env.step(np.array([0.0], dtype=np.float32))
    obs, r, _, _, info = env.step(np.array([-1.0], dtype=np.float32))  # close
    assert r > 0, f"Profitable close should give positive reward, got {r}"
    assert info["realized_pnl_this_step"] > 0


# ---- Test 4: losing close gives negative reward ----
def test_losing_long_close_gives_negative_reward():
    """Open long, price falls, close → negative reward."""
    env = _make_env(_make_data(100, price_drift=-1.0))
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))
    for _ in range(10):
        env.step(np.array([0.0], dtype=np.float32))
    obs, r, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
    assert r < 0, f"Losing close should give negative reward, got {r}"


# ---- Test 5: short close on price drop gives positive reward ----
def test_short_close_profit_on_price_drop():
    """Open short, price drops, close → positive realized PnL."""
    env = _make_env(_make_data(100, price_drift=-1.0))
    env.reset(seed=0)
    env.step(np.array([-1.0], dtype=np.float32))
    for _ in range(10):
        env.step(np.array([0.0], dtype=np.float32))
    obs, r, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert r > 0, f"Profitable short close should give positive reward, got {r}"


# ---- Test 6: partial close realizes proportional PnL ----
def test_partial_close_realizes_proportional_pnl():
    """Open full long, halve position when profitable → positive reward."""
    env = _make_env(_make_data(100, price_drift=1.0))
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))  # full long
    pos_before = env.current_position
    for _ in range(5):
        env.step(np.array([0.0], dtype=np.float32))
    obs, r, _, _, info = env.step(np.array([-0.5], dtype=np.float32))  # halve
    assert r > 0, f"Partial close should give positive reward, got {r}"
    assert env.current_position < pos_before


# ---- Test 7: reward magnitude = realized PnL / initial_capital ----
def test_reward_magnitude_normalized():
    """Reward = realized_pnl / initial_capital (portfolio fraction)."""
    env = _make_env(_make_data(100, price_drift=1.0), initial_capital=100_000.0)
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))
    for _ in range(5):
        env.step(np.array([0.0], dtype=np.float32))
    obs, r, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
    realized = info["realized_pnl_this_step"]
    expected = float(np.clip(realized / 100_000.0, -5.0, 5.0))
    assert abs(r - expected) < 1e-6, (
        f"reward should be clip(realized_pnl/initial_capital, ±5): {r} vs {expected}"
    )


# ---- Test 8: backward compat (sharpe_ratio path unchanged) ----
def test_sharpe_ratio_path_unchanged_when_realized_pnl_not_set():
    """reward_function='sharpe_ratio' (default) gives non-zero reward on hold
    while price is moving — regression guard for the existing path."""
    data = _make_data(100, price_drift=1.0)
    env = SingleAssetRLTradingEnv(
        data=data,
        initial_capital=100_000.0,
        window_size=20,
        partial_fills=False,
        apply_slippage=False,
        reward_function="sharpe_ratio",
    )
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))  # open long
    for _ in range(5):
        obs, r, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    # Hold while price rises — sharpe-based reward should be nonzero
    assert abs(r) > 1e-6, "sharpe_ratio path should give nonzero hold reward on rising price"
