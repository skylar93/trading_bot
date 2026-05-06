"""
Phase 8-Beta Stage 1 — reward shaping regression + unit tests.

Test 1: inactivity_penalty accumulates correctly on hold-only policy.
Test 2: defaults (inactivity_penalty=0, sharpe_clip_value=10) produce
        byte-identical reward stream to the pre-PR baseline fixture.
Test 3: sharpe_clip_value=2.0 caps the sharpe component at ±2.
Test 4: both knobs at defaults — _calculate_risk_adjusted_reward output
        byte-identical to baseline fixture (regression guard).
"""

import numpy as np
import pandas as pd
import pytest

from envs.single_asset_rl_env import SingleAssetRLTradingEnv

FIXTURE = "tests/fixtures/reward_stream_baseline.npy"
SEED = 42
N_STEPS = 150


def _make_env(**kwargs) -> SingleAssetRLTradingEnv:
    df = pd.read_csv("test_data.csv", index_col=0, parse_dates=True)
    defaults = dict(
        data=df,
        initial_capital=10000.0,
        window_size=20,
        risk_adjusted_reward=True,
        sharpe_weight=0.1,
        drawdown_penalty=True,
        cost_model="spot_taker",
    )
    defaults.update(kwargs)
    return SingleAssetRLTradingEnv(**defaults)


def _run_episode(env, n_steps=N_STEPS, seed=SEED):
    """Run n_steps with a fixed RNG, returning reward array."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(n_steps):
        action = np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            env.reset()
    return np.array(rewards, dtype=np.float64)


# ---------------------------------------------------------------------------
# Test 1: inactivity_penalty accumulates on a hold-only (flat) policy.
# ---------------------------------------------------------------------------

def test_inactivity_penalty_hold_only():
    penalty = 0.001
    env = _make_env(
        inactivity_penalty=penalty,
        risk_adjusted_reward=False,
        drawdown_penalty=False,
    )
    obs, _ = env.reset(seed=SEED)

    # Force flat policy: action=0 → no position
    n = 40
    inactivity_count = 0
    for _ in range(n):
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if abs(env.current_position) < 0.1:
            inactivity_count += 1
        if terminated or truncated:
            break

    # At least some steps should have been flat; each flat step loses `penalty`
    assert inactivity_count > 0, "Expected some flat steps"
    # All flat-step rewards should be exactly -penalty (no other components)
    env2 = _make_env(
        inactivity_penalty=penalty,
        risk_adjusted_reward=False,
        drawdown_penalty=False,
    )
    obs2, _ = env2.reset(seed=SEED)
    for _ in range(n):
        action = np.array([0.0], dtype=np.float32)
        obs2, reward2, term2, trunc2, _ = env2.step(action)
        if abs(env2.current_position) < 0.1:
            assert abs(reward2 - (-penalty)) < 1e-9, (
                f"Flat-step reward {reward2} != -penalty {-penalty}"
            )
        if term2 or trunc2:
            break


# ---------------------------------------------------------------------------
# Test 2: defaults produce byte-identical rewards to the baseline fixture.
# ---------------------------------------------------------------------------

def test_defaults_byte_identical_to_baseline():
    baseline = np.load(FIXTURE)
    env = _make_env()  # inactivity_penalty=0.0, sharpe_clip_value=10.0 (defaults)
    result = _run_episode(env)
    assert len(result) == len(baseline), (
        f"Length mismatch: {len(result)} vs {len(baseline)}"
    )
    np.testing.assert_array_equal(
        result,
        baseline,
        err_msg="Default reward stream diverged from baseline fixture — backward-compat broken",
    )


# ---------------------------------------------------------------------------
# Test 3: sharpe_clip_value=2.0 caps sharpe contribution at ±2.
# ---------------------------------------------------------------------------

def test_sharpe_clip_value_caps_sharpe():
    env_clipped = _make_env(sharpe_clip_value=2.0)
    env_default = _make_env(sharpe_clip_value=10.0)

    rng = np.random.default_rng(SEED)
    env_clipped.reset(seed=SEED)
    env_default.reset(seed=SEED)

    found_difference = False
    for _ in range(N_STEPS):
        action = np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)

        # Step both envs with same action (same RNG step for both)
        debug_clipped: dict = {}
        debug_default: dict = {}

        # reset rng so both get same action
        rng_step = np.random.default_rng(SEED + _)
        a = np.array([rng_step.uniform(-1.0, 1.0)], dtype=np.float32)

        _, r_clipped, tc, trc, _ = env_clipped.step(a)
        _, r_default, td, trd, _ = env_default.step(a)

        if tc or trc:
            env_clipped.reset()
        if td or trd:
            env_default.reset()

        # With clip=2 the sharpe component is bounded at ±0.2 (weight=0.1 × clip=2)
        # so total reward should differ from default when sharpe was >2 in magnitude
        if abs(r_clipped - r_default) > 1e-9:
            found_difference = True
            break

    assert found_difference, (
        "sharpe_clip_value=2.0 produced identical rewards to clip=10.0 throughout "
        "— clip is not active or not wired correctly"
    )


# ---------------------------------------------------------------------------
# Test 4: explicit defaults → byte-identical regression (guard against drift).
# ---------------------------------------------------------------------------

def test_explicit_defaults_byte_identical():
    baseline = np.load(FIXTURE)
    env = _make_env(inactivity_penalty=0.0, sharpe_clip_value=10.0)
    result = _run_episode(env)
    assert len(result) == len(baseline)
    np.testing.assert_array_equal(
        result,
        baseline,
        err_msg="Explicit-defaults reward stream diverged from fixture — regression",
    )
