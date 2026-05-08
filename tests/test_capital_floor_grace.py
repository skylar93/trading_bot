"""Phase 8 capital floor grace period bug fix tests.

Regression: ensures the grace period (min_episode_steps before capital floor
termination is allowed) counts from the episode's actual start step, not
from window_size. Bug was: random_start episodes had their grace period
evaluate as already-expired since start_step >> window_size in absolute
terms.
"""

import numpy as np
import pandas as pd
import pytest
from envs.single_asset_rl_env import SingleAssetRLTradingEnv


def _make_data(n: int = 1000) -> pd.DataFrame:
    np.random.seed(0)
    price = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    price = np.maximum(price, 1.0)
    return pd.DataFrame({
        "$open": price, "$high": price * 1.001, "$low": price * 0.999,
        "$close": price, "$volume": np.ones(n) * 1000.0,
    })


def _make_env(data, **kwargs):
    defaults = dict(
        initial_capital=100_000.0,
        window_size=20,
        min_episode_steps=30,
        max_position_size=1.0,
        partial_fills=False,
        apply_slippage=False,
    )
    defaults.update(kwargs)
    return SingleAssetRLTradingEnv(data=data, **defaults)


# ---- Test 1: episode_start_step set correctly in reset ----

def test_episode_start_step_fixed_start():
    """Fixed-start: _episode_start_step == window_size."""
    env = _make_env(_make_data(200))
    env.reset(seed=0)
    assert env._episode_start_step == env.window_size


def test_episode_start_step_random_start():
    """Random_start: _episode_start_step matches the picked current_step."""
    env = _make_env(_make_data(1000))
    env.reset(seed=0, options={"random_start": True})
    assert env._episode_start_step == env.current_step
    assert env._episode_start_step > env.window_size


# ---- Test 2: backward compat — fixed-start grace period unchanged ----

def test_fixed_start_grace_still_blocks_early_capital_floor():
    """Fixed-start episode: capital_floor termination should be delayed
    until min_episode_steps absolute steps elapsed (= same as old behavior)."""
    env = _make_env(_make_data(200))
    env.reset(seed=0)
    env.current_capital = env.initial_capital * 0.3  # well below 50% floor

    for i in range(env.min_episode_steps - 1):
        action = np.array([0.0], dtype=np.float32)
        _, _, term, trunc, _ = env.step(action)
        if i < env.min_episode_steps - 5:
            assert not term, (
                f"Grace period broken: terminated at i={i} (step={env.current_step}, "
                f"start={env._episode_start_step}, min_steps={env.min_episode_steps})"
            )


# ---- Test 3: random_start episode now grace-protected (THE BUG FIX) ----

def test_random_start_grace_protects_first_min_episode_steps():
    """Random_start episode: capital_floor termination MUST be delayed for
    min_episode_steps steps from episode start, not from window_size."""
    env = _make_env(_make_data(1000))
    env.reset(seed=0, options={"random_start": True})
    start = env._episode_start_step
    assert start > env.window_size, "test prerequisite: random_start picked non-trivial start"

    env.current_capital = env.initial_capital * 0.3

    for i in range(env.min_episode_steps - 5):
        action = np.array([0.0], dtype=np.float32)
        _, _, term, trunc, _ = env.step(action)
        rel_step = env.current_step - start
        if rel_step < env.min_episode_steps:
            assert not term, (
                f"BUG REGRESSION: random_start episode terminated at rel_step={rel_step} "
                f"(< min_episode_steps={env.min_episode_steps}). "
                f"absolute current_step={env.current_step}, start={start}."
            )


# ---- Test 4: termination DOES fire after min_episode_steps from random start ----

def test_random_start_grace_expires_after_min_episode_steps():
    """Random_start: after min_episode_steps from episode start with capital
    below floor, termination MUST fire (no infinite grace)."""
    env = _make_env(_make_data(1000))
    env.reset(seed=0, options={"random_start": True})
    start = env._episode_start_step
    env.current_capital = env.initial_capital * 0.3

    terminated_at = None
    for i in range(env.min_episode_steps + 50):
        action = np.array([0.0], dtype=np.float32)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            terminated_at = env.current_step - start
            break
    assert terminated_at is not None, "Episode should have terminated eventually"
    assert terminated_at >= env.min_episode_steps, \
        f"Terminated too early: rel_step={terminated_at} < min={env.min_episode_steps}"


# ---- Test 5: integration — random action × random_start mean episode length ----

def test_random_start_random_action_episodes_substantially_longer_than_buggy_baseline():
    """Integration regression: prior to fix, random_start × random action
    on a 730-row slice gave mean episode length ≈ 11 steps. Post-fix, with
    grace period correctly applied, episodes should average meaningfully
    longer (>= min_episode_steps when capital floor is the limiter)."""
    env = _make_env(_make_data(730))
    lengths = []
    for ep in range(20):
        env.reset(seed=ep, options={"random_start": True})
        n_steps = 0
        while True:
            a = env.action_space.sample()
            _, _, term, trunc, _ = env.step(a)
            n_steps += 1
            if term or trunc:
                break
        lengths.append(n_steps)
    mean_len = float(np.mean(lengths))
    assert mean_len >= 30.0, (
        f"Mean episode length {mean_len:.1f} < 30. "
        f"Either grace fix didn't take effect or test data is degenerate. "
        f"Distribution: {sorted(lengths)}"
    )


# ---- Test 6: ds_len boundary — episode runs to end of data ----

def test_random_start_episode_can_run_to_end_of_slice():
    """If capital is healthy and policy is benign, random_start episode
    should run to end of data slice (current_step >= ds_len)."""
    env = _make_env(_make_data(200))
    env.reset(seed=0, options={"random_start": True})
    start = env._episode_start_step
    expected_max_len = env._ds_len() - start

    n_steps = 0
    while True:
        a = np.array([0.0], dtype=np.float32)
        _, _, term, trunc, _ = env.step(a)
        n_steps += 1
        if term or trunc:
            break
        if n_steps > 10_000:
            pytest.fail("episode did not terminate")
    assert n_steps == expected_max_len, \
        f"Hold-action episode should run to end of data: got {n_steps}, expected {expected_max_len}"


# ---- Test 7: backward compat — reset state complete after fix ----

def test_reset_state_complete_after_fix():
    """After fix, reset() still sets all required state (no missed initializations)."""
    env = _make_env(_make_data(200))
    env.reset(seed=0)
    assert hasattr(env, '_episode_start_step')
    assert env._episode_start_step == env.window_size
    assert env.current_position == 0.0
    assert env.current_capital == env.initial_capital
    assert env.done is False
    assert env._gate_fires == 0


# ---- Test 8: portfolio FORCE_TERMINATION path also fixed (line 764) ----

def test_force_termination_grace_random_start():
    """The FORCE_TERMINATION path (portfolio_value <= CRITICAL_LOW_THRESHOLD)
    also uses min_steps_elapsed which had the same bug. Fix should apply here too."""
    env = _make_env(_make_data(1000))
    env.reset(seed=0, options={"random_start": True})
    start = env._episode_start_step
    assert start > env.window_size

    env.portfolio_value = 0.01
    env.current_capital = 0.01

    a = np.array([0.0], dtype=np.float32)
    _, _, term, trunc, _ = env.step(a)
    rel_step = env.current_step - start
    if rel_step < env.min_episode_steps:
        assert not term, (
            f"FORCE_TERMINATION grace not respected: terminated at rel_step={rel_step}"
        )
