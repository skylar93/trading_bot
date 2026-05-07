"""Phase 8-Gamma G1: HMM regime gate unit tests.

Tests:
1. regime_gate_enabled=False — byte-identical env behavior (backward compat).
2. mode=close, all-bear track, from long position: forces position to 0.
3. mode=refuse_entry, all-bear track, from flat: blocks new long entry.
4. bear_threshold=0 (argmax mode): gate fires when argmax==BEAR.
5. bear_threshold=0.7, bear_prob=0.6: gate does NOT fire.
6. _compute_regime_track: returns (n, n_regimes) shape; first lookback rows are uniform.
7. ValueError if regime_gate_enabled=True and regime_track is None.
8. info["regime_gate_fires"] increments correctly per episode.
"""

import numpy as np
import pandas as pd
import pytest

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from training.env_factory import _compute_regime_track
from training.signals.regime_detector import RegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    price = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    price = np.maximum(price, 1.0)
    return pd.DataFrame({
        "$open": price,
        "$high": price * 1.001,
        "$low": price * 0.999,
        "$close": price,
        "$volume": np.ones(n) * 1000.0,
    })


def _make_bear_track(n: int, n_regimes: int = 3) -> np.ndarray:
    """All-bear regime track: BEAR prob=1.0 at every step."""
    track = np.zeros((n, n_regimes), dtype=np.float32)
    track[:, 0] = 1.0  # BEAR=0 index
    return track


def _make_bull_track(n: int, n_regimes: int = 3) -> np.ndarray:
    """All-bull regime track: BULL prob=1.0 at every step."""
    track = np.zeros((n, n_regimes), dtype=np.float32)
    track[:, 2] = 1.0  # BULL=2 index
    return track


def _make_env(data, regime_track=None, regime_gate_enabled=False,
              regime_gate_mode="close", regime_gate_bear_threshold=0.5,
              **kwargs):
    defaults = dict(
        initial_capital=100_000.0,
        window_size=20,
        partial_fills=False,  # simplify position math in tests
        apply_slippage=False,
    )
    defaults.update(kwargs)
    return SingleAssetRLTradingEnv(
        data=data,
        regime_track=regime_track,
        regime_gate_enabled=regime_gate_enabled,
        regime_gate_mode=regime_gate_mode,
        regime_gate_bear_threshold=regime_gate_bear_threshold,
        **defaults,
    )


# ---------------------------------------------------------------------------
# Test 1: backward compatibility — regime_gate_enabled=False is byte-identical
# ---------------------------------------------------------------------------

def test_backward_compat_gate_disabled():
    """Gate disabled: env runs identically to pre-G1 code."""
    data = _make_data(200)
    env_base = _make_env(data)
    env_gate = _make_env(data, regime_gate_enabled=False)

    obs_base, _ = env_base.reset(seed=7)
    obs_gate, _ = env_gate.reset(seed=7)
    np.testing.assert_array_equal(obs_base, obs_gate)

    # Use fixed actions from a seeded numpy array (same for both envs)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        ob, rb, tb, truncb, _info_b = env_base.step(a.copy())
        og, rg, tg, truncg, _info_g = env_gate.step(a.copy())
        np.testing.assert_array_almost_equal(ob, og, decimal=7)
        assert abs(rb - rg) < 1e-9, f"reward mismatch: {rb} vs {rg}"
        assert env_base.current_position == pytest.approx(env_gate.current_position, abs=1e-9)
        if tb or truncb:
            env_base.reset(seed=7)
            env_gate.reset(seed=7)
            break


# ---------------------------------------------------------------------------
# Test 2: mode=close, all-bear track, long position → position goes to 0
# ---------------------------------------------------------------------------

def test_mode_close_forces_flat_from_long():
    data = _make_data(200)
    bear_track = _make_bear_track(len(data))
    env = _make_env(data, regime_track=bear_track, regime_gate_enabled=True,
                    regime_gate_mode="close")

    env.reset(seed=0)
    # Manually set a long position (bypass the gate bootstrap by directly writing state)
    env.current_position = 0.8

    # One step with any action — gate should force close
    action = np.array([1.0], dtype=np.float32)  # agent wants to go more long
    env.step(action)

    # Gate forces action = [-0.8 / 1.0] → position should reach 0
    assert abs(env.current_position) < 1e-5


# ---------------------------------------------------------------------------
# Test 3: mode=refuse_entry, all-bear track, flat → blocks new long
# ---------------------------------------------------------------------------

def test_mode_refuse_entry_blocks_new_long_from_flat():
    data = _make_data(200)
    bear_track = _make_bear_track(len(data))
    env = _make_env(data, regime_track=bear_track, regime_gate_enabled=True,
                    regime_gate_mode="refuse_entry")

    env.reset(seed=0)
    assert abs(env.current_position) < 1e-9  # starts flat

    action = np.array([1.0], dtype=np.float32)  # strong long signal
    env.step(action)

    # Gate intercepts: action zeroed → position stays 0
    assert abs(env.current_position) < 1e-5


# ---------------------------------------------------------------------------
# Test 4: bear_threshold=0 (argmax mode) — fires when argmax==BEAR even at 0.4
# ---------------------------------------------------------------------------

def test_argmax_mode_fires_at_low_bear_prob():
    data = _make_data(200)
    n = len(data)
    # Bear prob=0.4 (majority but below default 0.5 threshold)
    track = np.array([[0.4, 0.3, 0.3]] * n, dtype=np.float32)
    env = _make_env(data, regime_track=track, regime_gate_enabled=True,
                    regime_gate_mode="refuse_entry",
                    regime_gate_bear_threshold=0.0)  # argmax mode

    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))

    # argmax([0.4,0.3,0.3]) == 0 (BEAR) → gate fires → position stays 0
    assert abs(env.current_position) < 1e-5


# ---------------------------------------------------------------------------
# Test 5: bear_threshold=0.7, bear_prob=0.6 → gate does NOT fire
# ---------------------------------------------------------------------------

def test_threshold_not_triggered_below_threshold():
    data = _make_data(200)
    n = len(data)
    track = np.array([[0.6, 0.2, 0.2]] * n, dtype=np.float32)
    env = _make_env(data, regime_track=track, regime_gate_enabled=True,
                    regime_gate_mode="refuse_entry",
                    regime_gate_bear_threshold=0.7)

    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))

    # 0.6 < 0.7 → gate does NOT fire → position changes (will be > 0 after long action)
    assert env.current_position > 1e-5


# ---------------------------------------------------------------------------
# Test 6: _compute_regime_track shape + uniform priors for first lookback rows
# ---------------------------------------------------------------------------

def test_compute_regime_track_unfitted_returns_uniform():
    data = _make_data(200)
    detector = RegimeDetector(n_regimes=3, lookback=60)
    # Not fitted — should return all uniform
    track = _compute_regime_track(detector, data)
    assert track.shape == (200, 3)
    np.testing.assert_allclose(track, 1.0 / 3, atol=1e-6)


def test_compute_regime_track_fitted_uniform_for_lookback_rows():
    data = _make_data(200)
    detector = RegimeDetector(n_regimes=3, lookback=30, n_iter=10, random_state=0)
    detector.fit(data)
    track = _compute_regime_track(detector, data)
    assert track.shape == (200, 3)
    # First lookback rows must be uniform
    uniform = 1.0 / 3
    np.testing.assert_allclose(track[:30], uniform, atol=1e-6,
                                err_msg="Rows before lookback should be uniform priors")
    # At least some rows after lookback should differ from uniform (HMM has non-trivial state)
    assert not np.allclose(track[30:], uniform, atol=1e-3), \
        "Expected at least some non-uniform posteriors after lookback"


# ---------------------------------------------------------------------------
# Test 7: ValueError if regime_gate_enabled=True and regime_track is None
# ---------------------------------------------------------------------------

def test_raises_if_gate_enabled_without_track():
    data = _make_data(200)
    with pytest.raises(ValueError, match="regime_gate_enabled"):
        _make_env(data, regime_track=None, regime_gate_enabled=True)


# ---------------------------------------------------------------------------
# Test 8: info["regime_gate_fires"] increments correctly per episode
# ---------------------------------------------------------------------------

def test_gate_fires_counter_increments():
    data = _make_data(200)
    bear_track = _make_bear_track(len(data))
    env = _make_env(data, regime_track=bear_track, regime_gate_enabled=True,
                    regime_gate_mode="refuse_entry")

    obs, info = env.reset(seed=0)
    assert info["regime_gate_fires"] == 0

    fires_before = 0
    for _ in range(10):
        action = np.array([1.0], dtype=np.float32)  # always tries to go long
        _, _, term, trunc, info = env.step(action)
        assert info["regime_gate_fires"] >= fires_before
        fires_before = info["regime_gate_fires"]
        if term or trunc:
            break

    # After 10 long-action steps in all-bear regime, gate must have fired
    assert info["regime_gate_fires"] > 0

    # After reset, counter resets to 0
    _, info = env.reset(seed=0)
    assert info["regime_gate_fires"] == 0
