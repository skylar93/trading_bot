"""Week 37: Hard risk limit enforcement in SingleAssetRLTradingEnv.

Tests verify that the risk manager's hard limits (stop loss, trailing stop,
max drawdown) actually force-close positions and/or terminate episodes
when breached inside step().
"""

import numpy as np
import pandas as pd
import pytest

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from risk_management import create_risk_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(prices: np.ndarray) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close price array."""
    return pd.DataFrame(
        {
            "$open": prices,
            "$high": prices * 1.001,
            "$low": prices * 0.999,
            "$close": prices,
            "$volume": np.ones(len(prices)) * 1000.0,
        }
    )


@pytest.fixture
def declining_data():
    """200-bar linearly declining price: 100 → 50."""
    prices = np.linspace(100.0, 50.0, 200)
    return _make_data(prices)


@pytest.fixture
def rising_data():
    """200-bar linearly rising then dropping price for trailing-stop test."""
    rise = np.linspace(100.0, 150.0, 100)
    fall = np.linspace(150.0, 100.0, 100)
    prices = np.concatenate([rise, fall])
    return _make_data(prices)


@pytest.fixture
def flat_data():
    """200-bar flat price at 100."""
    prices = np.ones(200) * 100.0
    return _make_data(prices)


def _make_env(data, rm_config, **env_kwargs):
    rm = create_risk_manager("rl", rm_config)
    defaults = dict(
        initial_capital=10_000.0,
        window_size=20,
        apply_slippage=False,
        partial_fills=False,
        risk_adjusted_reward=False,
        drawdown_penalty=False,
        min_episode_steps=1,
    )
    defaults.update(env_kwargs)
    return SingleAssetRLTradingEnv(data=data, risk_manager=rm, **defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStopLoss:
    def test_stop_loss_triggers_on_declining_price(self, declining_data):
        """Stop loss should force-close a long position as price falls."""
        env = _make_env(
            declining_data,
            {
                "use_stop_loss": True,
                "stop_loss_threshold": 0.05,  # 5% loss
                "use_trailing_stop": False,
                "max_drawdown_pct": 1.0,  # effectively disabled
                "use_forced_liquidation": False,
            },
        )
        env.reset()

        # Open a long position
        env.step(np.array([1.0]))

        hit_stop_loss = False
        for _ in range(100):
            _, _, done, _, info = env.step(np.array([0.0]))
            if info.get("risk_limit_triggered") == "stop_loss":
                hit_stop_loss = True
                break
            if done:
                break

        assert hit_stop_loss, "Stop loss must trigger on persistently declining price"
        assert abs(env.current_position) < 1e-8, "Position must be 0 after stop loss"

    def test_stop_loss_does_not_trigger_without_risk_manager(self, declining_data):
        """Without a risk manager, no risk_limit_triggered key in info."""
        env = SingleAssetRLTradingEnv(
            data=declining_data,
            initial_capital=10_000.0,
            window_size=20,
            apply_slippage=False,
            partial_fills=False,
            risk_adjusted_reward=False,
            min_episode_steps=1,
        )
        env.reset()
        env.step(np.array([1.0]))
        for _ in range(50):
            _, _, done, _, info = env.step(np.array([0.0]))
            assert "risk_limit_triggered" not in info
            if done:
                break

    def test_stop_loss_clears_entry_price(self, declining_data):
        """After stop loss, _entry_price must be None."""
        env = _make_env(
            declining_data,
            {
                "use_stop_loss": True,
                "stop_loss_threshold": 0.05,
                "use_trailing_stop": False,
                "max_drawdown_pct": 1.0,
                "use_forced_liquidation": False,
            },
        )
        env.reset()
        env.step(np.array([1.0]))
        assert env._entry_price is not None  # entry price should have been set

        for _ in range(100):
            _, _, done, _, info = env.step(np.array([0.0]))
            if info.get("risk_limit_triggered") == "stop_loss":
                break
            if done:
                break

        assert env._entry_price is None, "_entry_price must be cleared after stop loss"


class TestTrailingStop:
    def test_trailing_stop_triggers_after_price_reversal(self, rising_data):
        """Trailing stop should trigger after price peaks and then falls enough."""
        env = _make_env(
            rising_data,
            {
                "use_stop_loss": False,
                "use_trailing_stop": True,
                "trailing_stop_buffer": 0.05,  # 5% drop from peak
                "max_drawdown_pct": 1.0,
                "use_forced_liquidation": False,
            },
        )
        env.reset()
        env.step(np.array([1.0]))  # open long at start of rising phase

        hit_trailing = False
        for _ in range(170):
            _, _, done, _, info = env.step(np.array([0.0]))
            if info.get("risk_limit_triggered") == "trailing_stop":
                hit_trailing = True
                break
            if done:
                break

        assert hit_trailing, "Trailing stop must trigger during the falling phase"
        assert abs(env.current_position) < 1e-8, "Position must be 0 after trailing stop"


class TestMaxDrawdown:
    # max_drawdown_pct=0.001 (0.1%) triggers when portfolio drops ~$10 from peak.
    # With 1 unit at ~$95 and price declining 0.25/bar, triggers in ~40 hold steps.
    _MAX_DD = 0.001

    def test_max_drawdown_terminates_episode(self, declining_data):
        """Max drawdown should terminate the episode when portfolio drops too far."""
        env = _make_env(
            declining_data,
            {
                "use_stop_loss": False,
                "use_trailing_stop": False,
                "max_drawdown_pct": self._MAX_DD,
                "use_forced_liquidation": True,
            },
        )
        env.reset()
        env.step(np.array([1.0]))  # open long position

        terminated = False
        for _ in range(150):
            _, _, done, _, info = env.step(np.array([0.0]))
            if info.get("risk_limit_triggered") == "max_drawdown" and done:
                terminated = True
                break
            if done:
                break

        assert terminated, "Max drawdown must terminate the episode"

    def test_max_drawdown_closes_position(self, declining_data):
        """Position must be 0 when max drawdown terminates the episode."""
        env = _make_env(
            declining_data,
            {
                "use_stop_loss": False,
                "use_trailing_stop": False,
                "max_drawdown_pct": self._MAX_DD,
                "use_forced_liquidation": True,
            },
        )
        env.reset()
        env.step(np.array([1.0]))

        for _ in range(150):
            _, _, done, _, info = env.step(np.array([0.0]))
            if info.get("risk_limit_triggered") == "max_drawdown":
                assert abs(env.current_position) < 1e-8
                break
            if done:
                break


class TestRiskManagerReset:
    def test_entry_price_reset_on_env_reset(self, declining_data):
        """env.reset() should clear _entry_price and reset the risk manager."""
        env = _make_env(
            declining_data,
            {
                "use_stop_loss": True,
                "stop_loss_threshold": 0.05,
                "use_trailing_stop": True,
                "trailing_stop_buffer": 0.05,
                "max_drawdown_pct": 1.0,
                "use_forced_liquidation": False,
            },
        )
        env.reset()
        env.step(np.array([1.0]))
        assert env._entry_price is not None

        env.reset()
        assert env._entry_price is None, "_entry_price must be None after reset"
        # Risk manager event counters should be cleared
        metrics = env._risk_manager._get_risk_events_info()
        assert metrics["stop_loss_events"] == 0
        assert metrics["trailing_stop_events"] == 0

    def test_no_cross_episode_state(self, declining_data):
        """Trailing stop state from episode N must not leak into episode N+1."""
        env = _make_env(
            declining_data,
            {
                "use_stop_loss": False,
                "use_trailing_stop": True,
                "trailing_stop_buffer": 0.05,
                "max_drawdown_pct": 1.0,
                "use_forced_liquidation": False,
            },
        )
        env.reset()
        env.step(np.array([1.0]))
        for _ in range(80):
            _, _, done, _, _ = env.step(np.array([0.0]))
            if done:
                break

        # Start fresh
        env.reset()
        # Trailing stop state should be cleared; no false trigger on flat hold
        for _ in range(30):
            _, _, done, _, info = env.step(np.array([0.0]))
            # No position open → risk limits should not trigger
            assert "risk_limit_triggered" not in info or env.current_position == 0.0
            if done:
                break
