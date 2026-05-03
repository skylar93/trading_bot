"""Phase 8-Alpha: futures_maker fee model correctness.

Verifies:
- Round-trip cost on buy-then-sell at constant price = 2 * trading_fee (no slippage)
- With trading_fee=0.00018: round-trip ≈ 0.00036 of notional (tolerance 1e-9)
- apply_slippage=True input is forced to False under futures_maker
- _calculate_dynamic_fee returns trading_fee unchanged regardless of trade size
"""

import pytest
import numpy as np
import pandas as pd

from envs.single_asset_rl_env import SingleAssetRLTradingEnv


MAKER_FEE = 0.00018
INITIAL = 100_000.0
CONST_PRICE = 50_000.0  # arbitrary constant BTC price


def _make_flat_data(n: int = 200) -> pd.DataFrame:
    """Flat price data — simplifies round-trip P&L verification."""
    return pd.DataFrame({
        "$open":   np.full(n, CONST_PRICE),
        "$high":   np.full(n, CONST_PRICE),
        "$low":    np.full(n, CONST_PRICE),
        "$close":  np.full(n, CONST_PRICE),
        "$volume": np.full(n, 1e6),
    })


def _make_futures_env(**kwargs) -> SingleAssetRLTradingEnv:
    defaults = dict(
        data=_make_flat_data(),
        window_size=10,
        initial_capital=INITIAL,
        trading_fee=MAKER_FEE,
        cost_model="futures_maker",
        partial_fills=False,
    )
    defaults.update(kwargs)
    return SingleAssetRLTradingEnv(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_round_trip_cost_equals_2x_fee():
    """Buy 0.5 BTC then sell 0.5 BTC at flat price → capital loss = 2 * fee * notional."""
    env = _make_futures_env()
    env.reset(seed=0)

    capital_before = env.current_capital

    # Buy: action=0.5 (50% of max_position_size=1.0 → actual_change ≈ 0.5)
    # Use a deterministic fixed action; partial_fills=False so fill is 100%
    buy_action = np.array([0.5], dtype=np.float32)
    env.step(buy_action)
    position_after_buy = env.current_position

    # Sell everything
    sell_action = np.array([-position_after_buy], dtype=np.float32)
    # cap to action space
    sell_action = np.clip(sell_action, env.action_space.low, env.action_space.high)
    env.step(sell_action)

    capital_after = env.current_capital
    notional = position_after_buy * CONST_PRICE

    expected_cost = 2 * MAKER_FEE * notional
    actual_cost = capital_before - capital_after

    assert actual_cost == pytest.approx(expected_cost, abs=1e-6), (
        f"Round-trip cost {actual_cost:.8f} ≠ expected {expected_cost:.8f}"
    )


def test_round_trip_fraction_is_0_00036():
    """With fee=0.00018: round-trip cost / notional ≈ 0.00036 (tolerance 1e-9)."""
    env = _make_futures_env()
    env.reset(seed=0)

    capital_before = env.current_capital

    buy_action = np.array([0.5], dtype=np.float32)
    env.step(buy_action)
    position_after_buy = env.current_position
    notional = position_after_buy * CONST_PRICE

    sell_action = np.clip(
        np.array([-position_after_buy], dtype=np.float32),
        env.action_space.low, env.action_space.high
    )
    env.step(sell_action)

    fraction = (capital_before - env.current_capital) / notional
    assert fraction == pytest.approx(2 * MAKER_FEE, abs=1e-9)


def test_dynamic_fee_returns_trading_fee_unchanged():
    """_calculate_dynamic_fee must return exactly trading_fee for any trade size."""
    env = _make_futures_env()
    env.reset(seed=0)

    for trade_value in [0.0, 1.0, 1_000.0, 100_000.0, 1e9]:
        result = env._calculate_dynamic_fee(trade_value)
        assert result == MAKER_FEE, (
            f"_calculate_dynamic_fee({trade_value}) returned {result}, expected {MAKER_FEE}"
        )


def test_apply_slippage_forced_false():
    env = _make_futures_env(apply_slippage=True)
    assert env.apply_slippage is False


def test_no_slippage_in_execution():
    """Executed price must equal close price (no slippage applied)."""
    env = _make_futures_env()
    env.reset(seed=0)

    buy_action = np.array([0.3], dtype=np.float32)
    _, _, _, _, info = env.step(buy_action)

    if env.trades:
        last = env.trades[-1]
        assert last["slippage"] == 0.0
        assert last["executed_price"] == pytest.approx(CONST_PRICE, rel=1e-9)
