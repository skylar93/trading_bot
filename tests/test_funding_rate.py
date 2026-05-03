"""Phase 8-Alpha: funding rate accrual tests.

Verifies:
- Constant long position 0.5 over 24 steps accrues 3 × funding payments (every 8 steps)
- Short position receives funding (capital increases)
- Flat position (0) accrues no funding
- reset() clears the step counter
"""

import pytest
import numpy as np
import pandas as pd

from envs.single_asset_rl_env import SingleAssetRLTradingEnv


FUNDING_RATE = 0.0001   # 0.01%/8h
PRICE = 100.0           # flat price for easy maths
INITIAL = 10_000.0


def _make_flat_data(n: int = 300) -> pd.DataFrame:
    return pd.DataFrame({
        "$open":   np.full(n, PRICE),
        "$high":   np.full(n, PRICE),
        "$low":    np.full(n, PRICE),
        "$close":  np.full(n, PRICE),
        "$volume": np.full(n, 1e6),
    })


def _make_env(**kwargs) -> SingleAssetRLTradingEnv:
    return SingleAssetRLTradingEnv(
        data=_make_flat_data(),
        window_size=10,
        initial_capital=INITIAL,
        trading_fee=0.0,          # zero fees so funding is isolated
        cost_model="futures_maker",
        apply_slippage=False,
        partial_fills=False,
        funding_rate_per_8h=FUNDING_RATE,
        **kwargs,
    )


def _open_and_hold(env, open_action: float, hold_steps: int):
    """Buy once, then hold with action=0 for hold_steps steps."""
    env.step(np.array([open_action], dtype=np.float32))
    hold = np.array([0.0], dtype=np.float32)
    for _ in range(hold_steps):
        env.step(hold)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_long_funding_3_payments_in_24_steps():
    """Long 0.5 BTC held for 24 hold-steps → 3 accruals (at hold-steps 8, 16, 24).

    Funding fires every 8 env steps.  We open on step 0 (env step 1 from perspective
    of _steps_since_funding) then do 24 hold steps, so funding fires at hold-steps 7,
    15, 23 (i.e., when the running counter hits 8).  That gives exactly 3 payments.
    """
    env = _make_env()
    env.reset(seed=0)

    # Buy 0.5 BTC: costs 50 capital (fee=0), position = 0.5
    env.step(np.array([0.5], dtype=np.float32))
    position = env.current_position   # ≈ 0.5
    capital_after_buy = env.current_capital   # 10000 - 50 = 9950
    funding_counter_after_buy = env._steps_since_funding  # 1

    # Hold for 23 more steps: funding fires at running counters 8, 16, 24
    # (buy step incremented to 1, so fires happen at hold-steps 7, 15, 23)
    hold = np.array([0.0], dtype=np.float32)
    for _ in range(23):
        env.step(hold)

    notional = position * PRICE       # 0.5 * 100 = 50
    expected_funding = 3 * notional * FUNDING_RATE   # 3 × 50 × 0.0001 = 0.015
    actual_funding = capital_after_buy - env.current_capital

    assert actual_funding == pytest.approx(expected_funding, abs=1e-6), (
        f"funding={actual_funding:.8f}, expected≈{expected_funding:.8f}"
    )


def test_short_funding_received():
    """Short position receives funding (capital net > capital after open)."""
    env = _make_env()
    env.reset(seed=0)

    # Open short -0.5 (sells 0.5 BTC → capital +50)
    env.step(np.array([-0.5], dtype=np.float32))
    capital_after_open = env.current_capital

    # Hold for 8 more steps → one funding payment fires; short receives
    hold = np.array([0.0], dtype=np.float32)
    for _ in range(8):
        env.step(hold)

    # After open (short sold, capital increased), then funding received → capital > after_open
    assert env.current_capital > capital_after_open, (
        "Short position should receive funding (capital should increase)"
    )


def test_flat_position_no_funding():
    """Zero position → no funding regardless of steps."""
    env = _make_env()
    env.reset(seed=0)
    capital_before = env.current_capital

    hold = np.array([0.0], dtype=np.float32)
    for _ in range(24):
        env.step(hold)

    assert env.current_capital == pytest.approx(capital_before, abs=1e-9), (
        "Zero position should incur no funding cost"
    )


def test_reset_clears_funding_counter():
    """After reset(), the funding step counter starts from 0."""
    env = _make_env()
    env.reset(seed=0)

    # Advance 7 steps without a trade (action=0)
    hold = np.array([0.0], dtype=np.float32)
    for _ in range(7):
        env.step(hold)
    assert env._steps_since_funding == 7

    env.reset(seed=1)
    assert env._steps_since_funding == 0


def test_funding_fires_at_step_8_exactly():
    """Counter increments from 0; funding debits on the 8th step then resets to 0."""
    env = _make_env()
    env.reset(seed=0)

    # Buy 0.5 BTC then hold; measure capital changes to isolate funding
    env.step(np.array([0.5], dtype=np.float32))   # counter = 1 after this step
    capital_after_buy = env.current_capital
    hold = np.array([0.0], dtype=np.float32)

    for i in range(1, 9):
        env.step(hold)
        if i < 7:
            # no funding yet (counter goes 2→3→...→7)
            assert env.current_capital == pytest.approx(capital_after_buy, abs=1e-6), (
                f"No funding should have fired before the 8th step (hold step {i})"
            )
        elif i == 7:
            # This is the step where counter reaches 8 → fires → resets to 0
            assert env._steps_since_funding == 0, "counter should reset after accrual"
            assert env.current_capital < capital_after_buy, "funding should fire at 8th step"
