"""
Week 42: Portfolio invariant property tests.

For 1 000 random price series, verifies:
  1. portfolio_value == current_capital + current_position * price  (at every step)
  2. Δcash + Δposition × exec_price + fee == 0  (per-trade value conservation)

Targets SingleAssetRLTradingEnv with apply_slippage=False for exact arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from envs.single_asset_rl_env import SingleAssetRLTradingEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 5
FEE = 0.001
INITIAL_CAPITAL = 10_000.0
N_SERIES = 1_000
STEPS_PER_SERIES = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_prices(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100.0 * np.cumprod(1.0 + returns)
    return np.clip(prices, 1.0, None).tolist()


def _make_env(prices) -> SingleAssetRLTradingEnv:
    """Build a minimal env from a price list; no slippage for exact math."""
    n = len(prices)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "$open": prices,
            "$high": prices,
            "$low": prices,
            "$close": prices,
            "$volume": [1e6] * n,
        },
        index=idx,
    )
    return SingleAssetRLTradingEnv(
        data=df,
        window_size=WINDOW_SIZE,
        initial_capital=INITIAL_CAPITAL,
        trading_fee=FEE,
        apply_slippage=False,
    )


def _portfolio_price(env: SingleAssetRLTradingEnv) -> float:
    """Return the price the env used to mark portfolio_value after the last step."""
    idx = max(0, min(env.current_step - 1, len(env.data) - 1))
    return float(env.data.iloc[idx]["$close"])


# ---------------------------------------------------------------------------
# 42.2a  portfolio_value invariant — deterministic single episode
# ---------------------------------------------------------------------------

class TestPortfolioValueInvariantEnv:

    def test_invariant_buy_and_hold(self):
        """Buying and holding: invariant holds at every step."""
        prices = _random_prices(WINDOW_SIZE + 20, seed=0)
        env = _make_env(prices)
        env.reset()

        for _ in range(15):
            _, _, done, trunc, _ = env.step(np.array([0.5]))
            price = _portfolio_price(env)
            expected = env.current_capital + env.current_position * price
            assert env.portfolio_value == pytest.approx(expected, rel=1e-6), (
                f"Invariant broken: pv={env.portfolio_value:.8f} "
                f"expected={expected:.8f}"
            )
            if done or trunc:
                break

    def test_invariant_sell_only(self):
        """Selling pressure from max long: invariant still holds."""
        prices = _random_prices(WINDOW_SIZE + 20, seed=1)
        env = _make_env(prices)
        env.reset()

        for step in range(15):
            action = np.array([1.0]) if step < 5 else np.array([-1.0])
            _, _, done, trunc, _ = env.step(action)
            price = _portfolio_price(env)
            expected = env.current_capital + env.current_position * price
            assert env.portfolio_value == pytest.approx(expected, rel=1e-6)
            if done or trunc:
                break

    def test_invariant_zero_action(self):
        """No action (0.0): position stays flat, invariant holds."""
        prices = _random_prices(WINDOW_SIZE + 20, seed=2)
        env = _make_env(prices)
        env.reset()

        for _ in range(10):
            _, _, done, trunc, _ = env.step(np.array([0.0]))
            price = _portfolio_price(env)
            expected = env.current_capital + env.current_position * price
            assert env.portfolio_value == pytest.approx(expected, rel=1e-6)
            if done or trunc:
                break


# ---------------------------------------------------------------------------
# 42.2b  Per-trade value conservation
# ---------------------------------------------------------------------------

class TestPerTradeConservation:
    """Δcash + Δposition × exec_price + fee == 0 for every executed trade."""

    def _run_conservation_check(self, env: SingleAssetRLTradingEnv, n_steps: int, rng):
        env.reset()
        prev_capital = env.current_capital
        prev_position = env.current_position

        for step_i in range(n_steps):
            action = np.array([rng.uniform(-1.0, 1.0)])
            _, _, done, trunc, _ = env.step(action)

            trade_step = env.current_step - 1  # step stored before increment
            trades_this_step = [
                t for t in env.trades if t.get("step") == trade_step
            ]

            if trades_this_step:
                t = trades_this_step[-1]
                d_capital = env.current_capital - prev_capital
                d_position = env.current_position - prev_position
                exec_price = float(t.get("executed_price", t.get("current_price", 0.0)))
                fee = float(t.get("cost", 0.0))

                conservation_err = d_capital + d_position * exec_price + fee
                assert abs(conservation_err) < 1e-6, (
                    f"Step {step_i}: conservation error = {conservation_err:.2e} "
                    f"(Δcash={d_capital:.4f}, Δpos={d_position:.6f}, "
                    f"exec={exec_price:.4f}, fee={fee:.4f})"
                )

            prev_capital = env.current_capital
            prev_position = env.current_position
            if done or trunc:
                break

    def test_conservation_fixed_seed(self):
        prices = _random_prices(WINDOW_SIZE + 50, seed=42)
        env = _make_env(prices)
        rng = np.random.default_rng(42)
        self._run_conservation_check(env, n_steps=40, rng=rng)

    def test_conservation_buy_only(self):
        """All-buy actions conserve value."""
        prices = _random_prices(WINDOW_SIZE + 30, seed=7)
        env = _make_env(prices)
        rng = np.random.default_rng(7)

        env.reset()
        prev_capital = env.current_capital
        prev_position = env.current_position

        for step_i in range(20):
            _, _, done, trunc, _ = env.step(np.array([1.0]))

            trade_step = env.current_step - 1
            trades_this_step = [t for t in env.trades if t.get("step") == trade_step]
            if trades_this_step:
                t = trades_this_step[-1]
                d_cap = env.current_capital - prev_capital
                d_pos = env.current_position - prev_position
                ep = float(t.get("executed_price", t.get("current_price", 0.0)))
                fee = float(t.get("cost", 0.0))
                assert abs(d_cap + d_pos * ep + fee) < 1e-6
            prev_capital = env.current_capital
            prev_position = env.current_position
            if done or trunc:
                break


# ---------------------------------------------------------------------------
# 42.2c  1 000-series property test
# ---------------------------------------------------------------------------

class TestPropertyTest1000Series:
    """Run both invariants across 1 000 random price series."""

    def test_portfolio_invariant_1000_series(self):
        """portfolio_value == capital + position * price for all 1 000 series."""
        failures = []

        for seed in range(N_SERIES):
            prices = _random_prices(WINDOW_SIZE + STEPS_PER_SERIES + 2, seed=seed)
            env = _make_env(prices)
            env.reset()
            rng = np.random.default_rng(seed)

            for _ in range(STEPS_PER_SERIES):
                action = np.array([rng.uniform(-1.0, 1.0)])
                _, _, done, trunc, _ = env.step(action)

                price = _portfolio_price(env)
                expected = env.current_capital + env.current_position * price
                err = abs(env.portfolio_value - expected)
                if err > 1e-5:
                    failures.append(
                        f"seed={seed}: pv={env.portfolio_value:.6f} expected={expected:.6f} err={err:.2e}"
                    )
                if done or trunc:
                    break

        assert not failures, (
            f"{len(failures)} invariant violations across 1000 series:\n"
            + "\n".join(failures[:5])
        )

    def test_conservation_1000_series(self):
        """Per-trade conservation holds for all 1 000 series."""
        failures = []

        for seed in range(N_SERIES):
            prices = _random_prices(WINDOW_SIZE + STEPS_PER_SERIES + 2, seed=seed)
            env = _make_env(prices)
            env.reset()
            rng = np.random.default_rng(seed)

            prev_cap = env.current_capital
            prev_pos = env.current_position

            for step_i in range(STEPS_PER_SERIES):
                action = np.array([rng.uniform(-1.0, 1.0)])
                _, _, done, trunc, _ = env.step(action)

                trade_step = env.current_step - 1
                for t in env.trades:
                    if t.get("step") == trade_step:
                        d_cap = env.current_capital - prev_cap
                        d_pos = env.current_position - prev_pos
                        ep = float(t.get("executed_price", t.get("current_price", 0.0)))
                        fee = float(t.get("cost", 0.0))
                        err = abs(d_cap + d_pos * ep + fee)
                        if err > 1e-6:
                            failures.append(
                                f"seed={seed} step={step_i}: err={err:.2e}"
                            )
                        break

                prev_cap = env.current_capital
                prev_pos = env.current_position
                if done or trunc:
                    break

        assert not failures, (
            f"{len(failures)} conservation violations across 1000 series:\n"
            + "\n".join(failures[:5])
        )
