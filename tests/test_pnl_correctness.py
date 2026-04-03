"""
Week 42: P&L mathematical correctness — golden tests.

Tests the BaseBacktester with known-good numbers:
  Buy 1.0 unit @ $100 (fee 0.1%) → cost  = $100.10
  Sell 1.0 unit @ $110 (fee 0.1%) → net  = $109.89
  Round-trip P&L                         = $9.79

Also verifies portfolio_value == cash + units * price at every step.
"""

import pytest
import pandas as pd
import numpy as np

from training.backtesting.base_backtester import BaseBacktester
from risk_management.backtesting_risk_manager import BacktestingRiskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEE = 0.001
INITIAL = 10_000.0

# Risk config: no slippage and no min_trade_size so golden numbers are exact
_NO_SLIPPAGE_CFG = BacktestingRiskConfig(slippage_std=0.0, min_trade_size=0.0)


def _bt() -> BaseBacktester:
    """Fresh backtester with no slippage so execution prices are deterministic."""
    bt = BaseBacktester(initial_capital=INITIAL, trading_fee=FEE, risk_config=_NO_SLIPPAGE_CFG)
    bt.reset()
    return bt


def _ts(day: int) -> pd.Timestamp:
    return pd.Timestamp(f"2024-01-{day:02d}")


def _buy(bt: BaseBacktester, price: float, day: int):
    """Buy ~1% of portfolio in 'default' asset at given price."""
    return bt.execute_trade(
        asset="default",
        action=0.01,
        price_data={"default": price},
        timestamp=_ts(day),
    )


def _sell_all(bt: BaseBacktester, price: float, day: int):
    """Sell all holdings of 'default' asset."""
    return bt.execute_trade(
        asset="default",
        action=0.0,
        price_data={"default": price},
        timestamp=_ts(day),
    )


def _pv(bt: BaseBacktester, price: float) -> float:
    return bt.get_portfolio_value({"default": price})


# ---------------------------------------------------------------------------
# 42.1a  Exact cost / revenue numbers
# ---------------------------------------------------------------------------

class TestPnLGoldenNumbers:

    def test_buy_cost_exact(self):
        """Buy 1.0 unit @ $100 → total cost = $100.10 (= $100 + $0.10 fee)."""
        bt = _bt()
        result = _buy(bt, price=100.0, day=1)

        assert result["success"] is True, f"Trade rejected: {result.get('reason')}"
        # cost = trade_value + fee = 100 + 0.1 = 100.10
        expected_cost = 100.0 * (1 + FEE)
        assert result["cost"] == pytest.approx(expected_cost, rel=1e-6)

    def test_sell_net_revenue_exact(self):
        """Sell 1.0 unit @ $110 → net revenue = $109.89 (= $110 − $0.11 fee)."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        result = _sell_all(bt, price=110.0, day=2)

        assert result["success"] is True, f"Trade rejected: {result.get('reason')}"
        net_revenue = result["revenue"] - result["fee"]
        expected_net = 110.0 * (1 - FEE)
        assert net_revenue == pytest.approx(expected_net, rel=1e-6)

    def test_roundtrip_pnl_exact(self):
        """Buy @ $100, sell @ $110 → P&L = $9.79."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        result = _sell_all(bt, price=110.0, day=2)

        assert result["success"] is True
        expected_pnl = 110.0 * (1 - FEE) - 100.0 * (1 + FEE)  # 109.89 − 100.10 = 9.79
        assert result["profit"] == pytest.approx(expected_pnl, rel=1e-4)

    def test_roundtrip_final_portfolio_value(self):
        """After round-trip, portfolio_value = initial + P&L."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        _sell_all(bt, price=110.0, day=2)

        final_pv = _pv(bt, price=110.0)
        expected_pnl = 110.0 * (1 - FEE) - 100.0 * (1 + FEE)
        assert final_pv == pytest.approx(INITIAL + expected_pnl, rel=1e-6)

    def test_loss_trade_reduces_portfolio(self):
        """Buy @ $100, sell @ $90 → portfolio value decreases."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        _sell_all(bt, price=90.0, day=2)

        final_pv = _pv(bt, price=90.0)
        loss = 90.0 * (1 - FEE) - 100.0 * (1 + FEE)  # negative
        assert loss < 0
        assert final_pv == pytest.approx(INITIAL + loss, rel=1e-6)
        assert final_pv < INITIAL


# ---------------------------------------------------------------------------
# 42.1b  portfolio_value == cash + units * price  at every step
# ---------------------------------------------------------------------------

class TestPortfolioValueInvariantBacktester:

    def _check_invariant(self, bt: BaseBacktester, price: float):
        pv = _pv(bt, price)
        units = bt.positions.get("default", {}).get("units", 0.0)
        expected = bt.cash + units * price
        assert pv == pytest.approx(expected, rel=1e-9), (
            f"Invariant broken: pv={pv:.8f} ≠ cash+pos={expected:.8f} "
            f"(cash={bt.cash:.4f}, units={units:.4f}, price={price:.4f})"
        )

    def test_invariant_before_any_trade(self):
        bt = _bt()
        self._check_invariant(bt, 100.0)

    def test_invariant_after_buy(self):
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        self._check_invariant(bt, 100.0)

    def test_invariant_after_price_change(self):
        """After buying, check at the new (higher) price."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        self._check_invariant(bt, 110.0)

    def test_invariant_after_sell(self):
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        _sell_all(bt, price=110.0, day=2)
        self._check_invariant(bt, 110.0)

    def test_invariant_price_series(self):
        """Invariant holds for every step in prices [100, 105, 103, 110]."""
        prices = [100.0, 105.0, 103.0, 110.0]
        actions = [0.01, 0.01, 0.01, 0.0]   # buy, hold (no-op near same frac), liquidate
        bt = _bt()
        for day, (price, action) in enumerate(zip(prices, actions), start=1):
            bt.execute_trade(
                asset="default",
                action=action,
                price_data={"default": price},
                timestamp=_ts(day),
            )
            self._check_invariant(bt, price)


# ---------------------------------------------------------------------------
# 42.1c  Value conservation: no money created or destroyed
# ---------------------------------------------------------------------------

class TestValueConservation:

    def test_fees_account_for_all_losses(self):
        """initial + price_gain − fees == final portfolio value."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        _sell_all(bt, price=110.0, day=2)

        total_fees = sum(float(t.get("fee", 0.0)) for t in bt.trades if t.get("success"))
        final_pv = _pv(bt, price=110.0)
        price_gain = 10.0  # 1 unit × ($110 − $100)

        assert INITIAL + price_gain - total_fees == pytest.approx(final_pv, rel=1e-6)

    def test_zero_gain_flat_prices(self):
        """Buy and sell at same price: only fees reduce portfolio."""
        bt = _bt()
        _buy(bt, price=100.0, day=1)
        _sell_all(bt, price=100.0, day=2)

        final_pv = _pv(bt, price=100.0)
        total_fees = sum(float(t.get("fee", 0.0)) for t in bt.trades if t.get("success"))
        assert final_pv == pytest.approx(INITIAL - total_fees, rel=1e-6)
        assert final_pv < INITIAL  # fees always reduce value
