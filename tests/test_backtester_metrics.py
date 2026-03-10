"""
Week 3: tests for corrected backtester metric calculations.

Each test verifies the metric against a hand-computed reference value so that
any future regression is immediately visible.
"""
import math

import numpy as np
import pandas as pd
import pytest

from training.backtesting.base_backtester import BaseBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtester_with_portfolio(portfolio_values: list) -> BaseBacktester:
    """Return a BaseBacktester whose portfolio_history is pre-populated."""
    data = pd.DataFrame(
        {
            "$open": [100.0] * 10,
            "$high": [101.0] * 10,
            "$low": [99.0] * 10,
            "$close": [100.0] * 10,
            "$volume": [1000.0] * 10,
        }
    )
    bt = BaseBacktester(initial_capital=portfolio_values[0], data=data)
    bt.portfolio_history = list(portfolio_values)
    bt.cash = portfolio_values[-1]
    return bt


def _manual_sharpe(values: list) -> float:
    """Reference Sharpe: sqrt(252) * mean(log_returns) / std(log_returns, ddof=1)."""
    v = np.array(values, dtype=float)
    lr = np.log(v[1:] / v[:-1])
    if lr.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(252) * lr.mean() / lr.std(ddof=1))


def _manual_sortino(values: list) -> float:
    """Reference Sortino: sqrt(252) * mean(log_returns) / std(downside, ddof=1)."""
    v = np.array(values, dtype=float)
    lr = np.log(v[1:] / v[:-1])
    down = lr[lr < 0]
    if len(down) <= 1:
        return 0.0
    return float(np.sqrt(252) * lr.mean() / down.std(ddof=1))


def _manual_calmar(values: list) -> float:
    """Reference Calmar: annualised_return / max_drawdown."""
    v = np.array(values, dtype=float)
    n = len(v)
    annual_ret = (v[-1] / v[0]) ** (252 / n) - 1
    running_max = np.maximum.accumulate(v)
    dd = (running_max - v) / running_max
    max_dd = dd.max()
    if max_dd < 1e-9:
        return 0.0
    return float(annual_ret / max_dd)


# ---------------------------------------------------------------------------
# Fixture: simple 10-step portfolio sequence
# ---------------------------------------------------------------------------

PORTFOLIO = [10_000, 10_100, 10_050, 10_200, 10_150, 10_300, 10_250, 10_400, 10_380, 10_500]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_matches_manual_log_return_calculation(self):
        bt = _make_backtester_with_portfolio(PORTFOLIO)
        metrics = bt._calculate_metrics()
        expected = _manual_sharpe(PORTFOLIO)
        assert math.isclose(metrics["sharpe_ratio"], expected, rel_tol=1e-6)

    def test_zero_variance_returns_zero(self):
        # Flat portfolio → std = 0 → Sharpe = 0
        flat = [10_000] * 5
        bt = _make_backtester_with_portfolio(flat)
        metrics = bt._calculate_metrics()
        assert metrics["sharpe_ratio"] == 0.0

    def test_positive_trending_portfolio_positive_sharpe(self):
        rising = [10_000 * (1.001 ** i) for i in range(50)]
        bt = _make_backtester_with_portfolio(rising)
        metrics = bt._calculate_metrics()
        assert metrics["sharpe_ratio"] > 0

    def test_negative_trending_portfolio_negative_sharpe(self):
        falling = [10_000 * (0.999 ** i) for i in range(50)]
        bt = _make_backtester_with_portfolio(falling)
        metrics = bt._calculate_metrics()
        assert metrics["sharpe_ratio"] < 0


class TestSortinoRatio:
    def test_matches_manual_calculation(self):
        # Sequence with both up and down moves
        seq = [10_000, 9_900, 10_100, 9_800, 10_300, 10_200, 10_500]
        bt = _make_backtester_with_portfolio(seq)
        metrics = bt._calculate_metrics()
        expected = _manual_sortino(seq)
        assert math.isclose(metrics["sortino_ratio"], expected, rel_tol=1e-6)

    def test_no_downside_returns_zero(self):
        # Only up moves → no downside → Sortino = 0 (convention)
        up = [10_000, 10_100, 10_200, 10_300, 10_400]
        bt = _make_backtester_with_portfolio(up)
        metrics = bt._calculate_metrics()
        assert metrics["sortino_ratio"] == 0.0

    def test_sortino_gt_sharpe_when_positive_drift(self):
        # With positive drift, Sortino should be ≥ Sharpe
        # (downside deviation ≤ total std when returns are asymmetric upwards)
        seq = [10_000, 9_900, 10_200, 10_100, 10_400, 10_300, 10_600]
        bt = _make_backtester_with_portfolio(seq)
        metrics = bt._calculate_metrics()
        # Sortino uses only downside std which is ≤ total std, so ratio ≥ Sharpe
        assert metrics["sortino_ratio"] >= metrics["sharpe_ratio"]


class TestCalmarRatio:
    def test_matches_manual_calculation(self):
        bt = _make_backtester_with_portfolio(PORTFOLIO)
        metrics = bt._calculate_metrics()
        expected = _manual_calmar(PORTFOLIO)
        assert math.isclose(metrics["calmar_ratio"], expected, rel_tol=1e-6)

    def test_zero_drawdown_portfolio(self):
        # Monotonically rising → max drawdown ≈ 0 → Calmar = 0 (guard clause)
        rising = [10_000 + i * 100 for i in range(10)]
        bt = _make_backtester_with_portfolio(rising)
        metrics = bt._calculate_metrics()
        assert metrics["calmar_ratio"] == 0.0  # max_dd < 1e-9

    def test_negative_return_negative_calmar(self):
        falling = [10_000, 9_500, 9_000, 8_500, 8_000]
        bt = _make_backtester_with_portfolio(falling)
        metrics = bt._calculate_metrics()
        assert metrics["calmar_ratio"] < 0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        # Peak 110, then 80 → drawdown = 30/110 ≈ 0.2727
        seq = [100, 110, 80]
        bt = _make_backtester_with_portfolio(seq)
        metrics = bt._calculate_metrics()
        assert math.isclose(metrics["max_drawdown"], 30 / 110, rel_tol=1e-6)

    def test_no_drawdown(self):
        rising = [100, 110, 120, 130]
        bt = _make_backtester_with_portfolio(rising)
        metrics = bt._calculate_metrics()
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


class TestProfitFactor:
    def test_profit_factor_computed_from_sell_trades(self):
        data = pd.DataFrame(
            {
                "$open": [100.0] * 5,
                "$high": [101.0] * 5,
                "$low": [99.0] * 5,
                "$close": [100.0] * 5,
                "$volume": [1000.0] * 5,
            }
        )
        bt = BaseBacktester(initial_capital=10_000, data=data)
        bt.portfolio_history = [10_000, 10_100, 10_050, 10_200]
        bt.cash = 10_200
        bt.trades = [
            {"success": True, "type": "sell", "profit": 200.0},
            {"success": True, "type": "sell", "profit": 100.0},
            {"success": True, "type": "sell", "profit": -50.0},
            {"success": False, "type": "sell", "profit": 500.0},  # not counted
        ]
        metrics = bt._calculate_metrics()
        # gross_profit=300, gross_loss=50 → profit_factor=6.0
        assert math.isclose(metrics["profit_factor"], 6.0, rel_tol=1e-6)

    def test_no_trades_profit_factor_zero(self):
        bt = _make_backtester_with_portfolio(PORTFOLIO)
        bt.trades = []
        metrics = bt._calculate_metrics()
        assert metrics["profit_factor"] == 0.0


class TestWinRate:
    def test_win_rate_only_counts_successful_sells(self):
        data = pd.DataFrame(
            {
                "$open": [100.0] * 5,
                "$high": [101.0] * 5,
                "$low": [99.0] * 5,
                "$close": [100.0] * 5,
                "$volume": [1000.0] * 5,
            }
        )
        bt = BaseBacktester(initial_capital=10_000, data=data)
        bt.portfolio_history = PORTFOLIO
        bt.cash = PORTFOLIO[-1]
        bt.trades = [
            {"success": True,  "type": "sell", "profit":  100},
            {"success": True,  "type": "sell", "profit": -50},
            {"success": True,  "type": "sell", "profit":  200},
            {"success": False, "type": "sell", "profit":  999},  # ignored
            {"success": True,  "type": "buy",  "profit":  300},  # buy ignored
        ]
        metrics = bt._calculate_metrics()
        # 2 profitable out of 3 completed sell trades
        assert math.isclose(metrics["win_rate"], 2 / 3, rel_tol=1e-6)

    def test_empty_trades_win_rate_zero(self):
        bt = _make_backtester_with_portfolio(PORTFOLIO)
        bt.trades = []
        metrics = bt._calculate_metrics()
        assert metrics["win_rate"] == 0.0


class TestMetricsReturnKeys:
    def test_all_required_keys_present(self):
        bt = _make_backtester_with_portfolio(PORTFOLIO)
        metrics = bt._calculate_metrics()
        required = {
            "total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "profit_factor", "total_trades", "win_rate",
            "final_balance", "final_portfolio_value",
            "successful_trades", "total_trade_attempts",
        }
        assert required.issubset(set(metrics.keys()))

    def test_single_step_returns_zeros(self):
        """Only 1 portfolio step → not enough data → return zeros gracefully."""
        data = pd.DataFrame(
            {
                "$open": [100.0],
                "$high": [101.0],
                "$low": [99.0],
                "$close": [100.0],
                "$volume": [1000.0],
            }
        )
        bt = BaseBacktester(initial_capital=10_000, data=data)
        bt.portfolio_history = [10_000]
        bt.cash = 10_000
        metrics = bt._calculate_metrics()
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
