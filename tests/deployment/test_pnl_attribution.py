"""
Week 66 tests — S55: P&L Attribution & Latency SLO.

Covers:
  - PnLAttributor: decomposition correctness (sum check), edge cases
  - MetricsExporter: latency fields, rolling Sharpe/Sortino, new PnL fields, backward compat
  - OrderManager: latency sample collection, compute_latency_percentiles()
  - ReconciliationReport: by_order / OrderDivergence population
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    side: str
    price: float
    quantity: float
    fee: float
    pnl: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class _Order:
    order_id: str
    side: str
    amount: float
    avg_fill_price: float
    filled_amount: float = 0.0
    status: str = "filled"


# ---------------------------------------------------------------------------
# S51: PnLAttributor
# ---------------------------------------------------------------------------

class TestPnLAttributor:
    def _make_attributor(self):
        from deployment.analysis.pnl_attribution import PnLAttributor
        return PnLAttributor()

    def test_no_trades_returns_empty(self):
        attr = self._make_attributor()
        result = attr.attribute([])
        assert result == []

    def test_only_buy_trades_returns_empty(self):
        attr = self._make_attributor()
        trades = [_Trade("buy", 100.0, 1.0, 0.1)]
        result = attr.attribute(trades)
        assert result == []

    def test_single_round_trip_sum_check(self):
        """market_move = net_pnl + slippage_cost + fees"""
        attr = self._make_attributor()
        # buy at 100, sell at 110, qty=1, fee=0.11 (0.1%), pnl=(110-100)*1=10
        entry_price = 100.0
        exit_price = 110.0
        qty = 1.0
        fee = exit_price * qty * 0.001  # 0.11
        raw_pnl = (exit_price - entry_price) * qty  # 10.0

        buy_trade = _Trade("buy", entry_price, qty, fee=entry_price * qty * 0.001)
        sell_trade = _Trade("sell", exit_price, qty, fee=fee, pnl=raw_pnl)
        trades = [buy_trade, sell_trade]

        result = attr.attribute(trades)
        assert len(result) == 1
        a = result[0]

        # Decomposition sum check: market_move = net_pnl + slippage_cost + fees
        assert abs(a.market_move - (a.net_pnl + a.slippage_cost + a.fees)) < 1e-8

    def test_slippage_applied_correctly(self):
        attr = self._make_attributor()
        exit_price = 110.0
        qty = 2.0
        fee = 0.22
        raw_pnl = 20.0  # (110-100)*2

        sell_trade = _Trade("sell", exit_price, qty, fee=fee, pnl=raw_pnl)
        slip_frac = 0.005  # 0.5%

        result = attr.attribute([sell_trade], slippage_records=[slip_frac])
        assert len(result) == 1
        a = result[0]

        expected_slip_cost = slip_frac * qty * exit_price
        assert abs(a.slippage_cost - expected_slip_cost) < 1e-8
        assert abs(a.net_pnl - (a.market_move - a.slippage_cost - a.fees)) < 1e-8

    def test_zero_slippage_in_paper_mode(self):
        """No slippage records → slippage_cost == 0."""
        attr = self._make_attributor()
        sell_trade = _Trade("sell", 110.0, 1.0, fee=0.11, pnl=10.0)
        result = attr.attribute([sell_trade])
        assert result[0].slippage_cost == 0.0

    def test_multiple_round_trips_sum_check(self):
        attr = self._make_attributor()
        trades = []
        for i in range(5):
            entry = 100.0 + i * 10
            exit_p = entry + 5.0
            qty = 0.5
            fee = exit_p * qty * 0.001
            trades.append(_Trade("buy", entry, qty, fee=entry * qty * 0.001))
            trades.append(_Trade("sell", exit_p, qty, fee=fee, pnl=(exit_p - entry) * qty))

        result = attr.attribute(trades)
        assert len(result) == 5
        for a in result:
            assert abs(a.market_move - (a.net_pnl + a.slippage_cost + a.fees)) < 1e-8

    def test_summarise_zero_trades(self):
        attr = self._make_attributor()
        summary = attr.summarise([])
        assert summary.num_closing_trades == 0
        assert summary.total_net_pnl == 0.0
        assert summary.slippage_pct_of_gross == 0.0

    def test_summarise_totals(self):
        attr = self._make_attributor()
        sell_trade = _Trade("sell", 110.0, 1.0, fee=0.11, pnl=10.0)
        attributions = attr.attribute([sell_trade], slippage_records=[0.001])
        summary = attr.summarise(attributions)

        assert summary.num_closing_trades == 1
        assert abs(summary.total_market_move - attributions[0].market_move) < 1e-8
        assert abs(summary.total_fees - attributions[0].fees) < 1e-8

    def test_to_exporter_fields(self):
        from deployment.analysis.pnl_attribution import PnLAttributor
        attr = PnLAttributor()
        sell_trade = _Trade("sell", 110.0, 1.0, fee=0.11, pnl=10.0)
        summary = attr.summarise(attr.attribute([sell_trade]))
        fields = attr.to_exporter_fields(summary)
        assert "pnl_market_move" in fields
        assert "pnl_slippage_cost" in fields
        assert "pnl_fees" in fields
        assert "pnl_net" in fields


# ---------------------------------------------------------------------------
# S52 / S53: MetricsExporter latency + rolling metrics
# ---------------------------------------------------------------------------

class TestMetricsExporterWeek66:
    def _make_exporter(self):
        from deployment.monitoring.metrics_exporter import MetricsExporter
        return MetricsExporter()

    def test_new_snapshot_fields_have_defaults(self):
        exp = self._make_exporter()
        snap = exp.update(portfolio_value=10000.0)
        assert snap.latency_p50_ms == 0.0
        assert snap.latency_p95_ms == 0.0
        assert snap.latency_p99_ms == 0.0
        assert snap.rolling_sharpe == 0.0
        assert snap.rolling_sortino == 0.0
        assert snap.pnl_market_move == 0.0
        assert snap.pnl_net == 0.0

    def test_update_latency_stores_values(self):
        exp = self._make_exporter()
        snap = exp.update_latency(p50=12.5, p95=45.0, p99=120.0)
        assert snap.latency_p50_ms == 12.5
        assert snap.latency_p95_ms == 45.0
        assert snap.latency_p99_ms == 120.0

    def test_update_latency_preserves_existing_fields(self):
        exp = self._make_exporter()
        exp.update(portfolio_value=99000.0, num_trades=5)
        snap = exp.update_latency(p50=10.0, p95=20.0, p99=30.0)
        assert snap.portfolio_value == 99000.0
        assert snap.num_trades == 5

    def test_rolling_sharpe_insufficient_history(self):
        exp = self._make_exporter()
        assert exp.rolling_sharpe(window=20) == 0.0

    def test_rolling_sharpe_positive_trend(self):
        exp = self._make_exporter()
        # Feed increasing portfolio values → positive returns → positive Sharpe
        for i in range(30):
            exp.update(portfolio_value=10000.0 + i * 50.0)
        sharpe = exp.rolling_sharpe(window=20)
        assert sharpe > 0.0

    def test_rolling_sortino_insufficient_history(self):
        exp = self._make_exporter()
        assert exp.rolling_sortino(window=20) == 0.0

    def test_rolling_sortino_no_downside(self):
        exp = self._make_exporter()
        for i in range(30):
            exp.update(portfolio_value=10000.0 + i * 100.0)
        sortino = exp.rolling_sortino(window=20)
        # monotonically increasing → no downside → high sortino
        assert sortino != 0.0

    def test_rolling_sharpe_negative_trend(self):
        exp = self._make_exporter()
        for i in range(30):
            exp.update(portfolio_value=10000.0 - i * 20.0)
        sharpe = exp.rolling_sharpe(window=20)
        assert sharpe < 0.0

    def test_pnl_attribution_fields_exposed_in_to_json(self):
        exp = self._make_exporter()
        snap = exp.update(pnl_market_move=500.0, pnl_slippage_cost=5.0, pnl_fees=2.5, pnl_net=492.5)
        data = exp.to_json()
        assert data["pnl_market_move"] == 500.0
        assert data["pnl_slippage_cost"] == 5.0
        assert data["pnl_net"] == 492.5

    def test_existing_fields_still_present_in_to_json(self):
        """Backward-compat: old fields must not disappear."""
        exp = self._make_exporter()
        exp.update(portfolio_value=12000.0, sharpe_ratio=1.5, is_halted=False)
        data = exp.to_json()
        for key in ("portfolio_value", "cash", "position", "unrealised_pnl",
                    "realised_pnl", "drawdown_pct", "num_trades", "win_rate",
                    "sharpe_ratio", "drift_detected", "alerts_fired",
                    "current_var", "daily_pnl", "is_halted"):
            assert key in data, f"Missing key: {key}"

    def test_thread_safe_update_latency(self):
        exp = self._make_exporter()
        results = []
        def worker():
            for _ in range(100):
                snap = exp.update_latency(p50=1.0, p95=2.0, p99=3.0)
                results.append(snap)
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 500


# ---------------------------------------------------------------------------
# S52: OrderManager latency tracking
# ---------------------------------------------------------------------------

class TestOrderManagerLatency:
    def _make_manager(self):
        from deployment.execution.order_manager import OrderManager
        return OrderManager(paper_mode=True)

    def test_latency_percentiles_empty(self):
        mgr = self._make_manager()
        result = mgr.compute_latency_percentiles()
        assert result["p50"] == 0.0
        assert result["p95"] == 0.0
        assert result["p99"] == 0.0
        assert result["count"] == 0.0

    def test_latency_recorded_after_fill(self):
        mgr = self._make_manager()
        mgr.update_paper_price(100.0)
        oid = mgr.submit_order("buy", 0.5, current_price=100.0)
        assert mgr.check_order(oid) == "filled"
        result = mgr.compute_latency_percentiles()
        assert result["count"] == 1.0
        assert result["p50"] >= 0.0   # paper mode = near-zero latency

    def test_multiple_fills_accumulate(self):
        mgr = self._make_manager()
        mgr.update_paper_price(100.0)
        for _ in range(10):
            mgr.submit_order("buy", 0.01, current_price=100.0)
        result = mgr.compute_latency_percentiles()
        assert result["count"] == 10.0

    def test_filled_at_set_on_paper_order(self):
        from deployment.execution.order_manager import OrderManager
        mgr = OrderManager(paper_mode=True)
        mgr.update_paper_price(50.0)
        oid = mgr.submit_order("buy", 0.1, current_price=50.0)
        order = mgr.get_order(oid)
        assert order.filled_at is not None
        assert order.submitted_at is not None
        assert order.filled_at >= order.submitted_at

    def test_rejected_order_no_latency_sample(self):
        """Rejected orders (status=failed) should not contribute latency samples."""
        from deployment.execution.order_manager import OrderManager
        from deployment.execution.circuit_breaker import VolatilityCircuitBreaker
        # Force circuit breaker to trip: very low threshold + large price moves
        # window=2 needs 3 prices (window+1) for ddof=1 std calculation
        cb = VolatilityCircuitBreaker(vol_threshold=0.001, window=2, cooldown=9999.0)
        cb.update(100.0)
        cb.update(150.0)  # +50%
        cb.update(80.0)   # -47%
        assert cb.is_tripped()
        mgr = OrderManager(paper_mode=True, circuit_breaker=cb)
        mgr.update_paper_price(100.0)
        oid = mgr.submit_order("buy", 0.1, current_price=100.0)
        assert mgr.check_order(oid) == "failed"
        result = mgr.compute_latency_percentiles()
        assert result["count"] == 0.0


# ---------------------------------------------------------------------------
# S54: ReconciliationReport.by_order / OrderDivergence
# ---------------------------------------------------------------------------

class TestReconciliationByOrder:
    def _base_reports(self):
        bt = {
            "total_return": 0.05,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.2,
            "max_drawdown": 0.02,
            "num_trades": 10,
            "win_rate": 0.6,
            "total_fees": 50.0,
            "avg_fill_slippage": 0.001,
            "final_portfolio_value": 10500.0,
        }
        live = dict(bt)
        live["total_return"] = 0.045
        live["avg_fill_slippage"] = 0.0015
        return bt, live

    def test_by_order_empty_when_no_orders(self):
        from training.analysis.reconciliation import ReconciliationReport
        bt, live = self._base_reports()
        report = ReconciliationReport.from_reports(bt, live)
        assert report.by_order == []

    def test_by_order_populated(self):
        from training.analysis.reconciliation import ReconciliationReport
        bt, live = self._base_reports()
        orders = [
            _Order("o1", "buy",  1.0, avg_fill_price=100.5, filled_amount=1.0),
            _Order("o2", "sell", 1.0, avg_fill_price=109.3, filled_amount=1.0),
        ]
        expected_prices = [100.0, 110.0]
        report = ReconciliationReport.from_reports(
            bt, live, orders=orders, expected_prices=expected_prices
        )
        assert len(report.by_order) == 2
        d0 = report.by_order[0]
        assert d0.order_id == "o1"
        assert abs(d0.slippage - abs(100.5 - 100.0) / 100.0) < 1e-8
        assert d0.side == "buy"

    def test_order_divergence_slippage_cost_sign(self):
        """Buy: overpaid → positive cost. Sell: underpaid → positive cost."""
        from training.analysis.reconciliation import ReconciliationReport
        bt, live = self._base_reports()
        # Buy filled 1% above expected → cost > 0
        buy_order = _Order("o1", "buy", 1.0, avg_fill_price=101.0, filled_amount=1.0)
        report = ReconciliationReport.from_reports(
            bt, live, orders=[buy_order], expected_prices=[100.0]
        )
        d = report.by_order[0]
        assert d.slippage_cost > 0.0  # overpaid

    def test_to_json_includes_by_order(self):
        from training.analysis.reconciliation import ReconciliationReport
        bt, live = self._base_reports()
        orders = [_Order("o1", "buy", 1.0, avg_fill_price=100.5, filled_amount=1.0)]
        report = ReconciliationReport.from_reports(
            bt, live, orders=orders, expected_prices=[100.0]
        )
        data = report.to_json()
        assert "by_order" in data
        assert len(data["by_order"]) == 1
        assert data["by_order"][0]["order_id"] == "o1"

    def test_by_order_warning_on_high_slippage(self):
        from training.analysis.reconciliation import ReconciliationReport
        bt, live = self._base_reports()
        # 5% slippage → should trigger warning
        orders = [_Order("o1", "buy", 1.0, avg_fill_price=105.0, filled_amount=1.0)]
        report = ReconciliationReport.from_reports(
            bt, live, orders=orders, expected_prices=[100.0]
        )
        assert any("slippage" in w.lower() for w in report.warnings)

    def test_backward_compat_no_orders_param(self):
        """Old callers that pass no orders must still work."""
        from training.analysis.reconciliation import ReconciliationReport
        bt, live = self._base_reports()
        report = ReconciliationReport.from_reports(bt, live)
        assert isinstance(report, ReconciliationReport)
        assert report.by_order == []


# ---------------------------------------------------------------------------
# S55: integration — PnL attribution total equals paper trader PnL
# ---------------------------------------------------------------------------

class TestPnLAttributionIntegration:
    """Verify that attribution sum equals observed PnL from PaperTrader."""

    def test_attribution_sum_matches_total_pnl(self):
        """sum(trade.pnl) ≈ sum(attribution.market_move) for sell trades."""
        from deployment.analysis.pnl_attribution import PnLAttributor

        # Simulate a set of trades manually
        trades = []
        for i in range(8):
            entry = 1000.0 + i * 5
            exit_p = entry + (3.0 if i % 2 == 0 else -1.0)
            qty = 1.0
            raw_pnl = (exit_p - entry) * qty
            buy_fee = entry * qty * 0.001
            sell_fee = exit_p * qty * 0.001
            trades.append(_Trade("buy", entry, qty, fee=buy_fee))
            trades.append(_Trade("sell", exit_p, qty, fee=sell_fee, pnl=raw_pnl))

        attr = PnLAttributor()
        attributions = attr.attribute(trades)
        summary = attr.summarise(attributions)

        # Selling trades: sum of raw_pnl
        expected_market_move = sum(t.pnl for t in trades if t.side == "sell")
        assert abs(summary.total_market_move - expected_market_move) < 1e-6

    def test_net_pnl_equals_market_move_minus_costs(self):
        from deployment.analysis.pnl_attribution import PnLAttributor
        slip = [0.001, 0.002]
        trades = [
            _Trade("buy",  100.0, 1.0, fee=0.1),
            _Trade("sell", 110.0, 1.0, fee=0.11, pnl=10.0),
            _Trade("buy",  115.0, 1.0, fee=0.115),
            _Trade("sell", 120.0, 1.0, fee=0.12, pnl=5.0),
        ]
        attr = PnLAttributor()
        attributions = attr.attribute(trades, slippage_records=slip)
        for a in attributions:
            expected_net = a.market_move - a.slippage_cost - a.fees
            assert abs(a.net_pnl - expected_net) < 1e-8
