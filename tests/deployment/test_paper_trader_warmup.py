"""E2: PaperTrader cold-start warmup guard tests."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


def _make_trader(warmup_minutes=1, size_fraction=0.5, max_qps=1.0):
    from deployment.paper_trader import PaperTrader

    agent = MagicMock()
    agent.predict.return_value = (1.0, None)
    config = {
        "paper_trading": {
            "initial_balance": 10_000,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
        },
        "warmup": {
            "enabled": True,
            "warmup_minutes": warmup_minutes,
            "size_fraction": size_fraction,
            "max_qps": max_qps,
            "progress_alerts": False,
        },
    }
    return PaperTrader(agent=agent, config=config, simulation_mode=True)


class TestWarmupGuardInit:
    def test_guard_created_when_enabled(self):
        trader = _make_trader()
        assert trader._warmup_guard is not None

    def test_guard_none_when_disabled(self):
        from deployment.paper_trader import PaperTrader

        agent = MagicMock()
        agent.predict.return_value = (0, None)
        config = {
            "paper_trading": {"initial_balance": 10_000, "trading_fee": 0.001},
            "warmup": {"enabled": False},
        }
        trader = PaperTrader(agent=agent, config=config, simulation_mode=True)
        assert trader._warmup_guard is None

    def test_guard_params_stored(self):
        trader = _make_trader(warmup_minutes=5, size_fraction=0.3, max_qps=2.0)
        g = trader._warmup_guard
        assert g.warmup_minutes == 5
        assert g.size_fraction == 0.3


class TestWarmupSizeCap:
    def test_buy_size_halved_during_warmup(self):
        """During warmup, _execute_buy should halve the requested strength."""
        trader = _make_trader(warmup_minutes=60, size_fraction=0.5, max_qps=100.0)
        # Manually start the guard
        trader._warmup_guard.start()
        assert trader._warmup_guard.in_warmup

        trader.state.pos._cash = 10_000.0
        trader.state.pos._position = 0.0
        initial_cash = trader.state.pos.cash

        trader._execute_buy(strength=1.0, price=100.0)
        assert len(trader.state.trades) == 1
        trade = trader.state.trades[0]
        # With strength=1.0 → capped to 0.5 → max_spend = 10000 * 0.5 = 5000
        expected_qty = 5000.0 / 100.0
        assert abs(trade.quantity - expected_qty) < 1e-6

    def test_sell_size_halved_during_warmup(self):
        trader = _make_trader(warmup_minutes=60, size_fraction=0.5, max_qps=100.0)
        trader._warmup_guard.start()

        # Give trader a position
        trader.state.pos._position = 1.0
        trader.state.pos._entry_price = 100.0

        trader._execute_sell(strength=1.0, price=100.0)
        assert len(trader.state.trades) == 1
        # sell_qty = 1.0 * min(0.5, 1.0) = 0.5
        assert abs(trader.state.trades[0].quantity - 0.5) < 1e-6

    def test_buy_full_size_after_warmup(self):
        """Once warmup window passes, orders execute at full size (not halved)."""
        trader = _make_trader(warmup_minutes=60, size_fraction=0.5, max_qps=100.0)
        trader._warmup_guard = None

        trader.state.pos._cash = 10_000.0
        trader.state.pos._position = 0.0

        # strength=0.5 → max_spend=5000, cost=5005 < 10000 ✓
        trader._execute_buy(strength=0.5, price=100.0)
        assert len(trader.state.trades) == 1
        # Without warmup cap, strength=0.5 is used as-is
        expected_qty = (10_000.0 * 0.5) / 100.0
        assert abs(trader.state.trades[0].quantity - expected_qty) < 1e-6

        # Verify warmup would have halved it further
        trader2 = _make_trader(warmup_minutes=60, size_fraction=0.5, max_qps=100.0)
        trader2._warmup_guard.start()
        trader2.state.pos._cash = 10_000.0
        trader2._execute_buy(strength=0.5, price=100.0)
        assert len(trader2.state.trades) == 1
        # With warmup: strength becomes 0.5 * 0.5 = 0.25 → max_spend = 2500
        expected_warmup_qty = (10_000.0 * 0.25) / 100.0
        assert abs(trader2.state.trades[0].quantity - expected_warmup_qty) < 1e-6


class TestWarmupQPS:
    def test_second_order_within_interval_rejected(self):
        """Two orders < 1 s apart → second is dropped."""
        trader = _make_trader(warmup_minutes=60, size_fraction=0.5, max_qps=1.0)
        trader._warmup_guard.start()
        trader.state.pos._cash = 10_000.0

        trader._execute_buy(strength=0.1, price=100.0)
        trader._execute_buy(strength=0.1, price=100.0)
        # Second buy should be throttled
        assert len(trader.state.trades) == 1

    def test_order_allowed_after_interval(self):
        """Orders separated by > min_order_interval are both allowed."""
        trader = _make_trader(warmup_minutes=60, size_fraction=0.5, max_qps=1.0)
        guard = trader._warmup_guard
        guard.start()
        # Force last_order_time far in the past
        guard._last_order_time = time.monotonic() - 2.0

        trader.state.pos._cash = 10_000.0
        trader._execute_buy(strength=0.1, price=100.0)
        assert len(trader.state.trades) == 1


class TestWarmupAlerts:
    def test_start_alert_fired(self):
        alerter = MagicMock()
        trader = _make_trader(warmup_minutes=60)
        trader._warmup_guard.alerter = alerter
        trader._warmup_guard.start()
        alerter.send_alert.assert_called_once()
        msg = alerter.send_alert.call_args[0][0]
        assert "Warmup mode ACTIVE" in msg

    def test_end_alert_fired(self):
        from deployment.execution.warmup_guard import WarmupGuard

        alerter = MagicMock()
        guard = WarmupGuard(warmup_minutes=1, alerter=alerter)
        guard._start_time = time.monotonic() - 61  # already elapsed
        guard._maybe_end()
        assert guard._ended
        end_calls = [
            c for c in alerter.send_alert.call_args_list
            if "ENDED" in c[0][0]
        ]
        assert len(end_calls) == 1
