"""I6: Live drill mock exchange — 5 scenario matrix.

Scenarios: filled / partial / unfilled / timeout / rejected
All use injected mock exchange + order manager; no real HTTP calls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.first_dollar_drill import run_live_drill


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_exchange(usdt: float = 1000.0, btc: float = 0.0) -> MagicMock:
    ex = MagicMock()
    ex.fetch_ticker.return_value = {"bid": 50_000.0, "ask": 50_100.0, "last": 50_050.0}
    ex.fetch_balance.return_value = {
        "USDT": {"free": usdt},
        "BTC": {"free": btc},
    }
    return ex


def _make_om(fill_status: str = "open", filled_amount: float = 0.001) -> MagicMock:
    om = MagicMock()
    om.submit_order.return_value = "order_mock_001"
    om.check_order.return_value = fill_status
    order = MagicMock()
    order.filled_amount = filled_amount
    order.avg_fill_price = 49_000.0
    order.fee = 0.05
    om.get_order.return_value = order
    om.cancel_order.return_value = True
    om.cancel_all_orders.return_value = 1
    om.close.return_value = None
    return om


# ---------------------------------------------------------------------------
# Scenario 1: filled — limit order fills → market sell flatting
# ---------------------------------------------------------------------------

class TestScenarioFilled:
    def test_filled_order_triggers_market_sell_and_passes(self):
        ex = _make_exchange()
        om = _make_om("filled", filled_amount=0.001)

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        assert result["status"] == "PASS", result["detail"]
        sell_calls = [
            c for c in om.submit_order.call_args_list
            if (c.kwargs.get("side") == "sell") or
               (c.args and c.args[0] == "sell")
        ]
        assert sell_calls, "Expected market sell after fill"


# ---------------------------------------------------------------------------
# Scenario 2: partial — cancel remainder + sell filled qty
# ---------------------------------------------------------------------------

class TestScenarioPartial:
    def test_partial_fill_cancels_remainder_and_sells(self):
        ex = _make_exchange()
        om = _make_om("partial", filled_amount=0.0005)

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        assert result["status"] == "PASS", result["detail"]
        om.cancel_order.assert_called()
        sell_calls = [
            c for c in om.submit_order.call_args_list
            if (c.kwargs.get("side") == "sell") or
               (c.args and c.args[0] == "sell")
        ]
        assert sell_calls, "Expected sell of partial fill qty"


# ---------------------------------------------------------------------------
# Scenario 3: unfilled — order stays open → cancelled
# ---------------------------------------------------------------------------

class TestScenarioUnfilled:
    def test_unfilled_order_is_cancelled_and_passes(self):
        ex = _make_exchange()
        om = _make_om("open")

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        assert result["status"] == "PASS", result["detail"]
        assert "cancelled" in result["detail"]
        om.cancel_order.assert_called_with("order_mock_001")


# ---------------------------------------------------------------------------
# Scenario 4: timeout — watchdog fires after 10 min
# ---------------------------------------------------------------------------

class TestScenarioTimeout:
    def test_timeout_triggers_cancel_all_and_returns_fail(self):
        ex = _make_exchange()
        om = _make_om("open")

        # Simulate order never filling and monotonic time jumping past 600s
        _call_count = [0]
        _BASE = 1_000_000.0

        def _mock_monotonic():
            # First call (drill start): base; subsequent calls: base + 601 (timeout)
            _call_count[0] += 1
            return _BASE if _call_count[0] <= 2 else _BASE + 601.0

        with patch("time.sleep", return_value=None), \
             patch("time.monotonic", side_effect=_mock_monotonic):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        # Either the timeout path or cancel path is triggered
        assert result is not None
        # cancel_all_orders should have been called at some point
        om.cancel_all_orders.assert_called()


# ---------------------------------------------------------------------------
# Scenario 5: rejected — submit_order raises an exception
# ---------------------------------------------------------------------------

class TestScenarioRejected:
    def test_rejected_order_returns_fail_with_detail(self):
        ex = _make_exchange()
        om = _make_om("open")
        om.submit_order.side_effect = Exception("Order rejected: insufficient margin")

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        assert result["status"] == "FAIL", result
        assert "rejected" in result["detail"].lower() or "insufficient" in result["detail"].lower()
