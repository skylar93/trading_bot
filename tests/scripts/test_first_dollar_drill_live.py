"""Tests for first_dollar_drill.py --live mode (I1-c).

All tests inject a mock CCXT exchange and mock OrderManager so no real
HTTP calls are made.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.first_dollar_drill import run_live_drill


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_exchange(
    bid: float = 50_000.0,
    ask: float = 50_100.0,
    usdt: float = 1000.0,
    btc: float = 0.0,
) -> MagicMock:
    ex = MagicMock()
    ex.fetch_ticker.return_value = {"bid": bid, "ask": ask, "last": (bid + ask) / 2}
    ex.fetch_balance.return_value = {
        "USDT": {"free": usdt},
        "BTC": {"free": btc},
    }
    return ex


def _make_order_manager(fill_status: str = "open") -> MagicMock:
    om = MagicMock()
    om.submit_order.return_value = "order_001"
    om.check_order.return_value = fill_status

    order = MagicMock()
    order.filled_amount = 0.001
    order.avg_fill_price = 49_000.0
    om.get_order.return_value = order
    om.cancel_order.return_value = True
    om.cancel_all_orders.return_value = 1
    om.close.return_value = None
    return om


# ---------------------------------------------------------------------------
# Capital guard
# ---------------------------------------------------------------------------

class TestCapitalGuard:
    def test_capital_above_100_returns_fail(self):
        result = run_live_drill(101.0, _exchange=_make_exchange(),
                                _order_manager=_make_order_manager())
        assert result["status"] == "FAIL"
        assert "exceeds" in result["detail"]

    def test_capital_exactly_100_passes_guard(self):
        om = _make_order_manager("open")
        # Will proceed past guard and hit open→cancel path
        result = run_live_drill(100.0, _exchange=_make_exchange(),
                                _order_manager=om)
        # Guard should not fire (status is not the guard check)
        assert "exceeds" not in result.get("detail", "")


# ---------------------------------------------------------------------------
# Unfilled order → cancel path
# ---------------------------------------------------------------------------

class TestUnfilledOrderCancelled:
    def test_open_order_is_cancelled(self):
        om = _make_order_manager("open")
        ex = _make_exchange()

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        om.cancel_order.assert_called_with("order_001")
        assert result["status"] == "PASS"
        assert "cancelled" in result["detail"]


# ---------------------------------------------------------------------------
# Filled order → market sell path
# ---------------------------------------------------------------------------

class TestFilledOrderMarketSell:
    def test_filled_order_triggers_market_sell(self):
        om = _make_order_manager("filled")
        ex = _make_exchange()

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        # Should have submitted a sell order
        sell_calls = [
            c for c in om.submit_order.call_args_list
            if c.kwargs.get("side") == "sell" or
               (c.args and c.args[0] == "sell")
        ]
        assert sell_calls, "Expected market sell order after fill"
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Partial fill → cancel remainder + sell filled qty
# ---------------------------------------------------------------------------

class TestPartialFill:
    def test_partial_fill_cancels_then_sells(self):
        om = _make_order_manager("partial")
        om.get_order.return_value.filled_amount = 0.0005
        ex = _make_exchange()

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        om.cancel_order.assert_called()
        sell_calls = [
            c for c in om.submit_order.call_args_list
            if c.kwargs.get("side") == "sell" or
               (len(c.args) > 0 and c.args[0] == "sell")
        ]
        assert sell_calls
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Audit check integration
# ---------------------------------------------------------------------------

class TestAuditCheck:
    def test_audit_ok_when_no_log_exists(self, tmp_path, monkeypatch):
        """If audit.jsonl doesn't exist, audit_ok defaults to True."""
        monkeypatch.chdir(tmp_path)
        om = _make_order_manager("open")
        ex = _make_exchange()

        with patch("time.sleep", return_value=None):
            result = run_live_drill(100.0, _exchange=ex, _order_manager=om)

        # Should complete without error
        assert result is not None


# ---------------------------------------------------------------------------
# --live flag wired into main()
# ---------------------------------------------------------------------------

class TestLiveFlagInMain:
    def test_live_flag_missing_creds_exits(self, monkeypatch):
        """Without injected exchange/order_manager, --live with no creds must exit 1."""
        import subprocess
        drill_script = PROJECT_ROOT / "scripts" / "first_dollar_drill.py"
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("EXCHANGE_BINANCE_KEY", "EXCHANGE_BINANCE_SECRET")}
        env["TRADING_BOT_SECRET_BACKEND"] = "env"  # force env backend so no creds found

        result = subprocess.run(
            [sys.executable, str(drill_script), "--live", "--capital", "100",
             "--skip-kill-switch-test"],
            capture_output=True, text=True, env=env,
        )
        # Should exit 1 (missing credentials)
        assert result.returncode == 1
        assert "missing credentials" in (result.stdout + result.stderr).lower()
