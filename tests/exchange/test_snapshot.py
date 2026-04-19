"""
Tests for ExchangeSnapshot (F7) and reconcile-on-boot / periodic reconcile (F8, F9, F10).

Scenarios covered:
  - ExchangeSnapshot.get_positions / get_open_orders / get_balance (happy path + errors)
  - ExchangeSnapshot.snapshot convenience wrapper
  - _do_reconcile: no mismatch, qty mismatch, price drift, open-orders mismatch
  - _reconcile_on_boot: halt | warn | ignore on mismatch
  - _periodic_reconcile: throttling, triggered after interval
  - Bootstrap: PaperTrader.restore() calls reconcile-on-boot
  - Bootstrap: forced position mismatch detected on startup
"""

from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from deployment.exchange.snapshot import ExchangeSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exchange(
    positions=None,
    open_orders=None,
    balance=None,
    positions_error=False,
    orders_error=False,
    balance_error=False,
):
    """Build a minimal mock CCXT exchange."""
    ex = MagicMock()
    if positions_error:
        ex.fetch_positions.side_effect = RuntimeError("positions unavailable")
    else:
        ex.fetch_positions.return_value = positions or []
    if orders_error:
        ex.fetch_open_orders.side_effect = RuntimeError("orders unavailable")
    else:
        ex.fetch_open_orders.return_value = open_orders or []
    if balance_error:
        ex.fetch_balance.side_effect = RuntimeError("balance unavailable")
    else:
        ex.fetch_balance.return_value = balance or {"free": {}, "used": {}, "total": {}}
    return ex


def _ccxt_position(symbol="BTC/USDT", qty=0.5, entry=50_000.0, side="long"):
    return {
        "symbol": symbol,
        "contracts": qty,
        "entryPrice": entry,
        "side": side,
        "unrealizedPnl": 100.0,
    }


def _ccxt_order(order_id="o1", symbol="BTC/USDT", side="buy", amount=0.01):
    return {
        "id": order_id,
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "filled": 0.0,
        "remaining": amount,
        "price": 50_000.0,
        "type": "limit",
        "status": "open",
    }


# ---------------------------------------------------------------------------
# F7 — ExchangeSnapshot unit tests
# ---------------------------------------------------------------------------

class TestExchangeSnapshotGetPositions:
    def test_happy_path_normalises_fields(self):
        ex = _make_exchange(positions=[_ccxt_position(qty=1.0, entry=40_000.0)])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_positions("BTC/USDT")
        assert len(result) == 1
        p = result[0]
        assert p["symbol"] == "BTC/USDT"
        assert p["qty"] == pytest.approx(1.0)
        assert p["entry_price"] == pytest.approx(40_000.0)
        assert p["side"] == "long"

    def test_returns_empty_list_on_error(self):
        ex = _make_exchange(positions_error=True)
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_positions()
        assert result == []

    def test_uses_default_symbol(self):
        ex = _make_exchange(positions=[_ccxt_position()])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        snap.get_positions()
        ex.fetch_positions.assert_called_once_with(["BTC/USDT"])

    def test_uses_provided_symbol(self):
        ex = _make_exchange(positions=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        snap.get_positions("ETH/USDT")
        ex.fetch_positions.assert_called_once_with(["ETH/USDT"])

    def test_zero_qty_position(self):
        raw = {"symbol": "BTC/USDT", "contracts": None, "amount": None,
               "entryPrice": None, "side": None, "unrealizedPnl": None}
        ex = _make_exchange(positions=[raw])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_positions()
        assert result[0]["qty"] == 0.0
        assert result[0]["side"] == "long"


class TestExchangeSnapshotGetOpenOrders:
    def test_happy_path(self):
        ex = _make_exchange(open_orders=[_ccxt_order(order_id="abc", amount=0.05)])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_open_orders()
        assert len(result) == 1
        o = result[0]
        assert o["order_id"] == "abc"
        assert o["amount"] == pytest.approx(0.05)
        assert o["side"] == "buy"

    def test_returns_empty_on_error(self):
        ex = _make_exchange(orders_error=True)
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        assert snap.get_open_orders() == []

    def test_multiple_orders(self):
        orders = [_ccxt_order(order_id=f"o{i}") for i in range(3)]
        ex = _make_exchange(open_orders=orders)
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        assert len(snap.get_open_orders()) == 3


class TestExchangeSnapshotGetBalance:
    def test_happy_path(self):
        raw = {
            "free":  {"BTC": 0.5, "USDT": 10_000.0},
            "used":  {"BTC": 0.0},
            "total": {"BTC": 0.5, "USDT": 10_000.0},
        }
        ex = _make_exchange(balance=raw)
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_balance()
        assert result["free"]["BTC"] == pytest.approx(0.5)
        assert result["free"]["USDT"] == pytest.approx(10_000.0)

    def test_returns_empty_on_error(self):
        ex = _make_exchange(balance_error=True)
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_balance()
        assert result == {"free": {}, "used": {}, "total": {}}

    def test_filters_none_values(self):
        raw = {"free": {"BTC": 0.5, "USDT": None}, "used": {}, "total": {}}
        ex = _make_exchange(balance=raw)
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.get_balance()
        assert "USDT" not in result["free"]


class TestExchangeSnapshotConvenience:
    def test_snapshot_returns_all_three(self):
        ex = _make_exchange(
            positions=[_ccxt_position()],
            open_orders=[_ccxt_order()],
            balance={"free": {"BTC": 0.5}, "used": {}, "total": {}},
        )
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        result = snap.snapshot()
        assert result["symbol"] == "BTC/USDT"
        assert len(result["positions"]) == 1
        assert len(result["open_orders"]) == 1
        assert "BTC" in result["balance"]["free"]


# ---------------------------------------------------------------------------
# F8/F9/F10 — PaperTrader reconciliation scenarios
# ---------------------------------------------------------------------------

def _make_trader(
    position: float = 0.0,
    entry_price: float = 0.0,
    exchange_snapshot=None,
    on_mismatch: str = "halt",
    qty_threshold: float = 0.001,
    price_threshold: float = 0.001,
    interval_sec: float = 60.0,
):
    """Build a minimal PaperTrader for reconciliation testing."""
    from deployment.paper_trader import PaperTrader

    agent = MagicMock()
    agent.predict.return_value = (0.0, None)
    config = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "window_size": 5,
        },
        "reconciliation": {
            "on_mismatch": on_mismatch,
            "qty_threshold": qty_threshold,
            "price_threshold": price_threshold,
            "interval_sec": interval_sec,
        },
    }
    trader = PaperTrader(
        agent=agent,
        config=config,
        simulation_mode=True,
        exchange_snapshot=exchange_snapshot,
    )
    # Manually set position state
    if position > 0:
        trader.state.pos.apply_buy(position, entry_price, fee=0.0)
    return trader


class TestDoReconcile:
    def test_no_mismatch_ok(self):
        """Local and exchange agree → ok=True, no diffs."""
        ex = _make_exchange(
            positions=[_ccxt_position(qty=0.5, entry=50_000.0)],
            open_orders=[],
        )
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.5, entry_price=50_000.0,
                               exchange_snapshot=snap)
        result = trader._do_reconcile()
        assert result["ok"] is True
        assert result["diffs"] == []

    def test_qty_mismatch_detected(self):
        """Exchange has 0.5, local has 0.0 → qty_mismatch diff."""
        ex = _make_exchange(
            positions=[_ccxt_position(qty=0.5, entry=50_000.0)],
            open_orders=[],
        )
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.0, exchange_snapshot=snap)
        result = trader._do_reconcile()
        assert result["ok"] is False
        types = [d["type"] for d in result["diffs"]]
        assert "qty_mismatch" in types

    def test_price_drift_detected(self):
        """Entry price differs by 10% → price_drift diff."""
        ex = _make_exchange(
            positions=[_ccxt_position(qty=0.5, entry=55_000.0)],
            open_orders=[],
        )
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.5, entry_price=50_000.0,
                               exchange_snapshot=snap, price_threshold=0.001)
        result = trader._do_reconcile()
        assert result["ok"] is False
        types = [d["type"] for d in result["diffs"]]
        assert "price_drift" in types

    def test_open_orders_mismatch(self):
        """Exchange has 1 open order, local has 0."""
        ex = _make_exchange(
            positions=[],
            open_orders=[_ccxt_order()],
        )
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.0, exchange_snapshot=snap)
        result = trader._do_reconcile()
        assert result["ok"] is False
        types = [d["type"] for d in result["diffs"]]
        assert "open_orders_mismatch" in types

    def test_snapshot_error_returns_ok(self):
        """If snapshot fetch throws, treat as no mismatch (conservative)."""
        snap = MagicMock()
        snap.snapshot.side_effect = RuntimeError("network error")
        trader = _make_trader(position=0.5, entry_price=50_000.0,
                               exchange_snapshot=snap)
        result = trader._do_reconcile()
        assert result["ok"] is True

    def test_audit_logger_called(self):
        """Each reconcile logs to audit_logger."""
        ex = _make_exchange(positions=[], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(exchange_snapshot=snap)
        audit = MagicMock()
        trader.audit_logger = audit
        trader._do_reconcile()
        audit.log_risk_event.assert_called_once()
        call_kwargs = audit.log_risk_event.call_args[0][0]
        assert call_kwargs["type"] == "reconcile"


class TestHandleMismatch:
    def test_halt_triggers_shutdown(self):
        trader = _make_trader(on_mismatch="halt")
        trader._trigger_shutdown = MagicMock()
        trader._handle_mismatch([{"type": "qty_mismatch"}], context="boot")
        trader._trigger_shutdown.assert_called_once()
        assert "reconcile_halt" in trader._trigger_shutdown.call_args[0][0]

    def test_warn_does_not_shutdown(self):
        trader = _make_trader(on_mismatch="warn")
        trader._trigger_shutdown = MagicMock()
        trader._handle_mismatch([{"type": "qty_mismatch"}], context="boot")
        trader._trigger_shutdown.assert_not_called()

    def test_ignore_does_not_shutdown(self):
        trader = _make_trader(on_mismatch="ignore")
        trader._trigger_shutdown = MagicMock()
        trader._handle_mismatch([{"type": "qty_mismatch"}], context="periodic")
        trader._trigger_shutdown.assert_not_called()

    def test_alerter_notified_on_mismatch(self):
        trader = _make_trader(on_mismatch="warn")
        alerter = MagicMock()
        trader.alerter = alerter
        trader._handle_mismatch([{"type": "qty_mismatch"}])
        alerter.send_alert.assert_called_once()
        args = alerter.send_alert.call_args
        assert args[1]["level"] == "ERROR" or args[0][1] == "ERROR"


class TestReconcileOnBoot:
    def test_no_snapshot_skips(self):
        """Without exchange_snapshot, reconcile_on_boot is a no-op."""
        trader = _make_trader(exchange_snapshot=None)
        trader._do_reconcile = MagicMock()
        trader._reconcile_on_boot()
        trader._do_reconcile.assert_not_called()

    def test_ok_reconcile_no_halt(self):
        """Clean reconcile on boot → trader keeps running."""
        ex = _make_exchange(positions=[_ccxt_position(qty=0.5)], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.5, entry_price=50_000.0,
                               exchange_snapshot=snap, on_mismatch="halt")
        trader._trigger_shutdown = MagicMock()
        trader._reconcile_on_boot()
        trader._trigger_shutdown.assert_not_called()

    def test_mismatch_on_boot_halts(self):
        """Qty mismatch on boot with on_mismatch=halt → shutdown triggered."""
        ex = _make_exchange(
            positions=[_ccxt_position(qty=0.9)],   # exchange has 0.9
            open_orders=[],
        )
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        # Local has 0.0 position — big mismatch
        trader = _make_trader(position=0.0, exchange_snapshot=snap, on_mismatch="halt")
        trader._trigger_shutdown = MagicMock()
        trader._reconcile_on_boot()
        trader._trigger_shutdown.assert_called_once()

    def test_mismatch_on_boot_warns(self):
        """Qty mismatch on boot with on_mismatch=warn → no shutdown."""
        ex = _make_exchange(positions=[_ccxt_position(qty=0.9)], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.0, exchange_snapshot=snap, on_mismatch="warn")
        trader._trigger_shutdown = MagicMock()
        trader._reconcile_on_boot()
        trader._trigger_shutdown.assert_not_called()

    def test_timestamp_updated_on_boot(self):
        """_last_reconcile_at is set after reconcile_on_boot."""
        ex = _make_exchange(positions=[], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(exchange_snapshot=snap)
        assert trader._last_reconcile_at == 0.0
        trader._reconcile_on_boot()
        assert trader._last_reconcile_at > 0.0


class TestPeriodicReconcile:
    def test_no_snapshot_skips(self):
        trader = _make_trader(exchange_snapshot=None)
        trader._do_reconcile = MagicMock()
        trader._periodic_reconcile(time.time())
        trader._do_reconcile.assert_not_called()

    def test_throttled_within_interval(self):
        """Not enough time elapsed → reconcile not called."""
        ex = _make_exchange(positions=[], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(exchange_snapshot=snap, interval_sec=60.0)
        trader._last_reconcile_at = time.time()   # just ran
        trader._do_reconcile = MagicMock()
        trader._periodic_reconcile(time.time())    # too soon
        trader._do_reconcile.assert_not_called()

    def test_fires_after_interval(self):
        """Interval elapsed → reconcile fires."""
        ex = _make_exchange(positions=[], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(exchange_snapshot=snap, interval_sec=60.0)
        trader._last_reconcile_at = time.time() - 61.0   # stale
        trader._do_reconcile = MagicMock(return_value={"ok": True, "diffs": []})
        trader._periodic_reconcile(time.time())
        trader._do_reconcile.assert_called_once()

    def test_timestamp_updated_after_periodic(self):
        ex = _make_exchange(positions=[], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(exchange_snapshot=snap, interval_sec=60.0)
        trader._last_reconcile_at = time.time() - 61.0
        trader._do_reconcile = MagicMock(return_value={"ok": True, "diffs": []})
        before = time.time()
        trader._periodic_reconcile(before + 1)
        assert trader._last_reconcile_at >= before

    def test_mismatch_during_periodic_warns(self):
        """Mismatch detected during periodic reconcile with on_mismatch=warn."""
        ex = _make_exchange(positions=[_ccxt_position(qty=0.5)], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.0, exchange_snapshot=snap, on_mismatch="warn")
        trader._last_reconcile_at = time.time() - 61.0
        trader._trigger_shutdown = MagicMock()
        trader._periodic_reconcile(time.time())
        trader._trigger_shutdown.assert_not_called()

    def test_mismatch_during_periodic_halts(self):
        """Mismatch detected during periodic reconcile with on_mismatch=halt."""
        ex = _make_exchange(positions=[_ccxt_position(qty=0.5)], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")
        trader = _make_trader(position=0.0, exchange_snapshot=snap, on_mismatch="halt")
        trader._last_reconcile_at = time.time() - 61.0
        trader._trigger_shutdown = MagicMock()
        trader._periodic_reconcile(time.time())
        trader._trigger_shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# F10 — Bootstrap: PaperTrader.restore() calls reconcile-on-boot
# ---------------------------------------------------------------------------

class TestBootstrapRestore:
    def test_restore_calls_reconcile_on_boot(self):
        """PaperTrader.restore() calls _reconcile_on_boot after state load."""
        from deployment.paper_trader import PaperTrader
        from unittest.mock import patch as _patch

        agent = MagicMock()
        config = {
            "paper_trading": {"symbol": "BTC/USDT", "initial_balance": 10_000.0,
                               "window_size": 5},
        }
        # Provide a real snapshot so restore() doesn't return early at "snap is None"
        snap_state = {
            "cash": 10_000.0, "equity": 10_000.0, "symbol": "BTC/USDT",
            "position": 0.0, "entry_price": 0.0, "current_price": 50_000.0,
            "peak_value": 10_000.0, "step": 5, "orders": [],
            "portfolio_history": [10_000.0] * 5, "trades": [],
            "shutdown_triggered": False, "shutdown_reason": "",
        }
        state_store = MagicMock()
        state_store.db_path = ":memory:"
        state_store.load_latest.return_value = snap_state

        ex = _make_exchange(positions=[], open_orders=[])
        snap = ExchangeSnapshot(ex, "BTC/USDT")

        with _patch.object(PaperTrader, "_reconcile_on_boot") as mock_boot:
            PaperTrader.restore(
                state_store, agent, config,
                simulation_mode=True,
                exchange_snapshot=snap,
            )
        mock_boot.assert_called_once()

    def test_restore_detects_position_mismatch(self):
        """restore() with snapshot mismatch → halt triggered (on_mismatch=halt)."""
        from deployment.paper_trader import PaperTrader, TradingState
        from deployment.execution.position_tracker import PositionTracker

        agent = MagicMock()
        config = {
            "paper_trading": {
                "symbol": "BTC/USDT",
                "initial_balance": 10_000.0,
                "window_size": 5,
            },
            "reconciliation": {"on_mismatch": "halt"},
        }

        # Saved state has 0.5 BTC position
        snap_state = {
            "cash": 5_000.0,
            "equity": 30_000.0,
            "symbol": "BTC/USDT",
            "position": 0.5,
            "entry_price": 50_000.0,
            "current_price": 50_000.0,
            "peak_value": 30_000.0,
            "step": 10,
            "orders": [],
            "portfolio_history": [10_000.0] * 10,
            "trades": [],
            "shutdown_triggered": False,
            "shutdown_reason": "",
        }
        state_store = MagicMock()
        state_store.db_path = ":memory:"
        state_store.load_latest.return_value = snap_state

        # Exchange says 0.0 position → mismatch
        ex = _make_exchange(positions=[], open_orders=[], balance={"free": {"BTC": 0.0}, "used": {}, "total": {}})
        exchange_snap = ExchangeSnapshot(ex, "BTC/USDT")

        trader = PaperTrader.restore(
            state_store, agent, config,
            simulation_mode=True,
            exchange_snapshot=exchange_snap,
        )
        # The mismatch should have triggered shutdown
        assert trader.state.shutdown_triggered is True
        assert "reconcile_halt" in trader.state.shutdown_reason


# ---------------------------------------------------------------------------
# F11 — ClockSync wire in OrderManager.submit_order
# ---------------------------------------------------------------------------

class TestClockSkewWire:
    def test_check_called_in_non_paper_mode(self):
        """ClockSync.check() is called at submit_order time (non-paper, throttle elapsed)."""
        from deployment.execution.order_manager import OrderManager
        from deployment.execution.clock_sync import ClockSync

        clock = MagicMock(spec=ClockSync)
        clock.is_halted = False
        clock.check.return_value = 0.5

        om = OrderManager(
            exchange_config={"exchange_mode": "paper"},  # paper so no real exchange
            paper_mode=True,
            clock_sync=clock,
        )
        # Force non-paper mode flag to exercise the check path
        om.paper_mode = False
        om._last_clock_check_at = 0.0   # expired
        om._clock_check_interval = 0.0

        try:
            om.submit_order("buy", 0.01, current_price=50_000.0)
        except Exception:
            pass  # execution may fail in mock; we only care about check() call

        clock.check.assert_called()

    def test_check_throttled_within_interval(self):
        """ClockSync.check() is NOT called if interval hasn't elapsed."""
        from deployment.execution.order_manager import OrderManager
        from deployment.execution.clock_sync import ClockSync
        import time as _time

        clock = MagicMock(spec=ClockSync)
        clock.is_halted = False

        om = OrderManager(paper_mode=True, clock_sync=clock)
        om.paper_mode = False
        om._last_clock_check_at = _time.monotonic()   # just checked
        om._clock_check_interval = 999.0              # very long

        try:
            om.submit_order("buy", 0.01, current_price=50_000.0)
        except Exception:
            pass

        clock.check.assert_not_called()

    def test_halted_clock_raises(self):
        """If is_halted is True, submit_order raises RuntimeError."""
        from deployment.execution.order_manager import OrderManager
        from deployment.execution.clock_sync import ClockSync

        clock = MagicMock(spec=ClockSync)
        clock.is_halted = True
        clock.check.return_value = 10.0

        # Use paper_mode=True to avoid real CCXT init, then override flags
        om = OrderManager(paper_mode=True, clock_sync=clock)
        om.paper_mode = False   # trick the F11 check path only
        om._last_clock_check_at = 0.0
        om._clock_check_interval = 0.0

        with pytest.raises(RuntimeError, match="clock skew"):
            om.submit_order("buy", 0.01, current_price=50_000.0)
