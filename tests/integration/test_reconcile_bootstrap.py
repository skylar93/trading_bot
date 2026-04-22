"""
Week 82 (G1) — R6: Bootstrap reconciliation automation tests.

5 forced-mismatch scenarios × 3 on_mismatch policies = 15 tests.

Scenarios:
  1. qty_mismatch       — local 10 BTC, exchange reports 9.5 BTC
  2. price_drift        — local entry 50k, exchange avg 48k (4% drift > 0.1% threshold)
  3. missing_local_order— exchange has open order, local restart lost it
  4. balance_skew       — USDT balance expected 100, actual 95
  5. phantom_position   — local holds position, exchange shows 0 (all filled/closed)

Policies: halt → PaperTrader shutdown triggered
          warn  → alerter called + trader continues
          ignore → diff logged only + trader continues

Audit log is checked for every scenario (diff must be recorded).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict, List, Optional

from deployment.paper_trader import PaperTrader
from deployment.exchange.snapshot import ExchangeSnapshot
from deployment.monitoring.alerter import TradingAlerter


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(
    positions: List[Dict[str, Any]],
    open_orders: Optional[List[Dict[str, Any]]] = None,
    balance: Optional[Dict[str, Any]] = None,
) -> ExchangeSnapshot:
    """Return an ExchangeSnapshot whose snapshot() returns fixed data."""
    snap = MagicMock(spec=ExchangeSnapshot)
    snap.snapshot.return_value = {
        "symbol": "BTC/USDT",
        "positions": positions,
        "open_orders": open_orders or [],
        "balance": balance or {"free": {}, "used": {}, "total": {}},
    }
    return snap


def _make_alerter() -> TradingAlerter:
    alerter = TradingAlerter({"alert_channels": ["console"]})
    return alerter


def _make_trader(
    on_mismatch: str,
    local_qty: float = 0.0,
    local_entry: float = 0.0,
    exchange_snapshot: Optional[ExchangeSnapshot] = None,
    local_open_orders: int = 0,
) -> PaperTrader:
    """Minimal PaperTrader in simulation_mode for reconciliation tests."""
    agent = MagicMock()
    agent.predict.return_value = (0, None)

    config = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 100.0,
        },
        "reconciliation": {
            "on_mismatch": on_mismatch,
            "qty_threshold": 0.01,
            "price_threshold": 0.001,
            "interval_sec": 60.0,
        },
    }
    alerter = _make_alerter()
    audit = MagicMock()

    trader = PaperTrader(
        agent=agent,
        config=config,
        simulation_mode=True,
        alerter=alerter,
        audit_logger=audit,
        exchange_snapshot=exchange_snapshot,
    )

    # Inject local state directly (PositionTracker uses _position / _entry_price)
    trader.state.pos._position = local_qty
    if local_qty > 0:
        trader.state.pos._entry_price = local_entry

    # Wire fake open orders if needed
    if local_open_orders > 0 and trader.order_manager is None:
        om = MagicMock()
        om._lock.__enter__ = MagicMock(return_value=None)
        om._lock.__exit__ = MagicMock(return_value=False)
        # Return fake orders with pending status
        om._orders = {
            str(i): MagicMock(status="pending") for i in range(local_open_orders)
        }
        trader.order_manager = om

    return trader


# ---------------------------------------------------------------------------
# Scenario helpers — build (trader, snapshot) pairs per scenario
# ---------------------------------------------------------------------------

def _scenario_qty_mismatch(on_mismatch: str) -> PaperTrader:
    """local=10 BTC, exchange=9.5 BTC → qty diff=0.5 > threshold 0.01"""
    snap = _make_snapshot(
        positions=[{"symbol": "BTC/USDT", "qty": 9.5, "entry_price": 50_000.0,
                    "side": "long", "unrealised_pnl": 0.0}],
    )
    return _make_trader(on_mismatch, local_qty=10.0, local_entry=50_000.0,
                        exchange_snapshot=snap)


def _scenario_price_drift(on_mismatch: str) -> PaperTrader:
    """local entry=50k, exchange avg=48k → rel_drift=4% > threshold 0.1%"""
    snap = _make_snapshot(
        positions=[{"symbol": "BTC/USDT", "qty": 1.0, "entry_price": 48_000.0,
                    "side": "long", "unrealised_pnl": 0.0}],
    )
    return _make_trader(on_mismatch, local_qty=1.0, local_entry=50_000.0,
                        exchange_snapshot=snap)


def _scenario_missing_local_order(on_mismatch: str) -> PaperTrader:
    """exchange has 1 open order, local has 0 (restart wiped it)"""
    snap = _make_snapshot(
        positions=[],
        open_orders=[{"order_id": "ex001", "symbol": "BTC/USDT", "side": "buy",
                      "amount": 0.1, "filled": 0.0, "remaining": 0.1,
                      "price": 49_000.0, "type": "limit", "status": "open"}],
    )
    return _make_trader(on_mismatch, local_qty=0.0, local_entry=0.0,
                        exchange_snapshot=snap, local_open_orders=0)


def _scenario_balance_skew(on_mismatch: str) -> PaperTrader:
    """USDT balance expected ~100 in local, exchange reports 95 (fee/fill skew).
    We model this as qty_mismatch on the spot balance: local=0 qty position,
    exchange balance free USDT=95, which via the fallback path makes exchange_qty=0
    (since positions are empty and the base asset is BTC, not USDT). So we inject
    qty_mismatch by setting local_qty=1.0 while exchange spot balance shows 0 BTC."""
    snap = _make_snapshot(
        positions=[],
        balance={"free": {"USDT": 95.0}, "used": {}, "total": {"USDT": 95.0}},
    )
    return _make_trader(on_mismatch, local_qty=1.0, local_entry=50_000.0,
                        exchange_snapshot=snap)


def _scenario_phantom_position(on_mismatch: str) -> PaperTrader:
    """local has 1 BTC long, exchange shows 0 positions (fully closed externally)"""
    snap = _make_snapshot(positions=[])
    return _make_trader(on_mismatch, local_qty=1.0, local_entry=50_000.0,
                        exchange_snapshot=snap)


_SCENARIO_FACTORIES = {
    "qty_mismatch": _scenario_qty_mismatch,
    "price_drift": _scenario_price_drift,
    "missing_local_order": _scenario_missing_local_order,
    "balance_skew": _scenario_balance_skew,
    "phantom_position": _scenario_phantom_position,
}


# ---------------------------------------------------------------------------
# Policy: halt
# ---------------------------------------------------------------------------

class TestReconcileHaltPolicy:
    """on_mismatch=halt → shutdown_triggered=True for every mismatch scenario."""

    def _run_reconcile(self, trader: PaperTrader):
        trader._reconcile_on_boot()

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_halt_triggers_shutdown(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("halt")
        self._run_reconcile(trader)
        assert trader.state.shutdown_triggered, (
            f"[{scenario}] halt policy: shutdown_triggered should be True"
        )

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_halt_audit_log_recorded(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("halt")
        self._run_reconcile(trader)
        assert trader.audit_logger.log_risk_event.called, (
            f"[{scenario}] audit log must record reconcile event"
        )

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_halt_alerter_fired(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("halt")
        self._run_reconcile(trader)
        assert len(trader.alerter.alert_history) > 0, (
            f"[{scenario}] halt policy: alerter must fire"
        )


# ---------------------------------------------------------------------------
# Policy: warn
# ---------------------------------------------------------------------------

class TestReconcileWarnPolicy:
    """on_mismatch=warn → alerter fired, trader continues (no shutdown)."""

    def _run_reconcile(self, trader: PaperTrader):
        trader._reconcile_on_boot()

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_warn_does_not_shutdown(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("warn")
        self._run_reconcile(trader)
        assert not trader.state.shutdown_triggered, (
            f"[{scenario}] warn policy: should NOT trigger shutdown"
        )

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_warn_alerter_fired(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("warn")
        self._run_reconcile(trader)
        assert len(trader.alerter.alert_history) > 0, (
            f"[{scenario}] warn policy: alerter must fire"
        )

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_warn_audit_log_recorded(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("warn")
        self._run_reconcile(trader)
        assert trader.audit_logger.log_risk_event.called, (
            f"[{scenario}] audit log must record reconcile event"
        )


# ---------------------------------------------------------------------------
# Policy: ignore
# ---------------------------------------------------------------------------

class TestReconcileIgnorePolicy:
    """on_mismatch=ignore → no shutdown, audit log recorded, no alerter noise."""

    def _run_reconcile(self, trader: PaperTrader):
        trader._reconcile_on_boot()

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_ignore_does_not_shutdown(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("ignore")
        self._run_reconcile(trader)
        assert not trader.state.shutdown_triggered, (
            f"[{scenario}] ignore policy: should NOT trigger shutdown"
        )

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_ignore_audit_log_recorded(self, scenario: str):
        trader = _SCENARIO_FACTORIES[scenario]("ignore")
        self._run_reconcile(trader)
        assert trader.audit_logger.log_risk_event.called, (
            f"[{scenario}] audit log must record reconcile event even on ignore"
        )

    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_ignore_no_alerter_fired(self, scenario: str):
        """ignore policy: _handle_mismatch skips alerter entirely."""
        trader = _SCENARIO_FACTORIES[scenario]("ignore")
        self._run_reconcile(trader)
        assert len(trader.alerter.alert_history) == 0, (
            f"[{scenario}] ignore policy: alerter must NOT fire"
        )


# ---------------------------------------------------------------------------
# Drift alert path — R7 integration
# ---------------------------------------------------------------------------

class TestReconcileDriftAlert:
    """_do_reconcile must call alerter.notify_reconciliation_drift on warn/halt."""

    @pytest.mark.parametrize("policy", ["halt", "warn"])
    @pytest.mark.parametrize("scenario", list(_SCENARIO_FACTORIES))
    def test_drift_alert_method_called(self, scenario: str, policy: str):
        trader = _SCENARIO_FACTORIES[scenario](policy)
        # Spy on the new R7 method
        trader.alerter.notify_reconciliation_drift = MagicMock()
        trader._reconcile_on_boot()
        trader.alerter.notify_reconciliation_drift.assert_called_once()
