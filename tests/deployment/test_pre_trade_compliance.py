"""
Week 76 (G6-G10): Pre-Trade Compliance Tests.

Coverage:
  G6  — Position limits: per-symbol notional, portfolio notional, leverage cap
  G7  — Self-trade prevention (resting order cross detection)
  G8  — Notional cap per unit time (hourly / daily rolling windows)
  G9  — Wash trade guard (same-direction cooldown)
  G10 — E2E: multiple guards active simultaneously, audit completeness

완료 조건:
  - 각 compliance rule에 대응 테스트 존재.
  - 거부 시 audit 완전 (order_rejected risk_event 포함).
"""

from __future__ import annotations

import threading
import time
from typing import Any, List
from unittest.mock import MagicMock, call

import pytest

from deployment.execution.order_manager import OrderManager
from risk_management.limits import ComplianceConfig, PreTradeComplianceChecker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_checker(**kwargs) -> PreTradeComplianceChecker:
    cfg = ComplianceConfig(**kwargs)
    return PreTradeComplianceChecker(cfg)


def make_manager(compliance_checker=None, **cfg_overrides) -> OrderManager:
    cfg = {"initial_cash": 100_000.0, "max_order_size": 100.0}
    cfg.update(cfg_overrides)
    m = OrderManager(
        exchange_config=cfg,
        paper_mode=True,
        compliance_checker=compliance_checker,
    )
    m.update_paper_price(1_000.0)
    return m


def rejected_order_ids(manager: OrderManager) -> List[str]:
    """Return order IDs whose status is 'failed'."""
    with manager._lock:
        return [oid for oid, o in manager._orders.items() if o.status == "failed"]


# ---------------------------------------------------------------------------
# G6: Position limits — unit tests
# ---------------------------------------------------------------------------

class TestPositionLimitsUnit:
    def test_per_symbol_notional_passes_below_max(self):
        checker = make_checker(per_symbol_notional_max=10_000)
        ok, reason = checker.check_position_limits(
            "BTC/USDT", order_notional=5_000, current_symbol_notional=0
        )
        assert ok
        assert reason == ""

    def test_per_symbol_notional_rejects_at_max(self):
        checker = make_checker(per_symbol_notional_max=10_000)
        ok, reason = checker.check_position_limits(
            "BTC/USDT", order_notional=5_001, current_symbol_notional=5_000
        )
        assert not ok
        assert "per_symbol" in reason
        assert "BTC/USDT" in reason

    def test_portfolio_notional_passes_below_max(self):
        checker = make_checker(portfolio_notional_max=50_000)
        ok, reason = checker.check_position_limits(
            "BTC/USDT", order_notional=10_000, current_portfolio_notional=39_999
        )
        assert ok

    def test_portfolio_notional_rejects_at_max(self):
        checker = make_checker(portfolio_notional_max=50_000)
        ok, reason = checker.check_position_limits(
            "BTC/USDT", order_notional=1_000, current_portfolio_notional=50_000
        )
        assert not ok
        assert "portfolio" in reason

    def test_leverage_cap_passes_below(self):
        checker = make_checker(leverage_max=3.0)
        ok, _ = checker.check_position_limits(
            "BTC/USDT", order_notional=1_000, leverage=2.9
        )
        assert ok

    def test_leverage_cap_rejects_above(self):
        checker = make_checker(leverage_max=3.0)
        ok, reason = checker.check_position_limits(
            "BTC/USDT", order_notional=1_000, leverage=3.1
        )
        assert not ok
        assert "leverage" in reason

    def test_defaults_allow_everything(self):
        checker = PreTradeComplianceChecker()  # all limits = inf
        ok, _ = checker.check_position_limits(
            "BTC/USDT", order_notional=1e15, current_symbol_notional=1e15, leverage=1000
        )
        assert ok


class TestPositionLimitsIntegration:
    def test_order_rejected_on_symbol_notional_breach(self):
        """OrderManager rejects when position would exceed per_symbol_notional_max."""
        checker = make_checker(per_symbol_notional_max=500)
        m = make_manager(compliance_checker=checker)
        # First buy: 0.3 BTC @ 1000 = 300 notional (OK)
        oid1 = m.submit_order("buy", 0.3, current_price=1_000.0)
        assert m.check_order(oid1) == "filled"
        # Second buy: would push to 600 notional (FAIL)
        oid2 = m.submit_order("buy", 0.3, current_price=1_000.0)
        assert m.check_order(oid2) == "failed"

    def test_rejected_order_logged_to_audit(self):
        audit = MagicMock()
        checker = make_checker(per_symbol_notional_max=100)
        m = make_manager(compliance_checker=checker)
        m._audit_logger = audit
        m.submit_order("buy", 0.2, current_price=1_000.0)  # 200 > 100: rejected
        risk_calls = [
            c for c in audit.log_risk_event.call_args_list
        ]
        assert len(risk_calls) >= 1
        last_event = risk_calls[-1].args[0]
        assert last_event["type"] == "order_rejected"
        assert "position_limit" in last_event["reason"]


# ---------------------------------------------------------------------------
# G7: Self-trade prevention — unit tests
# ---------------------------------------------------------------------------

class TestSelfTradeUnit:
    def test_no_cross_when_no_open_orders(self):
        checker = make_checker(self_trade_prevention=True)
        ok, _ = checker.check_self_trade("BTC/USDT", price=1_000.0, side="buy")
        assert ok

    def test_cross_detected_opposite_side(self):
        checker = make_checker(self_trade_prevention=True)
        checker.register_open_order("BTC/USDT", price=1_000.0, side="sell")
        ok, reason = checker.check_self_trade("BTC/USDT", price=1_000.0, side="buy")
        assert not ok
        assert "self_trade" in reason

    def test_same_side_not_a_cross(self):
        checker = make_checker(self_trade_prevention=True)
        checker.register_open_order("BTC/USDT", price=1_000.0, side="buy")
        ok, _ = checker.check_self_trade("BTC/USDT", price=1_000.0, side="buy")
        assert ok

    def test_different_price_not_a_cross(self):
        checker = make_checker(self_trade_prevention=True)
        checker.register_open_order("BTC/USDT", price=1_000.0, side="sell")
        ok, _ = checker.check_self_trade("BTC/USDT", price=999.0, side="buy")
        assert ok

    def test_deregister_clears_open_order(self):
        checker = make_checker(self_trade_prevention=True)
        checker.register_open_order("BTC/USDT", price=1_000.0, side="sell")
        checker.deregister_open_order("BTC/USDT", price=1_000.0)
        ok, _ = checker.check_self_trade("BTC/USDT", price=1_000.0, side="buy")
        assert ok

    def test_disabled_allows_cross(self):
        checker = make_checker(self_trade_prevention=False)
        checker.register_open_order("BTC/USDT", price=1_000.0, side="sell")
        ok, _ = checker.check_self_trade("BTC/USDT", price=1_000.0, side="buy")
        assert ok


class TestSelfTradeIntegration:
    def test_limit_order_registered_as_open(self):
        checker = make_checker(self_trade_prevention=True)
        m = make_manager(compliance_checker=checker)
        # Buy 1 BTC first so we have inventory to sell
        m.submit_order("buy", 1.0, current_price=1_000.0)
        # Place a pending limit sell at 2000; stays pending because market is 1000 < 2000
        oid_sell = m.submit_order(
            "sell", 0.5, order_type="limit", limit_price=2_000.0, current_price=1_000.0
        )
        assert m.check_order(oid_sell) == "pending"
        # Now try to buy at the same limit price — should be rejected (self-trade cross)
        oid_buy = m.submit_order(
            "buy", 0.5, order_type="limit", limit_price=2_000.0, current_price=1_000.0
        )
        assert m.check_order(oid_buy) == "failed"

    def test_cancel_deregisters_open_order(self):
        checker = make_checker(self_trade_prevention=True)
        m = make_manager(compliance_checker=checker)
        m.submit_order("buy", 1.0, current_price=1_000.0)
        oid_sell = m.submit_order(
            "sell", 0.5, order_type="limit", limit_price=2_000.0, current_price=1_000.0
        )
        assert m.check_order(oid_sell) == "pending"
        m.cancel_order(oid_sell)
        # After cancel, crossing buy should be accepted
        oid_buy = m.submit_order(
            "buy", 0.1, order_type="limit", limit_price=2_000.0, current_price=1_000.0
        )
        assert m.check_order(oid_buy) != "failed"


# ---------------------------------------------------------------------------
# G8: Notional cap — unit tests
# ---------------------------------------------------------------------------

class TestNotionalCapUnit:
    def test_first_order_passes(self):
        checker = make_checker(hourly_notional_cap=10_000)
        ok, _ = checker.check_notional_cap(5_000)
        assert ok

    def test_exceeds_hourly_cap(self):
        checker = make_checker(hourly_notional_cap=10_000)
        checker.record_order("X", "buy", notional=8_000)
        ok, reason = checker.check_notional_cap(3_000)
        assert not ok
        assert "hourly" in reason

    def test_exceeds_daily_cap(self):
        checker = make_checker(daily_notional_cap=20_000, hourly_notional_cap=float("inf"))
        for _ in range(4):
            checker.record_order("X", "buy", notional=5_000)
        ok, reason = checker.check_notional_cap(1)
        assert not ok
        assert "daily" in reason

    def test_hourly_window_evicts_old_entries(self):
        checker = make_checker(hourly_notional_cap=10_000)
        # Inject a stale entry older than 1 hour
        with checker._lock:
            checker._hourly_window.append((time.time() - 3700, 9_000))
        # Current order should now pass since old entry is expired
        ok, _ = checker.check_notional_cap(9_999)
        assert ok

    def test_defaults_allow_any_notional(self):
        checker = PreTradeComplianceChecker()
        ok, _ = checker.check_notional_cap(1e15)
        assert ok


class TestNotionalCapIntegration:
    def test_order_rejected_when_hourly_cap_breached(self):
        checker = make_checker(hourly_notional_cap=1_000)
        m = make_manager(compliance_checker=checker)
        # First order: 0.8 * 1000 = 800 (OK, fits under 1000 cap)
        oid1 = m.submit_order("buy", 0.8, current_price=1_000.0)
        assert m.check_order(oid1) == "filled"
        # Second order: would push to 1200 (FAIL)
        oid2 = m.submit_order("buy", 0.4, current_price=1_000.0)
        assert m.check_order(oid2) == "failed"

    def test_audit_contains_notional_cap_reason(self):
        audit = MagicMock()
        checker = make_checker(hourly_notional_cap=500)
        m = make_manager(compliance_checker=checker)
        m._audit_logger = audit
        m.submit_order("buy", 0.6, current_price=1_000.0)  # 600 > 500: rejected
        risk_calls = audit.log_risk_event.call_args_list
        assert any(
            "notional_cap" in c.args[0]["reason"]
            for c in risk_calls
        )


# ---------------------------------------------------------------------------
# G9: Wash trade guard — unit tests
# ---------------------------------------------------------------------------

class TestWashTradeUnit:
    def test_first_order_passes(self):
        checker = make_checker(wash_trade_cooldown_sec=5.0)
        ok, _ = checker.check_wash_trade("BTC/USDT", "buy")
        assert ok

    def test_second_order_within_cooldown_rejected(self):
        checker = make_checker(wash_trade_cooldown_sec=5.0)
        checker.record_order("BTC/USDT", "buy", notional=100)
        ok, reason = checker.check_wash_trade("BTC/USDT", "buy")
        assert not ok
        assert "wash_trade" in reason

    def test_opposite_side_not_blocked(self):
        checker = make_checker(wash_trade_cooldown_sec=5.0)
        checker.record_order("BTC/USDT", "buy", notional=100)
        ok, _ = checker.check_wash_trade("BTC/USDT", "sell")
        assert ok

    def test_different_symbol_not_blocked(self):
        checker = make_checker(wash_trade_cooldown_sec=5.0)
        checker.record_order("BTC/USDT", "buy", notional=100)
        ok, _ = checker.check_wash_trade("ETH/USDT", "buy")
        assert ok

    def test_order_passes_after_cooldown_expires(self):
        checker = make_checker(wash_trade_cooldown_sec=0.05)  # 50ms
        checker.record_order("BTC/USDT", "buy", notional=100)
        time.sleep(0.06)
        ok, _ = checker.check_wash_trade("BTC/USDT", "buy")
        assert ok

    def test_disabled_cooldown_always_passes(self):
        checker = make_checker(wash_trade_cooldown_sec=0.0)
        checker.record_order("BTC/USDT", "buy", notional=100)
        ok, _ = checker.check_wash_trade("BTC/USDT", "buy")
        assert ok


class TestWashTradeIntegration:
    def test_rapid_same_direction_orders_blocked(self):
        checker = make_checker(wash_trade_cooldown_sec=60.0)
        m = make_manager(compliance_checker=checker)
        oid1 = m.submit_order("buy", 0.1, current_price=1_000.0)
        assert m.check_order(oid1) == "filled"
        oid2 = m.submit_order("buy", 0.1, current_price=1_000.0)
        assert m.check_order(oid2) == "failed"

    def test_opposite_direction_not_blocked(self):
        checker = make_checker(wash_trade_cooldown_sec=60.0)
        m = make_manager(compliance_checker=checker)
        m.submit_order("buy", 1.0, current_price=1_000.0)
        # Sell should not be blocked by wash trade guard (different direction)
        oid_sell = m.submit_order("sell", 0.5, current_price=1_000.0)
        assert m.check_order(oid_sell) == "filled"


# ---------------------------------------------------------------------------
# G10: E2E — multiple guards + audit completeness
# ---------------------------------------------------------------------------

class TestComplianceE2E:
    def test_all_guards_can_fire_in_sequence(self):
        """Each guard fires in turn across separate scenarios."""
        # G6 fires
        checker_g6 = make_checker(per_symbol_notional_max=100)
        m_g6 = make_manager(compliance_checker=checker_g6)
        oid = m_g6.submit_order("buy", 0.2, current_price=1_000.0)  # 200 > 100
        assert m_g6.check_order(oid) == "failed"

        # G8 fires
        checker_g8 = make_checker(hourly_notional_cap=100)
        m_g8 = make_manager(compliance_checker=checker_g8)
        oid = m_g8.submit_order("buy", 0.2, current_price=1_000.0)  # 200 > 100
        assert m_g8.check_order(oid) == "failed"

        # G9 fires
        checker_g9 = make_checker(wash_trade_cooldown_sec=60.0)
        m_g9 = make_manager(compliance_checker=checker_g9)
        m_g9.submit_order("buy", 0.01, current_price=1_000.0)
        oid = m_g9.submit_order("buy", 0.01, current_price=1_000.0)
        assert m_g9.check_order(oid) == "failed"

    def test_all_rejected_orders_have_audit_risk_events(self):
        """Every compliance rejection must produce an audit risk event."""
        audit = MagicMock()
        scenarios = [
            # (config_kwargs, buy_amount)
            ({"per_symbol_notional_max": 100}, 0.2),        # G6
            ({"hourly_notional_cap": 100}, 0.2),            # G8
        ]
        for config_kwargs, buy_amount in scenarios:
            audit.reset_mock()
            checker = make_checker(**config_kwargs)
            m = make_manager(compliance_checker=checker)
            m._audit_logger = audit
            m.submit_order("buy", buy_amount, current_price=1_000.0)
            events = [c.args[0] for c in audit.log_risk_event.call_args_list]
            assert any(e["type"] == "order_rejected" for e in events), (
                f"No order_rejected audit event for config {config_kwargs}"
            )

    def test_multiple_guards_active_simultaneously(self):
        """When multiple guards are configured, the first one fires correctly."""
        checker = make_checker(
            per_symbol_notional_max=50_000,  # high — won't fire first
            hourly_notional_cap=100,         # low — fires first
            wash_trade_cooldown_sec=60.0,    # also would fire, but hourly wins
        )
        m = make_manager(compliance_checker=checker)
        oid = m.submit_order("buy", 0.2, current_price=1_000.0)  # 200 > 100 cap
        assert m.check_order(oid) == "failed"
        order = m.get_order(oid)
        assert order is not None

    def test_compliant_order_not_rejected(self):
        """Sanity: order within all limits is accepted."""
        checker = make_checker(
            per_symbol_notional_max=100_000,
            portfolio_notional_max=500_000,
            hourly_notional_cap=200_000,
            daily_notional_cap=500_000,
            wash_trade_cooldown_sec=0.0,
            self_trade_prevention=True,
        )
        m = make_manager(compliance_checker=checker)
        oid = m.submit_order("buy", 1.0, current_price=1_000.0)
        assert m.check_order(oid) == "filled"

    def test_no_compliance_checker_passes_all_orders(self):
        """OrderManager without compliance_checker behaves as before Week 76."""
        m = make_manager(compliance_checker=None)
        oid = m.submit_order("buy", 5.0, current_price=1_000.0)
        assert m.check_order(oid) == "filled"

    def test_thread_safety(self):
        """Concurrent order submissions under a tight hourly cap are consistent."""
        checker = make_checker(hourly_notional_cap=5_000)  # only 5 orders of 1000 each
        m = make_manager(compliance_checker=checker)
        results: List[str] = []
        lock = threading.Lock()

        def submit_one():
            oid = m.submit_order("buy", 1.0, current_price=1_000.0)
            with lock:
                results.append(m.check_order(oid))

        threads = [threading.Thread(target=submit_one) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        filled = results.count("filled")
        failed = results.count("failed")
        assert filled <= 5, f"Expected ≤5 fills, got {filled}"
        assert filled + failed == 10

    def test_rejected_order_fields(self):
        """A rejected order must have status=failed and a non-empty order_id."""
        checker = make_checker(hourly_notional_cap=100)
        m = make_manager(compliance_checker=checker)
        oid = m.submit_order("buy", 0.2, current_price=1_000.0)
        order = m.get_order(oid)
        assert order is not None
        assert order.status == "failed"
        assert order.order_id == oid
        assert order.side == "buy"
        assert order.amount == pytest.approx(0.2)
