"""
Week 64 — Live Risk Enforcement tests (S46)

Covers:
  S41  Correlation limit: order rejected + audit event logged
  S42  FatFingerGuard: hard cap + size multiplier
  S43  VolatilityCircuitBreaker: trip on high vol, auto-reset after cooldown
  S44  Idempotency key: deduplication, no duplicate live submissions
  S45  RateLimiter: token-bucket correctness; ClockSync: drift detection + halt
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from typing import List
from unittest.mock import MagicMock

import pytest

from deployment.execution.circuit_breaker import VolatilityCircuitBreaker
from deployment.execution.clock_sync import ClockSync
from deployment.execution.fat_finger_guard import FatFingerGuard
from deployment.execution.order_manager import Order, OrderManager
from deployment.execution.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manager(
    extra: dict | None = None,
    risk_manager=None,
    audit_logger=None,
    fat_finger_guard=None,
    circuit_breaker=None,
    clock_sync=None,
) -> OrderManager:
    cfg = {"initial_cash": 100_000.0, "max_order_size": 10.0}
    if extra:
        cfg.update(extra)
    return OrderManager(
        exchange_config=cfg,
        paper_mode=True,
        risk_manager=risk_manager,
        audit_logger=audit_logger,
        fat_finger_guard=fat_finger_guard,
        circuit_breaker=circuit_breaker,
        clock_sync=clock_sync,
    )


# ---------------------------------------------------------------------------
# S42 — FatFingerGuard unit tests
# ---------------------------------------------------------------------------


class TestFatFingerGuard:
    def test_always_passes_with_no_history_and_no_hard_cap(self):
        guard = FatFingerGuard(size_multiplier_limit=5.0, hard_cap=0.0)
        ok, reason = guard.check(1_000_000.0)
        assert ok, reason

    def test_hard_cap_rejects(self):
        guard = FatFingerGuard(hard_cap=1.0)
        ok, reason = guard.check(1.001)
        assert not ok
        assert "hard_cap" in reason

    def test_hard_cap_passes_at_boundary(self):
        guard = FatFingerGuard(hard_cap=1.0)
        ok, _ = guard.check(1.0)
        assert ok

    def test_multiplier_rejects_after_history(self):
        guard = FatFingerGuard(size_multiplier_limit=3.0, hard_cap=0.0)
        for _ in range(5):
            guard.record_fill(1.0)
        # avg = 1.0 → limit = 3.0; order of 3.01 should be rejected
        ok, reason = guard.check(3.01)
        assert not ok
        assert "avg" in reason

    def test_multiplier_passes_within_limit(self):
        guard = FatFingerGuard(size_multiplier_limit=3.0, hard_cap=0.0)
        for _ in range(5):
            guard.record_fill(1.0)
        ok, _ = guard.check(2.9)
        assert ok

    def test_history_capped_at_lookback(self):
        guard = FatFingerGuard(lookback=3)
        for v in [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]:
            guard.record_fill(v)
        assert guard.history_size == 3

    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError):
            FatFingerGuard(lookback=0)

    def test_thread_safety(self):
        guard = FatFingerGuard(size_multiplier_limit=5.0, hard_cap=100.0, lookback=50)
        errors: List[Exception] = []

        def worker():
            try:
                for i in range(20):
                    guard.record_fill(float(i + 1))
                    ok, _ = guard.check(float(i + 1))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# S42 — FatFingerGuard integration with OrderManager
# ---------------------------------------------------------------------------


class TestFatFingerInOrderManager:
    def test_hard_cap_rejects_order(self):
        guard = FatFingerGuard(hard_cap=0.5)
        mgr = _manager(fat_finger_guard=guard)
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=1.0, current_price=100.0)
        assert mgr.check_order(order_id) == "failed"

    def test_multiplier_rejects_after_fills(self):
        guard = FatFingerGuard(size_multiplier_limit=2.0, hard_cap=0.0)
        mgr = _manager(fat_finger_guard=guard)
        mgr.update_paper_price(100.0)
        # Record some fills via normal orders
        for _ in range(5):
            mgr.submit_order("buy", amount=1.0, current_price=100.0)
        # avg ≈ 1.0 → limit = 2.0; 2.01 should fail
        order_id = mgr.submit_order("buy", amount=2.01, current_price=100.0)
        assert mgr.check_order(order_id) == "failed"

    def test_audit_log_on_fat_finger_reject(self):
        audit = MagicMock()
        guard = FatFingerGuard(hard_cap=0.5)
        mgr = _manager(fat_finger_guard=guard, audit_logger=audit)
        mgr.update_paper_price(100.0)
        mgr.submit_order("buy", amount=1.0, current_price=100.0)
        # audit.log_risk_event should have been called with fat_finger reason
        calls = [c.args[0] for c in audit.log_risk_event.call_args_list]
        assert any("fat_finger" in str(c.get("reason", "")) for c in calls)


# ---------------------------------------------------------------------------
# S43 — VolatilityCircuitBreaker unit tests
# ---------------------------------------------------------------------------


class TestVolatilityCircuitBreaker:
    def test_not_tripped_with_low_vol(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.10, window=5)
        prices = [100.0, 100.1, 100.2, 100.1, 100.2, 100.1]
        for p in prices:
            cb.update(p)
        assert not cb.is_tripped()

    def test_trips_on_high_vol(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.01, window=5)
        prices = [100.0, 150.0, 50.0, 200.0, 25.0, 300.0]
        for p in prices:
            cb.update(p)
        assert cb.is_tripped()

    def test_auto_reset_after_cooldown(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.01, window=5, cooldown=0.0)
        prices = [100.0, 150.0, 50.0, 200.0, 25.0, 300.0]
        for p in prices:
            cb.update(p)
        assert cb.is_tripped()
        # Feed stable prices to drive vol down
        for _ in range(10):
            cb.update(100.0)
        # After cooldown=0 and stable prices, should auto-reset on next is_tripped()
        assert not cb.is_tripped()

    def test_current_vol_returns_none_insufficient_data(self):
        cb = VolatilityCircuitBreaker(window=10)
        cb.update(100.0)
        assert cb.current_vol is None

    def test_current_vol_returns_float_with_enough_data(self):
        cb = VolatilityCircuitBreaker(window=5)
        for p in [100, 101, 102, 101, 100, 101]:
            cb.update(float(p))
        assert cb.current_vol is not None
        assert cb.current_vol >= 0.0

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            VolatilityCircuitBreaker(window=1)
        with pytest.raises(ValueError):
            VolatilityCircuitBreaker(vol_threshold=0.0)

    def test_thread_safety(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.05, window=10)
        errors: List[Exception] = []

        def worker():
            try:
                for i in range(30):
                    cb.update(100.0 + (i % 5))
                    cb.is_tripped()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# S43 — Circuit breaker integration with OrderManager
# ---------------------------------------------------------------------------


class TestCircuitBreakerInOrderManager:
    def test_order_blocked_when_tripped(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.001, window=5)
        mgr = _manager(circuit_breaker=cb)
        # Trip the breaker via violent price moves
        for p in [100.0, 200.0, 50.0, 300.0, 10.0, 500.0]:
            mgr.update_paper_price(p)
        assert cb.is_tripped()
        order_id = mgr.submit_order("buy", amount=0.1, current_price=500.0)
        assert mgr.check_order(order_id) == "failed"

    def test_order_allowed_when_not_tripped(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.50, window=5)
        mgr = _manager(circuit_breaker=cb)
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(order_id) == "filled"

    def test_audit_log_on_circuit_breaker(self):
        audit = MagicMock()
        cb = VolatilityCircuitBreaker(vol_threshold=0.001, window=5)
        mgr = _manager(circuit_breaker=cb, audit_logger=audit)
        for p in [100.0, 200.0, 50.0, 300.0, 10.0, 500.0]:
            mgr.update_paper_price(p)
        mgr.submit_order("buy", amount=0.1, current_price=500.0)
        calls = [c.args[0] for c in audit.log_risk_event.call_args_list]
        assert any("volatility_circuit_breaker" in str(c.get("reason", "")) for c in calls)


# ---------------------------------------------------------------------------
# S41 — Correlation limit tests
# ---------------------------------------------------------------------------


class TestCorrelationLimit:
    def _make_risk_manager(self, check_result: bool):
        rm = MagicMock()
        rm.check_drawdown.return_value = False
        rm.check_correlation.return_value = check_result
        return rm

    def test_order_rejected_on_high_correlation(self):
        rm = self._make_risk_manager(True)
        mgr = _manager(risk_manager=rm)
        mgr.set_correlation(0.95)
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(order_id) == "failed"
        rm.check_correlation.assert_called_once()

    def test_order_passes_on_low_correlation(self):
        rm = self._make_risk_manager(False)
        mgr = _manager(risk_manager=rm)
        mgr.set_correlation(0.3)
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(order_id) == "filled"

    def test_no_check_without_set_correlation(self):
        rm = self._make_risk_manager(True)
        mgr = _manager(risk_manager=rm)
        # correlation not set → check_correlation should NOT be called
        mgr.update_paper_price(100.0)
        mgr.submit_order("buy", amount=0.1, current_price=100.0)
        rm.check_correlation.assert_not_called()

    def test_fallback_inline_check_without_risk_manager(self):
        # No risk_manager; inline abs(corr) > threshold check
        mgr = _manager(extra={"correlation_threshold": 0.7})
        mgr.set_correlation(0.9)  # exceeds 0.7
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(order_id) == "failed"

    def test_audit_log_on_correlation_reject(self):
        audit = MagicMock()
        mgr = _manager(extra={"correlation_threshold": 0.5}, audit_logger=audit)
        mgr.set_correlation(0.8)
        mgr.update_paper_price(100.0)
        mgr.submit_order("buy", amount=0.1, current_price=100.0)
        calls = [c.args[0] for c in audit.log_risk_event.call_args_list]
        assert any("correlation_limit" in str(c.get("type", "")) for c in calls)


# ---------------------------------------------------------------------------
# S44 — Idempotency key tests
# ---------------------------------------------------------------------------


class TestIdempotencyKey:
    def test_duplicate_key_returns_same_order_id(self):
        mgr = _manager()
        mgr.update_paper_price(100.0)
        key = "test-key-001"
        id1 = mgr.submit_order("buy", amount=0.1, current_price=100.0,
                                idempotency_key=key)
        id2 = mgr.submit_order("buy", amount=0.1, current_price=100.0,
                                idempotency_key=key)
        assert id1 == id2

    def test_different_keys_produce_different_orders(self):
        mgr = _manager()
        mgr.update_paper_price(100.0)
        id1 = mgr.submit_order("buy", amount=0.1, current_price=100.0,
                                idempotency_key="key-A")
        id2 = mgr.submit_order("buy", amount=0.1, current_price=100.0,
                                idempotency_key="key-B")
        assert id1 != id2

    def test_no_key_always_new_order(self):
        mgr = _manager()
        mgr.update_paper_price(100.0)
        id1 = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        id2 = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert id1 != id2

    def test_idempotency_key_stored_on_order(self):
        mgr = _manager()
        mgr.update_paper_price(100.0)
        key = "stored-key"
        order_id = mgr.submit_order("buy", amount=0.1, current_price=100.0,
                                     idempotency_key=key)
        order = mgr.get_order(order_id)
        assert order.idempotency_key == key

    def test_concurrent_duplicate_keys(self):
        """Race condition: two threads with same key → only one order created."""
        mgr = _manager()
        mgr.update_paper_price(100.0)
        key = "concurrent-key"
        results: List[str] = []

        def worker():
            oid = mgr.submit_order("buy", amount=0.01, current_price=100.0,
                                   idempotency_key=key)
            results.append(oid)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same order_id
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# S45 — RateLimiter unit tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_max_calls(self):
        rl = RateLimiter(max_calls=5, period=10.0)
        for _ in range(5):
            rl.acquire()  # should not block

    def test_blocks_when_limit_reached(self):
        rl = RateLimiter(max_calls=2, period=1.0)
        rl.acquire()
        rl.acquire()
        start = time.monotonic()
        rl.acquire()  # must wait for window to pass
        elapsed = time.monotonic() - start
        assert elapsed >= 0.9  # waited at least 0.9s (allow some slack)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            RateLimiter(max_calls=0)
        with pytest.raises(ValueError):
            RateLimiter(period=0)

    def test_thread_safety(self):
        rl = RateLimiter(max_calls=20, period=1.0)
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(3):
                    rl.acquire()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# S45 — ClockSync unit tests
# ---------------------------------------------------------------------------


class TestClockSync:
    def test_no_drift_when_server_close_to_local(self):
        cs = ClockSync(max_drift_sec=5.0, time_fn=lambda: time.time())
        drift = cs.check()
        assert drift < 1.0
        assert not cs.is_halted

    def test_warns_on_large_drift(self):
        future_time = time.time() + 100.0  # 100s ahead
        cs = ClockSync(max_drift_sec=5.0, time_fn=lambda: future_time)
        drift = cs.check()
        assert drift >= 5.0
        assert not cs.is_halted  # halt_on_skew=False by default

    def test_halts_on_skew_when_configured(self):
        future_time = time.time() + 100.0
        cs = ClockSync(max_drift_sec=5.0, halt_on_skew=True,
                       time_fn=lambda: future_time)
        cs.check()
        assert cs.is_halted

    def test_reset_halt_clears_flag(self):
        future_time = time.time() + 100.0
        cs = ClockSync(max_drift_sec=5.0, halt_on_skew=True,
                       time_fn=lambda: future_time)
        cs.check()
        assert cs.is_halted
        cs.reset_halt()
        assert not cs.is_halted

    def test_returns_zero_when_no_time_fn(self):
        cs = ClockSync()  # no time_fn, no exchange
        drift = cs.check()
        assert drift == 0.0

    def test_last_drift_updated(self):
        cs = ClockSync(time_fn=lambda: time.time() + 3.0)
        assert cs.last_drift is None
        cs.check()
        assert cs.last_drift is not None
        assert 2.5 < cs.last_drift < 4.0

    def test_time_fn_exception_returns_zero(self):
        def bad_fn():
            raise RuntimeError("network error")
        cs = ClockSync(time_fn=bad_fn)
        drift = cs.check()
        assert drift == 0.0

    def test_invalid_max_drift_raises(self):
        with pytest.raises(ValueError):
            ClockSync(max_drift_sec=0)


# ---------------------------------------------------------------------------
# S45 — ClockSync halt blocks OrderManager
# ---------------------------------------------------------------------------


class TestClockSyncInOrderManager:
    def test_order_blocked_when_clock_halted(self):
        future_time = time.time() + 100.0
        cs = ClockSync(max_drift_sec=1.0, halt_on_skew=True,
                       time_fn=lambda: future_time)
        cs.check()  # trip the halt
        mgr = _manager(clock_sync=cs)
        mgr.update_paper_price(100.0)
        with pytest.raises(RuntimeError, match="clock skew"):
            mgr.submit_order("buy", amount=0.1, current_price=100.0)

    def test_order_allowed_after_reset(self):
        future_time = time.time() + 100.0
        cs = ClockSync(max_drift_sec=1.0, halt_on_skew=True,
                       time_fn=lambda: future_time)
        cs.check()
        cs.reset_halt()
        mgr = _manager(clock_sync=cs)
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(order_id) == "filled"


# ---------------------------------------------------------------------------
# S46 — Integration: all safeguards in a single scenario
# ---------------------------------------------------------------------------


class TestAllSafeguardsIntegration:
    """Scenario: each safeguard fires independently and produces a failed order
    + audit event.  Normal orders still fill when no safeguard is active."""

    def _make_audit(self):
        audit = MagicMock()
        return audit

    def test_normal_order_fills(self):
        mgr = _manager()
        mgr.update_paper_price(100.0)
        oid = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(oid) == "filled"

    def test_correlation_then_normal(self):
        audit = self._make_audit()
        mgr = _manager(extra={"correlation_threshold": 0.5}, audit_logger=audit)
        mgr.set_correlation(0.9)
        mgr.update_paper_price(100.0)
        bad = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(bad) == "failed"
        # Fix correlation, order should pass
        mgr.set_correlation(0.3)
        good = mgr.submit_order("buy", amount=0.1, current_price=100.0)
        assert mgr.check_order(good) == "filled"

    def test_fat_finger_then_normal(self):
        guard = FatFingerGuard(hard_cap=1.0)
        mgr = _manager(fat_finger_guard=guard)
        mgr.update_paper_price(100.0)
        bad = mgr.submit_order("buy", amount=5.0, current_price=100.0)
        assert mgr.check_order(bad) == "failed"
        good = mgr.submit_order("buy", amount=0.5, current_price=100.0)
        assert mgr.check_order(good) == "filled"

    def test_circuit_breaker_then_normal(self):
        cb = VolatilityCircuitBreaker(vol_threshold=0.001, window=5, cooldown=0.0)
        mgr = _manager(circuit_breaker=cb)
        for p in [100.0, 200.0, 50.0, 300.0, 10.0, 500.0]:
            mgr.update_paper_price(p)
        bad = mgr.submit_order("buy", amount=0.1, current_price=500.0)
        assert mgr.check_order(bad) == "failed"
        # Feed stable prices until breaker resets
        for _ in range(20):
            cb.update(500.0)
        assert not cb.is_tripped()
        good = mgr.submit_order("buy", amount=0.1, current_price=500.0)
        assert mgr.check_order(good) == "filled"

    def test_audit_log_contains_rejection_events(self):
        audit = self._make_audit()
        guard = FatFingerGuard(hard_cap=0.5)
        mgr = _manager(fat_finger_guard=guard, audit_logger=audit)
        mgr.update_paper_price(100.0)
        mgr.submit_order("buy", amount=1.0, current_price=100.0)
        risk_calls = audit.log_risk_event.call_args_list
        assert len(risk_calls) >= 1
        reasons = [c.args[0].get("reason", "") for c in risk_calls]
        assert any("fat_finger" in r for r in reasons)
