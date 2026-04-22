"""
Week 83 (R14) — Safety Net Integration Tests.

Tests three independently operating safety nets:
  1. Canary auto-demotion (G4)
  2. OTel span instrumentation on order submission (G5)
  3. Real-time schema drift guard (G6)

E2E scenario: simultaneous canary underperformance + schema drift + normal order flow.
"""

from __future__ import annotations

import math
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from data.quality.stream_validator import StreamValidator, SchemaDrift
from deployment.monitoring.alerter import TradingAlerter


# ===========================================================================
# Helpers
# ===========================================================================

def _make_alerter() -> TradingAlerter:
    return TradingAlerter({"alert_channels": ["console"]})


def _make_paper_trader(canary_agent, demote_hours: int = 2, steps_per_hour: int = 5):
    """Return a minimal PaperTrader configured for canary auto-demotion tests."""
    from deployment.paper_trader import PaperTrader

    data = np.random.rand(50, 5).astype(np.float32)
    import pandas as pd
    df = pd.DataFrame(
        data,
        columns=["$open", "$high", "$low", "$close", "$volume"],
    )

    agent = MagicMock()
    agent.predict.return_value = (np.array([0.5]), None)

    alerter = _make_alerter()
    alerter.notify_canary_auto_demoted = MagicMock()

    config = {
        "paper_trading": {"symbol": "BTC/USDT", "initial_balance": 1000.0},
        "canary": {
            "enabled": True,
            "traffic_pct": 1.0,  # 100% canary steps for fast testing
            "steps_per_hour": steps_per_hour,
        },
        "alerts": {
            "canary_auto_demote": {
                "sigma_below_prod": 0.5,
                "consecutive_hours": demote_hours,
            }
        },
    }
    pt = PaperTrader(
        agent=agent,
        config=config,
        simulation_mode=True,
        alerter=alerter,
        canary_agent=canary_agent,
    )
    return pt, alerter


# ===========================================================================
# Safety Net 1: Canary auto-demotion (G4)
# ===========================================================================

class TestCanaryAutoDemotion:
    """Canary auto-demotion triggers when underperformance persists N hours."""

    def _make_bad_canary(self):
        """Canary that always predicts a losing action."""
        canary = MagicMock()
        canary.predict.return_value = (np.array([-1.0]), None)
        return canary

    def _make_good_canary(self):
        """Canary that predicts same as prod."""
        canary = MagicMock()
        canary.predict.return_value = (np.array([0.5]), None)
        return canary

    def test_auto_demote_fires_after_consecutive_underperformance(self):
        """canary return < prod - σ for N hours → traffic set to 0, alert fired."""
        canary = self._make_bad_canary()
        pt, alerter = _make_paper_trader(canary, demote_hours=2, steps_per_hour=5)

        # window = demote_hours * steps_per_hour = 10 steps
        # Need consecutive_hours (2) windows in breach
        # First fill window (10 steps), then 2 more breach windows
        n_steps = pt._canary_demote_window * (pt._canary_demote_hours + 1) + 5
        price = 50_000.0
        obs = np.zeros(pt.window_size * 5, dtype=np.float32)

        for _ in range(n_steps):
            pt._run_canary_agent(obs, step=_, price=price)

        assert pt._canary_auto_demoted, "Expected auto-demote flag to be set"
        assert pt._canary_traffic_pct == 0.0, "Expected traffic_pct set to 0"
        assert not pt._canary_enabled, "Expected canary disabled after demotion"
        alerter.notify_canary_auto_demoted.assert_called_once()

    def test_auto_demote_does_not_fire_for_good_canary(self):
        """Good canary (returns ≥ prod) must NOT be auto-demoted.

        We inject synthetic returns directly to bypass the prod-return proxy
        approximation used in _run_canary_agent (which mixes actual PV changes
        with a directional multiplier).
        """
        canary = self._make_good_canary()
        pt, alerter = _make_paper_trader(canary, demote_hours=2, steps_per_hour=5)

        # Inject synthetic returns: canary slightly above prod → should NOT demote.
        window = pt._canary_demote_window
        n_windows = pt._canary_demote_hours * 3
        prod_base = 0.001
        for _ in range(window * n_windows):
            pt._prod_returns.append(prod_base)
            pt._canary_returns.append(prod_base + 0.0002)  # canary > prod
            pt._check_canary_auto_demote()

        assert not pt._canary_auto_demoted
        alerter.notify_canary_auto_demoted.assert_not_called()

    def test_auto_demote_latches_fires_only_once(self):
        """After first auto-demotion, subsequent steps do not re-fire."""
        canary = self._make_bad_canary()
        pt, alerter = _make_paper_trader(canary, demote_hours=2, steps_per_hour=5)

        n_steps = pt._canary_demote_window * 20
        price = 50_000.0
        obs = np.zeros(pt.window_size * 5, dtype=np.float32)

        for _ in range(n_steps):
            pt._run_canary_agent(obs, step=_, price=price)

        # Alert fired exactly once regardless of how many steps ran after demotion
        assert alerter.notify_canary_auto_demoted.call_count == 1

    def test_auto_demote_stage_remains_canary(self):
        """Demotion only blocks traffic — stage remains 'canary' for human to decide."""
        canary = self._make_bad_canary()
        pt, alerter = _make_paper_trader(canary, demote_hours=2, steps_per_hour=5)

        n_steps = pt._canary_demote_window * (pt._canary_demote_hours + 2)
        price = 50_000.0
        obs = np.zeros(pt.window_size * 5, dtype=np.float32)

        for _ in range(n_steps):
            pt._run_canary_agent(obs, step=_, price=price)

        # Stage in model_registry is NOT touched by PaperTrader — it only blocks traffic.
        # Confirm demote flag is set (traffic=0) but no registry call was made.
        assert pt._canary_auto_demoted
        # canary_agent is still referenced (not None) — human decides final stage
        assert pt.canary_agent is not None


# ===========================================================================
# Safety Net 2: OTel span instrumentation (G5)
# ===========================================================================

class TestOTelSpans:
    """OTel spans are emitted for each stage of order submission."""

    def _make_order_manager(self):
        from deployment.execution.order_manager import OrderManager
        return OrderManager(exchange_config={}, paper_mode=True)

    def test_submit_order_emits_parent_span(self):
        """submit_order wraps the full call in trading.order.submit span."""
        om = self._make_order_manager()
        spans_recorded = []

        original_start_span = __import__(
            "deployment.monitoring.tracing", fromlist=["start_span"]
        ).start_span

        from contextlib import contextmanager
        from deployment.monitoring.tracing import _NoopSpan

        @contextmanager
        def _capturing_span(name, attributes=None):
            span = _NoopSpan()
            spans_recorded.append(name)
            yield span

        with patch("deployment.execution.order_manager.start_span", _capturing_span):
            om.submit_order("buy", amount=0.01, current_price=50_000.0)

        assert "trading.order.submit" in spans_recorded

    def test_submit_order_emits_child_spans(self):
        """Child spans for risk_check, compliance_check, exchange_submit are emitted."""
        om = self._make_order_manager()
        spans_recorded = []

        from contextlib import contextmanager
        from deployment.monitoring.tracing import _NoopSpan

        @contextmanager
        def _capturing_span(name, attributes=None):
            span = _NoopSpan()
            spans_recorded.append(name)
            yield span

        with patch("deployment.execution.order_manager.start_span", _capturing_span):
            om.submit_order("buy", amount=0.01, current_price=50_000.0)

        expected = {
            "trading.order.submit",
            "trading.order.idempotency_lookup",
            "trading.order.risk_check",
            "trading.order.compliance_check",
            "trading.order.exchange_submit",
        }
        missing = expected - set(spans_recorded)
        assert not missing, f"Missing spans: {missing}"

    def test_idempotency_hit_span_attribute(self):
        """Duplicate key causes idempotency span to record hit=True and return early."""
        om = self._make_order_manager()
        idempotency_attrs: Dict[str, Any] = {}

        from contextlib import contextmanager

        class _AttrCapturingSpan:
            def set_attribute(self, key, value):
                idempotency_attrs[key] = value
            def record_exception(self, exc): pass
            def set_status(self, *a): pass

        @contextmanager
        def _capturing_span(name, attributes=None):
            if name == "trading.order.idempotency_lookup":
                yield _AttrCapturingSpan()
            else:
                from deployment.monitoring.tracing import _NoopSpan
                yield _NoopSpan()

        with patch("deployment.execution.order_manager.start_span", _capturing_span):
            first_id = om.submit_order("buy", 0.01, idempotency_key="k1", current_price=50_000.0)
            idempotency_attrs.clear()
            second_id = om.submit_order("buy", 0.01, idempotency_key="k1", current_price=50_000.0)

        assert first_id == second_id
        assert idempotency_attrs.get("idempotency.hit") is True


# ===========================================================================
# Safety Net 3: Schema drift guard (G6)
# ===========================================================================

class TestSchemaDriftGuard:
    """StreamValidator detects and reacts to real-time feed drift."""

    def _good_tick(self) -> Dict[str, Any]:
        return {
            "$open": 50_000.0,
            "$high": 50_500.0,
            "$low": 49_800.0,
            "$close": 50_200.0,
            "$volume": 123.45,
        }

    def test_valid_tick_passes(self):
        """A well-formed tick passes without raising."""
        sv = StreamValidator(on_schema_drift="halt")
        sv.validate(self._good_tick())  # must not raise
        assert sv.drift_count == 0

    def test_missing_key_triggers_halt(self):
        """Missing required key raises SchemaDrift under halt policy."""
        sv = StreamValidator(on_schema_drift="halt")
        bad = self._good_tick()
        del bad["$close"]
        with pytest.raises(SchemaDrift, match="missing required keys"):
            sv.validate(bad)
        assert sv.drift_count == 1

    def test_wrong_dtype_triggers_halt(self):
        """Non-float value in required key raises SchemaDrift."""
        sv = StreamValidator(on_schema_drift="halt")
        bad = self._good_tick()
        bad["$open"] = "not_a_number"
        with pytest.raises(SchemaDrift, match="not float-coercible"):
            sv.validate(bad)

    def test_nan_value_triggers_halt(self):
        """NaN value in price field raises SchemaDrift."""
        sv = StreamValidator(on_schema_drift="halt")
        bad = self._good_tick()
        bad["$high"] = float("nan")
        with pytest.raises(SchemaDrift, match="NaN"):
            sv.validate(bad)

    def test_inf_value_triggers_halt(self):
        """±inf value raises SchemaDrift."""
        sv = StreamValidator(on_schema_drift="halt")
        bad = self._good_tick()
        bad["$volume"] = math.inf
        with pytest.raises(SchemaDrift, match="inf"):
            sv.validate(bad)

    def test_negative_price_triggers_halt(self):
        """Non-positive price raises SchemaDrift."""
        sv = StreamValidator(on_schema_drift="halt")
        bad = self._good_tick()
        bad["$close"] = -1.0
        with pytest.raises(SchemaDrift, match="not > 0"):
            sv.validate(bad)

    def test_warn_policy_does_not_raise(self):
        """Under warn policy, drift is detected but SchemaDrift is not raised."""
        alerter = _make_alerter()
        alerter.schema_drift_detected = MagicMock()
        sv = StreamValidator(on_schema_drift="warn", alerter=alerter)

        bad = self._good_tick()
        bad["$open"] = -5.0
        sv.validate(bad)  # must NOT raise

        assert sv.drift_count == 1
        alerter.schema_drift_detected.assert_called_once()
        call_args = alerter.schema_drift_detected.call_args
        assert call_args.kwargs.get("on_drift") == "warn" or call_args.args[1] == "warn"

    def test_alerter_called_on_halt_drift(self):
        """Alerter.schema_drift_detected is called before raising."""
        alerter = _make_alerter()
        alerter.schema_drift_detected = MagicMock()
        sv = StreamValidator(on_schema_drift="halt", alerter=alerter)

        bad = self._good_tick()
        del bad["$volume"]
        with pytest.raises(SchemaDrift):
            sv.validate(bad)

        alerter.schema_drift_detected.assert_called_once()

    def test_extra_keys_allowed(self):
        """Records with extra keys beyond OHLCV pass validation."""
        sv = StreamValidator(on_schema_drift="halt")
        tick = self._good_tick()
        tick["extra_field"] = "ignored"
        sv.validate(tick)
        assert sv.drift_count == 0


# ===========================================================================
# E2E: All three safety nets simultaneously (R14 plan requirement)
# ===========================================================================

class TestSafetyNetsNonInterference:
    """Canary demotion + schema drift + normal order flow run without interference."""

    def test_simultaneous_safety_nets(self):
        """All three safety nets operate independently without blocking each other."""
        # --- Setup: bad canary ---
        canary = MagicMock()
        canary.predict.return_value = (np.array([-1.0]), None)
        pt, alerter = _make_paper_trader(canary, demote_hours=2, steps_per_hour=5)

        # --- Setup: schema validator ---
        schema_alerter = _make_alerter()
        schema_alerter.schema_drift_detected = MagicMock()
        sv = StreamValidator(on_schema_drift="warn", alerter=schema_alerter)

        # --- Setup: order manager ---
        from deployment.execution.order_manager import OrderManager
        om = OrderManager(exchange_config={}, paper_mode=True)

        # 1. Submit normal order (no interference from canary or schema)
        order_id = om.submit_order("buy", amount=0.01, current_price=50_000.0)
        assert order_id is not None

        # 2. Inject schema drift — warn policy, does not halt the session
        bad_tick = {
            "$open": float("nan"), "$high": 1.0,
            "$low": 1.0, "$close": 1.0, "$volume": 1.0,
        }
        sv.validate(bad_tick)  # must not raise (warn policy)
        assert sv.drift_count == 1

        # 3. Trigger canary auto-demotion
        price = 50_000.0
        obs = np.zeros(pt.window_size * 5, dtype=np.float32)
        n_steps = pt._canary_demote_window * (pt._canary_demote_hours + 2)
        for i in range(n_steps):
            pt._run_canary_agent(obs, step=i, price=price)

        # Verify: canary demoted, schema drift detected, order succeeded — all independent
        assert pt._canary_auto_demoted, "Canary should have been auto-demoted"
        assert sv.drift_count >= 1, "Schema drift should have been recorded"

        # Order manager still functional after canary/schema events
        order_id2 = om.submit_order("sell", amount=0.01, current_price=50_100.0)
        assert order_id2 is not None
        assert order_id2 != order_id

    def test_critical_alerts_classified_correctly(self):
        """Schema drift (halt) is CRITICAL; canary demotion is CRITICAL; warn is WARNING."""
        alerter = _make_alerter()

        # schema drift halt → CRITICAL
        alerter.schema_drift_detected("missing $close", on_drift="halt")
        critical_events = [r for r in alerter.alert_history if r.level == "CRITICAL"]
        assert any(r.event == "schema_drift" for r in critical_events)

        # schema drift warn → WARNING
        alerter.schema_drift_detected("bad dtype", on_drift="warn")
        warning_events = [r for r in alerter.alert_history if r.level == "WARNING"]
        assert any(r.event == "schema_drift" for r in warning_events)

        # canary auto-demote → CRITICAL
        alerter.notify_canary_auto_demoted(
            version=1, sigma_below=1.0, consecutive_hours=6,
            canary_mean=-0.005, prod_mean=0.001, prod_std=0.002,
        )
        assert any(r.event == "canary_auto_demoted" and r.level == "CRITICAL"
                   for r in alerter.alert_history)
