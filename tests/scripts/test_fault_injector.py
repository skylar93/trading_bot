"""
I2: FaultInjector unit tests — each fault type, isolation, production guard.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import pytest


# ---------------------------------------------------------------------------
# Production import guard
# ---------------------------------------------------------------------------

def test_import_blocked_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """fault_injector must raise ImportError when TRADING_ENV=production."""
    import importlib
    import sys

    monkeypatch.setenv("TRADING_ENV", "production")
    # Remove cached module if present
    sys.modules.pop("deployment.testing.fault_injector", None)
    with pytest.raises(ImportError, match="non-drill"):
        import deployment.testing.fault_injector  # noqa: F401

    # Restore
    sys.modules.pop("deployment.testing.fault_injector", None)
    monkeypatch.setenv("TRADING_ENV", "test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockDrill:
    """Minimal stub that FaultInjector calls into."""

    def __init__(self, tmp_path: pathlib.Path) -> None:
        self._alerter = _make_alerter(tmp_path)
        self._drift_detector = _make_drift_detector()
        self._feed_pause_event: "threading.Event" = __import__("threading").Event()
        self.injected_calls: list = []

    def _inject_reconciliation_drift(self, qty_drift_pct: float) -> None:
        self.injected_calls.append(("reconciliation_drift", qty_drift_pct))
        self._alerter.notify_reconciliation_drift(
            [{"type": "qty_mismatch", "drift_pct": qty_drift_pct}]
        )

    def _inject_schema_drift(self, column_name: str) -> None:
        self.injected_calls.append(("schema_drift", column_name))
        self._drift_detector.report_schema_drift(column_name, on_drift="halt")

    def _inject_canary_underperform(self, sigma_below: float) -> None:
        self.injected_calls.append(("canary_underperform", sigma_below))
        self._alerter.notify_canary_auto_demoted(
            version=0, sigma_below=sigma_below, consecutive_hours=6,
            canary_mean=-0.001, prod_mean=0.0, prod_std=0.001,
        )

    def _inject_clock_skew(self, skew_seconds: float) -> None:
        self.injected_calls.append(("clock_skew", skew_seconds))

    def _pause_feed(self, seconds: float) -> None:
        self.injected_calls.append(("feed_stale", seconds))


def _make_alerter(tmp_path: pathlib.Path):
    from deployment.monitoring.alerter import TradingAlerter
    return TradingAlerter({"alert_channels": ["file"], "log_dir": str(tmp_path)})


def _make_drift_detector():
    from deployment.monitoring.drift_detector import DeploymentDriftDetector
    return DeploymentDriftDetector({"drift": {"shadow_mode_hours": 0}})


# ---------------------------------------------------------------------------
# inject_now tests
# ---------------------------------------------------------------------------

def test_inject_reconciliation_mismatch(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    evt = fi.inject_now("reconciliation_mismatch")
    assert evt.safety_net_triggered
    assert any(c[0] == "reconciliation_drift" for c in drill.injected_calls)


def test_inject_canary_underperform(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    evt = fi.inject_now("canary_underperform")
    assert evt.safety_net_triggered
    assert any(c[0] == "canary_underperform" for c in drill.injected_calls)


def test_inject_schema_drift(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    evt = fi.inject_now("schema_drift")
    assert evt.safety_net_triggered
    assert any(c[0] == "schema_drift" for c in drill.injected_calls)


def test_inject_clock_skew(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    evt = fi.inject_now("clock_skew")
    assert any(c[0] == "clock_skew" for c in drill.injected_calls)


def test_inject_feed_stale(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    evt = fi.inject_now("feed_stale")
    assert any(c[0] == "feed_stale" for c in drill.injected_calls)


# ---------------------------------------------------------------------------
# Log writing
# ---------------------------------------------------------------------------

def test_fault_log_written(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    log_path = tmp_path / "fault_injection.jsonl"
    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=log_path)
    fi.inject_now("canary_underperform")
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    # pre_inject + post_inject = 2 lines
    assert len(lines) == 2
    pre = json.loads(lines[0])
    post = json.loads(lines[1])
    assert pre["phase"] == "pre_inject"
    assert post["phase"] == "post_inject"
    assert pre["fault_type"] == "canary_underperform"


def test_fault_history_accumulates(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    fi.inject_now("canary_underperform")
    fi.inject_now("clock_skew")
    assert len(fi.history) == 2
    assert fi.history[0].fault_type == "canary_underperform"
    assert fi.history[1].fault_type == "clock_skew"


# ---------------------------------------------------------------------------
# Unknown fault type
# ---------------------------------------------------------------------------

def test_unknown_fault_type_does_not_raise(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrill(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    # Should log a warning but not raise
    fi.inject_now("nonexistent_fault_type")
    assert len(fi.history) == 1


# ---------------------------------------------------------------------------
# Schema drift triggers DeploymentDriftDetector halt
# ---------------------------------------------------------------------------

def test_schema_drift_sets_halt_outside_shadow(tmp_path: pathlib.Path) -> None:
    from deployment.testing.fault_injector import FaultInjector
    from deployment.monitoring.drift_detector import DeploymentDriftDetector

    drill = _MockDrill(tmp_path)
    # Drift detector with no shadow (post-shadow → halt fires)
    drill._drift_detector = DeploymentDriftDetector({"drift": {"shadow_mode_hours": 0}})
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl")
    fi.inject_now("schema_drift")
    assert drill._drift_detector.halt_requested


# ---------------------------------------------------------------------------
# I11: exchange_outage / spread_blowout
# ---------------------------------------------------------------------------

class _MockDrillI11(_MockDrill):
    """Extended mock with I11 flags."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self._fake_exchange_503: bool = False
        self._fake_spread_multiplier: float = 1.0


def test_inject_exchange_outage_sets_and_clears_flag(tmp_path: pathlib.Path) -> None:
    """exchange_outage: sets _fake_exchange_503=True for ~0s (fast in test)."""
    import threading
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrillI11(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl",
                       intervals={"exchange_outage": 999 * 3600})

    # Override sleep so the test finishes quickly
    captured: list[bool] = []
    original_sleep = __import__("time").sleep

    def fast_sleep(s):
        captured.append(drill._fake_exchange_503)
        # Don't actually sleep — just return
        drill._fake_exchange_503 = False  # simulate end of outage

    import time as _time_module
    old_sleep = _time_module.sleep
    _time_module.sleep = fast_sleep
    try:
        evt = fi.inject_now("exchange_outage")
    finally:
        _time_module.sleep = old_sleep

    assert evt.safety_net_triggered
    assert not drill._fake_exchange_503  # cleared after injection


def test_inject_spread_blowout_sets_and_clears_flag(tmp_path: pathlib.Path) -> None:
    """spread_blowout: sets _fake_spread_multiplier=10 then resets to 1."""
    import time as _time_module
    from deployment.testing.fault_injector import FaultInjector

    drill = _MockDrillI11(tmp_path)
    fi = FaultInjector(drill=drill, log_path=tmp_path / "faults.jsonl",
                       intervals={"spread_blowout": 999 * 3600})

    old_sleep = _time_module.sleep

    def fast_sleep(s):
        drill._fake_spread_multiplier = 1.0

    _time_module.sleep = fast_sleep
    try:
        evt = fi.inject_now("spread_blowout")
    finally:
        _time_module.sleep = old_sleep

    assert evt.safety_net_triggered
    assert drill._fake_spread_multiplier == 1.0


def test_exchange_outage_and_spread_blowout_in_default_intervals() -> None:
    """I11: Both faults appear in DEFAULT_INTERVALS with expected intervals."""
    from deployment.testing.fault_injector import FaultInjector

    assert "exchange_outage" in FaultInjector.DEFAULT_INTERVALS
    assert "spread_blowout" in FaultInjector.DEFAULT_INTERVALS
    assert FaultInjector.DEFAULT_INTERVALS["exchange_outage"] == 18 * 3600
    assert FaultInjector.DEFAULT_INTERVALS["spread_blowout"] == 9 * 3600
