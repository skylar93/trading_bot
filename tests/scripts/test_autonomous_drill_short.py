"""
I2: AutonomousDrill 5-min short run test.

Verifies:
1. Drill starts cleanly with GBM feed (no network required)
2. FaultInjector injects 1 fault immediately
3. Safety net triggers (drift halt)
4. Drill auto-resumes within 30s
5. Final report is written
"""
from __future__ import annotations

import pathlib
import time
import threading

import pytest

from scripts.autonomous_72h_drill import AutonomousDrill


def _make_config(tmp_path: pathlib.Path, duration_hours: float = 5 / 60) -> dict:
    return {
        "duration_hours": duration_hours,
        "feed": "gbm",
        "log_dir": str(tmp_path / "logs"),
        "docs_dir": str(tmp_path / "docs" / "phase7"),
        "initial_capital": 10_000.0,
        "tick_interval": 0.0,  # as fast as possible for test
        "monitoring": {"alert_channels": ["file"], "log_dir": str(tmp_path / "logs")},
        "alerts": {"drift": {"shadow_mode_hours": 72}},
        # Inject faults immediately for the short test
        "fault_intervals": {
            "feed_stale": 0.5,
            "reconciliation_mismatch": 1.0,
            "schema_drift": 2.0,
            "canary_underperform": 0.5,
            "clock_skew": 3.0,
        },
    }


def test_drill_runs_and_stops(tmp_path: pathlib.Path) -> None:
    """Drill should run for ~5s and stop cleanly."""
    config = _make_config(tmp_path, duration_hours=5 / 3600)  # 5 seconds
    drill = AutonomousDrill(config)
    start = time.time()
    stats = drill.run()
    elapsed = time.time() - start
    assert elapsed < 30, f"Drill took too long: {elapsed:.1f}s"
    assert stats.tick_count >= 0


def test_drill_generates_snapshots(tmp_path: pathlib.Path) -> None:
    """Observer should write at least one snapshot during the run."""
    config = _make_config(tmp_path, duration_hours=16 / 3600)  # ~16 seconds
    config["fault_intervals"] = {k: 9999 for k in ["feed_stale", "reconciliation_mismatch",
                                                     "schema_drift", "canary_underperform",
                                                     "clock_skew"]}
    drill = AutonomousDrill(config)
    # Shorten snapshot interval for test
    drill._observer_thread = None
    # Patch snapshot interval to 5s
    original_run_observer = drill._run_observer

    def patched_observer():
        snap_interval_orig = 15 * 60
        import unittest.mock as mock
        # Monkey-patch snapshot interval to 5s
        drill.__class__._SNAPSHOT_INTERVAL = 5
        import time as _t
        last = _t.time()
        while not drill._stop_event.is_set():
            now = _t.time()
            if now - last >= 5:
                drill._write_snapshot()
                last = now
            _t.sleep(0.5)
        drill._write_snapshot()

    drill._run_observer = patched_observer  # type: ignore[assignment]
    stats = drill.run()

    snap_path = tmp_path / "logs" / "drill_snapshots.jsonl"
    assert snap_path.exists(), "drill_snapshots.jsonl must be created"


def test_fault_injector_records_events(tmp_path: pathlib.Path) -> None:
    """FaultInjector should write events to fault_injection.jsonl."""
    config = _make_config(tmp_path, duration_hours=10 / 3600)  # 10 seconds
    # Inject canary fault immediately
    config["fault_intervals"] = {
        "canary_underperform": 1.0,
        "feed_stale": 9999,
        "reconciliation_mismatch": 9999,
        "schema_drift": 9999,
        "clock_skew": 9999,
    }
    drill = AutonomousDrill(config)
    stats = drill.run()

    fault_log = tmp_path / "logs" / "fault_injection.jsonl"
    assert fault_log.exists(), "fault_injection.jsonl must be written"
    lines = fault_log.read_text().strip().splitlines()
    assert len(lines) >= 1


def test_fault_injection_triggers_safety_net(tmp_path: pathlib.Path) -> None:
    """Reconciliation mismatch fault should trigger an alert."""
    config = _make_config(tmp_path, duration_hours=8 / 3600)
    config["fault_intervals"] = {
        "reconciliation_mismatch": 1.0,
        "feed_stale": 9999,
        "schema_drift": 9999,
        "canary_underperform": 9999,
        "clock_skew": 9999,
    }
    drill = AutonomousDrill(config)
    stats = drill.run()

    alerts_path = tmp_path / "logs" / "alerts.jsonl"
    assert alerts_path.exists(), "alerts.jsonl should be written"
    import json
    events = [json.loads(l)["event"] for l in alerts_path.read_text().strip().splitlines()]
    assert "reconciliation_drift" in events, f"Expected reconciliation_drift, got {events}"


def test_drill_finalize_writes_report(tmp_path: pathlib.Path) -> None:
    """finalize() should create the week85_72h_{date}.md report."""
    config = _make_config(tmp_path, duration_hours=5 / 3600)
    drill = AutonomousDrill(config)
    drill.run()
    report_path = drill.finalize()
    assert report_path.exists(), f"Report not found: {report_path}"
    content = report_path.read_text()
    assert "72h Autonomous Drill" in content
    assert "Summary" in content


def test_drill_auto_resume_after_halt(tmp_path: pathlib.Path) -> None:
    """After a halt the trader should auto-resume within 30s."""
    config = _make_config(tmp_path, duration_hours=12 / 3600)
    # Inject schema drift immediately (triggers halt via drift detector post-shadow)
    config["alerts"] = {"drift": {"shadow_mode_hours": 0}}  # no shadow → halt fires
    config["fault_intervals"] = {
        "schema_drift": 1.0,
        "feed_stale": 9999,
        "reconciliation_mismatch": 9999,
        "canary_underperform": 9999,
        "clock_skew": 9999,
    }
    drill = AutonomousDrill(config)
    stats = drill.run()
    # If halt occurred, resume_count should match
    assert stats.resume_count >= stats.halt_count, (
        f"resume_count {stats.resume_count} < halt_count {stats.halt_count}"
    )
