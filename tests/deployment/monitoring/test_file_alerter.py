"""I3: TradingAlerter file channel — append + rotation tests."""
from __future__ import annotations

import json
import pathlib

import pytest

from deployment.monitoring.alerter import TradingAlerter


def _make_alerter(tmp_path: pathlib.Path) -> TradingAlerter:
    return TradingAlerter(
        {
            "alert_channels": ["file"],
            "log_dir": str(tmp_path),
        }
    )


def test_file_channel_appends_jsonl(tmp_path: pathlib.Path) -> None:
    alerter = _make_alerter(tmp_path)
    alerter.notify_kill_switch(reason="test")
    alerts_path = tmp_path / "alerts.jsonl"
    assert alerts_path.exists(), "alerts.jsonl must be created"
    lines = alerts_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "kill_switch_activated"
    assert record["level"] == "CRITICAL"
    assert record["context_redacted"] is True


def test_file_channel_multiple_appends(tmp_path: pathlib.Path) -> None:
    alerter = _make_alerter(tmp_path)
    alerter.notify_kill_switch(reason="a")
    alerter.notify_error(error="boom")
    alerter.notify_drift(detector="adwin", signal_name="return")
    alerts_path = tmp_path / "alerts.jsonl"
    lines = alerts_path.read_text().strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(l)["event"] for l in lines]
    assert events == ["kill_switch_activated", "runtime_error", "drift_detected"]


def test_file_channel_rotation(tmp_path: pathlib.Path) -> None:
    alerter = _make_alerter(tmp_path)
    alerts_path = tmp_path / "alerts.jsonl"
    # Pre-create a file exceeding 10 MB
    alerts_path.write_bytes(b"x" * (11 * 1024 * 1024))
    alerter.notify_error(error="trigger rotation")
    # Original file should be renamed
    rotated = list(tmp_path.glob("alerts.jsonl.*"))
    assert len(rotated) == 1, "rotated file should exist"
    # New alerts.jsonl should be a fresh, small file
    assert alerts_path.exists()
    assert alerts_path.stat().st_size < 1024 * 1024


def test_default_channels_include_file(tmp_path: pathlib.Path) -> None:
    alerter = TradingAlerter({"log_dir": str(tmp_path)})
    assert "file" in alerter._channels
    assert "desktop_notify" in alerter._channels
    assert "console" in alerter._channels


def test_file_channel_creates_log_dir(tmp_path: pathlib.Path) -> None:
    nested = tmp_path / "a" / "b" / "logs"
    alerter = TradingAlerter({"alert_channels": ["file"], "log_dir": str(nested)})
    alerter.send_alert("test")
    assert (nested / "alerts.jsonl").exists()
