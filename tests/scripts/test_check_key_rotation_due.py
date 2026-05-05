"""
Tests for scripts/cron/check_key_rotation_due.py — E7 rotation alert.

Uses tmp_path for state/key_metadata.json via monkeypatch of STATE_DIR.
Does NOT touch the real alerter or keychain.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.cron.check_key_rotation_due as ckrd


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(ckrd, "_STATE_DIR", state_dir)
    monkeypatch.setattr(ckrd, "_KEY_METADATA_PATH", state_dir / "key_metadata.json")


@pytest.fixture()
def captured_alerts(monkeypatch: pytest.MonkeyPatch):
    """Capture send_alert calls without importing the real alerter."""
    alerts: list[dict] = []

    class _FakeAlerter:
        def send_alert(self, message: str, level: str = "WARNING"):
            alerts.append({"message": message, "level": level})

    monkeypatch.setattr(ckrd, "_load_alerter", lambda: _FakeAlerter(), raising=False)

    # Patch at the call site
    import unittest.mock as mock

    original_check = ckrd.check

    def patched_check():
        with mock.patch("deployment.monitoring.alerter.TradingAlerter", return_value=_FakeAlerter()):
            return original_check()

    monkeypatch.setattr(ckrd, "check", patched_check)
    return alerts


def _write_meta(tmp_path_state: Path, meta: dict) -> None:
    path = tmp_path_state / "key_metadata.json"
    with open(path, "w") as fh:
        json.dump(meta, fh)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: direct invocation of check() with alert patching
# ─────────────────────────────────────────────────────────────────────────────

def _run_check_with_fake_alerter(monkeypatch: pytest.MonkeyPatch):
    """Run check() with TradingAlerter replaced by a fake that captures calls."""
    captured: list[dict] = []

    class _FakeAlerter:
        def send_alert(self, message: str, level: str = "WARNING"):
            captured.append({"message": message, "level": level})

    import unittest.mock as mock
    with mock.patch("deployment.monitoring.alerter.TradingAlerter", return_value=_FakeAlerter()):
        rc = ckrd.check()

    return rc, captured


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — metadata missing → exit 0, no alert
# ─────────────────────────────────────────────────────────────────────────────

def test_metadata_missing_no_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, alerts = _run_check_with_fake_alerter(monkeypatch)
    assert rc == 0
    assert alerts == []


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — rotation_due_at in future → no alert
# ─────────────────────────────────────────────────────────────────────────────

def test_not_due_no_alert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = ckrd._STATE_DIR
    _write_meta(state_dir, {
        "exchange": "binance",
        "rotation_due_at": _iso(_now() + timedelta(days=30)),
    })
    rc, alerts = _run_check_with_fake_alerter(monkeypatch)
    assert rc == 0
    assert alerts == []


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — rotation_due_at past, last_alert_at < 24h ago → no alert (idempotent)
# ─────────────────────────────────────────────────────────────────────────────

def test_recent_alert_suppressed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = ckrd._STATE_DIR
    _write_meta(state_dir, {
        "exchange": "binance",
        "last_rotated_at": _iso(_now() - timedelta(days=95)),
        "rotation_due_at": _iso(_now() - timedelta(days=5)),
        "last_alert_at": _iso(_now() - timedelta(hours=2)),  # < 24h ago
    })
    rc, alerts = _run_check_with_fake_alerter(monkeypatch)
    assert rc == 0
    assert alerts == []


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — rotation_due_at past, no recent alert → WARNING sent + last_alert_at updated
# ─────────────────────────────────────────────────────────────────────────────

def test_warning_sent_and_last_alert_updated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = ckrd._STATE_DIR
    _write_meta(state_dir, {
        "exchange": "binance",
        "last_rotated_at": _iso(_now() - timedelta(days=95)),
        "rotation_due_at": _iso(_now() - timedelta(days=5)),
    })

    rc, alerts = _run_check_with_fake_alerter(monkeypatch)

    assert rc == 1
    assert len(alerts) == 1
    assert alerts[0]["level"] == "WARNING"
    assert "rotation due" in alerts[0]["message"].lower()

    # last_alert_at updated in metadata
    meta = json.loads((state_dir / "key_metadata.json").read_text())
    assert "last_alert_at" in meta


# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — rotation_due_at past + 14d → CRITICAL alert
# ─────────────────────────────────────────────────────────────────────────────

def test_critical_escalation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = ckrd._STATE_DIR
    _write_meta(state_dir, {
        "exchange": "binance",
        "last_rotated_at": _iso(_now() - timedelta(days=110)),
        "rotation_due_at": _iso(_now() - timedelta(days=20)),
    })

    rc, alerts = _run_check_with_fake_alerter(monkeypatch)

    assert rc == 1
    assert len(alerts) == 1
    assert alerts[0]["level"] == "CRITICAL"
