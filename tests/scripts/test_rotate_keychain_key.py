"""
Tests for scripts/rotate_keychain_key.py — E7 key rotation.

Monkeypatches keychain helpers and verify_exchange_key_scope.run_probes.
Does NOT touch the real macOS keychain.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.rotate_keychain_key as rkk


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect state dir and audit log to tmp_path."""
    monkeypatch.setattr(rkk, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(rkk, "_KEY_METADATA_PATH", tmp_path / "state" / "key_metadata.json")
    monkeypatch.setattr(rkk, "_AUDIT_LOG_PATH", tmp_path / "audit_log" / "audit.jsonl")


@pytest.fixture()
def fake_keychain(monkeypatch: pytest.MonkeyPatch):
    """In-memory fake keychain — no macOS calls."""
    store: dict[str, str] = {}

    def _set(key_name, value):
        store[key_name] = value

    def _get(key_name) -> Optional[str]:
        return store.get(key_name)

    def _delete(key_name):
        store.pop(key_name, None)

    monkeypatch.setattr(rkk, "_keychain_set", _set)
    monkeypatch.setattr(rkk, "_keychain_get", _get)
    monkeypatch.setattr(rkk, "_keychain_delete", _delete)
    return store


def _make_probes(ok: bool):
    probes = [
        {"name": "read",       "ok": ok, "msg": "ok" if ok else "failed"},
        {"name": "trade",      "ok": ok, "msg": "ok" if ok else "failed"},
        {"name": "no_withdraw", "ok": ok, "msg": "ok" if ok else "failed"},
    ]
    return probes, ok


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — happy path: stage → probe success → swap → metadata updated
# ─────────────────────────────────────────────────────────────────────────────

def test_happy_path(fake_keychain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "scripts.rotate_keychain_key.run_probes" if False else "scripts.verify_exchange_key_scope.run_probes",
        lambda **_kw: _make_probes(True),
        raising=False,
    )
    # Patch at the import site inside rotate_keychain_key
    import unittest.mock as mock
    with mock.patch("scripts.verify_exchange_key_scope.run_probes", return_value=_make_probes(True)):
        rc = rkk.rotate("binance", "new_key_abc", "new_secret_xyz")

    assert rc == 0

    # Active slot updated
    assert fake_keychain.get("EXCHANGE_BINANCE_KEY") == "new_key_abc"
    assert fake_keychain.get("EXCHANGE_BINANCE_SECRET") == "new_secret_xyz"

    # Pending slot cleaned up
    assert "EXCHANGE_BINANCE_KEY_PENDING" not in fake_keychain
    assert "EXCHANGE_BINANCE_SECRET_PENDING" not in fake_keychain

    # Metadata written
    meta_path = tmp_path / "state" / "key_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["exchange"] == "binance"
    assert meta["key_id"] == rkk._key_id("new_key_abc")
    assert meta["last_rotated_at"] is not None
    assert meta["rotation_due_at"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — probe fails → staged key cleaned up, active key untouched
# ─────────────────────────────────────────────────────────────────────────────

def test_probe_fail_cleans_staged(fake_keychain, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate an existing active key
    fake_keychain["EXCHANGE_BINANCE_KEY"] = "old_key"
    fake_keychain["EXCHANGE_BINANCE_SECRET"] = "old_secret"

    import unittest.mock as mock
    with mock.patch("scripts.verify_exchange_key_scope.run_probes", return_value=_make_probes(False)):
        rc = rkk.rotate("binance", "bad_key", "bad_secret")

    assert rc == 1

    # Active key unchanged
    assert fake_keychain["EXCHANGE_BINANCE_KEY"] == "old_key"
    assert fake_keychain["EXCHANGE_BINANCE_SECRET"] == "old_secret"

    # Pending cleaned up
    assert "EXCHANGE_BINANCE_KEY_PENDING" not in fake_keychain
    assert "EXCHANGE_BINANCE_SECRET_PENDING" not in fake_keychain


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — --skip-scope-check → WARNING emitted, swap proceeds
# ─────────────────────────────────────────────────────────────────────────────

def test_skip_scope_check_warns_and_swaps(
    fake_keychain, capsys: pytest.CaptureFixture
) -> None:
    rc = rkk.rotate("binance", "new_key", "new_secret", skip_scope_check=True)

    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "skip-scope-check" in captured.err
    assert fake_keychain["EXCHANGE_BINANCE_KEY"] == "new_key"


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — metadata file missing → created on first rotation
# ─────────────────────────────────────────────────────────────────────────────

def test_metadata_created_on_first_rotation(
    fake_keychain, tmp_path: Path
) -> None:
    meta_path = tmp_path / "state" / "key_metadata.json"
    assert not meta_path.exists()

    import unittest.mock as mock
    with mock.patch("scripts.verify_exchange_key_scope.run_probes", return_value=_make_probes(True)):
        rc = rkk.rotate("binance", "brand_new_key", "brand_new_secret")

    assert rc == 0
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["key_id"] == rkk._key_id("brand_new_key")


# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — audit log entry written with event key_rotated
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_event_written(fake_keychain, tmp_path: Path) -> None:
    audit_path = tmp_path / "audit_log" / "audit.jsonl"

    import unittest.mock as mock
    with mock.patch("scripts.verify_exchange_key_scope.run_probes", return_value=_make_probes(True)):
        rc = rkk.rotate("binance", "key_audit", "secret_audit")

    assert rc == 0
    assert audit_path.exists()
    lines = [l.strip() for l in audit_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "key_rotated"
    assert event["payload"]["exchange"] == "binance"
    assert event["payload"]["key_id_new"] == rkk._key_id("key_audit")
