"""
Tests for scripts/audit_replay.py.

Two main properties:
  1. Identity — replaying the same log produces the same final state.
  2. Drift detection — a modified fill is flagged as a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure repo root is importable so audit_replay can be imported directly.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_replay import (
    _apply_fill,
    _check_drift,
    _find_start,
    _load_records,
    _verify_chain,
    replay,
)
from deployment.execution.position_tracker import PositionTracker


# ---------------------------------------------------------------------------
# Helpers for building synthetic audit logs
# ---------------------------------------------------------------------------

_GENESIS = "0" * 64


def _sha256(prev: str, payload: Dict[str, Any]) -> str:
    raw = prev + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _make_record(
    prev_hash: str,
    record_type: str,
    payload: Dict[str, Any],
    ts: str = "2026-05-04T10:00:00+00:00",
) -> Dict[str, Any]:
    h = _sha256(prev_hash, payload)
    return {"ts": ts, "type": record_type, "payload": payload, "hash": h}


def _build_log(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn a list of (type, payload, ts?) tuples into a valid chained log."""
    records = []
    prev = _GENESIS
    for ev in events:
        rec = _make_record(prev, ev["type"], ev["payload"], ev.get("ts", "2026-05-04T10:00:00+00:00"))
        records.append(rec)
        prev = rec["hash"]
    return records


def _write_log(tmp_path: Path, records: List[Dict[str, Any]]) -> Path:
    p = tmp_path / "audit.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUY1 = {"type": "fill", "payload": {"side": "buy",  "qty": 0.1, "price": 50000.0, "fee": 5.0}}
SELL1 = {"type": "fill", "payload": {"side": "sell", "qty": 0.05, "price": 52000.0, "fee": 2.6}}
BUY2 = {"type": "fill", "payload": {"side": "buy",  "qty": 0.05, "price": 51000.0, "fee": 2.55}}
MODEL = {"type": "model_decision", "payload": {"action": 1, "obs_hash": "abc"}}
RISK  = {"type": "risk_event", "payload": {"type": "drawdown_breach", "value": 0.12}}


# ---------------------------------------------------------------------------
# Unit tests: chain verification
# ---------------------------------------------------------------------------

def test_verify_chain_ok(tmp_path):
    records = _build_log([BUY1, SELL1, BUY2])
    assert _verify_chain(records) is True


def test_verify_chain_tampered_payload(tmp_path):
    records = _build_log([BUY1, SELL1])
    # Tamper with the second record's payload without updating its hash.
    records[1] = dict(records[1])
    records[1]["payload"] = dict(records[1]["payload"], qty=999.0)
    assert _verify_chain(records) is False


def test_verify_chain_empty():
    assert _verify_chain([]) is True


# ---------------------------------------------------------------------------
# Unit tests: fill application
# ---------------------------------------------------------------------------

def test_apply_fill_buy():
    tracker = PositionTracker(initial_cash=10_000.0)
    _apply_fill(tracker, {"side": "buy", "qty": 0.1, "price": 50_000.0, "fee": 5.0})
    assert abs(tracker.position - 0.1) < 1e-9
    assert abs(tracker.cash - (10_000.0 - 0.1 * 50_000.0 - 5.0)) < 1e-6


def test_apply_fill_sell():
    tracker = PositionTracker(initial_cash=10_000.0)
    _apply_fill(tracker, {"side": "buy",  "qty": 0.1, "price": 50_000.0, "fee": 5.0})
    _apply_fill(tracker, {"side": "sell", "qty": 0.05, "price": 52_000.0, "fee": 2.6})
    assert abs(tracker.position - 0.05) < 1e-9


def test_apply_fill_zero_qty_ignored():
    tracker = PositionTracker(initial_cash=1_000.0)
    _apply_fill(tracker, {"side": "buy", "qty": 0.0, "price": 50_000.0, "fee": 0.0})
    assert tracker.position == 0.0
    assert tracker.cash == 1_000.0


# ---------------------------------------------------------------------------
# Integration: replay produces deterministic output
# ---------------------------------------------------------------------------

def test_replay_identity(tmp_path):
    """Replaying the same log twice gives the same snapshot."""
    records = _build_log([BUY1, SELL1, BUY2])
    state1 = replay(records, 0, initial_cash=10_000.0)
    state2 = replay(records, 0, initial_cash=10_000.0)
    assert state1 == state2


def test_replay_ignores_non_fill_records(tmp_path):
    """model_decision and risk_event records must not affect position state."""
    records_with_noise = _build_log([BUY1, MODEL, RISK, SELL1])
    records_clean = _build_log([BUY1, SELL1])
    state_noise = replay(records_with_noise, 0, initial_cash=10_000.0)
    state_clean = replay(records_clean, 0, initial_cash=10_000.0)
    assert state_noise == state_clean


def test_replay_from_line(tmp_path):
    """--from-line should skip fills before the start point."""
    records = _build_log([BUY1, SELL1, BUY2])
    # Replay from line 2 — skip BUY1, replay SELL1 + BUY2
    partial = replay(records, start_idx=1, initial_cash=10_000.0)
    # Manual calculation: no buy before sell, so sell is a no-op (qty=0 clamped)
    full = replay(records, start_idx=0, initial_cash=10_000.0)
    assert partial != full  # position histories differ


def test_replay_empty_log():
    state = replay([], start_idx=0, initial_cash=5_000.0)
    # No fills → initial cash, zero position, zero prices
    assert state["position"] == 0.0
    assert state["cash"] == 5_000.0


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def test_no_drift_identical():
    a = {"cash": 100.0, "position": 0.5, "entry_price": 200.0,
         "current_price": 210.0, "peak_value": 105.0}
    assert _check_drift(a, a) is False


def test_drift_detected_numeric():
    actual   = {"cash": 100.0, "position": 0.5}
    expected = {"cash": 99.0,  "position": 0.5}
    assert _check_drift(actual, expected) is True


def test_drift_within_tolerance():
    actual   = {"cash": 100.000000001, "position": 0.5}
    expected = {"cash": 100.0,         "position": 0.5}
    # delta ~1e-9 < 1e-8 tolerance
    assert _check_drift(actual, expected) is False


def test_drift_missing_key():
    actual   = {"cash": 100.0}
    expected = {"cash": 100.0, "position": 0.0}
    assert _check_drift(actual, expected) is True


# ---------------------------------------------------------------------------
# Start-point selection
# ---------------------------------------------------------------------------

def test_find_start_default():
    records = _build_log([BUY1, SELL1])
    assert _find_start(records, None, None, None) == 0


def test_find_start_by_line():
    records = _build_log([BUY1, SELL1, BUY2])
    assert _find_start(records, from_line=2, from_ts=None, from_hash=None) == 1


def test_find_start_by_hash():
    records = _build_log([BUY1, SELL1])
    prefix = records[1]["hash"][:8]
    assert _find_start(records, None, None, from_hash=prefix) == 1


def test_find_start_by_ts():
    events = [
        {**BUY1,  "ts": "2026-05-04T09:00:00+00:00"},
        {**SELL1, "ts": "2026-05-04T11:00:00+00:00"},
        {**BUY2,  "ts": "2026-05-04T12:00:00+00:00"},
    ]
    records = _build_log(events)
    assert _find_start(records, None, from_ts="2026-05-04T10:00:00", from_hash=None) == 1


# ---------------------------------------------------------------------------
# CLI end-to-end tests (subprocess)
# ---------------------------------------------------------------------------

_SCRIPT = str(REPO_ROOT / "scripts" / "audit_replay.py")


def test_cli_basic_output(tmp_path):
    records = _build_log([BUY1, SELL1])
    log_path = _write_log(tmp_path, records)
    result = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    state = json.loads(result.stdout)
    assert "cash" in state
    assert "position" in state


def test_cli_expected_match(tmp_path):
    records = _build_log([BUY1])
    log_path = _write_log(tmp_path, records)
    # First run to capture expected state
    r1 = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path)],
        capture_output=True, text=True,
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(r1.stdout, encoding="utf-8")
    # Second run with --expected should report no drift
    r2 = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path), "--expected", str(expected_path)],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0
    assert "no drift" in r2.stdout.lower()


def test_cli_expected_drift(tmp_path):
    records = _build_log([BUY1])
    log_path = _write_log(tmp_path, records)
    # Expected state with wrong cash
    expected = {"cash": 0.0, "position": 0.0, "entry_price": 0.0,
                "current_price": 0.0, "peak_value": 0.0}
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path), "--expected", str(expected_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "DRIFT" in r.stdout


def test_cli_broken_chain_exits_1(tmp_path):
    records = _build_log([BUY1, SELL1])
    # Tamper second record
    records[1] = dict(records[1], payload=dict(records[1]["payload"], qty=999.0))
    log_path = _write_log(tmp_path, records)
    r = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1


def test_cli_no_verify_skips_chain_check(tmp_path):
    records = _build_log([BUY1, SELL1])
    # Tamper second record — but --no-verify should still let it through
    records[1] = dict(records[1], payload=dict(records[1]["payload"], qty=999.0))
    log_path = _write_log(tmp_path, records)
    r = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path), "--no-verify"],
        capture_output=True, text=True,
    )
    # Should exit 0 (drift check not requested) even with tampered hash
    assert r.returncode == 0


def test_cli_empty_log(tmp_path):
    log_path = tmp_path / "empty.jsonl"
    log_path.write_text("", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, _SCRIPT, str(log_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout) == {}
