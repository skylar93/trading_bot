"""
Time-travel replay of an audit log for post-incident root-cause analysis.

Walks the .jsonl audit log, applies every ``fill`` record to a fresh
PositionTracker, and optionally compares the final state against an
expected snapshot.  Useful after an incident: given the log, reproduce
exactly what happened to the position and spot where it diverged.

Usage
-----
    python scripts/audit_replay.py <audit.jsonl>
    python scripts/audit_replay.py <audit.jsonl> --from-ts 2026-05-01T10:00:00
    python scripts/audit_replay.py <audit.jsonl> --from-hash abc123def
    python scripts/audit_replay.py <audit.jsonl> --from-line 42
    python scripts/audit_replay.py <audit.jsonl> --expected snapshot.json
    python scripts/audit_replay.py <audit.jsonl> --initial-cash 50000 --no-verify

Exit codes
----------
    0  replay complete, no drift
    1  drift detected, hash-chain break, or unreadable file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running from the repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deployment.execution.position_tracker import PositionTracker

_GENESIS_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Hash-chain helpers (mirrors verify_audit_log.py logic)
# ---------------------------------------------------------------------------

def _sha256(prev_hash: str, payload: Dict[str, Any]) -> str:
    raw = prev_hash + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_records(log_path: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Load all records.  Returns (records, ok) where ok=False on parse error."""
    records: List[Dict[str, Any]] = []
    try:
        with open(log_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"ERROR: line {lineno}: {exc}", file=sys.stderr)
                    return records, False
    except OSError as exc:
        print(f"ERROR: cannot open {log_path}: {exc}", file=sys.stderr)
        return [], False
    return records, True


def _verify_chain(records: List[Dict[str, Any]]) -> bool:
    """Verify hash chain integrity. Returns True if intact."""
    prev = _GENESIS_HASH
    for i, rec in enumerate(records, start=1):
        if "payload" not in rec or "hash" not in rec:
            print(f"ERROR: record {i}: missing payload or hash", file=sys.stderr)
            return False
        expected = _sha256(prev, rec["payload"])
        if rec["hash"] != expected:
            print(
                f"ERROR: record {i} hash mismatch\n"
                f"  type    : {rec.get('type', '?')}\n"
                f"  ts      : {rec.get('ts', '?')}\n"
                f"  stored  : {rec['hash']}\n"
                f"  expected: {expected}",
                file=sys.stderr,
            )
            return False
        prev = rec["hash"]
    return True


# ---------------------------------------------------------------------------
# Start-point selection
# ---------------------------------------------------------------------------

def _find_start(
    records: List[Dict[str, Any]],
    from_line: Optional[int],
    from_ts: Optional[str],
    from_hash: Optional[str],
) -> int:
    """Return the index into records where replay should begin (0-based)."""
    if from_line is not None:
        idx = from_line - 1  # user-facing lines are 1-based
        if idx < 0 or idx >= len(records):
            print(
                f"ERROR: --from-line {from_line} is out of range "
                f"(log has {len(records)} records)",
                file=sys.stderr,
            )
            sys.exit(1)
        return idx
    if from_hash is not None:
        for i, rec in enumerate(records):
            if rec.get("hash", "").startswith(from_hash):
                return i
        print(f"ERROR: --from-hash prefix '{from_hash}' not found", file=sys.stderr)
        sys.exit(1)
    if from_ts is not None:
        for i, rec in enumerate(records):
            if rec.get("ts", "") >= from_ts:
                return i
        print(f"ERROR: --from-ts '{from_ts}' is after the last record", file=sys.stderr)
        sys.exit(1)
    return 0


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

def _apply_fill(tracker: PositionTracker, payload: Dict[str, Any]) -> None:
    side = payload.get("side", "")
    qty = float(payload.get("qty", 0.0))
    price = float(payload.get("price", 0.0))
    fee = float(payload.get("fee", 0.0))
    if qty <= 0 or price <= 0:
        return
    tracker.update_price(price)
    if side == "buy":
        tracker.apply_buy(quantity=qty, price=price, fee=fee)
    elif side == "sell":
        tracker.apply_sell(quantity=qty, price=price, fee=fee)


def replay(
    records: List[Dict[str, Any]],
    start_idx: int,
    initial_cash: float,
) -> Dict[str, Any]:
    """Replay fill events from start_idx onward. Returns final tracker snapshot."""
    tracker = PositionTracker(initial_cash=initial_cash)
    fills_applied = 0
    for rec in records[start_idx:]:
        if rec.get("type") == "fill":
            _apply_fill(tracker, rec.get("payload", {}))
            fills_applied += 1
    print(
        f"Replayed {fills_applied} fill(s) from record "
        f"{start_idx + 1}/{len(records)}.",
        file=sys.stderr,
    )
    return tracker.snapshot()


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

_DRIFT_TOLERANCE = 1e-8


def _check_drift(
    actual: Dict[str, Any], expected: Dict[str, Any]
) -> bool:
    """Return True if there is a detectable drift between the two snapshots."""
    drifted = False
    keys = sorted(set(actual) | set(expected))
    for k in keys:
        a = actual.get(k)
        e = expected.get(k)
        if a is None or e is None:
            print(f"DRIFT  {k}: {e!r} → {a!r}", file=sys.stdout)
            drifted = True
            continue
        try:
            if abs(float(a) - float(e)) > _DRIFT_TOLERANCE:
                print(f"DRIFT  {k}: expected={e}  actual={a}  delta={float(a)-float(e):+.10f}")
                drifted = True
        except (TypeError, ValueError):
            if a != e:
                print(f"DRIFT  {k}: {e!r} → {a!r}")
                drifted = True
    return drifted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay an audit log and detect position state drift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("log_path", help="Path to the .jsonl audit log")
    p.add_argument(
        "--from-line", type=int, metavar="N",
        help="Start replay from line N (1-based)",
    )
    p.add_argument(
        "--from-ts", metavar="ISO_TS",
        help="Start replay from the first record at or after this timestamp",
    )
    p.add_argument(
        "--from-hash", metavar="HEX_PREFIX",
        help="Start replay from the record whose hash starts with this prefix",
    )
    p.add_argument(
        "--initial-cash", type=float, default=10_000.0, metavar="FLOAT",
        help="Starting cash for the replay PositionTracker (default: 10000)",
    )
    p.add_argument(
        "--expected", metavar="JSON_FILE",
        help="JSON file containing expected tracker snapshot; "
             "exits 1 if drift is detected",
    )
    p.add_argument(
        "--no-verify", action="store_true",
        help="Skip hash-chain integrity check (faster on large logs)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    records, ok = _load_records(args.log_path)
    if not ok:
        sys.exit(1)
    if not records:
        print("WARNING: log is empty — nothing to replay.", file=sys.stderr)
        print(json.dumps({}))
        sys.exit(0)

    if not args.no_verify:
        if not _verify_chain(records):
            sys.exit(1)
        print(f"Chain OK ({len(records)} records).", file=sys.stderr)

    start_idx = _find_start(records, args.from_line, args.from_ts, args.from_hash)
    final_state = replay(records, start_idx, args.initial_cash)

    if args.expected:
        try:
            with open(args.expected, encoding="utf-8") as fh:
                expected = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: cannot read --expected file: {exc}", file=sys.stderr)
            sys.exit(1)
        drifted = _check_drift(final_state, expected)
        if drifted:
            print("\nFinal replayed state:", file=sys.stderr)
            print(json.dumps(final_state, indent=2), file=sys.stderr)
            sys.exit(1)
        else:
            print("OK: no drift detected.", file=sys.stdout)
    else:
        print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
