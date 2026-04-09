#!/usr/bin/env python
"""
Verify the hash-chain integrity of an audit log (.jsonl).

Usage:
    python scripts/verify_audit_log.py <path/to/audit.jsonl>

Exit codes:
    0  — chain is intact
    1  — chain is broken (error printed to stderr) or file unreadable
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Optional


_GENESIS_HASH = "0" * 64


def _sha256(prev_hash: str, payload: dict) -> str:
    raw = prev_hash + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def verify(log_path: str) -> bool:
    """
    Walk the jsonl file and verify the hash chain.

    Returns True if intact, False if any break is found.
    Prints a description of the first broken record to stderr.
    """
    prev_hash = _GENESIS_HASH
    try:
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"ERROR: cannot read {log_path}: {e}", file=sys.stderr)
        return False

    if not lines:
        print(f"WARNING: {log_path} is empty — nothing to verify.", file=sys.stderr)
        return True

    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(
                f"ERROR: line {lineno}: JSON parse error: {e}", file=sys.stderr
            )
            return False

        if "payload" not in record or "hash" not in record:
            print(
                f"ERROR: line {lineno}: missing 'payload' or 'hash' field.",
                file=sys.stderr,
            )
            return False

        expected = _sha256(prev_hash, record["payload"])
        if record["hash"] != expected:
            print(
                f"ERROR: line {lineno}: hash mismatch.\n"
                f"  type       : {record.get('type', '?')}\n"
                f"  ts         : {record.get('ts', '?')}\n"
                f"  stored     : {record['hash']}\n"
                f"  recomputed : {expected}",
                file=sys.stderr,
            )
            return False

        prev_hash = record["hash"]

    print(f"OK: {len(lines)} records verified. Chain intact.", file=sys.stdout)
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <audit_log.jsonl>", file=sys.stderr)
        sys.exit(1)

    ok = verify(sys.argv[1])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
