#!/usr/bin/env python
"""
G13: Kill Switch — cancel all orders and flatten position within 5 seconds.

Usage:
    python scripts/kill_switch.py                        # reads default PID file
    python scripts/kill_switch.py --pid-file state/paper_trader.pid
    python scripts/kill_switch.py --pid 12345            # direct PID
    python scripts/kill_switch.py --no-wait              # fire-and-forget

Exit codes:
    0 — kill signal sent and process confirmed halted within timeout
    1 — process already gone (no-op)
    2 — signal sent but process did not halt within timeout
    3 — error (could not read PID / permission denied)
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

DEFAULT_PID_FILE = "state/paper_trader.pid"
HALT_TIMEOUT_SEC = 5.0
POLL_INTERVAL_SEC = 0.1


def _read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except FileNotFoundError:
        return None
    except ValueError as exc:
        print(f"ERROR: malformed PID file {pid_file}: {exc}", file=sys.stderr)
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't send signals — treat as alive
        return True


def kill(pid: int, wait: bool = True, timeout: float = HALT_TIMEOUT_SEC) -> int:
    """Send SIGUSR1 to *pid* and optionally wait for it to exit.

    Returns 0 on clean halt, 1 if already gone, 2 on timeout, 3 on error.
    """
    if not _process_alive(pid):
        print(f"Process {pid} is not running — nothing to do.", file=sys.stderr)
        return 1

    print(f"Sending SIGUSR1 to PaperTrader (pid={pid}) …")
    try:
        os.kill(pid, signal.SIGUSR1)
    except PermissionError:
        print(f"ERROR: permission denied sending SIGUSR1 to pid {pid}", file=sys.stderr)
        return 3
    except ProcessLookupError:
        print(f"Process {pid} disappeared before signal was sent.", file=sys.stderr)
        return 1

    if not wait:
        print("--no-wait: signal sent. Not waiting for confirmation.")
        return 0

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            elapsed = timeout - (deadline - time.monotonic())
            print(f"Kill switch confirmed: process {pid} halted in {elapsed:.2f}s ✓")
            return 0
        time.sleep(POLL_INTERVAL_SEC)

    print(
        f"WARNING: process {pid} still alive after {timeout}s. "
        "Check logs/paper_trader.log and kill manually if needed.",
        file=sys.stderr,
    )
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergency kill switch for PaperTrader.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pid-file", default=DEFAULT_PID_FILE, help="Path to PID file")
    group.add_argument("--pid", type=int, help="Direct process PID (bypasses PID file)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=HALT_TIMEOUT_SEC,
        help=f"Seconds to wait for clean halt (default: {HALT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Fire-and-forget: don't wait for confirmation",
    )
    args = parser.parse_args()

    if args.pid is not None:
        pid = args.pid
    else:
        pid_file = Path(args.pid_file)
        pid = _read_pid(pid_file)
        if pid is None:
            print(
                f"ERROR: could not read PID from {pid_file}. "
                "Is PaperTrader running?",
                file=sys.stderr,
            )
            sys.exit(3)

    rc = kill(pid, wait=not args.no_wait, timeout=args.timeout)
    sys.exit(rc)


if __name__ == "__main__":
    main()
