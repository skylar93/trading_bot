#!/usr/bin/env python3
"""Daily key-scope re-verification (I10-a).

Checks that the exchange API key does NOT have withdraw permission.
If withdraw is detected, fires a kill-switch alert and exits 1.

Run via launchd (see scripts/launchd/com.tradingbot.keyscope.plist).
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    from scripts.verify_exchange_key_scope import run_probes
    from deployment.monitoring.alerter import TradingAlerter

    alerter = TradingAlerter({})

    try:
        results = run_probes(sandbox=False)
    except Exception as exc:
        alerter.send_alert(f"Key-scope probe failed: {exc}", level="CRITICAL")
        return 1

    withdraw_detected = any(
        r.get("probe") == "no_withdraw" and not r.get("passed", True)
        for r in (results or [])
    )

    if withdraw_detected:
        alerter.notify_kill_switch(reason="withdraw_permission_detected")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
