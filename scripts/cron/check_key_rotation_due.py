#!/usr/bin/env python
"""
E7: Daily key-rotation-due alert.

Run via launchd (com.tradingbot.keyrotation.plist) at 03:00 UTC.

Exit codes:
  0  no alert needed
  1  alert was sent (WARNING or CRITICAL)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_STATE_DIR = Path(os.environ.get("TRADING_BOT_STATE_DIR", str(PROJECT_ROOT / "state")))
_KEY_METADATA_PATH = _STATE_DIR / "key_metadata.json"

_THRESHOLD_DAYS = 90
_ESCALATE_DAYS = 14       # days past due before escalating to CRITICAL
_COOLDOWN_HOURS = 24      # minimum hours between repeated alerts


def _load_metadata() -> dict:
    if not _KEY_METADATA_PATH.exists():
        return {}
    try:
        with open(_KEY_METADATA_PATH) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_metadata(meta: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_KEY_METADATA_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check() -> int:
    """Run the rotation-due check.  Returns exit code."""
    meta = _load_metadata()
    if not meta:
        return 0

    due_str = meta.get("rotation_due_at")
    if not due_str:
        return 0

    now = datetime.now(timezone.utc)
    try:
        due_at = _parse_dt(due_str)
    except ValueError:
        return 0

    if now <= due_at:
        return 0  # not due yet

    # Check idempotency: don't re-alert within cooldown window
    last_alert_str = meta.get("last_alert_at")
    if last_alert_str:
        try:
            last_alert = _parse_dt(last_alert_str)
            if (now - last_alert) < timedelta(hours=_COOLDOWN_HOURS):
                return 0
        except ValueError:
            pass

    last_rotated_str = meta.get("last_rotated_at") or meta.get("created_at")
    if last_rotated_str:
        try:
            days_since = (now - _parse_dt(last_rotated_str)).days
        except ValueError:
            days_since = _THRESHOLD_DAYS
    else:
        days_since = _THRESHOLD_DAYS

    overdue_days = (now - due_at).days
    if overdue_days >= _ESCALATE_DAYS:
        level = "CRITICAL"
    else:
        level = "WARNING"

    message = (
        f"API key rotation due — last rotated {days_since} days ago, "
        f"threshold {_THRESHOLD_DAYS} days. "
        f"Run: python scripts/rotate_keychain_key.py --exchange {meta.get('exchange', 'binance')} "
        f"--new-key-from-stdin"
    )

    try:
        from deployment.monitoring.alerter import TradingAlerter
        alerter = TradingAlerter()
        alerter.send_alert(message, level=level)
    except Exception as exc:
        print(f"WARNING: alerter raised {exc}; printing to stderr instead.", file=sys.stderr)
        print(f"[{level}] {message}", file=sys.stderr)

    # Update last_alert_at for idempotency
    meta["last_alert_at"] = now.isoformat()
    _save_metadata(meta)

    return 1


def main() -> None:
    sys.exit(check())


if __name__ == "__main__":
    main()
