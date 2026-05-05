#!/usr/bin/env python
"""
E7: API key rotation helper.

Stages a new key, validates scope, then swaps atomically into the active
keychain slot.  Never stores key material in files — keychain is source of truth.

Usage:
    python scripts/rotate_keychain_key.py \\
        --exchange binance \\
        --new-key-from-stdin \\
        [--skip-scope-check]

Exit codes:
    0  success — key rotated
    1  scope probe failed — active key unchanged
    2  unexpected error
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_STATE_DIR = PROJECT_ROOT / "state"
_KEY_METADATA_PATH = _STATE_DIR / "key_metadata.json"
_AUDIT_LOG_PATH = PROJECT_ROOT / "audit_log" / "audit.jsonl"
_KEY_TTL_DAYS = 90

# Keychain key naming convention (matches KeychainSecretProvider service "trading_bot")
# Active slots:   EXCHANGE_{EXCHANGE_UPPER}_KEY / EXCHANGE_{EXCHANGE_UPPER}_SECRET
# Pending slots:  EXCHANGE_{EXCHANGE_UPPER}_KEY_PENDING / EXCHANGE_{EXCHANGE_UPPER}_SECRET_PENDING


def _active_key_name(exchange: str) -> str:
    return f"EXCHANGE_{exchange.upper()}_KEY"


def _active_secret_name(exchange: str) -> str:
    return f"EXCHANGE_{exchange.upper()}_SECRET"


def _pending_key_name(exchange: str) -> str:
    return f"EXCHANGE_{exchange.upper()}_KEY_PENDING"


def _pending_secret_name(exchange: str) -> str:
    return f"EXCHANGE_{exchange.upper()}_SECRET_PENDING"


# ─────────────────────────────────────────────────────────────────────────────
# Keychain helpers (thin wrappers — monkeypatched in tests)
# ─────────────────────────────────────────────────────────────────────────────

def _keychain_set(key_name: str, value: str) -> None:
    from deployment.secrets.secret_provider import KeychainSecretProvider
    provider = KeychainSecretProvider()
    provider.set(key_name, value)


def _keychain_get(key_name: str) -> Optional[str]:
    from deployment.secrets.secret_provider import KeychainSecretProvider
    provider = KeychainSecretProvider()
    try:
        return provider.get(key_name)
    except KeyError:
        return None


def _keychain_delete(key_name: str) -> None:
    from deployment.secrets.secret_provider import KeychainSecretProvider
    provider = KeychainSecretProvider()
    provider.delete(key_name)


# ─────────────────────────────────────────────────────────────────────────────
# Key metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _load_metadata() -> dict:
    if not _KEY_METADATA_PATH.exists():
        return {}
    with open(_KEY_METADATA_PATH) as fh:
        return json.load(fh)


def _save_metadata(meta: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_KEY_METADATA_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)


def _update_metadata_after_rotation(exchange: str, old_key_id: Optional[str], new_api_key: str) -> None:
    now = datetime.now(timezone.utc)
    meta = _load_metadata()
    meta["exchange"] = exchange
    meta["key_id"] = _key_id(new_api_key)
    meta["created_at"] = meta.get("created_at") or now.isoformat()
    meta["last_rotated_at"] = now.isoformat()
    meta["last_verified_at"] = now.isoformat()
    meta["rotation_due_at"] = (now + timedelta(days=_KEY_TTL_DAYS)).isoformat()
    # Preserve last_alert_at if present (managed by check_key_rotation_due)
    _save_metadata(meta)


# ─────────────────────────────────────────────────────────────────────────────
# Audit log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_key_rotated_event(exchange: str, old_key_id: Optional[str], new_key_id: str) -> None:
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "key_rotated",
        "payload": {
            "exchange": exchange,
            "key_id_old": old_key_id,
            "key_id_new": new_key_id,
        },
    }
    with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main rotation logic
# ─────────────────────────────────────────────────────────────────────────────

def rotate(
    exchange: str,
    new_api_key: str,
    new_api_secret: str,
    skip_scope_check: bool = False,
) -> int:
    """Perform the key rotation.  Returns 0 on success, 1 on probe failure, 2 on error."""
    # Capture old key_id for the audit event (best-effort)
    old_meta = _load_metadata()
    old_key_id: Optional[str] = old_meta.get("key_id")

    # Stage pending key
    pending_key_name = _pending_key_name(exchange)
    pending_secret_name = _pending_secret_name(exchange)
    try:
        _keychain_set(pending_key_name, new_api_key)
        _keychain_set(pending_secret_name, new_api_secret)
    except Exception as exc:
        print(f"ERROR: failed to stage pending key: {exc}", file=sys.stderr)
        return 2

    if skip_scope_check:
        print(
            "WARNING: --skip-scope-check specified — skipping probe validation. "
            "Use only for emergency recovery.",
            file=sys.stderr,
        )
    else:
        # Run scope probes against staged key
        try:
            from scripts.verify_exchange_key_scope import run_probes
            probes, overall_ok = run_probes(
                exchange_id=exchange,
                api_key=new_api_key,
                api_secret=new_api_secret,
                sandbox=False,
                symbol="BTC/USDT",
            )
        except Exception as exc:
            print(f"ERROR: scope probe raised an exception: {exc}", file=sys.stderr)
            _cleanup_staged(exchange)
            return 1

        if not overall_ok:
            print(
                f"ERROR: scope probes FAILED for new key — active key unchanged.",
                file=sys.stderr,
            )
            for p in probes:
                print(f"  probe={p.get('name')} ok={p.get('ok')} msg={p.get('msg')}", file=sys.stderr)
            _cleanup_staged(exchange)
            return 1

    # Swap: write new key into active slot
    try:
        _keychain_set(_active_key_name(exchange), new_api_key)
        _keychain_set(_active_secret_name(exchange), new_api_secret)
    except Exception as exc:
        print(f"ERROR: failed to write new key into active slot: {exc}", file=sys.stderr)
        _cleanup_staged(exchange)
        return 2

    # Clean up staged entries
    _cleanup_staged(exchange)

    new_key_id = _key_id(new_api_key)
    _update_metadata_after_rotation(exchange, old_key_id, new_api_key)
    _append_key_rotated_event(exchange, old_key_id, new_key_id)

    print(
        f"Key rotated successfully. "
        f"exchange={exchange} key_id={new_key_id} "
        f"next_due_at={(datetime.now(timezone.utc) + timedelta(days=_KEY_TTL_DAYS)).date().isoformat()}",
        file=sys.stderr,
    )
    return 0


def _cleanup_staged(exchange: str) -> None:
    try:
        _keychain_delete(_pending_key_name(exchange))
    except Exception:
        pass
    try:
        _keychain_delete(_pending_secret_name(exchange))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate API key in keychain with scope validation."
    )
    parser.add_argument("--exchange", required=True, help="Exchange ID (e.g. binance)")
    parser.add_argument(
        "--new-key-from-stdin",
        action="store_true",
        required=True,
        help="Read new API key + secret from stdin (echo off for secret).",
    )
    parser.add_argument(
        "--skip-scope-check",
        action="store_true",
        default=False,
        help="Skip scope probe validation (emergency recovery only).",
    )
    args = parser.parse_args()

    if sys.stdin.isatty():
        new_api_key = input("New API key: ").strip()
        new_api_secret = getpass.getpass("New API secret: ").strip()
    else:
        lines = sys.stdin.read().splitlines()
        if len(lines) < 2:
            print("ERROR: stdin must contain two lines: <api_key>\\n<api_secret>", file=sys.stderr)
            sys.exit(2)
        new_api_key = lines[0].strip()
        new_api_secret = lines[1].strip()

    rc = rotate(
        exchange=args.exchange,
        new_api_key=new_api_key,
        new_api_secret=new_api_secret,
        skip_scope_check=args.skip_scope_check,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
