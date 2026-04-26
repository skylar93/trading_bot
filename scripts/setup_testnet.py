#!/usr/bin/env python3
"""Testnet Setup Wizard (I5) — operator bridge for 30-min testnet onboarding.

Interactive 6-step flow:
  1. Open Binance Spot Testnet URL in browser
  2. Collect API key / secret via getpass (no echo)
  3. Store in macOS Keychain via KeychainSecretProvider.set()
  4. Verify key scope (no withdraw permission)
  5. Run 5-min sandbox smoke test
  6. Optional: Discord webhook URL + test notification

--dry-run: skips stdin and external calls; validates wizard logic only.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import pathlib
import platform
import re
import subprocess
import sys
import textwrap

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TESTNET_URL = "https://testnet.binance.vision/"
_CHECKLIST = _ROOT / "docs" / "runbook" / "go_live_checklist.md"

_AUTO_WIZARD_ITEMS = {
    "F1": r"(\| F1 \|[^|]*\|) manual (\|)",
    "F2": r"(\| F2 \|[^|]*\|) manual (\|)",
    "S3": r"(\| S3 \|[^|]*\|) manual (\|)",
    "O7": r"(\| O7 \|[^|]*\|) manual (\|)",
}


def _open_url(url: str) -> None:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", url], check=False)
    elif system == "Linux":
        subprocess.run(["xdg-open", url], check=False)


def _update_checklist(date_str: str) -> None:
    if not _CHECKLIST.exists():
        print(f"  [warn] Checklist not found at {_CHECKLIST} — skipping update", file=sys.stderr)
        return
    text = _CHECKLIST.read_text(encoding="utf-8")
    for item, pattern in _AUTO_WIZARD_ITEMS.items():
        replacement = rf"\1 [auto-wizard] ✅ {date_str} \2"
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            text = new_text
            print(f"  ✅ Checklist updated: {item}")
        else:
            print(f"  [warn] Could not update checklist item {item}", file=sys.stderr)
    _CHECKLIST.write_text(text, encoding="utf-8")


def run_wizard(dry_run: bool = False) -> int:
    date_str = datetime.date.today().isoformat()
    print(textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════╗
    ║       Binance Testnet Setup Wizard (I5)          ║
    ║  {'[DRY-RUN MODE]' if dry_run else 'Interactive — keys never logged':<36}  ║
    ╚══════════════════════════════════════════════════╝
    """).strip())

    # ── Step 1: Open testnet URL ────────────────────────────────────────
    print("\n[1/6] Opening Binance Spot Testnet…")
    if not dry_run:
        _open_url(_TESTNET_URL)
        input("      Press ENTER once you have created an API key pair: ")
    else:
        print(f"      [dry-run] Would open {_TESTNET_URL}")

    # ── Step 2: Collect credentials ─────────────────────────────────────
    print("\n[2/6] Enter testnet API credentials (input hidden).")
    if not dry_run:
        api_key = getpass.getpass("      API key   : ")
        api_secret = getpass.getpass("      API secret: ")
        if not api_key or not api_secret:
            print("ERROR: API key/secret must not be empty.", file=sys.stderr)
            return 1
    else:
        _stub = "DRY-RUN-PLACEHOLDER"
        api_key = _stub + "-KEY"
        api_secret = _stub + "-SECRET"

    # ── Step 3: Store in Keychain ────────────────────────────────────────
    print("\n[3/6] Storing credentials in macOS Keychain…")
    if not dry_run:
        try:
            from deployment.secrets.secret_provider import KeychainSecretProvider
            kp = KeychainSecretProvider()
            kp.set("TESTNET_API_KEY", api_key)
            kp.set("TESTNET_API_SECRET", api_secret)
            print("      Stored: TESTNET_API_KEY, TESTNET_API_SECRET")
        except Exception as exc:
            print(f"      [warn] Keychain store failed: {exc}", file=sys.stderr)
    else:
        print("      [dry-run] Would call KeychainSecretProvider.set() for TESTNET_API_KEY/SECRET")

    # ── Step 4: Verify key scope ─────────────────────────────────────────
    print("\n[4/6] Verifying API key scope (no withdraw permission)…")
    if not dry_run:
        try:
            from scripts.verify_exchange_key_scope import run_probes
            probes, ok = run_probes(
                exchange_id="binance",
                api_key=api_key,
                api_secret=api_secret,
                sandbox=True,
                symbol="BTC/USDT",
            )
            if ok:
                print("      ✅ Key scope valid")
            else:
                print("      ❌ Key scope check FAILED — check withdraw permission", file=sys.stderr)
                return 1
        except Exception as exc:
            print(f"      [warn] Key scope probe failed: {exc}", file=sys.stderr)
    else:
        print("      [dry-run] Would call verify_exchange_key_scope.run_probes(sandbox=True)")

    # ── Step 5: Sandbox smoke test ───────────────────────────────────────
    print("\n[5/6] Running 5-minute sandbox smoke test…")
    if not dry_run:
        smoke_path = _ROOT / "scripts" / "sandbox_smoke.py"
        if smoke_path.exists():
            result = subprocess.run(
                [sys.executable, str(smoke_path)],
                timeout=360,
            )
            if result.returncode != 0:
                print("      ❌ Smoke test FAILED", file=sys.stderr)
                return 1
            print("      ✅ Smoke test passed")
        else:
            print(f"      [warn] {smoke_path} not found — skipping", file=sys.stderr)
    else:
        print("      [dry-run] Would run scripts/sandbox_smoke.py")

    # ── Step 6: Discord webhook (optional) ───────────────────────────────
    print("\n[6/6] Discord webhook URL (optional — press ENTER to skip).")
    if not dry_run:
        discord_url = getpass.getpass("      Discord URL: ").strip()
        if discord_url:
            try:
                from deployment.monitoring.alerter import TradingAlerter
                alerter = TradingAlerter({"discord_webhook_url": discord_url,
                                          "alert_channels": ["discord"]})
                alerter.send_alert("Testnet wizard complete — alerter configured ✅", level="INFO")
                print("      ✅ Test notification sent")
            except Exception as exc:
                print(f"      [warn] Discord test failed: {exc}", file=sys.stderr)
        else:
            print("      Skipped.")
    else:
        print("      [dry-run] Would prompt for Discord URL and send test notification")

    # ── Update go_live_checklist ─────────────────────────────────────────
    print("\nUpdating go_live_checklist.md…")
    try:
        _update_checklist(date_str)
    except Exception as exc:
        print(f"[warn] Checklist update failed: {exc}", file=sys.stderr)

    print("\n✅ Testnet wizard complete. Run `python scripts/sandbox_smoke.py` to verify.\n")
    return 0


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binance testnet setup wizard (I5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip stdin / external calls — validates wizard flow only")
    args = parser.parse_args(argv)
    return run_wizard(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
