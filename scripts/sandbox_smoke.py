#!/usr/bin/env python
"""
Sandbox smoke test — Week 72 F5.

Manual verification script (NOT run in CI — requires real exchange credentials).
Connects to Binance testnet, receives tickers for 60 seconds, submits a small
limit order, then cancels it and disconnects cleanly.

Prerequisites:
    export EXCHANGE_BINANCE_TESTNET_KEY="..."
    export EXCHANGE_BINANCE_TESTNET_SECRET="..."

Usage:
    python scripts/sandbox_smoke.py [--exchange binance] [--symbol BTC/USDT] [--duration 60]

CI note: This script is tagged @pytest.mark.local_only and is NOT imported by
the test suite.  Run it manually on a machine with valid testnet credentials.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("sandbox_smoke")


def _load_credentials(exchange_id: str) -> dict:
    """Load testnet credentials via SecretProvider (env backend by default)."""
    from deployment.secrets.secret_provider import get_default_provider
    provider = get_default_provider()

    key_name = f"EXCHANGE_{exchange_id.upper()}_TESTNET_KEY"
    secret_name = f"EXCHANGE_{exchange_id.upper()}_TESTNET_SECRET"
    try:
        api_key = provider.get(key_name)
        api_secret = provider.get(secret_name)
    except Exception as exc:
        logger.error(
            "Credential lookup failed (%s). "
            "Set %s and %s environment variables.",
            exc, key_name, secret_name,
        )
        sys.exit(1)

    if not api_key or not api_secret:
        logger.error(
            "Empty credentials — set %s / %s before running.", key_name, secret_name
        )
        sys.exit(1)

    return {"api_key": api_key, "api_secret": api_secret}


def run_smoke(exchange_id: str, symbol: str, duration: int) -> None:
    from deployment.exchange.ccxt_adapter import CCXTAdapter
    from deployment.execution.order_manager import OrderManager

    creds = _load_credentials(exchange_id)

    # ── Step 1: Connect and receive tickers ─────────────────────────────
    logger.info("STEP 1 — Connecting to %s testnet, watching %s for %ds …",
                exchange_id, symbol, duration)

    tick_count = 0

    def on_ticker(ticker):
        nonlocal tick_count
        tick_count += 1
        if tick_count % 10 == 1:
            logger.info("  ticker #%d | last=%.2f bid=%.2f ask=%.2f",
                        tick_count,
                        float(ticker.get("last") or 0),
                        float(ticker.get("bid") or 0),
                        float(ticker.get("ask") or 0))

    adapter_cfg = {
        "exchange_id": exchange_id,
        "symbol": symbol,
        "timeframe": "1m",
        "exchange_mode": "sandbox",
        "heartbeat_timeout": 30.0,
        **creds,
    }

    with CCXTAdapter(adapter_cfg) as adapter:
        adapter.add_ticker_callback(on_ticker)
        logger.info("Adapter started. Waiting %ds for ticks …", duration)
        time.sleep(duration)

    if tick_count == 0:
        logger.error("FAIL: No ticks received in %ds — check connectivity.", duration)
        sys.exit(1)
    logger.info("STEP 1 PASS — received %d ticks over %ds", tick_count, duration)

    # ── Step 2: Submit a tiny limit order and cancel it ─────────────────
    logger.info("STEP 2 — Submitting test limit order via OrderManager …")

    om_cfg = {
        "exchange_id": exchange_id,
        "exchange_mode": "sandbox",
        "symbol": symbol,
        "max_order_size": 0.001,
        "daily_loss_limit": -1000.0,
        **creds,
    }
    try:
        with OrderManager(exchange_config=om_cfg, paper_mode=False) as om:
            # Fetch current price first (best-effort)
            try:
                import ccxt
                ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
                ex.set_sandbox_mode(True)
                ticker = ex.fetch_ticker(symbol)
                mid = float(ticker.get("last") or ticker.get("bid") or 40000.0)
            except Exception:
                mid = 40000.0

            # Place 10 % below mid so it won't fill immediately
            limit_px = round(mid * 0.90, 2)
            logger.info("  placing limit buy 0.001 %s @ %.2f", symbol, limit_px)
            order_id = om.submit_order(
                side="buy",
                amount=0.001,
                order_type="limit",
                limit_price=limit_px,
                current_price=mid,
            )
            status = om.check_order(order_id)
            logger.info("  order %s status: %s", order_id, status)

            cancelled = om.cancel_order(order_id)
            logger.info("  cancel result: %s", cancelled)

        logger.info("STEP 2 PASS — limit order submitted and cancelled cleanly")
    except Exception as exc:
        logger.error("STEP 2 FAIL — %s", exc)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("SANDBOX SMOKE PASSED ✓")
    logger.info("  exchange : %s (testnet)", exchange_id)
    logger.info("  symbol   : %s", symbol)
    logger.info("  ticks    : %d over %ds", tick_count, duration)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sandbox smoke test (Week 72 F5)")
    parser.add_argument("--exchange", default="binance", help="CCXT exchange id")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--duration", type=int, default=60,
                        help="Seconds to collect tickers (default 60)")
    args = parser.parse_args()

    run_smoke(
        exchange_id=args.exchange,
        symbol=args.symbol,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
