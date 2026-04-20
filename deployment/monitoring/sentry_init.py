"""
Sentry runtime exception tracking (H5 — Week 78).

Initialises Sentry with:
- before_send scrubbing: credential fields and raw price data are stripped
- traces_sample_rate: 0.1 (10% of transactions sampled for performance)
- environment tag from TRADING_ENV env var (default "local")

Usage (in application entry point):
    from deployment.monitoring.sentry_init import init_sentry
    init_sentry(dsn=config.get("sentry_dsn"))

Or via config/monitoring.yaml:
    monitoring:
      sentry_dsn: "https://...@sentry.io/..."
      sentry_traces_sample_rate: 0.1
      sentry_environment: "paper"
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fields whose values must never appear in Sentry events
_SCRUB_KEYS = frozenset({
    # Credentials
    "api_key", "api_secret", "secret", "token", "password", "passphrase",
    "private_key", "secret_key", "exchange_key", "telegram_token",
    "discord_webhook_url", "webhook_url", "telegram_chat_id",
    # Exchange key patterns
    "BINANCE_API_KEY", "BINANCE_SECRET", "COINBASE_API_KEY",
    "EXCHANGE_BINANCE_TESTNET_KEY",
})

# Regex for values that look like API keys / secrets (base64-ish, long hex, etc.)
_SECRET_VALUE_RE = re.compile(
    r"^[A-Za-z0-9+/=_\-]{32,}$"
)

# Price data field names — strip raw OHLCV arrays to avoid quota bloat
_PRICE_KEYS = frozenset({
    "open", "high", "low", "close", "volume",
    "prices", "ohlcv", "candles", "ticks", "price_data",
    "observation", "obs",
})

_PLACEHOLDER = "[scrubbed]"


def _scrub_dict(data: Any, depth: int = 0) -> Any:
    """Recursively scrub sensitive fields from a Sentry event payload."""
    if depth > 10:
        return data
    if isinstance(data, dict):
        return {
            k: _PLACEHOLDER
            if (k.lower() in _SCRUB_KEYS or k in _SCRUB_KEYS)
            else _PLACEHOLDER
            if (k.lower() in _PRICE_KEYS and isinstance(v, (list, dict)) and len(str(v)) > 200)
            else _scrub_dict(v, depth + 1)
            for k, v in data.items()
            for v in [data[k]]  # bind v once
        }
    if isinstance(data, list):
        return [_scrub_dict(item, depth + 1) for item in data]
    # Scrub bare string values that look like secrets
    if isinstance(data, str) and _SECRET_VALUE_RE.match(data) and len(data) >= 32:
        return _PLACEHOLDER
    return data


def _before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sentry before_send hook — scrub credentials and price data."""
    # Scrub extra / request context
    for section in ("extra", "request", "user", "contexts"):
        if section in event:
            event[section] = _scrub_dict(event[section])

    # Scrub local variables in stack frames
    if "exception" in event:
        for exc_val in event["exception"].get("values", []):
            if "stacktrace" in exc_val:
                for frame in exc_val["stacktrace"].get("frames", []):
                    if "vars" in frame:
                        frame["vars"] = _scrub_dict(frame["vars"])

    return event


def init_sentry(
    dsn: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    environment: Optional[str] = None,
) -> bool:
    """Initialise Sentry SDK.

    Parameters
    ----------
    dsn:
        Sentry project DSN. Falls back to SENTRY_DSN env var.
        If neither is set, Sentry remains disabled (no-op).
    traces_sample_rate:
        Fraction of transactions to sample for performance monitoring (0–1).
    environment:
        Sentry environment tag (e.g. "paper", "live", "local").
        Falls back to TRADING_ENV env var, then "local".

    Returns
    -------
    bool
        True if Sentry was successfully initialised.
    """
    resolved_dsn = dsn or os.environ.get("SENTRY_DSN")
    if not resolved_dsn:
        logger.info("Sentry DSN not configured — exception tracking disabled")
        return False

    try:
        import sentry_sdk
    except ImportError:
        logger.warning("sentry-sdk not installed — exception tracking disabled")
        return False

    resolved_env = environment or os.environ.get("TRADING_ENV", "local")

    sentry_sdk.init(
        dsn=resolved_dsn,
        traces_sample_rate=max(0.0, min(1.0, traces_sample_rate)),
        environment=resolved_env,
        before_send=_before_send,
        # Never send raw request bodies (may contain order data)
        request_bodies="never",
        # Attach local variables to stack traces (scrubbed by before_send)
        attach_stacktrace=True,
        # Ignore common noisy exceptions
        ignore_errors=[KeyboardInterrupt, SystemExit],
    )

    logger.info(
        "Sentry initialised | env=%s traces_sample_rate=%.2f",
        resolved_env,
        traces_sample_rate,
    )
    return True


def capture_exception(exc: BaseException, context: Optional[Dict[str, Any]] = None) -> None:
    """Capture an exception to Sentry with optional scrubbed context."""
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            if context:
                scrubbed = _scrub_dict(context)
                for k, v in scrubbed.items():
                    scope.set_extra(k, v)
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass  # never let Sentry crash the trading loop
