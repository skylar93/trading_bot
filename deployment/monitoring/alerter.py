"""
Trading Alerter: condition-based notification dispatcher.

Supported channels:
    - console  : logger.warning (always available, used as fallback)
    - telegram : sends message via Telegram Bot API
    - webhook  : HTTP POST to a generic endpoint
    - discord  : HTTP POST to a Discord webhook URL

Triggers:
    - drawdown exceeds threshold (default 10 %)
    - daily P&L below loss limit
    - feature/reward drift detected
    - connection lost for more than N seconds
    - kill switch activated
    - audit chain integrity break
    - runtime error (notify_error)
    - trade executed (optional, verbose mode only)

Usage:
    alerter = TradingAlerter({
        "alert_channels": ["console", "discord"],
        "drawdown_alert_threshold": 0.10,
        "daily_loss_alert": -500,
        "discord_webhook_url": "https://discord.com/api/webhooks/...",
    })
    alerter.check_drawdown(current=9000, peak=10000)
    alerter.check_daily_pnl(pnl=-600)
    alerter.notify_drift(detector="adwin", signal_name="reward")
    alerter.notify_kill_switch()
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Discord embeds colour codes
_COLOUR_WARNING = 0xFFA500   # orange
_COLOUR_CRITICAL = 0xFF0000  # red
_COLOUR_INFO = 0x00B0F4      # blue
_COLOUR_ERROR = 0xFF0000     # red


@dataclass
class AlertRecord:
    """Record of a dispatched alert (used for testing and audit)."""
    level: str
    event: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"[{ts}] [{self.level}] {self.event}: {self.message}"


class TradingAlerter:
    """
    Dispatches alerts to one or more notification channels.

    Parameters
    ----------
    config : dict
        Alert configuration.
        Keys:
            alert_channels              – list of channels: ["console"] | ["telegram"] |
                                          ["webhook"] | ["discord"]
            drawdown_alert_threshold    – drawdown fraction to trigger alert (default 0.10)
            daily_loss_alert            – daily P&L threshold to trigger alert (default -500)
            connection_timeout_seconds  – seconds of lost connection before alert (default 60)
            telegram_token              – Telegram bot token (or TELEGRAM_BOT_TOKEN env var)
            telegram_chat_id            – Telegram chat id (or TELEGRAM_CHAT_ID env var)
            webhook_url                 – HTTP endpoint for generic webhook alerts
            discord_webhook_url         – Discord webhook URL (or DISCORD_WEBHOOK_URL env var)
            verbose                     – if True, also alert on every trade (default False)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._channels: List[str] = config.get("alert_channels", ["console"])
        self.drawdown_threshold: float = float(config.get("drawdown_alert_threshold", 0.10))
        self.daily_loss_limit: float = float(config.get("daily_loss_alert", -500.0))
        self.connection_timeout: float = float(config.get("connection_timeout_seconds", 60.0))
        self.verbose: bool = bool(config.get("verbose", False))

        # Telegram
        self._telegram_token: Optional[str] = (
            config.get("telegram_token") or os.environ.get("TELEGRAM_BOT_TOKEN")
        )
        self._telegram_chat_id: Optional[str] = (
            str(config.get("telegram_chat_id", "")) or os.environ.get("TELEGRAM_CHAT_ID")
        )

        # Generic webhook
        self._webhook_url: Optional[str] = config.get("webhook_url") or None

        # Discord webhook
        self._discord_url: Optional[str] = (
            config.get("discord_webhook_url")
            or os.environ.get("DISCORD_WEBHOOK_URL")
            or None
        )

        # Connection tracking
        self._last_connection_time: float = time.monotonic()
        self._connection_alert_sent: bool = False

        # Audit log
        self.alert_history: List[AlertRecord] = []

        logger.info(
            "TradingAlerter initialised | channels=%s drawdown_thresh=%.1f%% daily_loss=%.0f",
            self._channels,
            self.drawdown_threshold * 100,
            self.daily_loss_limit,
        )

    # ------------------------------------------------------------------
    # Trigger methods
    # ------------------------------------------------------------------

    def check_drawdown(self, current: float, peak: float) -> bool:
        """Alert if drawdown from peak exceeds the threshold. Returns True if fired."""
        if peak <= 0:
            return False
        drawdown = (peak - current) / peak
        if drawdown >= self.drawdown_threshold:
            self._dispatch(
                level="WARNING",
                event="drawdown_alert",
                message=(
                    f"Drawdown alert: {drawdown:.1%} drawdown "
                    f"(current={current:.2f}, peak={peak:.2f}, "
                    f"threshold={self.drawdown_threshold:.1%})"
                ),
            )
            return True
        return False

    def check_daily_pnl(self, pnl: float) -> bool:
        """Alert if daily P&L is below the loss limit. Returns True if fired."""
        if pnl <= self.daily_loss_limit:
            self._dispatch(
                level="CRITICAL",
                event="daily_loss_alert",
                message=(
                    f"Daily loss alert: P&L={pnl:.2f} "
                    f"(limit={self.daily_loss_limit:.2f})"
                ),
            )
            return True
        return False

    def check_connection_lost(self, seconds_since_last_tick: float) -> bool:
        """Alert if data feed has been silent for > connection_timeout. Returns True if fired."""
        if seconds_since_last_tick > self.connection_timeout:
            self._dispatch(
                level="CRITICAL",
                event="connection_lost",
                message=(
                    f"Connection lost for {seconds_since_last_tick:.0f}s "
                    f"(timeout={self.connection_timeout:.0f}s)"
                ),
            )
            return True
        return False

    def notify_drift(self, detector: str, signal_name: str, details: Optional[str] = None) -> None:
        """Alert that concept drift was detected."""
        msg = f"Drift detected by {detector} on signal '{signal_name}'"
        if details:
            msg += f" — {details}"
        self._dispatch(level="WARNING", event="drift_detected", message=msg)

    def notify_connection_restored(self) -> None:
        """Reset connection tracking after reconnect."""
        self._last_connection_time = time.monotonic()
        self._connection_alert_sent = False
        logger.info("Connection restored; alert state reset.")

    def notify_trade(self, side: str, amount: float, price: float, order_id: str = "") -> None:
        """Alert on trade execution (only in verbose mode)."""
        if not self.verbose:
            return
        self._dispatch(
            level="INFO",
            event="trade_executed",
            message=(
                f"Trade executed: {side.upper()} {amount:.6f} @ {price:.2f}"
                + (f" (id={order_id})" if order_id else "")
            ),
        )

    def notify_error(self, error: str, context: Optional[str] = None) -> None:
        """Alert on a runtime error (called from OrderManager and PaperTrader error paths)."""
        msg = f"Runtime error: {error}"
        if context:
            msg += f" | context={context}"
        self._dispatch(level="ERROR", event="runtime_error", message=msg)

    def notify_kill_switch(self, reason: str = "manual") -> None:
        """Alert that the kill switch has been activated — highest priority."""
        self._dispatch(
            level="CRITICAL",
            event="kill_switch_activated",
            message=f"KILL SWITCH ACTIVATED — reason={reason}. All orders cancelled.",
        )

    def notify_audit_chain_break(self, details: str = "") -> None:
        """Alert that audit log integrity has been compromised."""
        msg = "AUDIT CHAIN INTEGRITY BREAK detected — log continuity cannot be guaranteed."
        if details:
            msg += f" Details: {details}"
        self._dispatch(level="CRITICAL", event="audit_chain_break", message=msg)

    def notify_reconciliation_drift(self, drift_detail: Any) -> None:
        """Alert that a position/order mismatch was detected during reconciliation."""
        if isinstance(drift_detail, list):
            detail_str = "; ".join(
                f"{d.get('type', '?')}={d}" for d in drift_detail
            )
        else:
            detail_str = str(drift_detail)
        self._dispatch(
            level="ERROR",
            event="reconciliation_drift",
            message=f"Reconciliation mismatch detected: {detail_str}",
        )

    def notify_fee_refresh_failed(self, reason: str = "") -> None:
        """Alert that a scheduled fee-tier API refresh failed (fallback in effect)."""
        msg = "Fee tier refresh failed — retaining previous rates."
        if reason:
            msg += f" Reason: {reason}"
        self._dispatch(level="WARNING", event="fee_refresh_failed", message=msg)

    def notify_canary_auto_demoted(
        self,
        version: int,
        sigma_below: float,
        consecutive_hours: int,
        canary_mean: float,
        prod_mean: float,
        prod_std: float,
    ) -> None:
        """Alert that canary was auto-demoted due to sustained underperformance.

        Traffic has been set to 0 %; human must manually restore via
        ``promote_model.py --restore-traffic`` after investigating.
        """
        msg = (
            f"CANARY AUTO-DEMOTION: v{version} traffic set to 0%%. "
            f"Canary return ({canary_mean:.4f}) fell below prod - {sigma_below:.1f}σ "
            f"({prod_mean:.4f} - {sigma_below:.1f}×{prod_std:.4f} = "
            f"{prod_mean - sigma_below * prod_std:.4f}) "
            f"for {consecutive_hours}h. Stage remains 'canary'; "
            f"human sign-off required to restore traffic."
        )
        self._dispatch(level="CRITICAL", event="canary_auto_demoted", message=msg)

    def schema_drift_detected(self, drift_detail: str, on_drift: str = "halt") -> None:
        """Alert that real-time feed schema drift was detected.

        Parameters
        ----------
        drift_detail:
            Description of the drift (unexpected keys, wrong dtype, etc.).
        on_drift:
            Policy from config — ``"halt"`` or ``"warn"``.
        """
        level = "CRITICAL" if on_drift == "halt" else "WARNING"
        msg = f"Schema drift detected ({on_drift} policy): {drift_detail}"
        self._dispatch(level=level, event="schema_drift", message=msg)

    def send_alert(self, message: str, level: str = "WARNING") -> None:
        """Manually dispatch an alert message."""
        self._dispatch(level=level, event="manual_alert", message=message)

    # ------------------------------------------------------------------
    # Dispatch core
    # ------------------------------------------------------------------

    def _dispatch(self, level: str, event: str, message: str) -> None:
        """Route an alert to all configured channels and record it."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        full_message = f"[{timestamp}] [{level}] {event}: {message}"

        record = AlertRecord(level=level, event=event, message=message)
        self.alert_history.append(record)

        for channel in self._channels:
            if channel == "console":
                self._send_console(level, full_message)
            elif channel == "telegram":
                self._send_telegram(full_message)
            elif channel == "webhook":
                self._send_webhook(event, level, message, timestamp)
            elif channel == "discord":
                self._send_discord(event, level, message, timestamp)
            else:
                logger.warning("Unknown alert channel: %s", channel)

    def _send_console(self, level: str, message: str) -> None:
        if level in ("CRITICAL", "ERROR"):
            logger.error(message)
        else:
            logger.warning(message)

    def _send_telegram(self, message: str) -> None:
        if not self._telegram_token or not self._telegram_chat_id:
            logger.warning(
                "Telegram alert skipped: token or chat_id not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
            )
            return

        try:
            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self._telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Telegram API returned status %s", resp.status)
        except Exception as e:
            logger.error("Telegram alert failed: %s", e)

    def _send_webhook(self, event: str, level: str, message: str, timestamp: str) -> None:
        if not self._webhook_url:
            logger.warning("Webhook alert skipped: webhook_url not configured.")
            return

        try:
            payload = json.dumps({
                "event": event,
                "level": level,
                "message": message,
                "timestamp": timestamp,
            }).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 201, 204):
                    logger.warning("Webhook returned status %s", resp.status)
        except Exception as e:
            logger.error("Webhook alert failed: %s", e)

    def _send_discord(self, event: str, level: str, message: str, timestamp: str) -> None:
        """Send alert to a Discord channel via incoming webhook."""
        if not self._discord_url:
            logger.warning(
                "Discord alert skipped: discord_webhook_url not configured. "
                "Set DISCORD_WEBHOOK_URL env var."
            )
            return

        colour_map = {
            "CRITICAL": _COLOUR_CRITICAL,
            "ERROR": _COLOUR_ERROR,
            "WARNING": _COLOUR_WARNING,
            "INFO": _COLOUR_INFO,
        }
        colour = colour_map.get(level, _COLOUR_WARNING)

        try:
            payload = json.dumps({
                "username": "Trading Bot Alerter",
                "embeds": [
                    {
                        "title": f"[{level}] {event}",
                        "description": message,
                        "color": colour,
                        "footer": {"text": timestamp},
                    }
                ],
            }).encode("utf-8")
            req = urllib.request.Request(
                self._discord_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                # Discord returns 204 No Content on success
                if resp.status not in (200, 204):
                    logger.warning("Discord webhook returned status %s", resp.status)
        except Exception as e:
            logger.error("Discord alert failed: %s", e)
