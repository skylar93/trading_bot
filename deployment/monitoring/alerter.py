"""
Trading Alerter: 조건 충족 시 알림 발송.

지원 채널: console (기본), Telegram, generic webhook.
환경 변수로 토큰을 주입하는 것을 권장 (hardcode 금지).

Usage
-----
    from deployment.monitoring.alerter import TradingAlerter

    alerter = TradingAlerter({
        "alert_channels": ["console"],
        "drawdown_alert_threshold": 0.10,
        "daily_loss_alert": -500.0,
    })
    alerter.check_drawdown(current=8_800, peak=10_000)   # 12% → fires alert
    alerter.check_daily_pnl(-600)                         # below limit → fires alert
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal alert record
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    level: str          # "WARNING" | "CRITICAL"
    channel: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"[{ts}] [{self.level}] {self.message}"


# ---------------------------------------------------------------------------
# TradingAlerter
# ---------------------------------------------------------------------------

class TradingAlerter:
    """Monitors trading metrics and dispatches alerts to configured channels.

    Parameters
    ----------
    config : dict
        Configuration dict.  Expected keys (all optional):

        alert_channels : list of str
            ``["console"]`` (default).  Also accepts ``"telegram"`` and
            ``"webhook"``.
        drawdown_alert_threshold : float
            Drawdown fraction that triggers an alert (default 0.10 = 10 %).
        daily_loss_alert : float
            Absolute daily P&L below which an alert fires (default -500).
        telegram_token : str
            Telegram bot token.  Reads ``TELEGRAM_BOT_TOKEN`` env var if
            not provided.
        telegram_chat_id : str
            Telegram chat/channel ID.  Reads ``TELEGRAM_CHAT_ID`` env var.
        webhook_url : str
            HTTP URL for generic POST alerts.
    """

    def __init__(self, config: dict) -> None:
        self.channels: List[str] = config.get("alert_channels", ["console"])
        self.drawdown_threshold: float = float(
            config.get("drawdown_alert_threshold", 0.10)
        )
        self.daily_loss_limit: float = float(
            config.get("daily_loss_alert", -500.0)
        )

        # Telegram
        self._telegram_token: Optional[str] = config.get(
            "telegram_token", os.environ.get("TELEGRAM_BOT_TOKEN")
        )
        self._telegram_chat_id: Optional[str] = config.get(
            "telegram_chat_id", os.environ.get("TELEGRAM_CHAT_ID")
        )

        # Webhook
        self._webhook_url: Optional[str] = config.get("webhook_url")

        # History of fired alerts (useful for testing)
        self.alert_history: List[Alert] = []

    # ------------------------------------------------------------------
    # Trigger checks
    # ------------------------------------------------------------------

    def check_drawdown(self, current: float, peak: float) -> bool:
        """Fire an alert if drawdown from ``peak`` exceeds the threshold.

        Parameters
        ----------
        current : float
            Current portfolio value.
        peak : float
            Historical peak portfolio value.

        Returns
        -------
        bool
            ``True`` if an alert was fired.
        """
        if peak <= 0:
            return False
        drawdown = (peak - current) / peak
        if drawdown >= self.drawdown_threshold:
            msg = (
                f"Drawdown alert: current={current:.2f}, peak={peak:.2f}, "
                f"drawdown={drawdown:.1%} >= threshold={self.drawdown_threshold:.1%}"
            )
            self._fire("WARNING", msg)
            return True
        return False

    def check_daily_pnl(self, pnl: float) -> bool:
        """Fire an alert if today's realised P&L is below the limit.

        Parameters
        ----------
        pnl : float
            Today's P&L in account currency.

        Returns
        -------
        bool
            ``True`` if an alert was fired.
        """
        if pnl < self.daily_loss_limit:
            msg = (
                f"Daily loss alert: pnl={pnl:.2f} < limit={self.daily_loss_limit:.2f}"
            )
            self._fire("CRITICAL", msg)
            return True
        return False

    def check_connection_lost(self, seconds_since_last_tick: float) -> bool:
        """Fire an alert if the data feed has been silent for > 60 s."""
        if seconds_since_last_tick > 60.0:
            msg = f"Connection alert: no data for {seconds_since_last_tick:.0f} s"
            self._fire("CRITICAL", msg)
            return True
        return False

    def send_alert(self, message: str, level: str = "WARNING") -> None:
        """Manually dispatch an alert message."""
        self._fire(level, message)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _fire(self, level: str, message: str) -> None:
        alert = Alert(level=level, channel="", message=message)
        for channel in self.channels:
            alert.channel = channel
            self.alert_history.append(alert)
            try:
                if channel == "console":
                    self._send_console(alert)
                elif channel == "telegram":
                    self._send_telegram(alert)
                elif channel == "webhook":
                    self._send_webhook(alert)
                else:
                    logger.warning("Unknown alert channel: %s", channel)
            except Exception as exc:
                logger.error("Alert dispatch failed on channel '%s': %s", channel, exc)

    def _send_console(self, alert: Alert) -> None:
        log_fn = logger.critical if alert.level == "CRITICAL" else logger.warning
        log_fn("[ALERT] %s", alert.message)

    def _send_telegram(self, alert: Alert) -> None:
        if not self._telegram_token or not self._telegram_chat_id:
            logger.warning(
                "Telegram alert skipped: missing token or chat_id. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
            )
            return
        try:
            import urllib.request, json as _json, urllib.parse
            text = urllib.parse.quote(str(alert))
            url = (
                f"https://api.telegram.org/bot{self._telegram_token}"
                f"/sendMessage?chat_id={self._telegram_chat_id}&text={text}"
            )
            urllib.request.urlopen(url, timeout=5)
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)

    def _send_webhook(self, alert: Alert) -> None:
        if not self._webhook_url:
            logger.warning("Webhook alert skipped: no webhook_url configured.")
            return
        try:
            import urllib.request, json as _json
            payload = _json.dumps(
                {
                    "level": alert.level,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
            ).encode()
            req = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.error("Webhook send failed: %s", exc)
