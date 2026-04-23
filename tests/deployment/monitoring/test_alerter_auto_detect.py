"""I3: TradingAlerter env-var auto-detection of Discord / Telegram channels."""
from __future__ import annotations

import os

import pytest

from deployment.monitoring.alerter import TradingAlerter


def test_discord_env_var_auto_enables_channel(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test/token")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    alerter = TradingAlerter({"alert_channels": ["console"], "log_dir": str(tmp_path)})
    assert "discord" in alerter._channels


def test_telegram_env_vars_auto_enable_channel(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123456")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    alerter = TradingAlerter({"alert_channels": ["console"], "log_dir": str(tmp_path)})
    assert "telegram" in alerter._channels


def test_env_var_not_duplicated_if_already_in_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test/token")
    alerter = TradingAlerter(
        {"alert_channels": ["console", "discord"], "log_dir": str(tmp_path)}
    )
    assert alerter._channels.count("discord") == 1


def test_no_env_vars_no_extra_channels(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    alerter = TradingAlerter({"alert_channels": ["console"], "log_dir": str(tmp_path)})
    assert "discord" not in alerter._channels
    assert "telegram" not in alerter._channels


def test_partial_telegram_env_vars_not_auto_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Only token, no chat_id → should NOT auto-add telegram
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot123:abc")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    alerter = TradingAlerter({"alert_channels": ["console"], "log_dir": str(tmp_path)})
    assert "telegram" not in alerter._channels


def test_config_channels_take_precedence_no_auto_add_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Config explicitly sets channels; env-var auto-detection still adds if not present
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    alerter = TradingAlerter({"alert_channels": ["file"], "log_dir": str(tmp_path)})
    assert "discord" in alerter._channels  # env-detected


def test_three_channels_reached_on_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """notify_kill_switch must write to alerts.jsonl (file channel)."""
    import json

    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    alerter = TradingAlerter({"log_dir": str(tmp_path)})
    alerter.notify_kill_switch(reason="test_three_channels")
    alerts_path = tmp_path / "alerts.jsonl"
    assert alerts_path.exists()
    record = json.loads(alerts_path.read_text().strip())
    assert record["event"] == "kill_switch_activated"
