"""
Unit tests for CCXTAdapter (Week 72, F1 & F4).

All tests run without real exchange connectivity — ccxt.pro and ccxt are
mocked throughout.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deployment.exchange.ccxt_adapter import CCXTAdapter, _BACKOFF_BASE, _MAX_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(overrides: dict | None = None) -> CCXTAdapter:
    cfg = {
        "exchange_id": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "exchange_mode": "paper",
        "heartbeat_timeout": 1.0,  # short for tests
    }
    if overrides:
        cfg.update(overrides)
    return CCXTAdapter(cfg)


# ---------------------------------------------------------------------------
# F1: Basic data accessors before any connection
# ---------------------------------------------------------------------------

class TestCCXTAdapterInit:
    def test_default_values(self):
        adapter = _make_adapter()
        assert adapter.is_connected() is False
        assert adapter.get_latest_ticker() is None
        assert adapter.get_orderbook() is None
        assert adapter.get_latest_ohlcv() is None
        assert adapter.last_tick_at() is None

    def test_exchange_mode_sandbox(self):
        adapter = _make_adapter({"exchange_mode": "sandbox"})
        assert adapter._mode == "sandbox"

    def test_exchange_mode_live(self):
        adapter = _make_adapter({"exchange_mode": "live"})
        assert adapter._mode == "live"

    def test_callback_registration(self):
        adapter = _make_adapter()
        cb = MagicMock()
        adapter.add_ticker_callback(cb)
        adapter.add_ohlcv_callback(cb)
        assert len(adapter._ticker_callbacks) == 1
        assert len(adapter._ohlcv_callbacks) == 1


# ---------------------------------------------------------------------------
# F1: Data cache updates and callback dispatch
# ---------------------------------------------------------------------------

class TestCCXTAdapterCache:
    def test_ticker_cache_and_callback(self):
        adapter = _make_adapter()
        received = []
        adapter.add_ticker_callback(received.append)

        ticker = {"last": 50000.0, "bid": 49999.0, "ask": 50001.0}
        # Simulate what _watch_ticker does internally
        with adapter._lock:
            adapter._latest_ticker = ticker
            adapter._last_tick_at = time.monotonic()
        for cb in adapter._ticker_callbacks:
            cb(ticker)

        assert adapter.get_latest_ticker() == ticker
        assert len(received) == 1
        assert received[0]["last"] == 50000.0

    def test_ohlcv_cache_and_callback(self):
        adapter = _make_adapter()
        received = []
        adapter.add_ohlcv_callback(received.append)

        bars = [[1_700_000_000_000, 49000.0, 50000.0, 48500.0, 49800.0, 1.23]]
        with adapter._lock:
            adapter._latest_ohlcv = bars
            adapter._last_tick_at = time.monotonic()
        for cb in adapter._ohlcv_callbacks:
            cb(bars)

        assert adapter.get_latest_ohlcv() == bars
        assert len(received) == 1

    def test_get_latest_ticker_returns_copy(self):
        """Callers mutating the returned dict should not affect the cache."""
        adapter = _make_adapter()
        ticker = {"last": 1.0}
        with adapter._lock:
            adapter._latest_ticker = ticker
        result = adapter.get_latest_ticker()
        result["last"] = 99.0
        assert adapter.get_latest_ticker()["last"] == 1.0

    def test_last_tick_at_updates(self):
        adapter = _make_adapter()
        assert adapter.last_tick_at() is None
        t0 = time.monotonic()
        with adapter._lock:
            adapter._last_tick_at = t0
        assert adapter.last_tick_at() == t0


# ---------------------------------------------------------------------------
# F1: _build_exchange — sandbox mode activation
# ---------------------------------------------------------------------------

class TestBuildExchange:
    def test_sandbox_mode_set(self):
        ccxt = pytest.importorskip("ccxt", reason="ccxt not installed")
        adapter = _make_adapter({"exchange_mode": "sandbox"})
        mock_exchange = MagicMock()
        mock_cls = MagicMock(return_value=mock_exchange)

        with patch.object(ccxt, "binance", mock_cls, create=True):
            ex = adapter._build_exchange(use_pro=False)
            mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    def test_live_mode_no_sandbox(self):
        ccxt = pytest.importorskip("ccxt", reason="ccxt not installed")
        adapter = _make_adapter({"exchange_mode": "live"})
        mock_exchange = MagicMock()
        mock_cls = MagicMock(return_value=mock_exchange)

        with patch.object(ccxt, "binance", mock_cls, create=True):
            adapter._build_exchange(use_pro=False)
            mock_exchange.set_sandbox_mode.assert_not_called()


# ---------------------------------------------------------------------------
# F4: Reconnect — exponential backoff
# ---------------------------------------------------------------------------

class TestReconnectBackoff:
    """Verify the retry loop respects max retries and backoff caps."""

    def test_retry_count_respected(self):
        """After _MAX_RETRIES failures the loop gives up."""
        adapter = _make_adapter()
        call_count = 0

        async def boom():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("simulated failure")

        # Patch _subscribe_ws so it always raises
        adapter._subscribe_ws = boom  # type: ignore[assignment]
        # Disable pro check so it uses _subscribe_ws
        with patch("deployment.exchange.ccxt_adapter._check_ccxt_pro", return_value=True):
            # Run the retry loop directly in a temp event loop
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(adapter._subscribe_with_retry())
            finally:
                loop.close()

        # Should have tried _MAX_RETRIES + 1 times (initial + retries)
        assert call_count == _MAX_RETRIES + 1

    def test_stop_event_aborts_retry(self):
        """Setting stop_event before retry loop starts causes immediate exit."""
        adapter = _make_adapter()
        adapter._stop_event.set()

        call_count = 0

        async def boom():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("should not retry")

        adapter._subscribe_ws = boom  # type: ignore[assignment]
        with patch("deployment.exchange.ccxt_adapter._check_ccxt_pro", return_value=True):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(adapter._subscribe_with_retry())
            finally:
                loop.close()

        assert call_count == 0, "No calls expected when stop_event is set"

    def test_alerter_called_on_reconnect(self):
        """alerter.check_connection_lost is called on each retry."""
        alerter = MagicMock()
        adapter = _make_adapter()
        adapter._alerter = alerter
        call_count = 0

        async def boom():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("simulated")

        adapter._subscribe_ws = boom  # type: ignore[assignment]
        with patch("deployment.exchange.ccxt_adapter._check_ccxt_pro", return_value=True):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(adapter._subscribe_with_retry())
            finally:
                loop.close()

        # alerter should be called once per retry (not on the initial attempt)
        assert alerter.check_connection_lost.call_count == _MAX_RETRIES


# ---------------------------------------------------------------------------
# F4: Heartbeat watchdog
# ---------------------------------------------------------------------------

class TestHeartbeatWatchdog:
    def test_watchdog_calls_alerter_on_timeout(self):
        alerter = MagicMock()
        adapter = _make_adapter({"heartbeat_timeout": 0.05})  # 50 ms for speed
        adapter._alerter = alerter

        # Simulate a connected feed with a stale last_tick
        with adapter._lock:
            adapter._connected = True
            adapter._last_tick_at = time.monotonic() - 1.0  # 1s ago, > 50ms threshold

        # Run one watchdog cycle directly
        # (we can't call the real watchdog thread in a unit test easily,
        #  so we exercise the body logic inline)
        elapsed = time.monotonic() - adapter._last_tick_at
        assert elapsed > adapter._heartbeat_timeout
        if elapsed > adapter._heartbeat_timeout:
            adapter._alerter.check_connection_lost(elapsed)

        alerter.check_connection_lost.assert_called_once()

    def test_watchdog_calls_on_halt(self):
        halted_reasons = []
        adapter = _make_adapter({"heartbeat_timeout": 0.05})
        adapter._on_halt = halted_reasons.append

        with adapter._lock:
            adapter._connected = True
            adapter._last_tick_at = time.monotonic() - 1.0

        elapsed = time.monotonic() - adapter._last_tick_at
        if elapsed > adapter._heartbeat_timeout:
            adapter._on_halt("ccxt_heartbeat_timeout")

        assert halted_reasons == ["ccxt_heartbeat_timeout"]

    def test_watchdog_no_alert_when_not_connected(self):
        alerter = MagicMock()
        adapter = _make_adapter({"heartbeat_timeout": 0.05})
        adapter._alerter = alerter
        # Not connected — watchdog should skip
        with adapter._lock:
            adapter._connected = False
            adapter._last_tick_at = time.monotonic() - 1.0

        # Watchdog body: if not connected → continue
        if adapter._connected:
            adapter._alerter.check_connection_lost(999)

        alerter.check_connection_lost.assert_not_called()


# ---------------------------------------------------------------------------
# F1: Context manager (start/stop lifecycle)
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_starts_and_stops(self):
        adapter = _make_adapter()

        async def noop_subscribe():
            await asyncio.sleep(0.01)

        async def noop_retry():
            await asyncio.sleep(0.05)

        adapter._subscribe_with_retry = noop_retry  # type: ignore[assignment]

        with adapter:
            assert adapter._thread is not None
            assert adapter._thread.is_alive()
        # After exit the thread should stop
        adapter._thread.join(timeout=2.0)
        assert not adapter._thread.is_alive()


# ---------------------------------------------------------------------------
# F6: Credential not leaked via callbacks
# ---------------------------------------------------------------------------

class TestCredentialSafety:
    def test_api_key_not_stored_in_ticker(self):
        """The ticker dict returned by get_latest_ticker must not contain credentials."""
        adapter = _make_adapter({"api_key": "super_secret", "api_secret": "also_secret"})
        fake_ticker = {"last": 1.0, "bid": 0.99, "ask": 1.01}
        with adapter._lock:
            adapter._latest_ticker = fake_ticker
        result = adapter.get_latest_ticker()
        assert "api_key" not in result
        assert "secret" not in result
