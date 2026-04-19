"""
CCXTAdapter — WebSocket-first exchange connectivity (Week 72, F1 & F4).

Uses ccxt.pro (async WebSocket) when available; falls back to ccxt REST polling.

F1: Subscribe to ticker / orderbook / OHLCV on a public read-only channel.
    Callbacks let external subscribers (e.g. CCXTLiveDataSource) receive updates.
F4: Exponential-backoff reconnect (≤5 retries, capped at 30 s per wait).
    Heartbeat watchdog calls alerter.check_connection_lost() and on_halt()
    when feed goes silent beyond heartbeat_timeout seconds.

Usage (read-only, no credentials needed)::

    adapter = CCXTAdapter({"exchange_id": "binance", "symbol": "BTC/USDT"})
    adapter.add_ticker_callback(lambda t: print(t["last"]))
    adapter.start()
    time.sleep(60)
    adapter.stop()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BACKOFF_BASE = 1.0     # first wait (seconds)
_BACKOFF_MAX = 30.0     # per-attempt cap (seconds)
_WATCHDOG_POLL = 5.0    # heartbeat watchdog check interval (seconds)

# Lazy flag — set once after first import attempt
_has_ccxt_pro: Optional[bool] = None


def _check_ccxt_pro() -> bool:
    global _has_ccxt_pro
    if _has_ccxt_pro is None:
        try:
            import ccxt.pro  # noqa: F401
            _has_ccxt_pro = True
        except Exception:
            _has_ccxt_pro = False
    return _has_ccxt_pro  # type: ignore[return-value]


class CCXTAdapter:
    """
    Manages a live WebSocket connection to a CCXT-compatible exchange.

    Runs an asyncio event loop in a background daemon thread so callers
    remain fully synchronous.

    Parameters
    ----------
    config : dict
        exchange_id         – CCXT exchange id (default "binance")
        symbol              – trading pair (default "BTC/USDT")
        timeframe           – OHLCV bar size (default "1m")
        exchange_mode       – "paper" | "sandbox" | "live"  (default "paper")
        api_key             – optional; public feeds work without credentials
        api_secret          – optional
        heartbeat_timeout   – seconds of silence before watchdog fires (default 60)
    alerter : optional
        Any object with ``check_connection_lost(seconds: float) -> bool``.
    on_halt : optional
        Callable[[str], None] invoked by the heartbeat watchdog on timeout.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        alerter=None,
        on_halt: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._exchange_id: str = config.get("exchange_id", "binance")
        self._symbol: str = config.get("symbol", "BTC/USDT")
        self._timeframe: str = config.get("timeframe", "1m")
        self._mode: str = config.get("exchange_mode", "paper")
        self._api_key: str = config.get("api_key", "")
        self._api_secret: str = config.get("api_secret", "")
        self._heartbeat_timeout: float = float(config.get("heartbeat_timeout", 60.0))
        self._alerter = alerter
        self._on_halt = on_halt

        # Thread-safe caches
        self._lock = threading.RLock()
        self._latest_ticker: Optional[Dict] = None
        self._latest_orderbook: Optional[Dict] = None
        self._latest_ohlcv: Optional[List] = None
        self._last_tick_at: Optional[float] = None
        self._connected: bool = False

        # Callback registries (called from background thread, must be thread-safe)
        self._ticker_callbacks: List[Callable] = []
        self._ohlcv_callbacks: List[Callable] = []

        # Background infrastructure
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_ticker_callback(self, cb: Callable[[Dict], None]) -> None:
        """Register a function called on every new ticker update."""
        self._ticker_callbacks.append(cb)

    def add_ohlcv_callback(self, cb: Callable[[List], None]) -> None:
        """Register a function called on every new OHLCV update.

        cb receives the raw list of bars: [[ts, open, high, low, close, vol], ...]
        """
        self._ohlcv_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background subscriber thread and heartbeat watchdog."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run_event_loop, name="ccxt-ws", daemon=True
        )
        self._watchdog_thread = threading.Thread(
            target=self._heartbeat_watchdog, name="ccxt-heartbeat", daemon=True
        )
        self._thread.start()
        self._watchdog_thread.start()

        logger.info(
            "CCXTAdapter started | exchange=%s symbol=%s timeframe=%s mode=%s",
            self._exchange_id, self._symbol, self._timeframe, self._mode,
        )

    def stop(self) -> None:
        """Signal background threads to stop and join them."""
        self._stop_event.set()
        # Wake up the async loop if it's sleeping inside asyncio.sleep()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5.0)

        with self._lock:
            self._connected = False
        logger.info("CCXTAdapter stopped")

    # ------------------------------------------------------------------
    # Synchronous data accessors
    # ------------------------------------------------------------------

    def get_latest_ticker(self) -> Optional[Dict]:
        with self._lock:
            return dict(self._latest_ticker) if self._latest_ticker else None

    def get_orderbook(self) -> Optional[Dict]:
        with self._lock:
            return dict(self._latest_orderbook) if self._latest_orderbook else None

    def get_latest_ohlcv(self) -> Optional[List]:
        with self._lock:
            return list(self._latest_ohlcv) if self._latest_ohlcv else None

    def last_tick_at(self) -> Optional[float]:
        with self._lock:
            return self._last_tick_at

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    # ------------------------------------------------------------------
    # Exchange factory
    # ------------------------------------------------------------------

    def _build_exchange(self, use_pro: bool = True):
        creds: Dict[str, Any] = {"enableRateLimit": True}
        if self._api_key:
            creds["apiKey"] = self._api_key
        if self._api_secret:
            creds["secret"] = self._api_secret

        if use_pro:
            import ccxt.pro as ccxtpro
            exchange = getattr(ccxtpro, self._exchange_id)(creds)
        else:
            import ccxt
            exchange = getattr(ccxt, self._exchange_id)(creds)

        if self._mode == "sandbox":
            exchange.set_sandbox_mode(True)

        return exchange

    # ------------------------------------------------------------------
    # Background event loop thread
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._subscribe_with_retry())
        finally:
            loop.close()
            self._loop = None

    async def _subscribe_with_retry(self) -> None:
        """Outer retry loop with exponential backoff (F4)."""
        use_pro = _check_ccxt_pro()
        attempt = 0
        backoff = _BACKOFF_BASE

        while not self._stop_event.is_set():
            try:
                if use_pro:
                    await self._subscribe_ws()
                else:
                    await self._poll_rest()
                break  # clean stop requested
            except asyncio.CancelledError:
                break
            except Exception as exc:
                attempt += 1
                with self._lock:
                    self._connected = False

                if self._stop_event.is_set():
                    break
                if attempt > _MAX_RETRIES:
                    logger.error(
                        "CCXTAdapter: exceeded %d retries, giving up. Last: %s",
                        _MAX_RETRIES, exc,
                    )
                    break

                wait = min(backoff, _BACKOFF_MAX)
                logger.warning(
                    "CCXTAdapter: error (attempt %d/%d), retry in %.1fs: %s",
                    attempt, _MAX_RETRIES, wait, exc,
                )
                if self._alerter is not None:
                    try:
                        self._alerter.check_connection_lost(wait)
                    except Exception:
                        pass

                await asyncio.sleep(wait)
                backoff = min(backoff * 2.0, _BACKOFF_MAX)

    # ------------------------------------------------------------------
    # WebSocket subscriber (ccxt.pro)
    # ------------------------------------------------------------------

    async def _subscribe_ws(self) -> None:
        exchange = self._build_exchange(use_pro=True)
        try:
            with self._lock:
                self._connected = True
            logger.info(
                "CCXTAdapter: WebSocket connected | %s %s",
                self._exchange_id, self._symbol,
            )
            tasks = [
                asyncio.create_task(self._watch_ticker(exchange)),
                asyncio.create_task(self._watch_orderbook(exchange)),
                asyncio.create_task(self._watch_ohlcv(exchange)),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for t in pending:
                t.cancel()
            # Re-raise any exception so the retry loop can handle it
            for t in done:
                if t.exception() is not None:
                    raise t.exception()  # type: ignore[misc]
        finally:
            try:
                await exchange.close()
            except Exception:
                pass
            with self._lock:
                self._connected = False

    async def _watch_ticker(self, exchange) -> None:
        while not self._stop_event.is_set():
            ticker = await exchange.watch_ticker(self._symbol)
            with self._lock:
                self._latest_ticker = ticker
                self._last_tick_at = time.monotonic()
            for cb in self._ticker_callbacks:
                try:
                    cb(ticker)
                except Exception as e:
                    logger.warning("ticker callback error: %s", e)

    async def _watch_orderbook(self, exchange) -> None:
        while not self._stop_event.is_set():
            try:
                ob = await exchange.watch_order_book(self._symbol)
                with self._lock:
                    self._latest_orderbook = ob
            except Exception:
                # orderbook is best-effort; don't fail the whole connection
                await asyncio.sleep(1.0)

    async def _watch_ohlcv(self, exchange) -> None:
        while not self._stop_event.is_set():
            bars = await exchange.watch_ohlcv(self._symbol, self._timeframe)
            with self._lock:
                self._latest_ohlcv = bars
                self._last_tick_at = time.monotonic()
            for cb in self._ohlcv_callbacks:
                try:
                    cb(bars)
                except Exception as e:
                    logger.warning("ohlcv callback error: %s", e)

    # ------------------------------------------------------------------
    # REST polling fallback (when ccxt.pro is unavailable)
    # ------------------------------------------------------------------

    async def _poll_rest(self) -> None:
        exchange = self._build_exchange(use_pro=False)
        try:
            with self._lock:
                self._connected = True
            logger.info(
                "CCXTAdapter: REST polling %s (ccxt.pro not available)", self._exchange_id
            )
            while not self._stop_event.is_set():
                try:
                    ticker = exchange.fetch_ticker(self._symbol)
                    bars = exchange.fetch_ohlcv(self._symbol, self._timeframe, limit=2)
                    with self._lock:
                        self._latest_ticker = ticker
                        if bars:
                            self._latest_ohlcv = bars
                        self._last_tick_at = time.monotonic()
                    for cb in self._ticker_callbacks:
                        try:
                            cb(ticker)
                        except Exception as e:
                            logger.warning("ticker callback error: %s", e)
                    if bars:
                        for cb in self._ohlcv_callbacks:
                            try:
                                cb(bars)
                            except Exception as e:
                                logger.warning("ohlcv callback error: %s", e)
                except Exception as exc:
                    logger.warning("CCXTAdapter REST poll error: %s", exc)
                    raise  # let retry loop handle it
                await asyncio.sleep(5.0)
        finally:
            with self._lock:
                self._connected = False

    # ------------------------------------------------------------------
    # F4: Heartbeat watchdog thread
    # ------------------------------------------------------------------

    def _heartbeat_watchdog(self) -> None:
        """Separate thread: alerts + calls on_halt when feed goes silent."""
        while not self._stop_event.is_set():
            time.sleep(_WATCHDOG_POLL)
            with self._lock:
                last = self._last_tick_at
                connected = self._connected

            if not connected or last is None:
                continue

            elapsed = time.monotonic() - last
            if elapsed > self._heartbeat_timeout:
                logger.warning(
                    "CCXTAdapter: heartbeat timeout — no tick for %.1fs (limit %.1fs)",
                    elapsed, self._heartbeat_timeout,
                )
                if self._alerter is not None:
                    try:
                        self._alerter.check_connection_lost(elapsed)
                    except Exception:
                        pass
                if self._on_halt is not None:
                    try:
                        self._on_halt("ccxt_heartbeat_timeout")
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CCXTAdapter":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
