"""
CCXTLiveDataSource — Week 72 (F2)

Wraps a CCXTAdapter (WebSocket feed) to implement the DataSource contract
used by PaperTrader and SingleAssetRLTradingEnv.  Data arrives as OHLCV bars
pushed from the adapter callback; the buffer is thread-safe.

Staleness tracking (S47) is integrated: ``last_updated_at()`` and
``is_stale()`` work identically to MockLiveDataSource.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import List, Optional

import pandas as pd

from data.sources.base import DataSource

logger = logging.getLogger(__name__)

# Column names for live OHLCV bars.
_ENV_COLS: List[str] = ["open", "high", "low", "close", "volume"]

# Minimum bar length expected from CCXT: [timestamp, open, high, low, close, volume]
_MIN_BAR_LEN = 6


class CCXTLiveDataSource(DataSource):
    """
    Live OHLCV data source backed by a CCXTAdapter WebSocket feed.

    Parameters
    ----------
    adapter :
        A CCXTAdapter (or compatible mock) that provides
        ``add_ohlcv_callback(fn)``.
    max_bars : int
        Rolling buffer depth (default 500).
    max_staleness_sec : float
        If > 0, ``is_stale()`` uses this as the default threshold.
        Pass 0 to require an explicit threshold per call.
    """

    def __init__(
        self,
        adapter,
        max_bars: int = 500,
        max_staleness_sec: float = 0.0,
    ) -> None:
        self._buffer: deque = deque(maxlen=max_bars)
        self._lock = threading.Lock()
        self._last_updated_at: Optional[float] = None
        self.max_staleness_sec = max_staleness_sec

        adapter.add_ohlcv_callback(self._on_ohlcv)

    # ------------------------------------------------------------------
    # Callback from CCXTAdapter
    # ------------------------------------------------------------------

    def _on_ohlcv(self, bars: list) -> None:
        """Receive a batch of CCXT OHLCV bars and append to the buffer."""
        with self._lock:
            for bar in bars:
                if len(bar) < _MIN_BAR_LEN:
                    logger.debug("_on_ohlcv: skipping malformed bar (len=%d)", len(bar))
                    continue
                _, o, h, l, c, v = bar[0], bar[1], bar[2], bar[3], bar[4], bar[5]
                self._buffer.append([float(o), float(h), float(l), float(c), float(v)])
            if bars:
                self._last_updated_at = time.monotonic()

    # ------------------------------------------------------------------
    # S47 — Staleness interface (same contract as MockLiveDataSource)
    # ------------------------------------------------------------------

    def last_updated_at(self) -> Optional[float]:
        """Monotonic timestamp of the last received bar, or None if no data."""
        with self._lock:
            return self._last_updated_at

    def is_stale(self, max_staleness_sec: float = 0.0) -> bool:
        """True if the feed hasn't updated within *max_staleness_sec* seconds.

        Falls back to ``self.max_staleness_sec`` when the argument is 0.
        Returns False if both thresholds are 0 (staleness check disabled).
        """
        threshold = max_staleness_sec if max_staleness_sec > 0 else self.max_staleness_sec
        if threshold <= 0:
            return False
        with self._lock:
            last = self._last_updated_at
        if last is None:
            return True
        return (time.monotonic() - last) > threshold

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def is_live(self) -> bool:
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def latest(self) -> pd.Series:
        """Return the most recent bar as a Series.

        Raises RuntimeError if no data has been received yet.
        """
        with self._lock:
            if not self._buffer:
                raise RuntimeError("CCXTLiveDataSource: no data received yet")
            row = self._buffer[-1]
        return pd.Series(row, index=_ENV_COLS, dtype=float)

    def get_window(self, start: int, end: int) -> pd.DataFrame:
        """Return bars [start, end) as a DataFrame.

        Clamps *start* to 0 and *end* to ``len(self)``.
        Returns an empty DataFrame with correct columns if out of range.
        """
        with self._lock:
            rows = list(self._buffer)
        n = len(rows)
        s = max(0, start)
        e = min(end, n)
        if s >= e:
            return pd.DataFrame(columns=_ENV_COLS)
        return pd.DataFrame(rows[s:e], columns=_ENV_COLS, dtype=float)
