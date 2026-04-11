"""
MockLiveDataSource — simulates a live streaming feed for testing (Week 62, S32).

The "clock" advances one bar at a time via ``tick()``.  Only bars up to the
current tick are visible, mimicking a real-time data source where future data
is unknown.

Week 65 (S47): added ``last_updated_at()`` tracking so that staleness checks
work in tests.  ``tick()`` now records ``time.monotonic()`` as the update
timestamp.  Pass ``max_staleness_sec`` to enable the built-in staleness check
via ``is_stale()``.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from data.sources.base import DataSource


class MockLiveDataSource(DataSource):
    """
    Wraps a static DataFrame but exposes it as if data arrives bar by bar.

    Behaviour:
    - At initialisation the clock is at tick 0 (one bar visible: bar 0).
    - ``tick()`` advances the clock by one bar and records the update time.
    - ``get_window(start, end)`` returns rows in [start, end) **up to
      the current tick** — future bars raise IndexError.
    - ``latest()`` returns the bar at the current tick.
    - ``__len__()`` returns the *total* dataset length (how many bars exist
      in the underlying DataFrame), not the number of visible bars.
      Use ``current_tick`` to know how many bars have been seen.
    - ``is_live()`` returns True.
    - ``last_updated_at()`` returns the monotonic timestamp of the last tick.
    - ``is_stale(max_staleness_sec)`` returns True if the feed has not ticked
      within *max_staleness_sec* seconds.

    Args:
        df: Full OHLCV DataFrame (all bars, pre-loaded).
        initial_tick: Starting tick index (default 0).
        max_staleness_sec: If > 0, ``is_stale()`` will use this threshold.
            Pass 0 (default) to rely on caller-provided threshold.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_tick: int = 0,
        max_staleness_sec: float = 0.0,
    ) -> None:
        if df is None or len(df) == 0:
            raise ValueError("MockLiveDataSource requires a non-empty DataFrame")
        self._df = df.reset_index(drop=True)
        if not (0 <= initial_tick < len(self._df)):
            raise ValueError(
                f"initial_tick={initial_tick} out of range [0, {len(self._df) - 1}]"
            )
        self._tick: int = initial_tick
        self._last_updated_at: float = time.monotonic()
        self.max_staleness_sec: float = max_staleness_sec

    # ------------------------------------------------------------------
    # Clock control
    # ------------------------------------------------------------------

    @property
    def current_tick(self) -> int:
        """Index of the latest visible bar (0-based)."""
        return self._tick

    def tick(self, n: int = 1) -> None:
        """Advance the clock by *n* bars.  Stops at the last bar.

        Records the current monotonic time as ``last_updated_at``.
        """
        self._tick = min(self._tick + n, len(self._df) - 1)
        self._last_updated_at = time.monotonic()

    def reset(self, tick: int = 0) -> None:
        """Reset the clock to *tick* and refresh the update timestamp."""
        if not (0 <= tick < len(self._df)):
            raise ValueError(f"tick={tick} out of range")
        self._tick = tick
        self._last_updated_at = time.monotonic()

    # ------------------------------------------------------------------
    # Week 65 (S47) — staleness interface
    # ------------------------------------------------------------------

    def last_updated_at(self) -> Optional[float]:
        """Return the monotonic timestamp of the most recent ``tick()`` call."""
        return self._last_updated_at

    def is_stale(self, max_staleness_sec: float = 0.0) -> bool:
        """Return True if the feed has not ticked within *max_staleness_sec*
        seconds.  If *max_staleness_sec* is 0, falls back to
        ``self.max_staleness_sec``; if that is also 0, returns False.
        """
        threshold = max_staleness_sec if max_staleness_sec > 0 else self.max_staleness_sec
        if threshold <= 0:
            return False
        return (time.monotonic() - self._last_updated_at) > threshold

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def get_window(self, start: int, end: int) -> pd.DataFrame:
        """Return rows [start, end).  Clamps *end* to current_tick + 1."""
        visible_end = min(end, self._tick + 1)
        if start > visible_end:
            return self._df.iloc[0:0]  # empty frame with correct columns
        return self._df.iloc[start:visible_end]

    def latest(self) -> pd.Series:
        return self._df.iloc[self._tick]

    def __len__(self) -> int:
        """Total bars in the underlying dataset (not just visible ones)."""
        return len(self._df)

    def is_live(self) -> bool:
        return True
