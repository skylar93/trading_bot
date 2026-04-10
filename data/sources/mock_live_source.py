"""
MockLiveDataSource — simulates a live streaming feed for testing (Week 62, S32).

The "clock" advances one bar at a time via ``tick()``.  Only bars up to the
current tick are visible, mimicking a real-time data source where future data
is unknown.
"""

from __future__ import annotations

import pandas as pd

from data.sources.base import DataSource


class MockLiveDataSource(DataSource):
    """
    Wraps a static DataFrame but exposes it as if data arrives bar by bar.

    Behaviour:
    - At initialisation the clock is at tick 0 (one bar visible: bar 0).
    - ``tick()`` advances the clock by one bar.
    - ``get_window(start, end)`` returns rows in [start, end) **up to
      the current tick** — future bars raise IndexError.
    - ``latest()`` returns the bar at the current tick.
    - ``__len__()`` returns the *total* dataset length (how many bars exist
      in the underlying DataFrame), not the number of visible bars.
      Use ``current_tick`` to know how many bars have been seen.
    - ``is_live()`` returns True.

    Args:
        df: Full OHLCV DataFrame (all bars, pre-loaded).
        initial_tick: Starting tick index (default 0).
    """

    def __init__(self, df: pd.DataFrame, initial_tick: int = 0) -> None:
        if df is None or len(df) == 0:
            raise ValueError("MockLiveDataSource requires a non-empty DataFrame")
        self._df = df.reset_index(drop=True)
        if not (0 <= initial_tick < len(self._df)):
            raise ValueError(
                f"initial_tick={initial_tick} out of range [0, {len(self._df) - 1}]"
            )
        self._tick: int = initial_tick

    # ------------------------------------------------------------------
    # Clock control
    # ------------------------------------------------------------------

    @property
    def current_tick(self) -> int:
        """Index of the latest visible bar (0-based)."""
        return self._tick

    def tick(self, n: int = 1) -> None:
        """Advance the clock by *n* bars.  Stops at the last bar."""
        self._tick = min(self._tick + n, len(self._df) - 1)

    def reset(self, tick: int = 0) -> None:
        """Reset the clock to *tick*."""
        if not (0 <= tick < len(self._df)):
            raise ValueError(f"tick={tick} out of range")
        self._tick = tick

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
