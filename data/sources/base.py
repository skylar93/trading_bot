"""
DataSource abstraction layer (Week 61 — S27 skeleton, Week 62 — full expansion).

Provides a uniform interface for data access so that
SingleAssetRLTradingEnv and other consumers are decoupled from the
concrete data backend (in-memory DataFrame, CSV file, live feed, etc.).

Week 65 (S47): added optional staleness interface — ``last_updated_at()`` and
``is_stale()``.  Live sources override these; static sources are never stale.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataSource(ABC):
    """Abstract data source consumed by trading environments."""

    @abstractmethod
    def get_window(self, start: int, end: int) -> pd.DataFrame:
        """Return rows [start, end) as a DataFrame (integer-indexed)."""

    @abstractmethod
    def latest(self) -> pd.Series:
        """Return the last row as a Series."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of rows available."""

    @abstractmethod
    def is_live(self) -> bool:
        """Return True for streaming / live data sources."""

    # ------------------------------------------------------------------
    # Week 65 (S47) — optional staleness interface
    # ------------------------------------------------------------------

    def last_updated_at(self) -> Optional[float]:
        """Return the wall-clock time (``time.monotonic()``) of the most recent
        data update, or ``None`` if the source has never been updated.

        Static (non-live) sources return ``None`` by default; live sources
        should override this to expose their actual update timestamp.
        """
        return None

    def is_stale(self, max_staleness_sec: float) -> bool:
        """Return True if the feed has not been updated within
        *max_staleness_sec* seconds.

        Always returns False for static sources (``last_updated_at() is None``).
        """
        ts = self.last_updated_at()
        if ts is None:
            return False
        return (time.monotonic() - ts) > max_staleness_sec


class StaticDataSource(DataSource):
    """
    In-memory DataFrame wrapper.  Preserves the existing behavior of passing
    ``data=df`` to the environment while exposing the DataSource interface.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            raise ValueError("StaticDataSource requires a non-empty DataFrame")
        self._df = df.reset_index(drop=True)

    @property
    def df(self) -> pd.DataFrame:
        """Direct access to the underlying DataFrame (read-only contract)."""
        return self._df

    def get_window(self, start: int, end: int) -> pd.DataFrame:
        return self._df.iloc[start:end]

    def latest(self) -> pd.Series:
        return self._df.iloc[-1]

    def __len__(self) -> int:
        return len(self._df)

    def is_live(self) -> bool:
        return False
