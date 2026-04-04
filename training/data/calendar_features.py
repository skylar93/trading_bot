"""
Calendar and seasonality feature engine for financial/crypto trading.

Encodes temporal patterns known to influence market behavior:
time-of-day, day-of-week, month-end effects, quarterly rebalancing,
options/futures expiry, FOMC dates, and Bitcoin-specific cycles.

Features (8 per timestep, all in [-1, 1] or [0, 1]):
    1.  session_sin    — sin encoding of intraday position (24h cycle)
    2.  session_cos    — cos encoding of intraday position
    3.  session_flag   — active trading session {-1=Asia, 0=Europe, +1=US}
    4.  dow_sin        — sin encoding of day-of-week (Mon=0 … Sun=6)
    5.  dow_cos        — cos encoding of day-of-week
    6.  month_end_prox — proximity to month-end rebalancing [0, 1]
    7.  quarter_end_prox — proximity to quarter-end  [0, 1]
    8.  event_flag     — upcoming high-impact event {FOMC, CME expiry, halving}

All timestamps are handled in UTC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of calendar features
N_CALENDAR_FEATURES = 8

CALENDAR_COLS = [
    "session_sin",
    "session_cos",
    "session_flag",
    "dow_sin",
    "dow_cos",
    "month_end_prox",
    "quarter_end_prox",
    "event_flag",
]

# CME BTC futures expiry: last Friday of each month (approximate dates 2022-2026)
# Format: "YYYY-MM-DD"
_CME_EXPIRY_DATES: List[str] = [
    # 2023
    "2023-01-27", "2023-02-24", "2023-03-31", "2023-04-28",
    "2023-05-26", "2023-06-30", "2023-07-28", "2023-08-25",
    "2023-09-29", "2023-10-27", "2023-11-24", "2023-12-29",
    # 2024
    "2024-01-26", "2024-02-23", "2024-03-29", "2024-04-26",
    "2024-05-31", "2024-06-28", "2024-07-26", "2024-08-30",
    "2024-09-27", "2024-10-25", "2024-11-29", "2024-12-27",
    # 2025
    "2025-01-31", "2025-02-28", "2025-03-28", "2025-04-25",
    "2025-05-30", "2025-06-27", "2025-07-25", "2025-08-29",
    "2025-09-26", "2025-10-31", "2025-11-28", "2025-12-26",
    # 2026
    "2026-01-30", "2026-02-27", "2026-03-27", "2026-04-24",
    "2026-05-29", "2026-06-26", "2026-07-31", "2026-08-28",
    "2026-09-25", "2026-10-30", "2026-11-27", "2026-12-25",
]

# FOMC meeting dates (approximate, Fed announces ~8 per year)
_FOMC_DATES: List[str] = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# Bitcoin halving dates (historical + projected)
_BTC_HALVING_DATES: List[str] = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
    "2028-03-01",  # projected
]

# US market trading hours in UTC
_US_MARKET_OPEN_UTC = 13   # 9:30 AM ET ≈ 13:30 UTC (ignoring DST)
_US_MARKET_CLOSE_UTC = 20  # 4:00 PM ET ≈ 20:00 UTC

# Session boundaries (UTC hours)
_SESSION_BOUNDARIES = {
    "asia":   (0, 8),    # Tokyo / Hong Kong / Singapore
    "europe": (7, 16),   # London / Frankfurt
    "us":     (13, 22),  # New York
}


@dataclass
class CalendarConfig:
    """Configuration for the calendar feature engine."""
    # Event look-ahead window (how many days counts as "upcoming")
    event_lookahead_days: int = 0
    # Proximity decay rate for month/quarter end
    # prox = exp(-k * days_to_end); k controls steepness
    month_end_decay: float = 0.3
    quarter_end_decay: float = 0.2
    # Halving cycle encoding: include position within 4-year cycle
    include_halving_cycle: bool = True
    # Custom additional FOMC dates (extend the built-in list)
    extra_fomc_dates: List[str] = field(default_factory=list)
    # Custom CME expiry dates (extend the built-in list)
    extra_cme_dates: List[str] = field(default_factory=list)


class CalendarFeatureEngine:
    """
    Encodes calendar and seasonality patterns as normalized features.

    Usage::

        engine = CalendarFeatureEngine(config)
        df_features = engine.compute(price_df)   # (T, 8) DataFrame
        matrix = engine.get_feature_matrix(df_features)  # (T, 8) float32
    """

    def __init__(self, config: Optional[CalendarConfig] = None):
        self.cfg = config or CalendarConfig()
        self._fomc = self._parse_dates(
            _FOMC_DATES + self.cfg.extra_fomc_dates
        )
        self._cme = self._parse_dates(
            _CME_EXPIRY_DATES + self.cfg.extra_cme_dates
        )
        self._halvings = self._parse_dates(_BTC_HALVING_DATES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute calendar features for each row in *price_df*.

        *price_df* must have a DatetimeIndex (UTC preferred; naive interpreted as UTC).
        Returns a DataFrame with CALENDAR_COLS columns.
        """
        index = self._ensure_utc(price_df.index)

        df = pd.DataFrame(index=price_df.index, columns=CALENDAR_COLS, dtype=np.float32)

        df["session_sin"]     = self._session_sin(index)
        df["session_cos"]     = self._session_cos(index)
        df["session_flag"]    = self._session_flag(index)
        df["dow_sin"]         = self._dow_sin(index)
        df["dow_cos"]         = self._dow_cos(index)
        df["month_end_prox"]  = self._month_end_prox(index)
        df["quarter_end_prox"] = self._quarter_end_prox(index)
        df["event_flag"]      = self._event_flag(index)

        return df.astype(np.float32)

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (T, 8) float32 array from a DataFrame with CALENDAR_COLS."""
        cols = [c for c in CALENDAR_COLS if c in df.columns]
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return df[cols].values.astype(np.float32)

    def get_latest(self, ts: Optional[pd.Timestamp] = None) -> np.ndarray:
        """Return calendar features for a single timestamp (default: now UTC)."""
        if ts is None:
            ts = pd.Timestamp.utcnow()
        dummy = pd.DataFrame(index=pd.DatetimeIndex([ts]))
        return self.compute(dummy).values[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Feature implementations
    # ------------------------------------------------------------------

    def _session_sin(self, index: pd.DatetimeIndex) -> pd.Series:
        """sin encoding of hour-of-day (24h cycle)."""
        hour = index.hour + index.minute / 60.0
        return pd.Series(
            np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
            index=index,
        )

    def _session_cos(self, index: pd.DatetimeIndex) -> pd.Series:
        """cos encoding of hour-of-day (24h cycle)."""
        hour = index.hour + index.minute / 60.0
        return pd.Series(
            np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
            index=index,
        )

    def _session_flag(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Active trading session flag.
        -1 = Asia-dominant, 0 = Europe-dominant, +1 = US-dominant.
        Overlap hours get the later/more-liquid session.
        """
        hour = index.hour
        flags = np.where(
            (hour >= _SESSION_BOUNDARIES["us"][0]) & (hour < _SESSION_BOUNDARIES["us"][1]),
            1.0,
            np.where(
                (hour >= _SESSION_BOUNDARIES["europe"][0]) & (hour < _SESSION_BOUNDARIES["europe"][1]),
                0.0,
                -1.0,
            )
        )
        return pd.Series(flags.astype(np.float32), index=index)

    def _dow_sin(self, index: pd.DatetimeIndex) -> pd.Series:
        """sin encoding of day-of-week (Mon=0 … Sun=6)."""
        dow = index.dayofweek.astype(float)
        return pd.Series(
            np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
            index=index,
        )

    def _dow_cos(self, index: pd.DatetimeIndex) -> pd.Series:
        """cos encoding of day-of-week."""
        dow = index.dayofweek.astype(float)
        return pd.Series(
            np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
            index=index,
        )

    def _month_end_prox(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Proximity to month-end rebalancing.
        1.0 on last day of month, decays exponentially going back.
        """
        # days_to_month_end: number of days until last day of month
        month_end = pd.to_datetime(
            index.to_series().apply(lambda ts: ts + pd.offsets.MonthEnd(0))
        )
        days_to_end = (month_end.values - index.values) / np.timedelta64(1, 'D')
        days_to_end = np.maximum(days_to_end, 0.0)
        prox = np.exp(-self.cfg.month_end_decay * days_to_end)
        return pd.Series(prox.astype(np.float32), index=index)

    def _quarter_end_prox(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Proximity to quarter-end (Mar/Jun/Sep/Dec last day).
        1.0 on last day of quarter, decays exponentially.
        """
        quarter_end = pd.to_datetime(
            index.to_series().apply(lambda ts: ts + pd.offsets.QuarterEnd(0))
        )
        days_to_end = (quarter_end.values - index.values) / np.timedelta64(1, 'D')
        days_to_end = np.maximum(days_to_end, 0.0)
        prox = np.exp(-self.cfg.quarter_end_decay * days_to_end)
        return pd.Series(prox.astype(np.float32), index=index)

    def _event_flag(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        High-impact event proximity flag in [-1, 1].

        Positive: upcoming FOMC (+0.7) or CME expiry (+0.5) within lookahead.
        Negative: post-event (day after) → -0.3 (volatility decay).
        Zero: no nearby event.
        Bitcoin halving proximity: +1.0 within 7 days.
        """
        lookahead = pd.Timedelta(days=self.cfg.event_lookahead_days)
        if self.cfg.event_lookahead_days <= 0:
            # Only encode same-day and post-event signals (no future lookahead)
            lookahead = pd.Timedelta(0)
        postvent = pd.Timedelta(days=1)
        halving_window = pd.Timedelta(days=7)

        dates = index.normalize()  # strip time, keep date
        flags = np.zeros(len(index), dtype=np.float32)

        for i, d in enumerate(dates):
            val = 0.0

            # Check FOMC
            for fd in self._fomc:
                delta = fd - d
                if pd.Timedelta(0) <= delta <= lookahead:
                    ratio = float(delta / lookahead) if lookahead > pd.Timedelta(0) else 0.0
                    val = max(val, 0.7 * (1.0 - ratio))
                elif -postvent <= delta < pd.Timedelta(0):
                    val = min(val, -0.3)

            # Check CME expiry
            for cd in self._cme:
                delta = cd - d
                if pd.Timedelta(0) <= delta <= lookahead:
                    ratio = float(delta / lookahead) if lookahead > pd.Timedelta(0) else 0.0
                    val = max(val, 0.5 * (1.0 - ratio))

            # Check halving proximity
            # Use signed delta to avoid look-ahead: only fire for upcoming halving
            # or 1-day post-halving window (already happened).
            for hd in self._halvings:
                delta = hd - d  # positive = upcoming, negative = already happened
                if pd.Timedelta(0) <= delta <= halving_window:
                    # Upcoming halving: +1.0 decaying as event approaches
                    val = max(val, 1.0 * (1.0 - delta / halving_window))
                elif -pd.Timedelta(days=1) <= delta < pd.Timedelta(0):
                    # Day after halving: mild positive signal (event just occurred)
                    val = max(val, 0.5)

            flags[i] = float(np.clip(val, -1.0, 1.0))

        return pd.Series(flags, index=index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dates(date_strs: List[str]) -> List[pd.Timestamp]:
        result = []
        for s in date_strs:
            try:
                result.append(pd.Timestamp(s, tz="UTC").normalize())
            except Exception:
                pass
        return result

    @staticmethod
    def _ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Ensure DatetimeIndex is UTC-aware."""
        if index.tz is None:
            return index.tz_localize("UTC")
        return index.tz_convert("UTC")
