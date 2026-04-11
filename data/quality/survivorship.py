"""
Survivorship bias checker (Week 65, S49).

In backtesting, datasets that only contain currently-active instruments
over-represent survivors and under-represent assets that were delisted,
halted, or restructured during the period under test.  This leads to
inflated backtest performance.

This module provides a lightweight warning utility.  It does **not** halt
the backtest — it only reports potential bias to the caller (who may log
or print).

Checks implemented
------------------
1. ``short_history``:
   Dataset has fewer rows than *min_lookback_bars*.  Suggests the asset
   may have been recently listed and therefore lacks a representative
   history of different market regimes.

2. ``late_start``:
   Dataset start date is after *expected_start*.  If a backtest universe
   was selected based on assets that existed at the *end* of the period,
   some assets may appear to start mid-way through — a classic
   survivorship-bias pattern.

3. ``single_asset_universe``:
   A generic reminder (severity=info) that single-asset backtests
   implicitly select on the chosen asset's continued existence, which
   is itself a form of survivorship bias.

Usage::

    from data.quality.survivorship import SurvivorshipBiasChecker, check_survivorship

    checker = SurvivorshipBiasChecker(min_lookback_bars=200)
    warnings = checker.check(df, expected_start="2020-01-01")
    for w in warnings:
        print(w)

    # Module-level shortcut
    warnings = check_survivorship(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BiasWarning:
    """Describes a potential survivorship-bias issue in a dataset."""

    kind: str          # 'short_history' | 'late_start' | 'single_asset_universe'
    severity: str      # 'warning' | 'info'
    detail: str        # human-readable explanation

    def __str__(self) -> str:
        return f"[survivorship_bias/{self.kind}] ({self.severity}) {self.detail}"


class SurvivorshipBiasChecker:
    """
    Checks a dataset for potential survivorship bias and returns warnings.

    Parameters
    ----------
    min_lookback_bars : int
        Minimum number of rows expected before the backtest start.  If
        ``0`` (default), the ``short_history`` check is skipped.
    warn_single_asset : bool
        If True (default) emit a generic ``single_asset_universe`` info
        warning on every check, reminding the user that single-asset
        backtests implicitly condition on the asset's survival.
    """

    def __init__(
        self,
        min_lookback_bars: int = 0,
        warn_single_asset: bool = True,
    ) -> None:
        if min_lookback_bars < 0:
            raise ValueError(
                f"min_lookback_bars must be >= 0, got {min_lookback_bars}"
            )
        self.min_lookback_bars = min_lookback_bars
        self.warn_single_asset = warn_single_asset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        df: pd.DataFrame,
        expected_start: Optional[Union[str, pd.Timestamp]] = None,
        symbol: Optional[str] = None,
    ) -> List[BiasWarning]:
        """
        Run all checks and return any warnings found.  Never raises.

        Parameters
        ----------
        df :
            OHLCV DataFrame to inspect.
        expected_start :
            The earliest date the dataset should contain.  If the first
            timestamp in *df* is after this date, a ``late_start`` warning
            is emitted.  Accepts a string parseable by ``pd.Timestamp`` or
            an existing ``pd.Timestamp``.
        symbol :
            Optional symbol name for richer warning messages.
        """
        warnings: List[BiasWarning] = []

        if df is None or len(df) == 0:
            warnings.append(
                BiasWarning(
                    kind="short_history",
                    severity="warning",
                    detail="Dataset is empty; cannot perform bias checks.",
                )
            )
            return warnings

        asset = symbol or "dataset"

        # ── check 1: short history ─────────────────────────────────────
        if self.min_lookback_bars > 0 and len(df) < self.min_lookback_bars:
            warnings.append(
                BiasWarning(
                    kind="short_history",
                    severity="warning",
                    detail=(
                        f"{asset} has only {len(df)} rows; "
                        f"minimum recommended lookback is {self.min_lookback_bars} bars.  "
                        "Asset may be recently listed — backtest history may not "
                        "represent multiple market regimes."
                    ),
                )
            )

        # ── check 2: late start ────────────────────────────────────────
        if expected_start is not None:
            first_ts = self._first_timestamp(df)
            if first_ts is not None:
                try:
                    exp_ts = pd.Timestamp(expected_start)
                    if first_ts > exp_ts:
                        delta_days = (first_ts - exp_ts).days
                        warnings.append(
                            BiasWarning(
                                kind="late_start",
                                severity="warning",
                                detail=(
                                    f"{asset} data starts at {first_ts.date()} "
                                    f"but expected_start={exp_ts.date()} "
                                    f"({delta_days} day(s) of missing history).  "
                                    "If this asset was selected because it existed "
                                    "at the end of the backtest window, this is "
                                    "survivorship bias."
                                ),
                            )
                        )
                except Exception:
                    pass  # unparseable expected_start — skip check

        # ── check 3: single-asset universe reminder ────────────────────
        if self.warn_single_asset:
            warnings.append(
                BiasWarning(
                    kind="single_asset_universe",
                    severity="info",
                    detail=(
                        f"Backtest runs on a single asset ({asset}).  "
                        "The choice of this asset implicitly selects an instrument "
                        "that survived to the present — a mild form of survivorship "
                        "bias.  Consider multi-asset validation for robustness."
                    ),
                )
            )

        return warnings

    def log_warnings(
        self,
        df: pd.DataFrame,
        expected_start: Optional[Union[str, pd.Timestamp]] = None,
        symbol: Optional[str] = None,
    ) -> List[BiasWarning]:
        """Run checks and emit each warning via the module logger.

        Returns the same list as ``check()`` so callers can inspect them.
        """
        warnings = self.check(df, expected_start=expected_start, symbol=symbol)
        for w in warnings:
            if w.severity == "warning":
                logger.warning("%s", w)
            else:
                logger.info("%s", w)
        return warnings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
        """Try to extract the first timestamp from the DataFrame index or a
        recognised timestamp column.  Returns None if no datetime axis found.
        """
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            return pd.Timestamp(df.index[0])

        _TIME_COLS = {"timestamp", "time", "date", "datetime"}
        for col in df.columns:
            if col.lower() in _TIME_COLS:
                try:
                    return pd.Timestamp(pd.to_datetime(df[col].iloc[0]))
                except Exception:
                    pass
        return None


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_CHECKER = SurvivorshipBiasChecker()


def check_survivorship(
    df: pd.DataFrame,
    expected_start: Optional[Union[str, pd.Timestamp]] = None,
    symbol: Optional[str] = None,
    min_lookback_bars: int = 0,
) -> List[BiasWarning]:
    """Module-level shortcut: return survivorship bias warnings for *df*.

    Parameters
    ----------
    df :
        OHLCV DataFrame to inspect.
    expected_start :
        If provided, warns when the dataset starts after this date.
    symbol :
        Optional asset symbol for richer messages.
    min_lookback_bars :
        Minimum bar count; 0 disables the short-history check.
    """
    checker = SurvivorshipBiasChecker(
        min_lookback_bars=min_lookback_bars,
        warn_single_asset=True,
    )
    return checker.check(df, expected_start=expected_start, symbol=symbol)
