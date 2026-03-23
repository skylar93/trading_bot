"""
Multi-Timeframe Feature Generator.

Aggregates 1H OHLCV data into higher timeframes (4H, 1D) and computes
technical indicators on each.  All values are forward-filled from the
higher timeframe back to the 1H index with NO look-ahead bias.

Usage
-----
    from training.features.multi_timeframe import MultiTimeframeFeatures
    import pandas as pd

    df_1h = pd.read_csv("data/btc_1h.csv", parse_dates=["date"], index_col="date")
    mtf = MultiTimeframeFeatures()
    df_out = mtf.generate(df_1h)   # adds 4H_rsi, 4H_macd_signal, ... columns
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column name aliases: the env uses $-prefixed names
_OHLCV_MAP = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
}


def _ensure_plain_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with $ prefixes stripped from column names."""
    return df.rename(columns=_OHLCV_MAP)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_signal(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line.ewm(span=signal, adjust=False).mean()


def _bb_position(series: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger Band position: (price - lower) / (upper - lower), in [0, 1]."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    band_width = upper - lower
    pos = (series - lower) / band_width.replace(0, np.nan)
    return pos.clip(0.0, 1.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _compute_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute RSI, MACD-signal, BB-position, ATR for a given OHLCV DataFrame."""
    c = df["close"]
    result = pd.DataFrame(index=df.index)
    result[f"{prefix}_rsi"] = _rsi(c)
    result[f"{prefix}_macd_signal"] = _macd_signal(c)
    result[f"{prefix}_bb_pos"] = _bb_position(c)
    result[f"{prefix}_atr"] = _atr(df["high"], df["low"], c)
    return result


# Resample rule → label
_RESAMPLE_RULES = {
    "4H": "4h",
    "1D": "1D",
}

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


class MultiTimeframeFeatures:
    """Generate higher-timeframe technical indicator features from 1H OHLCV data.

    Parameters
    ----------
    base_timeframe : str
        Label of the input data timeframe (informational only).
    higher_timeframes : list of str
        Timeframes to aggregate up to.  Supported: "4H", "1D".
    """

    def __init__(
        self,
        base_timeframe: str = "1H",
        higher_timeframes: List[str] | None = None,
    ) -> None:
        self.base_timeframe = base_timeframe
        self.higher_timeframes: List[str] = higher_timeframes or ["4H", "1D"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """Attach higher-timeframe features to a 1H DataFrame.

        Parameters
        ----------
        df_1h : pd.DataFrame
            1H OHLCV data.  Must contain either ``$open / $high / $low /
            $close / $volume`` columns **or** ``open / high / low / close /
            volume`` columns.  The index is used for resampling; if it is not
            a DatetimeIndex a simple integer range is used instead (unit-test
            friendly).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional columns named
            ``<TF>_rsi``, ``<TF>_macd_signal``, ``<TF>_bb_pos``, ``<TF>_atr``
            for each requested higher timeframe.

        Notes
        -----
        All higher-timeframe values are forward-filled from the bar that
        **closed** just before or at the 1H bar, ensuring no look-ahead.
        """
        df = df_1h.copy()

        # Normalise column names
        has_dollar = any(c.startswith("$") for c in df.columns)
        if has_dollar:
            plain = _ensure_plain_columns(df)
            # Keep original columns in the output; work on a plain copy
        else:
            plain = df.rename(
                columns={
                    c: c.lower()
                    for c in df.columns
                    if c.lower() in {"open", "high", "low", "close", "volume"}
                }
            )

        required = {"open", "high", "low", "close"}
        missing = required - set(plain.columns)
        if missing:
            raise ValueError(f"MultiTimeframeFeatures: missing columns {missing}")

        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

        for tf in self.higher_timeframes:
            rule = _RESAMPLE_RULES.get(tf)
            if rule is None:
                logger.warning("Unsupported timeframe '%s' — skipping.", tf)
                continue

            if has_datetime_index:
                htf_df = (
                    plain[list(required | {"volume"})]
                    .resample(rule, label="right", closed="right")
                    .agg(
                        {
                            k: v
                            for k, v in _OHLCV_AGG.items()
                            if k in plain.columns
                        }
                    )
                    .dropna(how="all")
                )
                htf_indicators = _compute_indicators(htf_df, prefix=tf)
                # Reindex to 1H and forward-fill (no look-ahead)
                htf_ff = htf_indicators.reindex(df.index, method="ffill")
            else:
                # No DatetimeIndex: use a simple stride-based pseudo-aggregation
                htf_ff = self._pseudo_aggregate(plain, tf, df.index)

            df = pd.concat([df, htf_ff], axis=1)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pseudo_aggregate(
        self,
        plain: pd.DataFrame,
        tf: str,
        orig_index: pd.Index,
    ) -> pd.DataFrame:
        """Fallback aggregation for DataFrames without a DatetimeIndex.

        Uses a rolling window whose length approximates the higher timeframe
        (4 bars for 4H, 24 bars for 1D) and computes indicators on that
        window.  Values are computed at each 1H bar using only past data.
        """
        stride = {"4H": 4, "1D": 24}.get(tf, 4)
        n = len(plain)
        result = pd.DataFrame(index=orig_index)
        prefix = tf

        close = plain["close"].values
        high = plain["high"].values
        low = plain["low"].values

        rsi_vals = np.full(n, np.nan)
        macd_vals = np.full(n, np.nan)
        bb_vals = np.full(n, np.nan)
        atr_vals = np.full(n, np.nan)

        # Compute on the full series then sample every `stride` bars
        close_s = pd.Series(close)
        high_s = pd.Series(high)
        low_s = pd.Series(low)

        rsi_full = _rsi(close_s).values
        macd_full = _macd_signal(close_s).values
        bb_full = _bb_position(close_s).values
        atr_full = _atr(high_s, low_s, close_s).values

        # Forward-fill at stride boundaries (no look-ahead)
        last_vals = [np.nan, np.nan, np.nan, np.nan]
        for i in range(n):
            if i % stride == 0:
                last_vals = [rsi_full[i], macd_full[i], bb_full[i], atr_full[i]]
            rsi_vals[i] = last_vals[0]
            macd_vals[i] = last_vals[1]
            bb_vals[i] = last_vals[2]
            atr_vals[i] = last_vals[3]

        result[f"{prefix}_rsi"] = rsi_vals
        result[f"{prefix}_macd_signal"] = macd_vals
        result[f"{prefix}_bb_pos"] = bb_vals
        result[f"{prefix}_atr"] = atr_vals
        return result
