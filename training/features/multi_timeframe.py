"""
Multi-timeframe feature generator.

Aggregates 1H OHLCV data into higher timeframes (4H, 1D) and computes
technical indicators for each timeframe. All features are forward-filled
to the base timeframe (no look-ahead) and tanh-normalized to [-1, 1].

Week 31: 4 indicators × 2 higher timeframes = 8 new feature columns.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Pandas resample rule strings for each supported timeframe label
_RESAMPLE_RULES: dict[str, str] = {
    "4H": "4h",
    "1D": "1D",
    "2H": "2h",
    "8H": "8h",
    "1W": "1W",
}

# Column prefix used in output DataFrame for each timeframe label
_COL_PREFIX: dict[str, str] = {
    "4H": "4H",
    "1D": "1D",
    "2H": "2H",
    "8H": "8H",
    "1W": "1W",
}

# 4 indicators per timeframe
MTF_INDICATOR_SUFFIXES = ["rsi", "macd_signal", "bb_pos", "atr"]

# $-prefixed column map
_OHLCV_DOLLAR = {
    "$open": "open", "$high": "high", "$low": "low",
    "$close": "close", "$volume": "volume",
}


class MultiTimeframeFeatures:
    """
    Generates higher-timeframe features from 1H OHLCV data.

    For each higher timeframe, computes 4 indicators:
    - RSI(14):         momentum oscillator
    - MACD signal(9):  trend direction
    - BB position:     price within Bollinger Bands (20, 2σ)
    - ATR(14):         normalised volatility

    Default: 4 indicators × 2 timeframes (4H, 1D) = 8 new columns.

    All values are forward-filled onto the base (1H) index so there is
    no look-ahead bias.

    Usage::

        mtf = MultiTimeframeFeatures()
        df_with_mtf = mtf.generate(df_1h)  # adds 8 columns to df_1h
    """

    def __init__(
        self,
        base_timeframe: str = "1H",
        higher_timeframes: Optional[List[str]] = None,
    ):
        self.base_timeframe = base_timeframe
        self.higher_timeframes = higher_timeframes or ["4H", "1D"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-timeframe indicator columns to df_1h.

        Args:
            df_1h: OHLCV DataFrame with $open/$high/$low/$close/$volume columns.
                   Should have a DatetimeIndex. If it does not, the method
                   attempts to auto-detect a date column and set the index.

        Returns:
            Copy of df_1h with additional columns named
            ``{prefix}_{indicator}`` (e.g. ``4H_rsi``, ``1D_atr``).
        """
        if _TA_AVAILABLE:
            return self._generate_ta(df_1h)
        else:
            return self._generate_fallback(df_1h)

    # ------------------------------------------------------------------
    # ta-based implementation
    # ------------------------------------------------------------------

    def _generate_ta(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        df_indexed = self._ensure_datetime_index(df_1h)
        out = df_1h.copy()

        for tf in self.higher_timeframes:
            rule = _RESAMPLE_RULES.get(tf, tf.lower())
            prefix = _COL_PREFIX.get(tf, tf)
            try:
                tf_df = self._resample(df_indexed, rule)
                if len(tf_df) < 2:
                    logger.warning(
                        "MultiTimeframeFeatures: %s produced only %d rows — skipping.", tf, len(tf_df)
                    )
                    continue
                tf_features = self._compute_indicators_ta(tf_df, prefix)
                tf_features_aligned = tf_features.reindex(df_indexed.index, method="ffill")
                for col in tf_features_aligned.columns:
                    out[col] = tf_features_aligned[col].values
                logger.info(
                    "MultiTimeframeFeatures: %s → %d bars, %d indicators added",
                    tf, len(tf_df), len(tf_features_aligned.columns),
                )
            except Exception as e:
                logger.warning("MultiTimeframeFeatures: %s failed (%s) — skipping.", tf, e)

        return out

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV columns to a higher timeframe."""
        # Support both $-prefixed and plain column names
        has_dollar = any(c.startswith("$") for c in df.columns)
        if has_dollar:
            agg = {k: v for k, v in {
                "$open": "first", "$high": "max", "$low": "min",
                "$close": "last", "$volume": "sum",
            }.items() if k in df.columns}
        else:
            agg = {k: v for k, v in {
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }.items() if k in df.columns}

        return (
            df[list(agg.keys())]
            .resample(rule)
            .agg(agg)
            .dropna(subset=[list(agg.keys())[3]])   # drop rows with no close
        )

    def _compute_indicators_ta(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Compute RSI, MACD signal, BB position, ATR using `ta` library."""
        has_dollar = any(c.startswith("$") for c in df.columns)
        close = df["$close" if has_dollar else "close"].astype(float)
        high  = df["$high"  if has_dollar else "high"].astype(float)
        low   = df["$low"   if has_dollar else "low"].astype(float)
        result = pd.DataFrame(index=df.index)

        # RSI(14) → tanh((rsi - 50) / 50)
        try:
            rsi_raw = ta.momentum.RSIIndicator(close=close, window=14, fillna=False).rsi()
            result[f"{prefix}_rsi"] = np.tanh((rsi_raw - 50.0) / 50.0)
        except Exception as e:
            logger.warning("RSI failed for %s: %s", prefix, e)
            result[f"{prefix}_rsi"] = 0.0

        # MACD signal → tanh(signal / rolling_std)
        try:
            macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12,
                                     window_sign=9, fillna=False)
            signal = macd_obj.macd_signal()
            rolling_std = signal.rolling(window=20, min_periods=1).std().replace(0, 1.0)
            result[f"{prefix}_macd_signal"] = np.tanh(signal / rolling_std)
        except Exception as e:
            logger.warning("MACD failed for %s: %s", prefix, e)
            result[f"{prefix}_macd_signal"] = 0.0

        # Bollinger Band position → tanh((pos - 0.5) * 4)
        try:
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0, fillna=False)
            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            band_width = (upper - lower).replace(0, np.nan).ffill().fillna(1.0)
            bb_pos = (close - lower) / band_width
            result[f"{prefix}_bb_pos"] = np.tanh((bb_pos - 0.5) * 4.0)
        except Exception as e:
            logger.warning("BB position failed for %s: %s", prefix, e)
            result[f"{prefix}_bb_pos"] = 0.0

        # ATR(14) / close → tanh(norm_atr * 10)
        try:
            atr_raw = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14, fillna=False
            ).average_true_range()
            close_safe = close.replace(0, np.nan).ffill().fillna(1.0)
            norm_atr = atr_raw / close_safe
            result[f"{prefix}_atr"] = np.tanh(norm_atr * 10.0)
        except Exception as e:
            logger.warning("ATR failed for %s: %s", prefix, e)
            result[f"{prefix}_atr"] = 0.0

        return result.ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # Fallback implementation (no ta library)
    # ------------------------------------------------------------------

    def _generate_fallback(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """Pure-numpy fallback when `ta` is not installed."""
        df = df_1h.copy()
        has_dollar = any(c.startswith("$") for c in df.columns)
        close_col = "$close" if has_dollar else "close"
        high_col  = "$high"  if has_dollar else "high"
        low_col   = "$low"   if has_dollar else "low"

        close = df[close_col]
        high  = df[high_col]
        low   = df[low_col]

        for tf in self.higher_timeframes:
            stride = {"4H": 4, "1D": 24, "2H": 2, "8H": 8, "1W": 168}.get(tf, 4)
            prefix = _COL_PREFIX.get(tf, tf)
            n = len(df)

            rsi_full   = self._rsi_numpy(close.values)
            macd_full  = self._macd_signal_numpy(close.values)
            bb_full    = self._bb_pos_numpy(close.values)
            atr_full   = self._atr_numpy(high.values, low.values, close.values)

            # Forward-fill at stride boundaries (no look-ahead)
            rsi_vals = np.full(n, np.nan)
            macd_vals = np.full(n, np.nan)
            bb_vals = np.full(n, np.nan)
            atr_vals = np.full(n, np.nan)

            last = [np.nan, np.nan, np.nan, np.nan]
            for i in range(n):
                if i % stride == 0:
                    last = [rsi_full[i], macd_full[i], bb_full[i], atr_full[i]]
                rsi_vals[i], macd_vals[i], bb_vals[i], atr_vals[i] = last

            df[f"{prefix}_rsi"]         = rsi_vals
            df[f"{prefix}_macd_signal"] = macd_vals
            df[f"{prefix}_bb_pos"]      = bb_vals
            df[f"{prefix}_atr"]         = atr_vals

        return df

    @staticmethod
    def _rsi_numpy(close: np.ndarray, period: int = 14) -> np.ndarray:
        s = pd.Series(close)
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return np.tanh((rsi.values - 50.0) / 50.0)

    @staticmethod
    def _macd_signal_numpy(close: np.ndarray) -> np.ndarray:
        s = pd.Series(close)
        fast = s.ewm(span=12, adjust=False).mean()
        slow = s.ewm(span=26, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False).mean()
        std = signal.rolling(20, min_periods=1).std().replace(0, 1.0)
        return np.tanh((signal / std).values)

    @staticmethod
    def _bb_pos_numpy(close: np.ndarray, period: int = 20) -> np.ndarray:
        s = pd.Series(close)
        ma = s.rolling(period).mean()
        std = s.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        band_width = (upper - lower).replace(0, np.nan)
        bb_pos = (s - lower) / band_width
        return np.tanh(((bb_pos - 0.5) * 4.0).values)

    @staticmethod
    def _atr_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 14) -> np.ndarray:
        s_high = pd.Series(high)
        s_low  = pd.Series(low)
        s_close = pd.Series(close)
        prev_close = s_close.shift(1)
        tr = pd.concat(
            [s_high - s_low, (s_high - prev_close).abs(), (s_low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        close_safe = s_close.replace(0, np.nan).ffill().fillna(1.0)
        return np.tanh((atr / close_safe * 10.0).values)

    # ------------------------------------------------------------------
    # Datetime index helper
    # ------------------------------------------------------------------

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        df = df.copy()
        for col in df.columns:
            if any(kw in col.lower() for kw in ("unnamed", "date", "time", "timestamp")):
                try:
                    df.index = pd.to_datetime(df[col])
                    return df.drop(columns=[col])
                except Exception:
                    pass
        try:
            df.index = pd.to_datetime(df.iloc[:, 0])
            return df.drop(columns=[df.columns[0]])
        except Exception:
            pass
        logger.warning("MultiTimeframeFeatures: could not detect a DatetimeIndex.")
        return df
