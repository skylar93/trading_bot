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
    "4H": "4h",
    "1D": "1d",
    "2H": "2h",
    "8H": "8h",
    "1W": "1w",
}

# 4 indicators per timeframe
MTF_INDICATOR_SUFFIXES = ["rsi", "macd_signal", "bb_pos", "atr"]


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
        if not _TA_AVAILABLE:  # pragma: no cover
            raise ImportError("Install the 'ta' library: pip install ta")
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
                   Should have a DatetimeIndex.  If it does not, the method
                   attempts to auto-detect a date column and set the index.

        Returns:
            Copy of df_1h with additional columns named
            ``{prefix}_{indicator}`` (e.g. ``4h_rsi``, ``1d_atr``).
        """
        df_indexed = self._ensure_datetime_index(df_1h)
        out = df_1h.copy()

        for tf in self.higher_timeframes:
            rule = _RESAMPLE_RULES.get(tf, tf.lower())
            prefix = _COL_PREFIX.get(tf, tf.lower())
            try:
                tf_df = self._resample(df_indexed, rule)
                if len(tf_df) < 2:
                    logger.warning(
                        "MultiTimeframeFeatures: %s produced only %d rows after resample"
                        " — skipping.", tf, len(tf_df)
                    )
                    continue
                tf_features = self._compute_indicators(tf_df, prefix)
                # Align back to base index: forward-fill only (no look-ahead)
                tf_features_aligned = tf_features.reindex(
                    df_indexed.index, method="ffill"
                )
                # Copy feature values into output using positional alignment
                for col in tf_features_aligned.columns:
                    out[col] = tf_features_aligned[col].values
                logger.info(
                    "MultiTimeframeFeatures: %s → %d bars, %d indicators added",
                    tf, len(tf_df), len(tf_features_aligned.columns),
                )
            except Exception as e:
                logger.warning(
                    "MultiTimeframeFeatures: %s failed (%s) — skipping.", tf, e
                )

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV columns to a higher timeframe."""
        ohlcv_agg = {
            "$open": "first",
            "$high": "max",
            "$low": "min",
            "$close": "last",
            "$volume": "sum",
        }
        present = {k: v for k, v in ohlcv_agg.items() if k in df.columns}
        resampled = (
            df[list(present.keys())]
            .resample(rule)
            .agg(present)
            .dropna(subset=["$close"])
        )
        return resampled

    def _compute_indicators(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Compute RSI, MACD signal, BB position, ATR for a resampled DataFrame."""
        close = df["$close"].astype(float)
        high = df["$high"].astype(float)
        low = df["$low"].astype(float)
        result = pd.DataFrame(index=df.index)

        # RSI(14) → tanh((rsi - 50) / 50)
        try:
            rsi_raw = ta.momentum.RSIIndicator(
                close=close, window=14, fillna=False
            ).rsi()
            result[f"{prefix}_rsi"] = np.tanh((rsi_raw - 50.0) / 50.0)
        except Exception as e:
            logger.warning("RSI failed for %s: %s", prefix, e)
            result[f"{prefix}_rsi"] = 0.0

        # MACD signal line → tanh(signal / rolling_std)
        try:
            macd_obj = ta.trend.MACD(
                close=close, window_slow=26, window_fast=12,
                window_sign=9, fillna=False,
            )
            signal = macd_obj.macd_signal()
            rolling_std = (
                signal.rolling(window=20, min_periods=1).std().replace(0, 1.0)
            )
            result[f"{prefix}_macd_signal"] = np.tanh(signal / rolling_std)
        except Exception as e:
            logger.warning("MACD signal failed for %s: %s", prefix, e)
            result[f"{prefix}_macd_signal"] = 0.0

        # Bollinger Band position: (close - lower) / (upper - lower) → tanh((pos - 0.5) * 4)
        try:
            bb = ta.volatility.BollingerBands(
                close=close, window=20, window_dev=2.0, fillna=False
            )
            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            band_width = (upper - lower).replace(0, np.nan).ffill().fillna(1.0)
            bb_pos = (close - lower) / band_width  # 0 = lower band, 1 = upper band
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

        # Forward-fill NaN (from warm-up period), then zero-fill residuals
        result = result.ffill().fillna(0.0)
        return result

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return df with a DatetimeIndex.

        If df already has a DatetimeIndex, returns it unchanged.
        Otherwise tries to find a date/time column (including 'Unnamed: 0')
        and sets it as the index.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        df = df.copy()

        # Look for a column that likely contains timestamps
        for col in df.columns:
            if any(kw in col.lower() for kw in ("unnamed", "date", "time", "timestamp")):
                try:
                    df.index = pd.to_datetime(df[col])
                    df = df.drop(columns=[col])
                    logger.debug(
                        "MultiTimeframeFeatures: DatetimeIndex set from column '%s'", col
                    )
                    return df
                except Exception:
                    pass

        # Last resort: try the first column
        first_col = df.columns[0]
        try:
            df.index = pd.to_datetime(df[first_col])
            df = df.drop(columns=[first_col])
            return df
        except Exception:
            pass

        logger.warning(
            "MultiTimeframeFeatures: could not detect a DatetimeIndex — "
            "resample will likely fail. Ensure df has a DatetimeIndex."
        )
        return df
