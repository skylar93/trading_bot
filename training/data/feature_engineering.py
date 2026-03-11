"""
Technical indicator feature engineering for trading environments.

All indicators are:
- Computed without look-ahead (shift(1) applied where needed)
- Normalized to [-1, 1] via tanh
- NaN-filled with 0 (safe default in tanh space)

Requires: ta>=0.10.0 (pip install ta)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Feature column names produced by FeatureEngineer
FEATURE_COLS = ["rsi", "macd", "bb_width", "atr", "obv", "vwap_dev"]


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Toggle individual indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_obv: bool = True
    use_vwap: bool = True

    # Indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Normalisation scales (passed to tanh)
    rsi_scale: float = 50.0       # RSI 0-100 → centre at 50 → /50 → tanh input
    macd_scale: float = 1.0       # will be auto-scaled by rolling std
    atr_scale: float = 1.0        # will be auto-scaled by close price
    obv_scale: float = 1.0        # will be auto-scaled by rolling std
    vwap_scale: float = 1.0       # will be auto-scaled by close price

    # Which indicators to include in the final feature matrix (ordered)
    enabled_features: List[str] = field(default_factory=lambda: list(FEATURE_COLS))


class FeatureEngineer:
    """
    Computes technical indicators and normalises them to [-1, 1].

    Usage::

        fe = FeatureEngineer(config)
        df_with_features = fe.compute_features(df)   # adds 6 new columns
        feature_matrix = fe.get_feature_matrix(df_with_features)  # (T, n_features)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        if not _TA_AVAILABLE:  # pragma: no cover
            raise ImportError("Install the 'ta' library: pip install ta")
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicator columns to *df* (returns a copy).

        All new columns are in the set defined by FEATURE_COLS.
        Pre-existing columns are preserved.
        """
        _require_ohlcv(df)
        out = df.copy()
        cfg = self.config

        if cfg.use_rsi:
            out = self._add_rsi(out)
        if cfg.use_macd:
            out = self._add_macd(out)
        if cfg.use_bollinger:
            out = self._add_bollinger(out)
        if cfg.use_atr:
            out = self._add_atr(out)
        if cfg.use_obv:
            out = self._add_obv(out)
        if cfg.use_vwap:
            out = self._add_vwap(out)

        # Forward-fill any residual NaN, then zero-fill
        for col in FEATURE_COLS:
            if col in out.columns:
                out[col] = out[col].ffill().fillna(0.0)

        return out

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a (T, n_features) float32 array for the enabled features.

        *df* must have been processed by :meth:`compute_features` first.
        """
        cols = [c for c in self.config.enabled_features if c in df.columns]
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return df[cols].values.astype(np.float32)

    def n_features(self) -> int:
        """Number of enabled feature columns."""
        return len(self.config.enabled_features)

    # ------------------------------------------------------------------
    # Private: each indicator adds a single normalised column
    # ------------------------------------------------------------------

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI(period) → tanh((rsi - 50) / rsi_scale).  Range ≈ [-1, 1]."""
        close = _get_close(df)
        rsi_raw = ta.momentum.RSIIndicator(
            close=close, window=self.config.rsi_period, fillna=False
        ).rsi()
        # Centre at 50 and scale
        df["rsi"] = np.tanh((rsi_raw - 50.0) / self.config.rsi_scale)
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD histogram → tanh(hist / rolling_std).  Sign encodes direction."""
        close = _get_close(df)
        macd_obj = ta.trend.MACD(
            close=close,
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal,
            fillna=False,
        )
        hist = macd_obj.macd_diff()
        # Normalise by rolling std to make scale-independent
        rolling_std = hist.rolling(window=50, min_periods=1).std().replace(0, 1.0)
        df["macd"] = np.tanh(hist / rolling_std)
        return df

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band width = (upper - lower) / middle → tanh(width - 1)."""
        close = _get_close(df)
        bb = ta.volatility.BollingerBands(
            close=close,
            window=self.config.bb_period,
            window_dev=self.config.bb_std,
            fillna=False,
        )
        width = bb.bollinger_wband()  # already (upper-lower)/middle
        # width=0 → contracted; typical ~0.04; tanh(width-1) centres near mean
        df["bb_width"] = np.tanh(width - 1.0)
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR(period) / close → tanh(normalised_atr * atr_scale)."""
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        atr_raw = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close,
            window=self.config.atr_period, fillna=False
        ).average_true_range()
        close_safe = close.replace(0, np.nan).ffill().fillna(1.0)
        norm_atr = atr_raw / close_safe
        # typical norm_atr ~0.01-0.05; scale by 10 to spread tanh input
        df["atr"] = np.tanh(norm_atr * 10.0)
        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV direction: tanh(Δobv / rolling_std(Δobv))."""
        close = _get_close(df)
        vol = _get_col(df, "$volume")
        obv_raw = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=vol, fillna=False
        ).on_balance_volume()
        delta = obv_raw.diff()
        rolling_std = delta.rolling(window=50, min_periods=1).std().replace(0, 1.0)
        df["obv"] = np.tanh(delta / rolling_std)
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """(close - VWAP) / VWAP → tanh(deviation * vwap_scale * 10)."""
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        vol = _get_col(df, "$volume")
        vwap_raw = ta.volume.VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=vol, fillna=False
        ).volume_weighted_average_price()
        vwap_safe = vwap_raw.replace(0, np.nan).ffill().fillna(close)
        deviation = (close - vwap_safe) / vwap_safe
        df["vwap_dev"] = np.tanh(deviation * 10.0)
        return df


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_ohlcv(df: pd.DataFrame) -> None:
    required = {"$open", "$high", "$low", "$close", "$volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _get_close(df: pd.DataFrame) -> pd.Series:
    return _get_col(df, "$close")


def _get_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(float)
