"""
Technical indicator feature engineering for trading environments.

All indicators are:
- Computed without look-ahead (shift(1) applied where needed)
- Normalized to [-1, 1] via tanh
- NaN-filled with 0 (safe default in tanh space)

Requires: ta>=0.10.0 (pip install ta)

Week 25: Added 10 extended indicators — ADX, Stochastic %K/%D, CCI, Williams %R,
         MFI, CMF, Aroon Oscillator, EMA ratio, Keltner Channel position.
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

# Original 6 feature column names (Week 12)
FEATURE_COLS = ["rsi", "macd", "bb_width", "atr", "obv", "vwap_dev"]

# Week 25: 10 extended indicator columns
EXTENDED_FEATURE_COLS = [
    "adx",         # Average Directional Index (trend strength)
    "stoch_k",     # Stochastic %K (momentum)
    "stoch_d",     # Stochastic %D (signal line)
    "cci",         # Commodity Channel Index
    "williams_r",  # Williams %R
    "mfi",         # Money Flow Index (volume-weighted RSI)
    "cmf",         # Chaikin Money Flow
    "aroon",       # Aroon Oscillator (aroon_up - aroon_down)
    "ema_ratio",   # EMA(fast) / EMA(slow) - 1, normalized
    "keltner",     # Keltner Channel position
]

# All feature columns (original + extended)
ALL_FEATURE_COLS = FEATURE_COLS + EXTENDED_FEATURE_COLS


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Toggle original indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_obv: bool = True
    use_vwap: bool = True

    # Toggle Week 25 extended indicators (default False for backward compatibility)
    use_adx: bool = False
    use_stochastic: bool = False
    use_cci: bool = False
    use_williams_r: bool = False
    use_mfi: bool = False
    use_cmf: bool = False
    use_aroon: bool = False
    use_ema_ratio: bool = False
    use_keltner: bool = False

    # Original indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Week 25 indicator parameters
    adx_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    cci_period: int = 20
    williams_r_period: int = 14
    mfi_period: int = 14
    cmf_period: int = 20
    aroon_period: int = 25
    ema_fast: int = 9
    ema_slow: int = 21
    keltner_period: int = 20
    keltner_atr_multiplier: float = 2.0

    # Normalisation scales (passed to tanh)
    rsi_scale: float = 50.0       # RSI 0-100 → centre at 50 → /50 → tanh input
    macd_scale: float = 1.0       # will be auto-scaled by rolling std
    atr_scale: float = 1.0        # will be auto-scaled by close price
    obv_scale: float = 1.0        # will be auto-scaled by rolling std
    vwap_scale: float = 1.0       # will be auto-scaled by close price

    # Which indicators to include in the final feature matrix (ordered).
    # Defaults to original 6 for backward compatibility.
    # Use FeatureConfig.with_extended() to enable all 16.
    enabled_features: List[str] = field(default_factory=lambda: list(FEATURE_COLS))

    @classmethod
    def with_extended(cls) -> "FeatureConfig":
        """Return a config with all 16 indicators (original 6 + extended 10) enabled."""
        return cls(
            use_adx=True,
            use_stochastic=True,
            use_cci=True,
            use_williams_r=True,
            use_mfi=True,
            use_cmf=True,
            use_aroon=True,
            use_ema_ratio=True,
            use_keltner=True,
            enabled_features=list(ALL_FEATURE_COLS),
        )


class FeatureEngineer:
    """
    Computes technical indicators and normalises them to [-1, 1].

    Usage::

        fe = FeatureEngineer(config)
        df_with_features = fe.compute_features(df)   # adds indicator columns
        feature_matrix = fe.get_feature_matrix(df_with_features)  # (T, n_features)

    Week 25 extended usage::

        fe = FeatureEngineer(FeatureConfig.with_extended())
        df_with_features = fe.compute_features(df)   # adds all 16 columns
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

        All new columns are in ALL_FEATURE_COLS.
        Pre-existing columns are preserved.
        """
        _require_ohlcv(df)
        out = df.copy()
        cfg = self.config

        # Original indicators
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

        # Week 25 extended indicators
        if cfg.use_adx:
            out = self._add_adx(out)
        if cfg.use_stochastic:
            out = self._add_stochastic(out)
        if cfg.use_cci:
            out = self._add_cci(out)
        if cfg.use_williams_r:
            out = self._add_williams_r(out)
        if cfg.use_mfi:
            out = self._add_mfi(out)
        if cfg.use_cmf:
            out = self._add_cmf(out)
        if cfg.use_aroon:
            out = self._add_aroon(out)
        if cfg.use_ema_ratio:
            out = self._add_ema_ratio(out)
        if cfg.use_keltner:
            out = self._add_keltner(out)

        # Forward-fill any residual NaN, then zero-fill
        for col in ALL_FEATURE_COLS:
            if col in out.columns:
                out[col] = out[col].ffill(limit=5).fillna(0.0)

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
    # Private: original indicators
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
    # Private: Week 25 extended indicators
    # ------------------------------------------------------------------

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX(period) → tanh((adx - 25) / 25).

        ADX 0-100: <25 weak trend, >25 strong trend.
        Centred at 25 so 0 ≈ threshold; positive = stronger trend.
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        adx_raw = ta.trend.ADXIndicator(
            high=high, low=low, close=close,
            window=self.config.adx_period, fillna=False
        ).adx()
        df["adx"] = np.tanh((adx_raw - 25.0) / 25.0)
        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic %K and %D → tanh((value - 50) / 50).

        Both centred at 50 (midpoint of 0-100 range).
        >80 overbought, <20 oversold.
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close,
            window=self.config.stoch_k_period,
            smooth_window=self.config.stoch_d_period,
            fillna=False,
        )
        df["stoch_k"] = np.tanh((stoch.stoch() - 50.0) / 50.0)
        df["stoch_d"] = np.tanh((stoch.stoch_signal() - 50.0) / 50.0)
        return df

    def _add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI(period) → tanh(cci / 200).

        CCI typically oscillates in [-200, +200]; dividing by 200 maps to [-1, 1].
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        cci_raw = ta.trend.CCIIndicator(
            high=high, low=low, close=close,
            window=self.config.cci_period, fillna=False
        ).cci()
        df["cci"] = np.tanh(cci_raw / 200.0)
        return df

    def _add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Williams %R → tanh((wr + 50) / 50).

        Williams %R oscillates in [-100, 0]; shift by +50 to centre at 0.
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        wr_raw = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close,
            lbp=self.config.williams_r_period, fillna=False
        ).williams_r()
        # wr_raw ∈ [-100, 0] → shift to [-50, 50] → divide by 50
        df["williams_r"] = np.tanh((wr_raw + 50.0) / 50.0)
        return df

    def _add_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """MFI(period) → tanh((mfi - 50) / 50).

        Money Flow Index: 0-100, centred at 50. >80 overbought, <20 oversold.
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        vol = _get_col(df, "$volume")
        mfi_raw = ta.volume.MFIIndicator(
            high=high, low=low, close=close, volume=vol,
            window=self.config.mfi_period, fillna=False
        ).money_flow_index()
        df["mfi"] = np.tanh((mfi_raw - 50.0) / 50.0)
        return df

    def _add_cmf(self, df: pd.DataFrame) -> pd.DataFrame:
        """CMF(period) → tanh(cmf * 5).

        Chaikin Money Flow: already in [-1, 1] range but with small values.
        Scale by 5 to spread the tanh input.
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        vol = _get_col(df, "$volume")
        cmf_raw = ta.volume.ChaikinMoneyFlowIndicator(
            high=high, low=low, close=close, volume=vol,
            window=self.config.cmf_period, fillna=False
        ).chaikin_money_flow()
        df["cmf"] = np.tanh(cmf_raw * 5.0)
        return df

    def _add_aroon(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aroon Oscillator = aroon_up - aroon_down → tanh(oscillator / 100).

        Oscillator range [-100, 100]; divide by 100 for tanh scale.
        Positive = bullish trend, negative = bearish trend.
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        aroon = ta.trend.AroonIndicator(
            high=high, low=low,
            window=self.config.aroon_period, fillna=False
        )
        oscillator = aroon.aroon_indicator()  # up - down, in [-100, 100]
        df["aroon"] = np.tanh(oscillator / 100.0)
        return df

    def _add_ema_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA(fast) / EMA(slow) - 1 → tanh(ratio * 50).

        Captures momentum crossover: positive when fast EMA above slow EMA.
        Typical range ±0.05; scale by 50 to spread tanh input.
        """
        close = _get_close(df)
        ema_fast = ta.trend.EMAIndicator(
            close=close, window=self.config.ema_fast, fillna=False
        ).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(
            close=close, window=self.config.ema_slow, fillna=False
        ).ema_indicator()
        ema_slow_safe = ema_slow.replace(0, np.nan).ffill().fillna(1.0)
        ratio = (ema_fast / ema_slow_safe) - 1.0
        df["ema_ratio"] = np.tanh(ratio * 50.0)
        return df

    def _add_keltner(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keltner Channel position: (close - midline) / half_width → tanh.

        +1 ≈ upper band (overbought), 0 = midline, -1 ≈ lower band (oversold).
        """
        high = _get_col(df, "$high")
        low = _get_col(df, "$low")
        close = _get_close(df)
        kc = ta.volatility.KeltnerChannel(
            high=high, low=low, close=close,
            window=self.config.keltner_period,
            window_atr=self.config.keltner_period,
            multiplier=self.config.keltner_atr_multiplier,
            fillna=False,
        )
        mband = kc.keltner_channel_mband()
        hband = kc.keltner_channel_hband()
        half_width = (hband - mband).replace(0, np.nan).ffill().fillna(1.0)
        position = (close - mband) / half_width
        df["keltner"] = np.tanh(position)
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
