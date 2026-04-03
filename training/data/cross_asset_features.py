"""
Cross-asset correlation feature engineering for trading environments.

Computes rolling statistical relationships between a primary asset and auxiliary
assets (e.g. BTC vs SPY, VIX). Features capture:
  - Rolling Pearson correlation of log-returns
  - Rolling beta (sensitivity of primary to each auxiliary)
  - Relative strength (primary return / auxiliary return, rolling)
  - VIX-style fear gauge normalization (when VIX data supplied)

All features are normalized to [-1, 1] via tanh.

Week 25 implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature name templates for cross-asset columns
# Actual column names are generated dynamically from asset names.
_CORR_SUFFIX = "_corr"       # e.g. "spy_corr", "btc_corr"
_BETA_SUFFIX = "_beta"       # e.g. "spy_beta"
_RELSTR_SUFFIX = "_relstr"   # e.g. "spy_relstr"
_VIX_COL = "vix_norm"        # special column for VIX-like fear gauge


@dataclass
class CrossAssetConfig:
    """Configuration for cross-asset feature engineering.

    Parameters
    ----------
    aux_assets:
        Dict mapping asset name → OHLCV DataFrame (must have $close column).
        The asset name becomes the column prefix (e.g. "spy" → "spy_corr").
    correlation_window:
        Rolling window length (bars) for Pearson correlation computation.
    beta_window:
        Rolling window length (bars) for OLS beta estimation.
    relstr_window:
        Rolling window length (bars) for relative-strength ratio.
    vix_asset:
        Optional name (key in aux_assets) of a VIX-like volatility index.
        When set, its level is added as a normalised fear gauge feature.
    vix_centre:
        VIX level to centre around (default 20; historical median ≈ 18-20).
    vix_scale:
        Scaling factor: tanh((vix - vix_centre) / vix_scale).
    min_periods:
        Minimum non-NaN observations required before producing a value.
    """
    aux_assets: Dict[str, pd.DataFrame] = field(default_factory=dict)
    correlation_window: int = 60
    beta_window: int = 60
    relstr_window: int = 20
    vix_asset: Optional[str] = None
    vix_centre: float = 20.0
    vix_scale: float = 15.0
    min_periods: int = 10

    def feature_names(self) -> List[str]:
        """Return the list of feature column names this config will produce."""
        names: List[str] = []
        for asset in self.aux_assets:
            names.append(f"{asset}{_CORR_SUFFIX}")
            names.append(f"{asset}{_BETA_SUFFIX}")
            names.append(f"{asset}{_RELSTR_SUFFIX}")
        if self.vix_asset is not None:
            names.append(_VIX_COL)
        return names


class CrossAssetFeatureEngineer:
    """Computes cross-asset correlation features and appends them to a DataFrame.

    Usage::

        spy_df = pd.DataFrame({"$close": ...})
        vix_df = pd.DataFrame({"$close": ...})

        config = CrossAssetConfig(
            aux_assets={"spy": spy_df, "vix": vix_df},
            vix_asset="vix",
            correlation_window=60,
        )
        cross_fe = CrossAssetFeatureEngineer(config)
        main_df_enriched = cross_fe.compute_features(main_df)
        feature_matrix = cross_fe.get_feature_matrix(main_df_enriched)  # (T, n)
    """

    def __init__(self, config: Optional[CrossAssetConfig] = None):
        self.config = config or CrossAssetConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append cross-asset features to *df* (returns a copy).

        *df* must have a '$close' column.  An index aligned with the
        aux_asset DataFrames is recommended; otherwise a positional
        alignment is used as fallback.
        """
        if "$close" not in df.columns:
            raise ValueError("Primary DataFrame must have a '$close' column.")

        out = df.copy()
        cfg = self.config

        primary_ret = _log_returns(out["$close"].astype(float))

        for asset_name, aux_df in cfg.aux_assets.items():
            if "$close" not in aux_df.columns:
                logger.warning("Aux asset '%s' missing $close column — skipping.", asset_name)
                continue

            aux_ret = _log_returns(aux_df["$close"].astype(float))

            # Align to primary index
            aux_ret_aligned = _align_series(primary_ret, aux_ret)

            # Correlation
            out[f"{asset_name}{_CORR_SUFFIX}"] = _rolling_correlation(
                primary_ret, aux_ret_aligned,
                window=cfg.correlation_window,
                min_periods=cfg.min_periods,
            )

            # Beta
            out[f"{asset_name}{_BETA_SUFFIX}"] = _rolling_beta(
                primary_ret, aux_ret_aligned,
                window=cfg.beta_window,
                min_periods=cfg.min_periods,
            )

            # Relative strength
            out[f"{asset_name}{_RELSTR_SUFFIX}"] = _rolling_relative_strength(
                primary_ret, aux_ret_aligned,
                window=cfg.relstr_window,
                min_periods=cfg.min_periods,
            )

        # VIX fear gauge
        if cfg.vix_asset is not None and cfg.vix_asset in cfg.aux_assets:
            vix_df = cfg.aux_assets[cfg.vix_asset]
            if "$close" in vix_df.columns:
                vix_level = vix_df["$close"].astype(float)
                vix_aligned = _align_series(out["$close"], vix_level)
                out[_VIX_COL] = np.tanh(
                    (vix_aligned - cfg.vix_centre) / cfg.vix_scale
                ).values

        # Forward-fill then zero-fill all new feature columns
        for col in cfg.feature_names():
            if col in out.columns:
                out[col] = out[col].ffill().fillna(0.0)

        return out

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (T, n_features) float32 array of cross-asset features."""
        cols = [c for c in self.config.feature_names() if c in df.columns]
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return df[cols].values.astype(np.float32)

    def n_features(self) -> int:
        """Number of cross-asset feature columns this config produces."""
        return len(self.config.feature_names())


# ------------------------------------------------------------------
# Cross-asset statistics helpers
# ------------------------------------------------------------------

def _rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling Pearson correlation → output already in [-1, 1], no tanh needed."""
    corr = x.rolling(window=window, min_periods=min_periods).corr(y)
    return corr.clip(-1.0, 1.0)


def _rolling_beta(
    primary: pd.Series,
    market: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling OLS beta = Cov(primary, market) / Var(market).

    Normalised via tanh(beta / 2) so beta=1 (market-neutral) → 0.46,
    beta=2 → 0.96, beta=-1 → -0.46.
    """
    cov = primary.rolling(window=window, min_periods=min_periods).cov(market)
    var = market.rolling(window=window, min_periods=min_periods).var()
    var_safe = var.replace(0, np.nan).ffill().fillna(1.0)
    beta = cov / var_safe
    return np.tanh(beta / 2.0)


def _rolling_relative_strength(
    primary: pd.Series,
    other: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling relative strength: cumulative return of primary vs other.

    RS = sum(primary_ret, w) - sum(other_ret, w)
    Normalised via tanh(RS * 10) to compress to [-1, 1].
    Positive = primary outperforming, negative = underperforming.
    """
    sum_primary = primary.rolling(window=window, min_periods=min_periods).sum()
    sum_other = other.rolling(window=window, min_periods=min_periods).sum()
    rs = sum_primary - sum_other
    return np.tanh(rs * 10.0)


def _log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns; first value is NaN."""
    return np.log(prices / prices.shift(1))


def _align_series(reference: pd.Series, other: pd.Series) -> pd.Series:
    """Align *other* to *reference* index.

    - If both have the same index, returns *other* as-is.
    - If indices share dtype (e.g. DatetimeIndex), use reindex + ffill.
    - Otherwise falls back to positional alignment via reset_index.
    """
    if reference.index.equals(other.index):
        return other

    try:
        aligned = other.reindex(reference.index).ffill()
    except Exception:
        # Fallback: positional alignment
        n = len(reference)
        vals = other.values
        if len(vals) >= n:
            aligned = pd.Series(vals[:n], index=reference.index)
        else:
            pad = np.full(n - len(vals), np.nan)
            aligned = pd.Series(np.concatenate([pad, vals]), index=reference.index)

    return aligned


# ------------------------------------------------------------------
# Convenience factory: build from price dict
# ------------------------------------------------------------------

def make_cross_asset_config(
    aux_prices: Dict[str, pd.Series],
    vix_name: Optional[str] = None,
    correlation_window: int = 60,
    beta_window: int = 60,
    relstr_window: int = 20,
) -> CrossAssetConfig:
    """Build a CrossAssetConfig from a dict of {name: close_price_series}.

    Each Series is wrapped in a minimal DataFrame with a '$close' column.

    Example::

        config = make_cross_asset_config(
            {"spy": spy_close, "btc": btc_close},
            vix_name="vix",
        )
    """
    aux_dfs: Dict[str, pd.DataFrame] = {}
    for name, series in aux_prices.items():
        aux_dfs[name] = pd.DataFrame({"$close": series})

    return CrossAssetConfig(
        aux_assets=aux_dfs,
        vix_asset=vix_name,
        correlation_window=correlation_window,
        beta_window=beta_window,
        relstr_window=relstr_window,
    )
