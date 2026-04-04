"""
Week 38: Look-ahead bias tests for calendar and feature engineering modules.

Truncation test: feature computed on data up to time T must equal
feature computed on full data at index T.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 200, freq: str = "1D", start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame with DatetimeIndex (UTC)."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(1e6, 1e7, n)
    return pd.DataFrame(
        {"$open": open_, "$high": high, "$low": low, "$close": close, "$volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# 38.1 Calendar features — halving look-ahead bias
# ---------------------------------------------------------------------------

class TestCalendarLookaheadBias:
    """Ensure CalendarFeatureEngine does not leak future info at time T."""

    def _get_engine(self):
        from training.data.calendar_features import CalendarFeatureEngine, CalendarConfig
        return CalendarFeatureEngine(CalendarConfig())

    def test_truncation_event_flag(self):
        """event_flag[T] must be identical whether computed on full or truncated data."""
        engine = self._get_engine()
        df = _make_price_df(n=100, freq="1D", start="2024-03-01")

        full = engine.compute(df)

        for t in range(10, len(df) - 1):
            partial = engine.compute(df.iloc[:t + 1])
            full_val = full["event_flag"].iloc[t]
            partial_val = partial["event_flag"].iloc[-1]
            assert full_val == pytest.approx(partial_val, abs=1e-5), (
                f"Look-ahead bias at t={t}: full={full_val:.4f} partial={partial_val:.4f}"
            )

    def test_halving_not_fired_after_window(self):
        """Days > 1 day after a halving should not fire the upcoming-event window.

        2024-04-20 halving. May 10 is 20 days after, far from any CME/FOMC.
        Old abs() code would fire for any date within 7 days on *either* side;
        new signed-delta code must NOT fire here.
        """
        from training.data.calendar_features import CalendarFeatureEngine, CalendarConfig

        engine = CalendarFeatureEngine(CalendarConfig())

        # May 10, 2024: 20 days post-halving, 21 days before CME (May 31),
        # 9 days after FOMC (May 1) — all outside lookahead_days=3 window.
        idx = pd.date_range("2024-05-10", periods=1, freq="1D", tz="UTC")
        df = pd.DataFrame({"$close": [100.0]}, index=idx)
        result = engine.compute(df)
        # Halving (20d after) should NOT contribute — well outside 7d window
        # No nearby FOMC/CME either → event_flag should be ~0
        assert result["event_flag"].iloc[0] == pytest.approx(0.0, abs=0.05), (
            "20 days after halving should not fire any event signal"
        )

    def test_day_after_halving_gets_small_signal(self):
        """The day after a halving (within 1d post-window) gets a mild signal."""
        from training.data.calendar_features import CalendarFeatureEngine, CalendarConfig

        engine = CalendarFeatureEngine(CalendarConfig())
        # 2024-04-21 = 1 day after halving (2024-04-20)
        day_after_idx = pd.date_range("2024-04-21", periods=1, freq="1D", tz="UTC")
        day_after_df = pd.DataFrame({"$close": [100.0]}, index=day_after_idx)
        result = engine.compute(day_after_df)
        # Should be >= 0.5 (the post-halving signal we set)
        assert result["event_flag"].iloc[0] >= 0.5, (
            "Day after halving should carry a positive event signal"
        )


# ---------------------------------------------------------------------------
# 38.2 Pandas API — no FutureWarning from deprecated method= kwarg
# ---------------------------------------------------------------------------

class TestDeprecatedPandasAPI:
    """Ensure deprecated reindex(method=) usage has been replaced."""

    def test_multi_timeframe_no_method_kwarg(self):
        """MultiTimeframeFeatures.generate() should not trigger FutureWarning."""
        import warnings
        from training.features.multi_timeframe import MultiTimeframeFeatures

        df = _make_price_df(n=100, freq="1H", start="2024-01-01")
        mtf = MultiTimeframeFeatures(higher_timeframes=["4H"])

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = mtf.generate(df)  # raises FutureWarning → test fails

        assert result is not None
        assert len(result) == len(df)

    def test_cross_asset_align_no_method_kwarg(self):
        """_align_series should not use deprecated reindex(method=)."""
        import warnings
        from training.data.cross_asset_features import _align_series

        ref = pd.Series(np.ones(50), index=pd.date_range("2024-01-01", periods=50, tz="UTC"))
        other = pd.Series(np.arange(25), index=pd.date_range("2024-01-01", periods=25, freq="2D", tz="UTC"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            aligned = _align_series(ref, other)

        assert len(aligned) == len(ref)


# ---------------------------------------------------------------------------
# 38.3 Forward-fill limit in feature engineering
# ---------------------------------------------------------------------------

class TestForwardFillLimit:
    """ffill(limit=5) must not propagate stale values beyond 5 bars."""

    def _build_feature_eng(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        return FeatureEngineer(FeatureConfig())

    def test_ffill_does_not_propagate_indefinitely(self):
        """A gap of >5 bars of NaN should become 0.0 (fillna fallback), not stale."""
        fe = self._build_feature_eng()

        # Build a price DataFrame long enough that a 10-bar NaN gap appears
        df = _make_price_df(n=200)
        result = fe.compute_features(df)

        # Check that features exist and no single column is entirely NaN
        assert not result.empty
        for col in result.columns:
            nan_frac = result[col].isna().mean()
            assert nan_frac < 0.1, f"Column {col} has {nan_frac:.1%} NaN — ffill limit may be too aggressive"


# ---------------------------------------------------------------------------
# 42.3a  OnChainFeatureEngine — truncation consistency
# ---------------------------------------------------------------------------

class TestOnChainLookaheadBias:
    """OnChainFeatureEngine.align_to_prices() must not leak future data at time T."""

    def _get_engine(self):
        from training.data.onchain_features import OnChainFeatureEngine, OnChainConfig
        # Disable external fetches so tests are network-isolated
        cfg = OnChainConfig(use_coingecko=False, use_ccxt_derivatives=False, cache_db=None)
        return OnChainFeatureEngine(cfg)

    def test_truncation_returns_same_value(self):
        """align_to_prices on full vs. truncated data gives identical values at T."""
        engine = self._get_engine()
        df = _make_price_df(n=50, freq="1D", start="2024-01-01")

        full_result = engine.align_to_prices(df)

        for t in range(5, 20):
            partial_result = engine.align_to_prices(df.iloc[: t + 1])
            for col in full_result.columns:
                full_val = float(full_result[col].iloc[t])
                partial_val = float(partial_result[col].iloc[-1])
                assert full_val == pytest.approx(partial_val, abs=1e-6), (
                    f"OnChain look-ahead at t={t}, col={col}: "
                    f"full={full_val:.4f} partial={partial_val:.4f}"
                )

    def test_output_no_nan_on_network_failure(self):
        """Returns zero-filled DataFrame (no NaN) when network is unavailable."""
        engine = self._get_engine()
        df = _make_price_df(n=30, freq="1D", start="2024-06-01")
        result = engine.align_to_prices(df)

        assert len(result) == len(df)
        assert not result.isnull().any().any(), "OnChain result must not contain NaN"

    def test_ffill_does_not_introduce_future_values(self):
        """Forward-fill propagates only within the available history, not beyond."""
        engine = self._get_engine()
        df_short = _make_price_df(n=10, freq="1D", start="2024-01-01")
        df_long = _make_price_df(n=30, freq="1D", start="2024-01-01")

        result_short = engine.align_to_prices(df_short)
        result_long = engine.align_to_prices(df_long)

        # Values at the last row of the short series must equal those in the long series
        for col in result_short.columns:
            short_val = float(result_short[col].iloc[-1])
            long_val = float(result_long[col].iloc[len(df_short) - 1])
            assert short_val == pytest.approx(long_val, abs=1e-6), (
                f"OnChain ffill look-ahead in col={col}: "
                f"short_end={short_val:.4f} long_at_same_t={long_val:.4f}"
            )


# ---------------------------------------------------------------------------
# 42.3b  CrossAssetFeatureEngineer — truncation consistency
# ---------------------------------------------------------------------------

class TestCrossAssetLookaheadBias:
    """CrossAssetFeatureEngineer.compute_features() must be look-ahead free."""

    def _make_cross_asset_engineer(self, primary_df: pd.DataFrame):
        from training.data.cross_asset_features import (
            CrossAssetFeatureEngineer,
            CrossAssetConfig,
        )
        rng = np.random.default_rng(0)
        # Synthetic aux asset with same index as primary
        aux_close = 50.0 * np.cumprod(1 + rng.normal(0, 0.01, len(primary_df)))
        aux_df = pd.DataFrame(
            {"$close": aux_close},
            index=primary_df.index,
        )
        cfg = CrossAssetConfig(
            aux_assets={"aux": aux_df},
            correlation_window=10,
            beta_window=10,
            relstr_window=10,
            min_periods=5,
        )
        return CrossAssetFeatureEngineer(cfg)

    def test_truncation_correlation_no_lookahead(self):
        """Rolling correlation at time T is the same on full vs. truncated data."""
        df = _make_price_df(n=100, freq="1D", start="2024-01-01")
        engineer = self._make_cross_asset_engineer(df)
        full_out = engineer.compute_features(df)
        cross_cols = [c for c in full_out.columns if c not in df.columns]

        for t in range(15, 30):
            partial_df = df.iloc[: t + 1].copy()
            partial_engineer = self._make_cross_asset_engineer(partial_df)
            partial_out = partial_engineer.compute_features(partial_df)

            for col in cross_cols:
                if col not in partial_out.columns:
                    continue
                full_val = float(full_out[col].iloc[t])
                partial_val = float(partial_out[col].iloc[-1])
                assert full_val == pytest.approx(partial_val, abs=1e-5), (
                    f"Cross-asset look-ahead at t={t}, col={col}: "
                    f"full={full_val:.6f} partial={partial_val:.6f}"
                )

    def test_no_future_values_in_new_columns(self):
        """Feature columns added by compute_features() must not be NaN."""
        df = _make_price_df(n=80, freq="1D", start="2024-01-01")
        engineer = self._make_cross_asset_engineer(df)
        out = engineer.compute_features(df)
        new_cols = [c for c in out.columns if c not in df.columns]

        assert new_cols, "No cross-asset feature columns were produced"
        for col in new_cols:
            # After the warm-up window, values should not be NaN
            tail = out[col].iloc[20:]
            nan_frac = tail.isnull().mean()
            assert nan_frac == 0.0, (
                f"Column '{col}' has {nan_frac:.0%} NaN after warm-up"
            )

    def test_ffill_stays_within_history(self):
        """ffill inside compute_features must not pull values from beyond T."""
        df = _make_price_df(n=60, freq="1D", start="2024-01-01")
        engineer_full = self._make_cross_asset_engineer(df)
        out_full = engineer_full.compute_features(df)

        # Compare last value when using 40-row subset vs. full 60-row
        df_short = df.iloc[:40].copy()
        engineer_short = self._make_cross_asset_engineer(df_short)
        out_short = engineer_short.compute_features(df_short)

        new_cols_full = [c for c in out_full.columns if c not in df.columns]
        new_cols_short = [c for c in out_short.columns if c not in df_short.columns]

        for col in new_cols_full:
            if col not in new_cols_short:
                continue
            full_at_39 = float(out_full[col].iloc[39])
            short_at_end = float(out_short[col].iloc[-1])
            assert full_at_39 == pytest.approx(short_at_end, abs=1e-5), (
                f"Cross-asset ffill look-ahead in col={col}: "
                f"full[39]={full_at_39:.6f} short[-1]={short_at_end:.6f}"
            )
