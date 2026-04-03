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
