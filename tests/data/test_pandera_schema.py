"""Tests for data.quality.pandera_schema — Week 79 (H6)."""
import math

import numpy as np
import pandas as pd
import pytest

from data.quality.pandera_schema import validate_ohlcv, HAS_PANDERA, OHLCV_SCHEMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_df(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "$open":   np.linspace(100, 110, n),
            "$high":   np.linspace(101, 111, n),
            "$low":    np.linspace(99, 109, n),
            "$close":  np.linspace(100.5, 110.5, n),
            "$volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# not_null / no_inf
# ---------------------------------------------------------------------------

class TestPanderaSchemaNotNull:
    def test_clean_returns_no_errors(self):
        errors = validate_ohlcv(_clean_df())
        assert errors == [], errors

    def test_nan_in_close_is_detected(self):
        df = _clean_df()
        df.loc[df.index[5], "$close"] = float("nan")
        errors = validate_ohlcv(df)
        assert errors, "Expected NaN error"

    def test_inf_in_open_is_detected(self):
        df = _clean_df()
        df.loc[df.index[3], "$open"] = math.inf
        errors = validate_ohlcv(df)
        assert errors

    def test_neg_inf_in_volume_is_detected(self):
        df = _clean_df()
        df.loc[df.index[0], "$volume"] = -math.inf
        errors = validate_ohlcv(df)
        assert errors


# ---------------------------------------------------------------------------
# positive check
# ---------------------------------------------------------------------------

class TestPanderaPositive:
    def test_zero_close_is_detected(self):
        df = _clean_df()
        df.loc[df.index[2], "$close"] = 0.0
        errors = validate_ohlcv(df)
        assert errors

    def test_negative_volume_is_detected(self):
        df = _clean_df()
        df.loc[df.index[1], "$volume"] = -50.0
        errors = validate_ohlcv(df)
        assert errors


# ---------------------------------------------------------------------------
# monotonic_ts
# ---------------------------------------------------------------------------

class TestPanderaMonotonicTs:
    def test_non_monotonic_index_is_detected(self):
        df = _clean_df(10)
        # Swap two timestamps to break monotonicity
        idx = df.index.tolist()
        idx[3], idx[4] = idx[4], idx[3]
        df.index = pd.DatetimeIndex(idx)
        errors = validate_ohlcv(df)
        assert any("monotonic_ts" in e for e in errors), errors

    def test_strictly_increasing_ok(self):
        df = _clean_df(10)
        errors = validate_ohlcv(df)
        ts_errors = [e for e in errors if "monotonic_ts" in e]
        assert not ts_errors


# ---------------------------------------------------------------------------
# no_gap
# ---------------------------------------------------------------------------

class TestPanderaNoGap:
    def test_large_gap_is_detected(self):
        # 20 bars of 1-min, then skip 2 hours
        df1 = _clean_df(10)
        df2 = _clean_df(10)
        df2.index = pd.date_range("2024-01-01 02:00", periods=10, freq="1min")
        df = pd.concat([df1, df2])
        # max_gap_multiplier=3 → threshold = 3min, gap = 2h → should trigger
        errors = validate_ohlcv(df, max_gap_multiplier=3.0)
        assert any("no_gap" in e for e in errors), errors

    def test_uniform_bars_no_gap_error(self):
        df = _clean_df(30)
        errors = validate_ohlcv(df)
        gap_errors = [e for e in errors if "no_gap" in e]
        assert not gap_errors


# ---------------------------------------------------------------------------
# Extra columns are allowed
# ---------------------------------------------------------------------------

class TestPanderaExtraColumns:
    def test_extra_column_ok(self):
        df = _clean_df()
        df["rsi_14"] = 50.0
        errors = validate_ohlcv(df)
        assert errors == []


# ---------------------------------------------------------------------------
# OHLCV_SCHEMA availability
# ---------------------------------------------------------------------------

class TestSchemaObject:
    def test_schema_available_when_pandera_installed(self):
        if HAS_PANDERA:
            assert OHLCV_SCHEMA is not None
        else:
            assert OHLCV_SCHEMA is None
