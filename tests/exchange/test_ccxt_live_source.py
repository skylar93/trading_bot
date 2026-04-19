"""
Unit tests for CCXTLiveDataSource (Week 72, F2).

No real exchange connectivity — the adapter is a stub that fires callbacks
directly so we can verify the DataSource contract.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.sources.ccxt_live import CCXTLiveDataSource, _ENV_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(close: float = 1.0) -> list:
    """Return a minimal CCXT-style OHLCV bar: [ts, o, h, l, c, v]."""
    return [1_700_000_000_000, close * 0.99, close * 1.01, close * 0.98, close, 10.0]


def _make_source(max_bars: int = 500, max_staleness_sec: float = 0.0) -> tuple:
    """Return (adapter_stub, CCXTLiveDataSource)."""
    adapter = MagicMock()
    adapter.add_ohlcv_callback = MagicMock()
    ds = CCXTLiveDataSource(adapter, max_bars=max_bars, max_staleness_sec=max_staleness_sec)
    return adapter, ds


# ---------------------------------------------------------------------------
# F2: DataSource contract
# ---------------------------------------------------------------------------

class TestCCXTLiveDataSourceContract:
    def test_is_live(self):
        _, ds = _make_source()
        assert ds.is_live() is True

    def test_initial_len_zero(self):
        _, ds = _make_source()
        assert len(ds) == 0

    def test_latest_raises_when_empty(self):
        _, ds = _make_source()
        with pytest.raises(RuntimeError, match="no data"):
            ds.latest()

    def test_get_window_empty_returns_empty_df(self):
        _, ds = _make_source()
        result = ds.get_window(0, 10)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_ohlcv_callback_updates_buffer(self):
        _, ds = _make_source()
        bars = [_make_bar(100.0), _make_bar(101.0)]
        ds._on_ohlcv(bars)

        assert len(ds) == 2
        latest = ds.latest()
        assert latest["close"] == 101.0

    def test_get_window_returns_correct_slice(self):
        _, ds = _make_source()
        for i in range(10):
            ds._on_ohlcv([_make_bar(float(i))])

        window = ds.get_window(0, 5)
        assert isinstance(window, pd.DataFrame)
        assert len(window) == 5
        assert list(window.columns) == _ENV_COLS

    def test_get_window_clamps_end(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        window = ds.get_window(0, 100)  # end > len
        assert len(window) == 1

    def test_get_window_clamps_start(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        window = ds.get_window(-5, 1)  # negative start
        assert len(window) == 1

    def test_rolling_buffer_respects_max_bars(self):
        _, ds = _make_source(max_bars=5)
        for i in range(10):
            ds._on_ohlcv([_make_bar(float(i))])
        assert len(ds) == 5  # deque(maxlen=5) evicts oldest

    def test_latest_after_multiple_updates(self):
        _, ds = _make_source()
        for close in [10.0, 20.0, 30.0]:
            ds._on_ohlcv([_make_bar(close)])
        assert ds.latest()["close"] == 30.0

    def test_callback_registered_on_adapter(self):
        adapter = MagicMock()
        adapter.add_ohlcv_callback = MagicMock()
        ds = CCXTLiveDataSource(adapter, max_bars=100)
        adapter.add_ohlcv_callback.assert_called_once_with(ds._on_ohlcv)

    def test_multi_bar_batch(self):
        """A batch of 5 bars in one callback should produce 5 buffer entries."""
        _, ds = _make_source()
        bars = [_make_bar(float(i)) for i in range(5)]
        ds._on_ohlcv(bars)
        assert len(ds) == 5

    def test_empty_callback_noop(self):
        _, ds = _make_source()
        ds._on_ohlcv([])
        assert len(ds) == 0

    def test_malformed_bar_skipped(self):
        """Bars shorter than 6 elements are skipped silently."""
        _, ds = _make_source()
        ds._on_ohlcv([[1, 2, 3]])   # only 3 elements
        ds._on_ohlcv([_make_bar(42.0)])  # valid bar
        assert len(ds) == 1
        assert ds.latest()["close"] == 42.0


# ---------------------------------------------------------------------------
# F2: Staleness (S47 contract)
# ---------------------------------------------------------------------------

class TestCCXTLiveDataSourceStaleness:
    def test_last_updated_at_none_before_data(self):
        _, ds = _make_source()
        assert ds.last_updated_at() is None

    def test_last_updated_at_set_after_callback(self):
        _, ds = _make_source()
        t_before = time.monotonic()
        ds._on_ohlcv([_make_bar(1.0)])
        t_after = time.monotonic()
        last = ds.last_updated_at()
        assert last is not None
        assert t_before <= last <= t_after

    def test_is_stale_returns_false_when_no_threshold(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        assert ds.is_stale(0.0) is False

    def test_is_stale_returns_true_when_no_data_and_threshold_set(self):
        _, ds = _make_source(max_staleness_sec=1.0)
        # No data ever received → last_updated_at is None → stale
        assert ds.is_stale() is True

    def test_is_stale_false_after_recent_update(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        assert ds.is_stale(max_staleness_sec=60.0) is False

    def test_is_stale_true_after_forced_old_timestamp(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        # Force the last update to be far in the past
        with ds._lock:
            ds._last_updated_at = time.monotonic() - 100.0
        assert ds.is_stale(max_staleness_sec=5.0) is True

    def test_is_stale_uses_instance_threshold(self):
        _, ds = _make_source(max_staleness_sec=5.0)
        ds._on_ohlcv([_make_bar(1.0)])
        with ds._lock:
            ds._last_updated_at = time.monotonic() - 10.0
        assert ds.is_stale() is True  # uses instance max_staleness_sec=5.0

    def test_caller_threshold_overrides_instance(self):
        _, ds = _make_source(max_staleness_sec=5.0)
        ds._on_ohlcv([_make_bar(1.0)])
        with ds._lock:
            ds._last_updated_at = time.monotonic() - 3.0
        # instance says stale after 5s, but caller says 60s → not stale
        assert ds.is_stale(max_staleness_sec=60.0) is False


# ---------------------------------------------------------------------------
# F2: Column schema
# ---------------------------------------------------------------------------

class TestColumnSchema:
    def test_columns_match_env_cols(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        window = ds.get_window(0, 1)
        assert list(window.columns) == _ENV_COLS

    def test_latest_series_index_matches_env_cols(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(1.0)])
        row = ds.latest()
        for col in _ENV_COLS:
            assert col in row.index

    def test_ohlcv_values_stored_as_float(self):
        _, ds = _make_source()
        ds._on_ohlcv([_make_bar(12345.6)])
        latest = ds.latest()
        assert isinstance(latest["close"], float)
        assert isinstance(latest["volume"], float)
