"""
Week 62 (S35) — DataSource unit tests:
  - StaticDataSource
  - CSVDataSource
  - MockLiveDataSource
  - DataQualityGate
  - Env + MockLiveDataSource integration
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from data.sources.base import DataSource, StaticDataSource
from data.sources.csv_source import CSVDataSource
from data.sources.mock_live_source import MockLiveDataSource
from data.quality.gate import DataIssue, DataQualityError, DataQualityGate, validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 50, prefix: bool = True) -> pd.DataFrame:
    """Return a simple deterministic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.standard_normal(n))
    col = "$" if prefix else ""
    return pd.DataFrame(
        {
            f"{col}open": prices,
            f"{col}high": prices + rng.uniform(0.1, 1.0, n),
            f"{col}low": prices - rng.uniform(0.1, 1.0, n),
            f"{col}close": prices + rng.standard_normal(n) * 0.5,
            f"{col}volume": rng.uniform(100, 1000, n),
        }
    )


# ===========================================================================
# StaticDataSource
# ===========================================================================

class TestStaticDataSource:
    def test_interface_compliance(self):
        ds = StaticDataSource(_make_ohlcv())
        assert isinstance(ds, DataSource)

    def test_len(self):
        df = _make_ohlcv(30)
        ds = StaticDataSource(df)
        assert len(ds) == 30

    def test_is_live_false(self):
        assert StaticDataSource(_make_ohlcv()).is_live() is False

    def test_get_window_normal(self):
        df = _make_ohlcv(50)
        ds = StaticDataSource(df)
        win = ds.get_window(5, 15)
        assert len(win) == 10
        pd.testing.assert_frame_equal(win.reset_index(drop=True), df.iloc[5:15].reset_index(drop=True))

    def test_get_window_full(self):
        df = _make_ohlcv(20)
        ds = StaticDataSource(df)
        win = ds.get_window(0, 20)
        assert len(win) == 20

    def test_latest(self):
        df = _make_ohlcv(20)
        ds = StaticDataSource(df)
        latest = ds.latest()
        pd.testing.assert_series_equal(latest, df.iloc[-1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            StaticDataSource(pd.DataFrame())

    def test_df_property(self):
        df = _make_ohlcv()
        ds = StaticDataSource(df)
        assert ds.df is not None
        assert len(ds.df) == len(df)


# ===========================================================================
# CSVDataSource
# ===========================================================================

class TestCSVDataSource:
    def _write_csv(self, df: pd.DataFrame, path: str, use_dollar: bool = True) -> None:
        df.to_csv(path, index=False)

    def test_loads_dollar_prefixed_cols(self, tmp_path):
        df = _make_ohlcv(30, prefix=True)
        p = str(tmp_path / "data.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        assert len(ds) == 30
        assert "$close" in ds.df.columns

    def test_renames_plain_cols(self, tmp_path):
        df = _make_ohlcv(30, prefix=False)
        p = str(tmp_path / "plain.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        assert "$close" in ds.df.columns
        assert "close" not in ds.df.columns

    def test_get_window(self, tmp_path):
        df = _make_ohlcv(50, prefix=True)
        p = str(tmp_path / "data.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        win = ds.get_window(10, 20)
        assert len(win) == 10

    def test_latest(self, tmp_path):
        df = _make_ohlcv(20, prefix=True)
        p = str(tmp_path / "data.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        latest = ds.latest()
        assert latest["$close"] == pytest.approx(df.iloc[-1]["$close"])

    def test_is_live_false(self, tmp_path):
        df = _make_ohlcv(10, prefix=True)
        p = str(tmp_path / "data.csv")
        df.to_csv(p, index=False)
        assert CSVDataSource(p).is_live() is False

    def test_missing_required_col_raises(self, tmp_path):
        df = pd.DataFrame({"$open": [1.0], "$close": [1.0]})  # missing others
        p = str(tmp_path / "bad.csv")
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            CSVDataSource(p).get_window(0, 1)

    def test_lazy_load(self, tmp_path):
        df = _make_ohlcv(10, prefix=True)
        p = str(tmp_path / "data.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        # File not accessed until first use
        assert ds._df is None
        _ = ds.latest()
        assert ds._df is not None


# ===========================================================================
# MockLiveDataSource
# ===========================================================================

class TestMockLiveDataSource:
    def test_is_live_true(self):
        ds = MockLiveDataSource(_make_ohlcv(20))
        assert ds.is_live() is True

    def test_len_is_total(self):
        ds = MockLiveDataSource(_make_ohlcv(20))
        assert len(ds) == 20

    def test_initial_tick_default(self):
        ds = MockLiveDataSource(_make_ohlcv(20))
        assert ds.current_tick == 0

    def test_latest_at_tick(self):
        df = _make_ohlcv(20)
        ds = MockLiveDataSource(df)
        ds.tick(5)
        assert ds.current_tick == 5
        pd.testing.assert_series_equal(ds.latest(), df.iloc[5])

    def test_tick_advances(self):
        ds = MockLiveDataSource(_make_ohlcv(20))
        ds.tick()
        ds.tick()
        ds.tick()
        assert ds.current_tick == 3

    def test_tick_clamps_at_last_bar(self):
        ds = MockLiveDataSource(_make_ohlcv(5))
        ds.tick(100)
        assert ds.current_tick == 4

    def test_get_window_respects_tick(self):
        df = _make_ohlcv(20)
        ds = MockLiveDataSource(df)
        ds.tick(9)  # tick=9 → visible bars 0..9
        win = ds.get_window(0, 20)  # request all 20, but only 10 visible
        assert len(win) == 10

    def test_get_window_within_visible(self):
        df = _make_ohlcv(20)
        ds = MockLiveDataSource(df)
        ds.tick(14)
        win = ds.get_window(5, 10)
        assert len(win) == 5
        pd.testing.assert_frame_equal(
            win.reset_index(drop=True), df.iloc[5:10].reset_index(drop=True)
        )

    def test_reset(self):
        ds = MockLiveDataSource(_make_ohlcv(20))
        ds.tick(10)
        ds.reset(3)
        assert ds.current_tick == 3

    def test_initial_tick_out_of_range_raises(self):
        with pytest.raises(ValueError):
            MockLiveDataSource(_make_ohlcv(5), initial_tick=10)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MockLiveDataSource(pd.DataFrame())


# ===========================================================================
# DataQualityGate
# ===========================================================================

class TestDataQualityGate:
    def test_clean_data_no_issues(self):
        df = _make_ohlcv(50)
        issues = validate(df)
        assert issues == []

    def test_nan_detected(self):
        df = _make_ohlcv(10)
        df.iloc[3, df.columns.get_loc("$close")] = float("nan")
        issues = validate(df)
        kinds = [i.kind for i in issues]
        assert "nan_inf" in kinds

    def test_inf_detected(self):
        df = _make_ohlcv(10)
        df.iloc[0, df.columns.get_loc("$open")] = math.inf
        issues = validate(df)
        assert any(i.kind == "nan_inf" for i in issues)

    def test_negative_price_detected(self):
        df = _make_ohlcv(10)
        df.iloc[2, df.columns.get_loc("$close")] = -5.0
        issues = validate(df)
        assert any(i.kind == "negative_price" for i in issues)

    def test_zero_volume_detected(self):
        df = _make_ohlcv(10)
        df.iloc[5, df.columns.get_loc("$volume")] = 0.0
        issues = validate(df)
        assert any(i.kind == "zero_volume" for i in issues)

    def test_check_raises_on_issue(self):
        df = _make_ohlcv(10)
        df.iloc[0, df.columns.get_loc("$close")] = float("nan")
        gate = DataQualityGate()
        with pytest.raises(DataQualityError) as exc_info:
            gate.check(df)
        assert len(exc_info.value.issues) >= 1

    def test_time_gap_detected(self):
        n = 20
        df = _make_ohlcv(n)
        # Add timestamps with a 1-hour frequency, then insert a 10-hour gap at row 10
        times = pd.date_range("2024-01-01", periods=n, freq="1h")
        times_list = list(times)
        times_list[10] = times_list[9] + pd.Timedelta(hours=11)
        for i in range(11, n):
            times_list[i] = times_list[i - 1] + pd.Timedelta(hours=1)
        df["timestamp"] = times_list
        gate = DataQualityGate(max_gap_bars=5)
        issues = gate.validate(df)
        assert any(i.kind == "time_gap" for i in issues)

    def test_no_time_gap_with_regular_timestamps(self):
        n = 20
        df = _make_ohlcv(n)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="1h")
        gate = DataQualityGate(max_gap_bars=5)
        issues = gate.validate(df)
        assert not any(i.kind == "time_gap" for i in issues)

    def test_multiple_issues_collected(self):
        df = _make_ohlcv(10)
        df.iloc[0, df.columns.get_loc("$close")] = float("nan")
        df.iloc[1, df.columns.get_loc("$open")] = -1.0
        df.iloc[2, df.columns.get_loc("$volume")] = 0.0
        issues = validate(df)
        kinds = {i.kind for i in issues}
        assert kinds >= {"nan_inf", "negative_price", "zero_volume"}

    def test_issue_str_representation(self):
        issue = DataIssue(kind="nan_inf", row=5, column="$close", detail="value=nan")
        s = str(issue)
        assert "nan_inf" in s
        assert "row=5" in s
        assert "$close" in s


# ===========================================================================
# Env + MockLiveDataSource integration (S35)
# ===========================================================================

class TestEnvWithMockLiveDataSource:
    """Integration: SingleAssetRLTradingEnv driven by MockLiveDataSource."""

    def _make_env_with_live_ds(self, n: int = 100, window_size: int = 10):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        df = _make_ohlcv(n)
        ds = MockLiveDataSource(df)
        # Advance tick so env has enough history for window
        ds.tick(n - 1)  # all bars visible from start for simplicity
        env = SingleAssetRLTradingEnv(
            data_source=ds,
            window_size=window_size,
            initial_capital=10_000,
        )
        return env, ds

    def test_env_accepts_mock_live_source(self):
        env, _ = self._make_env_with_live_ds()
        assert env.data_source is not None
        assert env.data_source.is_live() is True

    def test_env_reset_with_live_source(self):
        env, _ = self._make_env_with_live_ds()
        obs, info = env.reset()
        assert obs.shape == (10, 5)

    def test_env_step_returns_valid_obs(self):
        env, _ = self._make_env_with_live_ds(n=100, window_size=10)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (10, 5)
        assert isinstance(reward, float)
        assert not np.isnan(reward)

    def test_observation_changes_with_tick(self):
        """As tick advances, the observation window should shift."""
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        df = _make_ohlcv(80)
        ds = MockLiveDataSource(df)
        ds.tick(79)  # all bars visible

        env = SingleAssetRLTradingEnv(data_source=ds, window_size=5, initial_capital=10_000)
        obs1, _ = env.reset()

        # Do one step to advance current_step
        action = env.action_space.sample()
        obs2, *_ = env.step(action)

        # obs1 and obs2 should differ (different window)
        assert not np.allclose(obs1, obs2), "Observation should change after step"

    def test_env_uses_ds_len_not_self_data(self):
        """Env with non-StaticDataSource must work even if self.data is None."""
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        df = _make_ohlcv(60)
        ds = MockLiveDataSource(df)
        ds.tick(59)

        env = SingleAssetRLTradingEnv(data_source=ds, window_size=5, initial_capital=10_000)
        # self.data should be None since ds is not StaticDataSource
        assert env.data is None
        # but _ds_len() should work
        assert env._ds_len() == 60

        obs, _ = env.reset()
        assert obs.shape == (5, 5)


# ===========================================================================
# DataSource gate integration (gate check before DataSource returns data)
# ===========================================================================

class TestGateIntegration:
    def test_gate_rejects_nan_df(self):
        df = _make_ohlcv(10)
        df.iloc[2, df.columns.get_loc("$close")] = float("nan")
        gate = DataQualityGate()
        with pytest.raises(DataQualityError):
            gate.check(df)

    def test_csv_source_with_gate_check(self, tmp_path):
        """CSVDataSource + gate: simulate pre-use validation."""
        df = _make_ohlcv(20)
        p = str(tmp_path / "clean.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        gate = DataQualityGate()
        # Should not raise
        gate.check(ds.df)

    def test_csv_source_bad_data_raises(self, tmp_path):
        df = _make_ohlcv(10)
        df.iloc[0, df.columns.get_loc("$open")] = float("nan")
        p = str(tmp_path / "bad.csv")
        df.to_csv(p, index=False)
        ds = CSVDataSource(p)
        gate = DataQualityGate()
        with pytest.raises(DataQualityError):
            gate.check(ds.df)
