"""
Week 65 — Data Pipeline Safety tests (S50)

Covers:
  S47  Feed staleness halt: PaperTrader shuts down when live DataSource is stale
  S48  NaN/inf in features: step skipped + warning; N consecutive → halt
  S49  SurvivorshipBiasChecker: short_history / late_start / single_asset warnings
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.quality.survivorship import (
    BiasWarning,
    SurvivorshipBiasChecker,
    check_survivorship,
)
from data.sources.base import DataSource, StaticDataSource
from data.sources.mock_live_source import MockLiveDataSource

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DATA_CSV = Path(__file__).parents[2] / "test_data.csv"


def _load_df(nrows: int = 100) -> pd.DataFrame:
    df = pd.read_csv(_DATA_CSV, index_col=0, nrows=nrows)
    return df.reset_index(drop=True)


def _make_trader(config_overrides: dict | None = None, audit_logger=None, data_source=None):
    """Build a minimal PaperTrader in simulation mode."""
    from deployment.paper_trader import PaperTrader

    cfg = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.20,
            "window_size": 10,
        },
        "monitoring": {},
        "data_pipeline_safety": {
            "staleness_enabled": True,
            "max_staleness_sec": 0.2,  # 200 ms — fast for tests
            "nan_check_enabled": True,
            "nan_halt_after_n": 3,
        },
    }
    if config_overrides:
        cfg.update(config_overrides)

    agent = MagicMock()
    agent.predict.return_value = (np.array([0.0]), None)

    return PaperTrader(
        agent=agent,
        config=cfg,
        simulation_mode=True,
        audit_logger=audit_logger,
        data_source=data_source,
    )


# ===========================================================================
# S47 — Feed staleness halt
# ===========================================================================


class TestFeedStalenessHalt:
    def _make_live_source(self) -> MockLiveDataSource:
        df = _load_df(50)
        return MockLiveDataSource(df, initial_tick=10)

    def test_static_source_never_stale(self):
        src = StaticDataSource(_load_df(50))
        assert not src.is_stale(0.001)

    def test_mock_live_not_stale_immediately_after_tick(self):
        src = self._make_live_source()
        src.tick()
        assert not src.is_stale(60.0)

    def test_mock_live_stale_after_delay(self):
        src = self._make_live_source()
        # Record update, then wait longer than threshold
        src.tick()
        time.sleep(0.05)
        assert src.is_stale(0.01)   # threshold = 10 ms

    def test_last_updated_at_set_on_init(self):
        src = self._make_live_source()
        ts = src.last_updated_at()
        assert ts is not None
        assert isinstance(ts, float)

    def test_last_updated_at_updated_on_tick(self):
        src = self._make_live_source()
        t0 = src.last_updated_at()
        time.sleep(0.01)
        src.tick()
        t1 = src.last_updated_at()
        assert t1 > t0

    def test_static_source_last_updated_at_is_none(self):
        src = StaticDataSource(_load_df(50))
        assert src.last_updated_at() is None

    def test_paper_trader_halts_on_stale_feed(self):
        """Trader should shutdown before executing any trades when feed is stale."""
        df = _load_df(50)
        src = MockLiveDataSource(df, initial_tick=10)
        # Make it immediately stale: last_updated_at is in the past
        src._last_updated_at = time.monotonic() - 100.0

        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": True,
                    "max_staleness_sec": 1.0,  # stale after 1s; source is 100s stale
                    "nan_check_enabled": False,
                    "nan_halt_after_n": 5,
                }
            },
            data_source=src,
        )

        prices = list(df["$close"].iloc[10:30])
        report = trader.run(price_stream=iter(prices))

        assert trader.state.shutdown_triggered
        assert "stale" in trader.state.shutdown_reason.lower()

    def test_paper_trader_halts_on_stale_feed_with_audit(self):
        """Staleness halt records a risk_event in the audit log."""
        df = _load_df(50)
        src = MockLiveDataSource(df, initial_tick=10)
        src._last_updated_at = time.monotonic() - 100.0

        audit = MagicMock()
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": True,
                    "max_staleness_sec": 1.0,
                    "nan_check_enabled": False,
                    "nan_halt_after_n": 5,
                }
            },
            audit_logger=audit,
            data_source=src,
        )

        prices = list(df["$close"].iloc[10:25])
        trader.run(price_stream=iter(prices))

        assert audit.log_risk_event.called
        calls = [c.args[0] for c in audit.log_risk_event.call_args_list]
        assert any(c.get("type") == "feed_staleness_halt" for c in calls)

    def test_paper_trader_no_halt_when_staleness_disabled(self):
        """When staleness_enabled=False the feed age is ignored."""
        df = _load_df(50)
        src = MockLiveDataSource(df, initial_tick=10)
        src._last_updated_at = time.monotonic() - 100.0

        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 1.0,
                    "nan_check_enabled": False,
                    "nan_halt_after_n": 5,
                }
            },
            data_source=src,
        )

        prices = list(df["$close"].iloc[10:30])
        trader.run(price_stream=iter(prices))

        assert not trader.state.shutdown_triggered

    def test_paper_trader_no_halt_without_data_source(self):
        """Staleness check is skipped when no data_source is attached."""
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": True,
                    "max_staleness_sec": 0.001,
                    "nan_check_enabled": False,
                    "nan_halt_after_n": 5,
                }
            },
        )

        df = _load_df(30)
        prices = list(df["$close"].values)
        trader.run(price_stream=iter(prices))

        assert not trader.state.shutdown_triggered


# ===========================================================================
# S48 — NaN/inf in computed features
# ===========================================================================


class TestNanInfFeatureCheck:
    def test_trader_skips_step_on_nan_obs(self):
        """Steps with NaN observations are skipped; no trade executed."""
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 0,
                    "nan_check_enabled": True,
                    "nan_halt_after_n": 0,  # never halt
                }
            }
        )

        # Force _build_observation to return NaN obs for first N steps, then normal
        df = _load_df(50)
        prices = list(df["$close"].values)

        nan_obs = np.full(11, np.nan)  # window_size=10 → log_returns(10) + 2 = 11 features
        good_obs = np.zeros(11, dtype=np.float32)

        call_count = [0]

        def mock_build_obs(self_trader):
            call_count[0] += 1
            if call_count[0] <= 3:
                return nan_obs
            return good_obs

        # Patch _build_observation
        original_build = trader._build_observation
        call_idx = [0]

        def patched_build():
            call_idx[0] += 1
            if call_idx[0] <= 3:
                return nan_obs
            return original_build()

        trader._build_observation = patched_build
        trader.agent.predict.return_value = (np.array([0.0]), None)

        trader.run(price_stream=iter(prices))

        assert call_idx[0] > 3
        assert not trader.state.shutdown_triggered

    def test_trader_halts_after_n_consecutive_nans(self):
        """N consecutive NaN observations trigger halt."""
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 0,
                    "nan_check_enabled": True,
                    "nan_halt_after_n": 3,
                }
            }
        )

        df = _load_df(50)
        prices = list(df["$close"].values)
        nan_obs = np.full(11, np.nan)

        trader._build_observation = lambda: nan_obs
        trader.run(price_stream=iter(prices))

        assert trader.state.shutdown_triggered
        assert "nan_in_features" in trader.state.shutdown_reason.lower()

    def test_consecutive_nan_counter_resets_on_good_obs(self):
        """A single good observation resets the consecutive counter."""
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 0,
                    "nan_check_enabled": True,
                    "nan_halt_after_n": 4,
                }
            }
        )

        df = _load_df(50)
        prices = list(df["$close"].values)
        nan_obs = np.full(11, np.nan)
        good_obs = np.zeros(11, dtype=np.float32)

        call_idx = [0]

        def patched_build():
            call_idx[0] += 1
            # 3 nans, then 1 good, then more nans — should NOT halt
            if call_idx[0] in (1, 2, 3):
                return nan_obs
            if call_idx[0] == 4:
                return good_obs
            if call_idx[0] in (5, 6, 7):
                return nan_obs
            return good_obs

        trader._build_observation = patched_build
        trader.agent.predict.return_value = (np.array([0.0]), None)
        trader.run(price_stream=iter(prices))

        # consecutive counter was reset at step 4, so 3 more nans should NOT trigger halt
        assert not trader.state.shutdown_triggered

    def test_nan_halt_emits_audit_events(self):
        """NaN observations produce audit events for each bad step."""
        audit = MagicMock()
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 0,
                    "nan_check_enabled": True,
                    "nan_halt_after_n": 3,
                }
            },
            audit_logger=audit,
        )

        df = _load_df(30)
        prices = list(df["$close"].values)
        trader._build_observation = lambda: np.full(11, np.nan)
        trader.run(price_stream=iter(prices))

        calls = [c.args[0] for c in audit.log_risk_event.call_args_list]
        nan_events = [c for c in calls if c.get("type") in ("nan_in_features", "nan_halt")]
        assert len(nan_events) >= 3

    def test_inf_in_obs_treated_as_nan(self):
        """Observations with inf values are treated the same as NaN."""
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 0,
                    "nan_check_enabled": True,
                    "nan_halt_after_n": 3,
                }
            }
        )

        df = _load_df(30)
        prices = list(df["$close"].values)
        inf_obs = np.full(11, np.inf)
        trader._build_observation = lambda: inf_obs
        trader.run(price_stream=iter(prices))

        assert trader.state.shutdown_triggered

    def test_nan_check_disabled_passes_bad_obs(self):
        """When nan_check_enabled=False, NaN obs are forwarded to the agent."""
        trader = _make_trader(
            config_overrides={
                "data_pipeline_safety": {
                    "staleness_enabled": False,
                    "max_staleness_sec": 0,
                    "nan_check_enabled": False,
                    "nan_halt_after_n": 3,
                }
            }
        )

        df = _load_df(30)
        prices = list(df["$close"].values)
        nan_obs = np.full(11, np.nan)
        trader._build_observation = lambda: nan_obs
        trader.agent.predict.return_value = (np.array([0.0]), None)
        trader.run(price_stream=iter(prices))

        # trader did not halt due to nan check
        assert not trader.state.shutdown_triggered
        # agent.predict was called (NaN obs passed through)
        assert trader.agent.predict.called


# ===========================================================================
# S49 — SurvivorshipBiasChecker
# ===========================================================================


class TestSurvivorshipBiasChecker:
    def _df_with_dates(self, n: int, start: str) -> pd.DataFrame:
        dates = pd.date_range(start=start, periods=n, freq="h")
        df = pd.DataFrame(
            {
                "$open": np.random.uniform(100, 200, n),
                "$high": np.random.uniform(200, 300, n),
                "$low": np.random.uniform(50, 100, n),
                "$close": np.random.uniform(100, 200, n),
                "$volume": np.random.uniform(1, 100, n),
                "timestamp": dates,
            }
        )
        return df

    def test_short_history_warning(self):
        df = self._df_with_dates(50, "2024-01-01")
        checker = SurvivorshipBiasChecker(min_lookback_bars=200)
        warnings = checker.check(df)
        kinds = [w.kind for w in warnings]
        assert "short_history" in kinds

    def test_no_short_history_warning_when_enough_bars(self):
        df = self._df_with_dates(300, "2024-01-01")
        checker = SurvivorshipBiasChecker(min_lookback_bars=200)
        warnings = checker.check(df)
        kinds = [w.kind for w in warnings]
        assert "short_history" not in kinds

    def test_no_short_history_when_min_lookback_zero(self):
        df = self._df_with_dates(10, "2024-01-01")
        checker = SurvivorshipBiasChecker(min_lookback_bars=0)
        warnings = checker.check(df)
        kinds = [w.kind for w in warnings]
        assert "short_history" not in kinds

    def test_late_start_warning(self):
        df = self._df_with_dates(100, "2024-06-01")  # starts in June
        checker = SurvivorshipBiasChecker()
        warnings = checker.check(df, expected_start="2024-01-01")
        kinds = [w.kind for w in warnings]
        assert "late_start" in kinds

    def test_no_late_start_warning_when_data_starts_on_time(self):
        df = self._df_with_dates(100, "2024-01-01")
        checker = SurvivorshipBiasChecker()
        warnings = checker.check(df, expected_start="2024-01-01")
        kinds = [w.kind for w in warnings]
        assert "late_start" not in kinds

    def test_single_asset_universe_warning_always_present(self):
        df = self._df_with_dates(100, "2024-01-01")
        checker = SurvivorshipBiasChecker(warn_single_asset=True)
        warnings = checker.check(df)
        kinds = [w.kind for w in warnings]
        assert "single_asset_universe" in kinds

    def test_single_asset_warning_suppressed(self):
        df = self._df_with_dates(100, "2024-01-01")
        checker = SurvivorshipBiasChecker(warn_single_asset=False)
        warnings = checker.check(df)
        kinds = [w.kind for w in warnings]
        assert "single_asset_universe" not in kinds

    def test_empty_dataframe_returns_warning(self):
        checker = SurvivorshipBiasChecker()
        warnings = checker.check(pd.DataFrame())
        assert len(warnings) >= 1
        assert warnings[0].kind == "short_history"

    def test_severity_fields(self):
        df = self._df_with_dates(50, "2024-06-01")
        checker = SurvivorshipBiasChecker(min_lookback_bars=200)
        warnings = checker.check(df, expected_start="2024-01-01")
        for w in warnings:
            assert w.severity in ("warning", "info")

    def test_str_representation(self):
        w = BiasWarning(kind="short_history", severity="warning", detail="test detail")
        s = str(w)
        assert "short_history" in s
        assert "warning" in s
        assert "test detail" in s

    def test_invalid_min_lookback_raises(self):
        with pytest.raises(ValueError):
            SurvivorshipBiasChecker(min_lookback_bars=-1)

    def test_datetime_index_works(self):
        dates = pd.date_range("2024-06-01", periods=50, freq="h")
        df = pd.DataFrame(
            {"$close": np.random.uniform(100, 200, 50)},
            index=dates,
        )
        checker = SurvivorshipBiasChecker()
        warnings = checker.check(df, expected_start="2024-01-01")
        kinds = [w.kind for w in warnings]
        assert "late_start" in kinds

    def test_module_shortcut_function(self):
        df = self._df_with_dates(50, "2024-01-01")
        warnings = check_survivorship(df, min_lookback_bars=200)
        assert any(w.kind == "short_history" for w in warnings)

    def test_log_warnings_calls_logger(self):
        import logging
        df = self._df_with_dates(50, "2024-06-01")
        checker = SurvivorshipBiasChecker(min_lookback_bars=200)
        with patch("data.quality.survivorship.logger") as mock_logger:
            result = checker.log_warnings(df, expected_start="2024-01-01")
        assert len(result) > 0
        assert mock_logger.warning.called or mock_logger.info.called

    def test_symbol_appears_in_warning_detail(self):
        df = self._df_with_dates(50, "2024-01-01")
        checker = SurvivorshipBiasChecker(min_lookback_bars=200)
        warnings = checker.check(df, symbol="AAPL")
        text = " ".join(str(w) for w in warnings)
        assert "AAPL" in text


# ===========================================================================
# S49 — SurvivorshipBiasChecker used at backtest start (integration)
# ===========================================================================


class TestSurvivorshipAtBacktestStart:
    """Verify that the checker can be called during env/backtest initialisation."""

    def test_checker_runs_on_test_data(self):
        """Run the checker against real test_data.csv and expect only info-level
        issues (the dataset is fine for tests)."""
        df = pd.read_csv(_DATA_CSV, index_col=0, nrows=200)
        checker = SurvivorshipBiasChecker(min_lookback_bars=100)
        warnings = checker.check(df, expected_start="2024-01-01")
        # Should only produce single_asset_universe (info), not halt.
        severe = [w for w in warnings if w.severity == "warning"]
        # We expect no severe warnings because dataset has ≥ 100 bars
        # and its start matches expected_start range.
        assert all(w.kind != "short_history" for w in severe), severe

    def test_check_does_not_raise(self):
        """Checker must never raise even on adversarial input."""
        cases = [
            pd.DataFrame(),
            pd.DataFrame({"$close": [1.0]}),
            _load_df(5),
        ]
        checker = SurvivorshipBiasChecker(min_lookback_bars=500)
        for df in cases:
            try:
                checker.check(df)
            except Exception as exc:
                pytest.fail(f"check() raised unexpectedly: {exc}")
