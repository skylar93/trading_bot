"""
Week 61 (S30): Integration tests for build_system DI entry-point.

Tests verify:
- build_system wires components in canonical order
- Mock risk_manager / mock data_source injection works
- Fallback to None when required pieces are missing
- DataSource (StaticDataSource) round-trip
- SingleAssetRLTradingEnv accepts data_source kwarg (S27)
- PaperTrader accepts Optional[RiskManagerBase] (S26)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.sources.base import DataSource, StaticDataSource
from training.factories.build_system import SystemComponents, build_system


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "$open": price,
        "$high": price * 1.01,
        "$low": price * 0.99,
        "$close": price,
        "$volume": rng.uniform(1e5, 1e6, n),
    })


@pytest.fixture()
def minimal_config() -> dict:
    return {
        "env": {
            "type": "single_asset_rl",
            "window_size": 20,
            "initial_capital": 10_000.0,
        },
        "risk_management": {
            "type": "rl",
        },
        "paper_trading": {
            "enabled": False,
        },
    }


# ---------------------------------------------------------------------------
# StaticDataSource (S27)
# ---------------------------------------------------------------------------

class TestStaticDataSource:
    def test_len(self, sample_df):
        ds = StaticDataSource(sample_df)
        assert len(ds) == len(sample_df)

    def test_is_live_false(self, sample_df):
        ds = StaticDataSource(sample_df)
        assert ds.is_live() is False

    def test_get_window_shape(self, sample_df):
        ds = StaticDataSource(sample_df)
        window = ds.get_window(10, 20)
        assert len(window) == 10

    def test_latest_returns_series(self, sample_df):
        ds = StaticDataSource(sample_df)
        latest = ds.latest()
        assert isinstance(latest, pd.Series)
        assert latest["$close"] == pytest.approx(sample_df["$close"].iloc[-1])

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            StaticDataSource(pd.DataFrame())

    def test_index_reset(self, sample_df):
        # Even if original df has non-zero index, StaticDataSource resets it
        df_shifted = sample_df.copy()
        df_shifted.index = range(50, 50 + len(df_shifted))
        ds = StaticDataSource(df_shifted)
        window = ds.get_window(0, 5)
        assert list(window.index) == [0, 1, 2, 3, 4]

    def test_datasource_is_abstract(self):
        with pytest.raises(TypeError):
            DataSource()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SingleAssetRLTradingEnv — data_source injection (S27)
# ---------------------------------------------------------------------------

class TestEnvDataSourceInjection:
    def test_data_kwarg_creates_data_source(self, sample_df):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        env = SingleAssetRLTradingEnv(data=sample_df, window_size=20)
        assert env.data_source is not None
        assert isinstance(env.data_source, StaticDataSource)
        assert len(env.data_source) == len(sample_df)

    def test_data_source_kwarg_accepted(self, sample_df):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        ds = StaticDataSource(sample_df)
        env = SingleAssetRLTradingEnv(data_source=ds, window_size=20)
        assert env.data_source is ds

    def test_data_source_takes_priority_over_data(self, sample_df):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        ds = StaticDataSource(sample_df)
        other_df = sample_df.copy()
        other_df["$close"] = 999.0
        env = SingleAssetRLTradingEnv(data=other_df, data_source=ds, window_size=20)
        # data_source wins; close should NOT be 999
        assert env.data_source is ds

    def test_no_data_env_data_source_none(self):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        env = SingleAssetRLTradingEnv(window_size=20)
        assert env.data_source is None

    def test_env_reset_with_data_source(self, sample_df):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        ds = StaticDataSource(sample_df)
        env = SingleAssetRLTradingEnv(data_source=ds, window_size=20)
        obs, _ = env.reset()
        assert obs is not None


# ---------------------------------------------------------------------------
# PaperTrader — Optional[RiskManagerBase] type annotation (S26)
# ---------------------------------------------------------------------------

class TestPaperTraderRiskManagerInjection:
    def _make_mock_agent(self):
        """Minimal duck-typed agent."""
        class _Agent:
            def predict(self, obs, deterministic=True):
                return 0, None
        return _Agent()

    def test_paper_trader_accepts_none_risk_manager(self):
        from deployment.paper_trader import PaperTrader

        agent = self._make_mock_agent()
        trader = PaperTrader(agent=agent, config={}, risk_manager=None)
        assert trader.risk_manager is None

    def test_paper_trader_accepts_risk_manager_instance(self):
        from deployment.paper_trader import PaperTrader
        from risk_management.factory import create_risk_manager

        agent = self._make_mock_agent()
        rm = create_risk_manager("rl")
        trader = PaperTrader(agent=agent, config={}, risk_manager=rm)
        assert trader.risk_manager is rm


# ---------------------------------------------------------------------------
# build_system — component assembly (S28)
# ---------------------------------------------------------------------------

class TestBuildSystem:
    def test_returns_system_components(self, sample_df, minimal_config):
        result = build_system(minimal_config, data=sample_df)
        assert isinstance(result, SystemComponents)

    def test_data_source_populated(self, sample_df, minimal_config):
        result = build_system(minimal_config, data=sample_df)
        assert result.data_source is not None
        assert isinstance(result.data_source, StaticDataSource)

    def test_env_populated(self, sample_df, minimal_config):
        result = build_system(minimal_config, data=sample_df)
        assert result.env is not None

    def test_risk_manager_populated(self, sample_df, minimal_config):
        result = build_system(minimal_config, data=sample_df)
        assert result.risk_manager is not None

    def test_agent_none_when_no_model_path(self, sample_df, minimal_config):
        result = build_system(minimal_config, data=sample_df)
        assert result.agent is None

    def test_trader_none_when_no_agent(self, sample_df, minimal_config):
        result = build_system(minimal_config, data=sample_df)
        assert result.trader is None

    def test_mock_risk_manager_injection(self, sample_df, minimal_config):
        from unittest.mock import MagicMock
        from risk_management.risk_manager_base import RiskManagerBase

        mock_rm = MagicMock(spec=RiskManagerBase)
        result = build_system(minimal_config, data=sample_df, risk_manager=mock_rm)
        assert result.risk_manager is mock_rm

    def test_mock_data_source_injection(self, sample_df, minimal_config):
        ds = StaticDataSource(sample_df)
        result = build_system(minimal_config, data_source=ds)
        assert result.data_source is ds

    def test_no_data_gives_none_env(self, minimal_config):
        result = build_system(minimal_config)
        assert result.data_source is None
        assert result.env is None

    def test_data_override_takes_precedence_over_data_source(self, sample_df, minimal_config):
        """When both data= and data_source= given, data_source wins."""
        ds = StaticDataSource(sample_df)
        other = sample_df.copy()
        result = build_system(minimal_config, data=other, data_source=ds)
        assert result.data_source is ds

    def test_mock_agent_creates_trader(self, sample_df, minimal_config):
        """Providing a mock agent should let build_system create a PaperTrader."""
        class _MockAgent:
            def predict(self, obs, deterministic=True):
                return 0, None

        result = build_system(minimal_config, data=sample_df, agent=_MockAgent())
        assert result.agent is not None
        assert result.trader is not None
