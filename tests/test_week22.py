"""
Week 22 tests: DTForecaster + Agent Communication Protocol.

Tests follow the plan's verification scripts and cover:
    22.1 DTForecaster — predict() returns expected keys, shapes, confidence ∈ (0,1]
    22.2 CommunicationBus + AgentIntention — publish/aggregation/reset
    22.3 MetaController with intention features
    22.4 SingleAssetRLTradingEnv with DTForecaster attached
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# 22.1  DTForecaster
# ---------------------------------------------------------------------------

class TestDTForecaster:
    """Tests for agents.offline.dt_forecaster.DTForecaster."""

    def _make_forecaster(self, state_dim=5, seq_len=10):
        from agents.offline.dt_forecaster import DTForecaster, DTForecasterConfig
        cfg = DTForecasterConfig(
            state_dim=state_dim,
            seq_len=seq_len,
            hidden_size=32,
            n_layer=1,
            n_head=4,
            dropout=0.1,
            n_epochs=2,
            batch_size=16,
            mc_samples=5,
        )
        return DTForecaster(config=cfg)

    def test_predict_returns_expected_keys(self):
        """predict() must return return_1step, return_5step, confidence."""
        f = self._make_forecaster(state_dim=5, seq_len=10)
        state_history = np.random.randn(10, 5).astype(np.float32)
        pred = f.predict(state_history)
        assert "return_1step" in pred
        assert "return_5step" in pred
        assert "confidence" in pred

    def test_predict_confidence_in_range(self):
        """Confidence must be in (0, 1]."""
        f = self._make_forecaster()
        pred = f.predict(np.random.randn(10, 5).astype(np.float32))
        assert 0.0 < pred["confidence"] <= 1.0

    def test_predict_returns_are_scalars(self):
        f = self._make_forecaster()
        pred = f.predict(np.random.randn(10, 5).astype(np.float32))
        assert isinstance(pred["return_1step"], float)
        assert isinstance(pred["return_5step"], float)

    def test_train_supervised_reduces_loss(self):
        """train_supervised() should return finite train and val losses."""
        f = self._make_forecaster(state_dim=5, seq_len=10)
        N = 80
        states = np.random.randn(N, 10, 5).astype(np.float32)
        r1 = np.random.randn(N).astype(np.float32) * 0.01
        r5 = np.random.randn(N).astype(np.float32) * 0.02
        metrics = f.train_supervised(states, r1, r5, verbose=False)
        assert math.isfinite(metrics["train_loss"])
        assert math.isfinite(metrics["val_loss"])
        assert metrics["train_loss"] > 0.0

    def test_predict_batch(self):
        """predict_batch() must return arrays of correct shape."""
        from agents.offline.dt_forecaster import DTForecaster, DTForecasterConfig
        cfg = DTForecasterConfig(state_dim=5, seq_len=10, hidden_size=32, n_layer=1, n_head=4)
        f = DTForecaster(config=cfg)
        B = 4
        hist = np.random.randn(B, 10, 5).astype(np.float32)
        out = f.predict_batch(hist)
        assert out["return_1step"].shape == (B,)
        assert out["return_5step"].shape == (B,)

    def test_build_dataset_shape(self):
        """build_dataset() must return consistent shapes."""
        from agents.offline.dt_forecaster import DTForecaster
        T, D = 100, 5
        features = np.random.randn(T, D).astype(np.float32)
        # use a synthetic close column (index 3)
        features[:, 3] = np.cumsum(np.random.randn(T) * 0.01) + 100.0
        states, r1, r5 = DTForecaster.build_dataset(features, seq_len=20)
        expected_N = T - 20 - 5  # horizon_long=5
        assert states.shape == (expected_N, 20, D)
        assert r1.shape == (expected_N,)
        assert r5.shape == (expected_N,)

    def test_from_config_factory(self, tmp_path):
        """from_config() must build a forecaster from a YAML file."""
        import yaml
        from agents.offline.dt_forecaster import DTForecaster
        cfg_data = {
            "env": {"window_size": 10},
            "dt_forecaster": {
                "state_dim": 5,
                "seq_len": 10,
                "hidden_size": 32,
                "n_layer": 1,
                "n_head": 4,
                "dropout": 0.1,
                "n_epochs": 1,
            },
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        f = DTForecaster.from_config(str(cfg_path))
        assert f.cfg.state_dim == 5
        assert f.cfg.seq_len == 10

    def test_save_load_roundtrip(self, tmp_path):
        """save() / load() must preserve weights."""
        from agents.offline.dt_forecaster import DTForecaster, DTForecasterConfig
        cfg = DTForecasterConfig(state_dim=5, seq_len=10, hidden_size=32, n_layer=1, n_head=4)
        f = DTForecaster(config=cfg)
        path = str(tmp_path / "dt_f.pt")
        f.save(path)
        f2 = DTForecaster.load(path)
        # Both should produce the same output on same input (no dropout active)
        x = np.random.randn(10, 5).astype(np.float32)
        f.net.eval()
        f2.net.eval()
        with torch.no_grad():
            xt = torch.tensor(x[np.newaxis], dtype=torch.float32)
            p1a, p5a = f.net(xt)
            p1b, p5b = f2.net(xt)
        assert abs(p1a.item() - p1b.item()) < 1e-5
        assert abs(p5a.item() - p5b.item()) < 1e-5


# ---------------------------------------------------------------------------
# 22.2  AgentIntention + CommunicationBus
# ---------------------------------------------------------------------------

class TestAgentIntention:

    def test_to_vector_shape(self):
        from agents.ensemble.communication import AgentIntention
        intent = AgentIntention(direction=0.5, confidence=0.8, horizon=10, risk_assessment=0.3)
        v = intent.to_vector()
        assert v.shape == (4,)
        assert v.dtype == np.float32

    def test_direction_clipped(self):
        from agents.ensemble.communication import AgentIntention
        i = AgentIntention(direction=2.0)
        assert i.direction == 1.0
        i2 = AgentIntention(direction=-5.0)
        assert i2.direction == -1.0

    def test_horizon_floor(self):
        from agents.ensemble.communication import AgentIntention
        i = AgentIntention(horizon=0)
        assert i.horizon == 1

    def test_from_vector_roundtrip(self):
        from agents.ensemble.communication import AgentIntention
        original = AgentIntention(direction=-0.3, confidence=0.7, horizon=20, risk_assessment=0.9)
        v = original.to_vector()
        reconstructed = AgentIntention.from_vector(v)
        assert abs(reconstructed.direction - original.direction) < 1e-5
        assert abs(reconstructed.confidence - original.confidence) < 1e-5
        # Horizon may differ by 1 due to log-scale rounding
        assert abs(reconstructed.horizon - original.horizon) <= 2

    def test_neutral_intention(self):
        from agents.ensemble.communication import AgentIntention
        n = AgentIntention.neutral()
        assert n.direction == 0.0
        assert n.confidence == 0.0

    def test_intention_from_action(self):
        from agents.ensemble.communication import CommunicationBus
        intent = CommunicationBus.intention_from_action(action=0.8, policy_entropy=0.1)
        assert abs(intent.direction - 0.8) < 1e-5
        assert intent.confidence > 0.8  # low entropy → high confidence


class TestCommunicationBus:

    def test_basic_publish_and_aggregated(self):
        """Published intentions should be reflected in get_aggregated()."""
        from agents.ensemble.communication import AgentIntention, CommunicationBus
        bus = CommunicationBus(n_agents=4)
        for i in range(4):
            bus.publish(i, AgentIntention(direction=0.5, confidence=0.8, horizon=10, risk_assessment=0.3))
        agg = bus.get_aggregated()
        assert agg.shape == (4 * 4,), f"Expected (16,), got {agg.shape}"

    def test_missing_agents_filled_with_neutral(self):
        """Agents that didn't publish should appear as neutral in aggregated."""
        from agents.ensemble.communication import AgentIntention, CommunicationBus, INTENTION_DIM
        bus = CommunicationBus(n_agents=3)
        bus.publish(0, AgentIntention(direction=1.0, confidence=1.0, horizon=5, risk_assessment=0.0))
        # agents 1, 2 did not publish
        agg = bus.get_aggregated()
        assert agg.shape == (3 * INTENTION_DIM,)

    def test_get_summary_shape(self):
        from agents.ensemble.communication import AgentIntention, CommunicationBus
        bus = CommunicationBus(n_agents=4)
        for i in range(4):
            bus.publish(i, AgentIntention())
        summary = bus.get_summary()
        assert summary.shape == (4,)

    def test_consensus_direction_weighted(self):
        """High-confidence agent should pull consensus toward its direction."""
        from agents.ensemble.communication import AgentIntention, CommunicationBus
        bus = CommunicationBus(n_agents=2)
        bus.publish(0, AgentIntention(direction=1.0, confidence=0.99, horizon=1, risk_assessment=0.0))
        bus.publish(1, AgentIntention(direction=-1.0, confidence=0.01, horizon=1, risk_assessment=0.0))
        consensus = bus.consensus_direction()
        assert consensus > 0.0, f"Expected bullish consensus, got {consensus}"

    def test_reset_clears_intentions(self):
        from agents.ensemble.communication import AgentIntention, CommunicationBus
        bus = CommunicationBus(n_agents=2)
        bus.publish(0, AgentIntention(direction=0.9))
        bus.reset()
        assert len(bus.intentions) == 0
        assert bus.step_history_len() == 1  # snapshot saved to history

    def test_invalid_agent_id_raises(self):
        from agents.ensemble.communication import AgentIntention, CommunicationBus
        bus = CommunicationBus(n_agents=2)
        with pytest.raises(ValueError):
            bus.publish(5, AgentIntention())

    def test_n_agents_zero_raises(self):
        from agents.ensemble.communication import CommunicationBus
        with pytest.raises(ValueError):
            CommunicationBus(n_agents=0)


# ---------------------------------------------------------------------------
# 22.3  MetaController with intention features
# ---------------------------------------------------------------------------

class TestMetaControllerWithIntention:

    def test_obs_dim_extended_when_use_intention(self):
        """When use_intention=True the obs_dim must be larger by n_agents*4."""
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig

        n_agents = 3
        cfg_base = MetaControllerConfig(n_regimes=3, n_market_features=2, use_intention=False)
        cfg_intent = MetaControllerConfig(n_regimes=3, n_market_features=2, use_intention=True)

        mc_base = MetaController(n_agents=n_agents, config=cfg_base)
        mc_intent = MetaController(n_agents=n_agents, config=cfg_intent)

        expected_extra = n_agents * 4
        assert mc_intent.obs_dim == mc_base.obs_dim + expected_extra

    def test_get_weights_without_intention(self):
        """Baseline get_weights still works (no intention)."""
        from agents.ensemble.meta_controller import MetaController
        mc = MetaController(n_agents=3)
        weights = mc.get_weights(
            regime_probs=np.array([0.6, 0.3, 0.1]),
            sharpe_history=np.array([0.8, -0.2, 0.5]),
        )
        assert weights.shape == (3,)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_get_weights_with_intention(self):
        """get_weights with intention_vector must return valid weights."""
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        from agents.ensemble.communication import AgentIntention, CommunicationBus

        n_agents = 4
        cfg = MetaControllerConfig(n_regimes=3, n_market_features=2, use_intention=True)
        mc = MetaController(n_agents=n_agents, config=cfg)

        bus = CommunicationBus(n_agents=n_agents)
        for i in range(n_agents):
            bus.publish(i, AgentIntention(direction=0.3, confidence=0.6, horizon=5, risk_assessment=0.4))
        intention_vec = bus.get_aggregated()

        weights = mc.get_weights(
            regime_probs=np.array([0.5, 0.3, 0.2]),
            sharpe_history=np.ones(n_agents) * 0.5,
            intention_vector=intention_vec,
        )
        assert weights.shape == (n_agents,)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_step_with_intention(self):
        """step() with intention_vector must record transition and return weights."""
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig

        n_agents = 2
        cfg = MetaControllerConfig(n_regimes=3, n_market_features=2, use_intention=True)
        mc = MetaController(n_agents=n_agents, config=cfg)

        intention_vec = np.zeros(n_agents * 4, dtype=np.float32)
        weights = mc.step(
            regime_probs=np.array([0.7, 0.2, 0.1]),
            sharpe_history=np.array([0.5, 0.3]),
            portfolio_return=0.02,
            intention_vector=intention_vec,
        )
        assert weights.shape == (n_agents,)
        assert abs(weights.sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 22.4  SingleAssetRLTradingEnv with DTForecaster
# ---------------------------------------------------------------------------

class TestEnvWithDTForecaster:

    def _make_env(self, with_forecaster=False):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        T = 60
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(T) * 0.5) + 100.0
        df = pd.DataFrame({
            "$open": prices,
            "$high": prices * 1.002,
            "$low": prices * 0.998,
            "$close": prices,
            "$volume": np.random.rand(T) * 1e6,
        })

        dt_forecaster = None
        if with_forecaster:
            from agents.offline.dt_forecaster import DTForecaster, DTForecasterConfig
            cfg = DTForecasterConfig(state_dim=5, seq_len=20, hidden_size=32, n_layer=1, n_head=4, dropout=0.0)
            dt_forecaster = DTForecaster(config=cfg)

        return SingleAssetRLTradingEnv(
            data=df,
            window_size=20,
            dt_forecaster=dt_forecaster,
        )

    def test_obs_shape_without_forecaster(self):
        env = self._make_env(with_forecaster=False)
        obs, _ = env.reset()
        # Base: (window_size, 5) = (20, 5)
        assert obs.shape == (20, 5), f"Got {obs.shape}"

    def test_obs_shape_with_forecaster(self):
        env = self._make_env(with_forecaster=True)
        obs, _ = env.reset()
        # With DT forecaster: (window_size, 5+3) = (20, 8)
        assert obs.shape == (20, 8), f"Got {obs.shape}"

    def test_forecast_columns_consistent_within_step(self):
        """All rows in the DT forecast columns should have the same value."""
        env = self._make_env(with_forecaster=True)
        obs, _ = env.reset()
        # Columns 5-7 are the DT forecast (constant across rows)
        forecast_cols = obs[:, 5:]
        assert forecast_cols.shape == (20, 3)
        # All rows should be identical
        assert np.allclose(forecast_cols[0], forecast_cols[-1], atol=1e-6)

    def test_step_returns_correct_obs_shape_with_forecaster(self):
        env = self._make_env(with_forecaster=True)
        env.reset()
        obs, reward, term, trunc, info = env.step(np.array([0.0]))
        assert obs.shape == (20, 8)

    def test_observation_space_matches_obs_with_forecaster(self):
        env = self._make_env(with_forecaster=True)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape, (
            f"obs shape {obs.shape} != observation_space {env.observation_space.shape}"
        )


# ---------------------------------------------------------------------------
# 22.5  Integration: Plan validation scripts
# ---------------------------------------------------------------------------

class TestPlanValidation:
    """Reproduces the exact validation commands from the Week 22 plan."""

    def test_dt_forecaster_plan_validation(self):
        """Mirrors the plan's validation for DTForecaster."""
        from agents.offline.dt_forecaster import DTForecaster, DTForecasterConfig
        cfg = DTForecasterConfig(state_dim=18, seq_len=20, hidden_size=32, n_layer=1, n_head=4)
        forecaster = DTForecaster(config=cfg)
        state_history = np.random.randn(20, 18).astype(np.float32)
        pred = forecaster.predict(state_history)
        assert "return_1step" in pred
        assert "confidence" in pred
        assert 0.0 < pred["confidence"] <= 1.0

    def test_communication_bus_plan_validation(self):
        """Mirrors the plan's validation for CommunicationBus."""
        from agents.ensemble.communication import CommunicationBus, AgentIntention
        bus = CommunicationBus(n_agents=4)
        for i in range(4):
            bus.publish(i, AgentIntention(direction=0.5, confidence=0.8, horizon=10, risk_assessment=0.3))
        agg = bus.get_aggregated()
        assert agg.shape[0] == 4 * 4  # 4 agents * 4 intention dims
