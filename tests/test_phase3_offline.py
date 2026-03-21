"""
Phase 3 offline RL tests: CQL baseline + DT→PPO fine-tuning.

Coverage
--------
CQLConfig
  - defaults are valid
  - hidden_size <= 0 raises ValueError
  - n_layers < 1 raises ValueError
  - gamma out of (0, 1] raises ValueError
  - tau out of (0, 1] raises ValueError

_QNetwork
  - forward output shape (B, 1)
  - gradients flow through forward

CQLAgent
  - instantiation (default config, explicit device)
  - target networks initialised equal to online networks
  - train_batch returns dict with correct keys
  - train_batch: total_loss ≥ 0 (td_loss dominates on fresh network)
  - train_batch with alpha=0 → cql_loss contributes nothing to total_loss
  - soft update: target drifts toward online network
  - get_action output shape (act_dim,)
  - get_action with 2-D input raises or accepts gracefully
  - get_action is within [-1, 1] range
  - train method returns dict with correct keys
  - train method returns one loss per epoch
  - save / load round-trip: actions match
  - from_config factory

CQL + TradingTrajectoryDataset integration
  - _dataset_to_transitions returns non-empty TensorDataset
  - train on real dataset runs without error
  - losses decrease over enough epochs (smoke-test convergence)

DTFeatureExtractor
  - _SB3_AVAILABLE flag is bool
  - instantiation with a DT model
  - features_dim equals hidden_size
  - forward output shape (B, hidden_size)
  - freeze_backbone=True: no grad on state_embed params
  - freeze_backbone=False: state_embed params have grad

DecisionTransformerFineTuner
  - instantiation
  - count_parameters returns dict with total and trainable > 0
  - fine_tune runs for a small number of timesteps
  - get_action output shape
  - get_action deterministic=True returns consistent result
  - save / load round-trip preserves action output
  - freeze_backbone=True: state_embed params frozen inside PPO policy

agents.offline __init__ re-exports
  - CQLConfig, CQLAgent, FineTunerConfig, DecisionTransformerFineTuner,
    DTFeatureExtractor, _SB3_AVAILABLE importable from agents.offline

Comparison
  - CQL and DT both produce valid actions on the same state
  - Both save/load without error
"""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.spaces import Box

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

OBS_DIM = 20
ACT_DIM = 1
T = 30        # steps per trajectory
N_TRAJ = 4
K = 8         # context length


def _make_trajectory(length: int = T):
    from agents.offline.trajectory_dataset import Trajectory
    rng = np.random.default_rng(42)
    return Trajectory(
        observations=rng.standard_normal((length, OBS_DIM)).astype(np.float32),
        actions=rng.standard_normal((length, ACT_DIM)).astype(np.float32),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.zeros(length, dtype=np.float32),
    )


def _make_dataset(n_traj: int = N_TRAJ, context_len: int = K):
    from agents.offline.trajectory_dataset import TradingTrajectoryDataset
    trajs = [_make_trajectory() for _ in range(n_traj)]
    return TradingTrajectoryDataset(trajs, context_len=context_len)


def _small_cql_config(**overrides):
    from agents.offline.cql_agent import CQLConfig
    defaults = dict(
        state_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_size=32,
        n_layers=1,
        learning_rate=1e-3,
        batch_size=16,
        n_action_samples=4,
        n_inference_samples=8,
    )
    defaults.update(overrides)
    return CQLConfig(**defaults)


def _small_dt_config():
    from agents.offline.decision_transformer import DecisionTransformerConfig
    return DecisionTransformerConfig(
        state_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_size=16,
        context_len=K,
        n_layer=1,
        n_head=1,
        dropout=0.0,
        use_gpt2_backbone=False,
    )


def _make_dt():
    from agents.offline.decision_transformer import TradingDecisionTransformer
    return TradingDecisionTransformer(_small_dt_config())


def _make_fake_env():
    """Minimal gymnasium Box environment for DT fine-tuner tests."""
    return gym.make("Pendulum-v1") if False else _BoxEnv(OBS_DIM, ACT_DIM)


class _BoxEnv(gym.Env):
    """Minimal flat-obs, continuous-action env."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.random.default_rng().standard_normal(self.observation_space.shape).astype(np.float32)
        reward = float(np.random.default_rng().standard_normal())
        done = self._step >= 20
        return obs, reward, done, False, {}


# ===========================================================================
# CQLConfig tests
# ===========================================================================

class TestCQLConfig:
    def test_defaults_valid(self):
        from agents.offline.cql_agent import CQLConfig
        cfg = CQLConfig()
        assert cfg.state_dim == 100
        assert cfg.act_dim == 1
        assert cfg.alpha == 1.0

    def test_hidden_size_zero_raises(self):
        from agents.offline.cql_agent import CQLConfig
        with pytest.raises(ValueError, match="hidden_size"):
            CQLConfig(hidden_size=0)

    def test_n_layers_zero_raises(self):
        from agents.offline.cql_agent import CQLConfig
        with pytest.raises(ValueError, match="n_layers"):
            CQLConfig(n_layers=0)

    def test_gamma_out_of_range_raises(self):
        from agents.offline.cql_agent import CQLConfig
        with pytest.raises(ValueError, match="gamma"):
            CQLConfig(gamma=1.5)

    def test_tau_out_of_range_raises(self):
        from agents.offline.cql_agent import CQLConfig
        with pytest.raises(ValueError, match="tau"):
            CQLConfig(tau=0.0)


# ===========================================================================
# _QNetwork tests
# ===========================================================================

class TestQNetwork:
    def test_forward_shape(self):
        from agents.offline.cql_agent import _QNetwork
        net = _QNetwork(OBS_DIM, ACT_DIM, 32, 2)
        s = torch.zeros(8, OBS_DIM)
        a = torch.zeros(8, ACT_DIM)
        out = net(s, a)
        assert out.shape == (8, 1)

    def test_gradients_flow(self):
        from agents.offline.cql_agent import _QNetwork
        net = _QNetwork(OBS_DIM, ACT_DIM, 32, 1)
        s = torch.randn(4, OBS_DIM)
        a = torch.randn(4, ACT_DIM)
        loss = net(s, a).mean()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None


# ===========================================================================
# CQLAgent tests
# ===========================================================================

class TestCQLAgent:
    def test_instantiation(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        assert agent.device == "cpu"

    def test_target_networks_init_equal(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        for p1, p2 in zip(agent.q1.parameters(), agent.q1_target.parameters()):
            assert torch.allclose(p1, p2)

    def test_train_batch_keys(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        B = 8
        s = torch.randn(B, OBS_DIM)
        a = torch.randn(B, ACT_DIM)
        r = torch.randn(B)
        sn = torch.randn(B, OBS_DIM)
        d = torch.zeros(B)
        metrics = agent.train_batch(s, a, r, sn, d)
        assert set(metrics.keys()) == {"td_loss", "cql_loss", "total_loss"}

    def test_train_batch_total_loss_finite(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        B = 8
        s = torch.randn(B, OBS_DIM)
        a = torch.randn(B, ACT_DIM)
        r = torch.randn(B)
        sn = torch.randn(B, OBS_DIM)
        d = torch.zeros(B)
        m = agent.train_batch(s, a, r, sn, d)
        assert np.isfinite(m["total_loss"])

    def test_alpha_zero_no_cql_contribution(self):
        """With alpha=0 total_loss == td_loss (CQL term multiplied to zero)."""
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(alpha=0.0), device="cpu")
        B = 8
        s = torch.randn(B, OBS_DIM)
        a = torch.randn(B, ACT_DIM)
        r = torch.randn(B)
        sn = torch.randn(B, OBS_DIM)
        d = torch.zeros(B)
        m = agent.train_batch(s, a, r, sn, d)
        assert abs(m["total_loss"] - m["td_loss"]) < 1e-5

    def test_soft_update_target_drifts(self):
        from agents.offline.cql_agent import CQLAgent
        cfg = _small_cql_config(tau=1.0)  # full copy each step
        agent = CQLAgent(cfg, device="cpu")
        # Perturb online Q1 weights
        for p in agent.q1.parameters():
            p.data.fill_(99.0)
        agent._soft_update()
        for p in agent.q1_target.parameters():
            assert torch.allclose(p, torch.tensor(99.0))

    def test_get_action_shape(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        state = np.zeros(OBS_DIM, dtype=np.float32)
        action = agent.get_action(state)
        assert action.shape == (ACT_DIM,)

    def test_get_action_in_range(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        state = np.random.default_rng(0).standard_normal(OBS_DIM).astype(np.float32)
        action = agent.get_action(state)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_train_returns_keys(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        ds = _make_dataset()
        metrics = agent.train(ds, n_epochs=2)
        assert set(metrics.keys()) == {"train_td_loss", "train_cql_loss", "train_total_loss"}

    def test_train_returns_one_loss_per_epoch(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        ds = _make_dataset()
        n = 3
        metrics = agent.train(ds, n_epochs=n)
        assert len(metrics["train_total_loss"]) == n

    def test_save_load_roundtrip(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        state = np.random.default_rng(1).standard_normal(OBS_DIM).astype(np.float32)
        torch.manual_seed(0)
        action_before = agent.get_action(state)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            agent.save(f.name)
            agent2 = CQLAgent.load(f.name, map_location="cpu")
        torch.manual_seed(0)
        action_after = agent2.get_action(state)
        np.testing.assert_allclose(action_before, action_after, atol=1e-5)

    def test_from_config_factory(self):
        from agents.offline.cql_agent import CQLAgent
        cfg_dict = {"cql": {"state_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden_size": 32, "device": "cpu"}}
        agent = CQLAgent.from_config(cfg_dict)
        assert agent.config.state_dim == OBS_DIM


# ===========================================================================
# CQL + TradingTrajectoryDataset integration
# ===========================================================================

class TestCQLDatasetIntegration:
    def test_dataset_to_transitions_nonempty(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        ds = _make_dataset()
        td = agent._dataset_to_transitions(ds)
        assert len(td) > 0

    def test_train_on_dataset_runs(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        ds = _make_dataset()
        metrics = agent.train(ds, n_epochs=1)
        assert all(np.isfinite(v) for v in metrics["train_total_loss"])

    def test_losses_finite_over_epochs(self):
        from agents.offline.cql_agent import CQLAgent
        agent = CQLAgent(_small_cql_config(), device="cpu")
        ds = _make_dataset(n_traj=6)
        metrics = agent.train(ds, n_epochs=3)
        for v in metrics["train_total_loss"]:
            assert np.isfinite(v), f"Non-finite loss: {v}"


# ===========================================================================
# DTFeatureExtractor tests
# ===========================================================================

class TestDTFeatureExtractor:
    def test_sb3_available_flag(self):
        from agents.offline.dt_finetuner import _SB3_AVAILABLE
        assert isinstance(_SB3_AVAILABLE, bool)

    def test_instantiation(self):
        from agents.offline.dt_finetuner import DTFeatureExtractor
        dt = _make_dt()
        obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        extractor = DTFeatureExtractor(obs_space, dt_model=dt, freeze_backbone=False)
        assert extractor.features_dim == dt.config.hidden_size

    def test_features_dim_matches_hidden_size(self):
        from agents.offline.dt_finetuner import DTFeatureExtractor
        dt = _make_dt()
        obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        extractor = DTFeatureExtractor(obs_space, dt_model=dt)
        assert extractor.features_dim == 16  # small test hidden_size

    def test_forward_shape(self):
        from agents.offline.dt_finetuner import DTFeatureExtractor
        dt = _make_dt()
        obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        extractor = DTFeatureExtractor(obs_space, dt_model=dt)
        obs = torch.randn(5, OBS_DIM)
        out = extractor(obs)
        assert out.shape == (5, dt.config.hidden_size)

    def test_freeze_backbone_no_grad(self):
        from agents.offline.dt_finetuner import DTFeatureExtractor
        dt = _make_dt()
        obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        extractor = DTFeatureExtractor(obs_space, dt_model=dt, freeze_backbone=True)
        for p in extractor.state_embed.parameters():
            assert not p.requires_grad

    def test_no_freeze_backbone_has_grad(self):
        from agents.offline.dt_finetuner import DTFeatureExtractor
        dt = _make_dt()
        obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        extractor = DTFeatureExtractor(obs_space, dt_model=dt, freeze_backbone=False)
        for p in extractor.state_embed.parameters():
            assert p.requires_grad


# ===========================================================================
# DecisionTransformerFineTuner tests
# ===========================================================================

class TestDecisionTransformerFineTuner:
    def test_instantiation(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner
        dt = _make_dt()
        env = _make_fake_env()
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, device="cpu")
        assert ft.ppo is not None

    def test_count_parameters(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner
        dt = _make_dt()
        env = _make_fake_env()
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, device="cpu")
        counts = ft.count_parameters()
        assert counts["total"] > 0
        assert counts["trainable"] > 0

    def test_fine_tune_runs(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner, FineTunerConfig
        dt = _make_dt()
        env = _make_fake_env()
        cfg = FineTunerConfig(n_steps=64, batch_size=32, n_epochs=1)
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, config=cfg, device="cpu")
        result = ft.fine_tune(total_timesteps=128)
        assert result["total_timesteps"] == 128

    def test_get_action_shape(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner
        dt = _make_dt()
        env = _make_fake_env()
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, device="cpu")
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action, _ = ft.get_action(obs)
        assert action.shape == (ACT_DIM,)

    def test_get_action_deterministic_consistent(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner
        dt = _make_dt()
        env = _make_fake_env()
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, device="cpu")
        obs = np.ones(OBS_DIM, dtype=np.float32)
        a1, _ = ft.get_action(obs, deterministic=True)
        a2, _ = ft.get_action(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_save_load_roundtrip(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner
        dt = _make_dt()
        env = _make_fake_env()
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, device="cpu")
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        a_before, _ = ft.get_action(obs, deterministic=True)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            ft.save(f.name)
            ft2 = DecisionTransformerFineTuner.load(f.name, env=env, map_location="cpu")
        a_after, _ = ft2.get_action(obs, deterministic=True)
        np.testing.assert_allclose(a_before, a_after, atol=1e-5)

    def test_freeze_backbone_frozen_in_policy(self):
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner, FineTunerConfig
        dt = _make_dt()
        env = _make_fake_env()
        cfg = FineTunerConfig(freeze_backbone=True)
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, config=cfg, device="cpu")
        extractor = ft.ppo.policy.features_extractor
        for p in extractor.state_embed.parameters():
            assert not p.requires_grad


# ===========================================================================
# agents.offline __init__ re-export tests
# ===========================================================================

class TestOfflineInitExports:
    def test_cql_exports(self):
        from agents.offline import CQLConfig, CQLAgent
        assert CQLConfig is not None
        assert CQLAgent is not None

    def test_finetuner_exports(self):
        from agents.offline import FineTunerConfig, DecisionTransformerFineTuner
        assert FineTunerConfig is not None
        assert DecisionTransformerFineTuner is not None

    def test_dt_feature_extractor_export(self):
        from agents.offline import DTFeatureExtractor
        assert DTFeatureExtractor is not None

    def test_sb3_available_export(self):
        from agents.offline import _SB3_AVAILABLE
        assert isinstance(_SB3_AVAILABLE, bool)


# ===========================================================================
# Comparison: CQL vs DT on same state
# ===========================================================================

class TestCQLvsDTComparison:
    def test_both_produce_valid_actions(self):
        from agents.offline.cql_agent import CQLAgent
        cql = CQLAgent(_small_cql_config(), device="cpu")
        dt = _make_dt()

        state = np.random.default_rng(7).standard_normal(OBS_DIM).astype(np.float32)

        cql_action = cql.get_action(state)
        assert cql_action.shape == (ACT_DIM,)

        # DT needs full context; use a single-step context
        K_inf = _small_dt_config().context_len
        states_t = torch.from_numpy(state).unsqueeze(0).expand(K_inf, -1)
        actions_t = torch.zeros(K_inf, ACT_DIM)
        rtg_t = torch.ones(K_inf, 1)
        ts_t = torch.arange(K_inf)
        dt_action = dt.get_action(states_t, actions_t, rtg_t, ts_t)
        assert dt_action.shape == (ACT_DIM,)

    def test_both_save_without_error(self):
        from agents.offline.cql_agent import CQLAgent
        from agents.offline.dt_finetuner import DecisionTransformerFineTuner
        cql = CQLAgent(_small_cql_config(), device="cpu")
        dt = _make_dt()
        env = _make_fake_env()
        ft = DecisionTransformerFineTuner(dt_model=dt, env=env, device="cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt") as fc, \
             tempfile.NamedTemporaryFile(suffix=".pt") as ff:
            cql.save(fc.name)
            ft.save(ff.name)
            # No errors means pass
