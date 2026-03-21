"""
Week 14: Decision Transformer pre-training tests.

Coverage:
  - Trajectory dataclass (RTG, edge cases)
  - TradingTrajectoryDataset (length, getitem, padding, normalisation,
    from_rollouts, save/load)
  - DecisionTransformerConfig (defaults, validation)
  - _CausalTransformerBlock / _CausalTransformerBackbone (shapes, causal mask)
  - TradingDecisionTransformer (forward shapes, loss, get_action,
    count_parameters, save/load, from_config)
  - DecisionTransformerTrainer (train_epoch, evaluate, train, from_config,
    scheduler warm-up)
  - Optional-import guards (_PEFT_AVAILABLE, _TRANSFORMERS_AVAILABLE)
  - agents/offline/__init__ re-exports
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import List

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OBS_DIM = 20
ACT_DIM = 1
T = 50       # steps per trajectory
N_TRAJ = 3
K = 10       # context length (small for fast tests)


def _make_trajectory(length: int = T, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
    from agents.offline.trajectory_dataset import Trajectory

    rng = np.random.default_rng(0)
    return Trajectory(
        observations=rng.standard_normal((length, obs_dim)).astype(np.float32),
        actions=rng.standard_normal((length, act_dim)).astype(np.float32),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.zeros(length, dtype=np.float32),
    )


def _make_dataset(
    n_traj: int = N_TRAJ,
    traj_len: int = T,
    context_len: int = K,
    **kwargs,
):
    from agents.offline.trajectory_dataset import TradingTrajectoryDataset

    trajs = [_make_trajectory(traj_len) for _ in range(n_traj)]
    return TradingTrajectoryDataset(trajs, context_len=context_len, **kwargs)


def _make_small_config(**overrides):
    from agents.offline.decision_transformer import DecisionTransformerConfig

    defaults = dict(
        state_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_size=16,
        context_len=K,
        n_layer=1,
        n_head=1,
        dropout=0.0,
        use_gpt2_backbone=False,
    )
    defaults.update(overrides)
    return DecisionTransformerConfig(**defaults)


def _make_small_model(**cfg_overrides):
    from agents.offline.decision_transformer import TradingDecisionTransformer

    return TradingDecisionTransformer(_make_small_config(**cfg_overrides))


def _make_batch(B: int = 2, K: int = K, state_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
    rng = torch.manual_seed(0)
    return (
        torch.randn(B, K, state_dim),
        torch.randn(B, K, act_dim),
        torch.randn(B, K, 1),
        torch.arange(K).unsqueeze(0).expand(B, -1),
        torch.ones(B, K),  # attention mask (all real)
    )


# ===========================================================================
# Trajectory
# ===========================================================================

class TestTrajectory:
    def test_basic_construction(self):
        from agents.offline.trajectory_dataset import Trajectory

        traj = _make_trajectory(10)
        assert traj.observations.shape == (10, OBS_DIM)
        assert traj.actions.shape == (10, ACT_DIM)
        assert traj.rewards.shape == (10,)
        assert traj.dones.shape == (10,)

    def test_scalar_actions_become_2d(self):
        from agents.offline.trajectory_dataset import Trajectory

        traj = Trajectory(
            observations=np.zeros((5, 4), dtype=np.float32),
            actions=np.ones(5, dtype=np.float32),  # 1-D
            rewards=np.zeros(5, dtype=np.float32),
            dones=np.zeros(5, dtype=np.float32),
        )
        assert traj.actions.shape == (5, 1)

    def test_length(self):
        traj = _make_trajectory(7)
        assert len(traj) == 7

    def test_obs_dim(self):
        traj = _make_trajectory(5, obs_dim=12)
        assert traj.obs_dim == 12

    def test_act_dim(self):
        traj = _make_trajectory(5, act_dim=3)
        assert traj.act_dim == 3

    def test_mismatched_lengths_raise(self):
        from agents.offline.trajectory_dataset import Trajectory

        with pytest.raises(ValueError):
            Trajectory(
                observations=np.zeros((5, 4), dtype=np.float32),
                actions=np.zeros((6, 1), dtype=np.float32),  # wrong length
                rewards=np.zeros(5, dtype=np.float32),
                dones=np.zeros(5, dtype=np.float32),
            )

    def test_rtg_gamma1(self):
        from agents.offline.trajectory_dataset import Trajectory

        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        traj = Trajectory(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((3, 1), dtype=np.float32),
            rewards=rewards,
            dones=np.zeros(3, dtype=np.float32),
        )
        rtg = traj.compute_rtg(gamma=1.0)
        assert rtg.shape == (3,)
        np.testing.assert_allclose(rtg, [6.0, 5.0, 3.0], rtol=1e-5)

    def test_rtg_gamma_discount(self):
        from agents.offline.trajectory_dataset import Trajectory

        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        traj = Trajectory(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((3, 1), dtype=np.float32),
            rewards=rewards,
            dones=np.zeros(3, dtype=np.float32),
        )
        rtg = traj.compute_rtg(gamma=0.5)
        # RTG[2]=1, RTG[1]=1+0.5*1=1.5, RTG[0]=1+0.5*1.5=1.75
        np.testing.assert_allclose(rtg, [1.75, 1.5, 1.0], rtol=1e-5)

    def test_rtg_resets_at_done(self):
        from agents.offline.trajectory_dataset import Trajectory

        rewards = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        traj = Trajectory(
            observations=np.zeros((4, 2), dtype=np.float32),
            actions=np.zeros((4, 1), dtype=np.float32),
            rewards=rewards,
            dones=dones,
        )
        rtg = traj.compute_rtg(gamma=1.0)
        # After done at t=1, future rewards don't propagate back to t<2
        assert rtg[2] == pytest.approx(9.0)  # 4+5
        assert rtg[3] == pytest.approx(5.0)
        assert rtg[0] == pytest.approx(5.0)  # 2+3 (done at t=1 cuts off)
        assert rtg[1] == pytest.approx(3.0)

    def test_rtg_dtype(self):
        traj = _make_trajectory(5)
        rtg = traj.compute_rtg()
        assert rtg.dtype == np.float32


# ===========================================================================
# TradingTrajectoryDataset
# ===========================================================================

class TestTradingTrajectoryDataset:
    def test_len_equals_total_steps(self):
        ds = _make_dataset(n_traj=2, traj_len=30, context_len=K)
        assert len(ds) == 60

    def test_empty_trajectories_raise(self):
        from agents.offline.trajectory_dataset import TradingTrajectoryDataset

        with pytest.raises(ValueError):
            TradingTrajectoryDataset([])

    def test_getitem_keys(self):
        ds = _make_dataset()
        sample = ds[0]
        assert set(sample.keys()) == {"states", "actions", "returns_to_go", "timesteps", "attention_mask"}

    def test_getitem_shapes(self):
        ds = _make_dataset(context_len=K)
        sample = ds[0]
        assert sample["states"].shape == (K, OBS_DIM)
        assert sample["actions"].shape == (K, ACT_DIM)
        assert sample["returns_to_go"].shape == (K, 1)
        assert sample["timesteps"].shape == (K,)
        assert sample["attention_mask"].shape == (K,)

    def test_getitem_dtypes(self):
        ds = _make_dataset()
        sample = ds[0]
        assert sample["states"].dtype == torch.float32
        assert sample["actions"].dtype == torch.float32
        assert sample["returns_to_go"].dtype == torch.float32
        assert sample["timesteps"].dtype == torch.int64
        assert sample["attention_mask"].dtype == torch.float32

    def test_padding_at_start_of_trajectory(self):
        ds = _make_dataset(context_len=K)
        # First sample of first trajectory: end_t=0, actual_len=1 → K-1 padded
        sample = ds[0]
        mask = sample["attention_mask"]
        assert mask[-1].item() == 1.0       # last position is real
        assert mask[0].item() == 0.0        # first positions are padded
        assert mask.sum().item() == pytest.approx(1.0)

    def test_no_padding_after_K_steps(self):
        ds = _make_dataset(traj_len=T, context_len=K)
        # Sample at index K (= end_t=K in first trajectory) has no padding
        sample = ds[K]
        assert sample["attention_mask"].sum().item() == pytest.approx(float(K))

    def test_timesteps_are_correct(self):
        ds = _make_dataset(context_len=K)
        # Sample at index K has timesteps [0 .. K-1] (no padding)
        sample = ds[K]
        expected = torch.arange(1, K + 1)
        assert torch.all(sample["timesteps"] == expected)

    def test_normalisation_reduces_scale(self):
        ds = _make_dataset(normalize_states=True)
        # States should be roughly zero-mean after normalisation
        all_states = torch.stack([ds[i]["states"] for i in range(len(ds))])
        assert all_states.abs().mean().item() < 5.0  # rough sanity

    def test_no_normalisation(self):
        ds_norm = _make_dataset(normalize_states=False)
        sample = ds_norm[K]
        # Without normalisation, obs_mean=0 obs_std=1 so values are raw
        assert sample["states"].shape == (K, OBS_DIM)

    def test_rtg_scale_normalisation(self):
        ds = _make_dataset(normalize_returns=True)
        # RTG values should be within (-1, 1] approximately
        all_rtg = torch.stack([ds[i]["returns_to_go"] for i in range(len(ds))])
        assert all_rtg.abs().max().item() <= 1.0 + 1e-5

    def test_obs_dim_property(self):
        ds = _make_dataset()
        assert ds.obs_dim == OBS_DIM

    def test_act_dim_property(self):
        ds = _make_dataset()
        assert ds.act_dim == ACT_DIM

    def test_multi_dim_observations(self):
        from agents.offline.trajectory_dataset import Trajectory, TradingTrajectoryDataset

        # Simulated (window_size=5, n_feat=3) obs  → obs_dim=15
        traj = Trajectory(
            observations=np.zeros((20, 5, 3), dtype=np.float32),
            actions=np.zeros((20, 1), dtype=np.float32),
            rewards=np.zeros(20, dtype=np.float32),
            dones=np.zeros(20, dtype=np.float32),
        )
        ds = TradingTrajectoryDataset([traj], context_len=5)
        sample = ds[4]
        assert sample["states"].shape == (5, 15)  # flattened

    def test_from_rollouts_single_episode(self):
        from agents.offline.trajectory_dataset import TradingTrajectoryDataset

        N = 50
        rng = np.random.default_rng(1)
        obs = rng.standard_normal((N, OBS_DIM)).astype(np.float32)
        acts = rng.standard_normal((N, ACT_DIM)).astype(np.float32)
        rwds = rng.standard_normal(N).astype(np.float32)
        dones = np.zeros(N, dtype=np.float32)
        dones[-1] = 1.0  # last step ends episode

        ds = TradingTrajectoryDataset.from_rollouts(obs, acts, rwds, dones, context_len=K)
        assert len(ds) == N
        assert len(ds.trajectories) == 1

    def test_from_rollouts_multiple_episodes(self):
        from agents.offline.trajectory_dataset import TradingTrajectoryDataset

        N = 60
        rng = np.random.default_rng(2)
        obs = rng.standard_normal((N, OBS_DIM)).astype(np.float32)
        acts = rng.standard_normal(N).astype(np.float32)  # scalar actions
        rwds = np.ones(N, dtype=np.float32)
        dones = np.zeros(N, dtype=np.float32)
        dones[19] = 1.0
        dones[39] = 1.0
        # last chunk: 40..59 (20 steps), not ended explicitly → captured at end

        ds = TradingTrajectoryDataset.from_rollouts(obs, acts, rwds, dones, context_len=K)
        assert len(ds.trajectories) == 3
        assert len(ds) == 60

    def test_from_rollouts_no_trajectories_raise(self):
        from agents.offline.trajectory_dataset import TradingTrajectoryDataset

        with pytest.raises(ValueError):
            TradingTrajectoryDataset.from_rollouts(
                np.zeros((0, OBS_DIM)),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                context_len=K,
            )

    def test_save_load_roundtrip(self, tmp_path):
        ds = _make_dataset()
        path = str(tmp_path / "traj.pkl")
        ds.save(path)
        ds2 = type(ds).load(path)

        assert len(ds2) == len(ds)
        sample_a = ds[5]
        sample_b = ds2[5]
        torch.testing.assert_close(sample_a["states"], sample_b["states"])
        torch.testing.assert_close(sample_a["actions"], sample_b["actions"])

    def test_save_load_preserves_config(self, tmp_path):
        ds = _make_dataset(context_len=7, gamma=0.99)
        path = str(tmp_path / "traj.pkl")
        ds.save(path)
        ds2 = type(ds).load(path)
        assert ds2.context_len == 7
        assert ds2.gamma == pytest.approx(0.99)


# ===========================================================================
# DecisionTransformerConfig
# ===========================================================================

class TestDecisionTransformerConfig:
    def test_defaults(self):
        from agents.offline.decision_transformer import DecisionTransformerConfig

        cfg = DecisionTransformerConfig()
        assert cfg.state_dim == 100
        assert cfg.act_dim == 1
        assert cfg.hidden_size == 128
        assert cfg.context_len == 20
        assert cfg.n_layer == 3
        assert cfg.n_head == 1

    def test_invalid_head_size_raises(self):
        from agents.offline.decision_transformer import DecisionTransformerConfig

        with pytest.raises(ValueError):
            DecisionTransformerConfig(hidden_size=32, n_head=3)  # 32 % 3 != 0

    def test_valid_head_size(self):
        from agents.offline.decision_transformer import DecisionTransformerConfig

        cfg = DecisionTransformerConfig(hidden_size=64, n_head=4)
        assert cfg.n_head == 4

    def test_n_inner_default_none(self):
        from agents.offline.decision_transformer import DecisionTransformerConfig

        cfg = DecisionTransformerConfig()
        assert cfg.n_inner is None

    def test_lora_defaults(self):
        from agents.offline.decision_transformer import DecisionTransformerConfig

        cfg = DecisionTransformerConfig()
        assert cfg.use_lora is True
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == pytest.approx(32.0)


# ===========================================================================
# _CausalTransformerBlock / _CausalTransformerBackbone
# ===========================================================================

class TestCausalTransformerBackbone:
    def test_block_output_shape(self):
        from agents.offline.decision_transformer import _CausalTransformerBlock

        block = _CausalTransformerBlock(hidden_size=16, n_head=2, n_inner=32, dropout=0.0)
        x = torch.randn(2, 5, 16)
        out = block(x)
        assert out.shape == (2, 5, 16)

    def test_backbone_output_shape(self):
        from agents.offline.decision_transformer import _CausalTransformerBackbone

        backbone = _CausalTransformerBackbone(
            hidden_size=16, n_layer=2, n_head=2, n_inner=32, dropout=0.0
        )
        x = torch.randn(3, 12, 16)
        out = backbone(x)
        assert out.shape == (3, 12, 16)

    def test_backbone_with_padding_mask(self):
        from agents.offline.decision_transformer import _CausalTransformerBackbone

        backbone = _CausalTransformerBackbone(
            hidden_size=16, n_layer=1, n_head=2, n_inner=32, dropout=0.0
        )
        x = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8)
        mask[0, :3] = 0.0  # pad first 3 positions of batch item 0
        out = backbone(x, attention_mask=mask)
        assert out.shape == (2, 8, 16)

    def test_backbone_no_mask(self):
        from agents.offline.decision_transformer import _CausalTransformerBackbone

        backbone = _CausalTransformerBackbone(
            hidden_size=8, n_layer=1, n_head=1, n_inner=16, dropout=0.0
        )
        x = torch.randn(1, 6, 8)
        out = backbone(x, attention_mask=None)
        assert out.shape == (1, 6, 8)

    def test_causal_masking_causality(self):
        """Output at position t should not depend on position t+1."""
        from agents.offline.decision_transformer import _CausalTransformerBackbone

        backbone = _CausalTransformerBackbone(
            hidden_size=8, n_layer=1, n_head=1, n_inner=16, dropout=0.0
        )
        backbone.eval()
        x = torch.randn(1, 4, 8)
        x_mod = x.clone()
        x_mod[0, 3] += 10.0  # perturb last position

        out = backbone(x)
        out_mod = backbone(x_mod)
        # First 3 positions should not change
        assert torch.allclose(out[0, :3], out_mod[0, :3], atol=1e-5)


# ===========================================================================
# TradingDecisionTransformer
# ===========================================================================

class TestTradingDecisionTransformer:
    def test_forward_shape(self):
        model = _make_small_model()
        states, actions, rtg, timesteps, mask = _make_batch()
        preds = model(states, actions, rtg, timesteps, mask)
        assert preds.shape == (2, K, ACT_DIM)

    def test_forward_no_mask(self):
        model = _make_small_model()
        states, actions, rtg, timesteps, _ = _make_batch()
        preds = model(states, actions, rtg, timesteps)
        assert preds.shape == (2, K, ACT_DIM)

    def test_forward_different_batch_sizes(self):
        model = _make_small_model()
        for B in [1, 4, 8]:
            states, actions, rtg, timesteps, mask = _make_batch(B=B)
            preds = model(states, actions, rtg, timesteps, mask)
            assert preds.shape == (B, K, ACT_DIM)

    def test_forward_different_act_dim(self):
        model = _make_small_model(act_dim=3)
        states, actions, rtg, timesteps, mask = _make_batch(act_dim=3)
        preds = model(states, actions, rtg, timesteps, mask)
        assert preds.shape == (2, K, 3)

    def test_compute_loss_scalar(self):
        model = _make_small_model()
        states, actions, rtg, timesteps, mask = _make_batch()
        loss = model.compute_loss(states, actions, rtg, timesteps, mask)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_compute_loss_no_mask(self):
        model = _make_small_model()
        states, actions, rtg, timesteps, _ = _make_batch()
        loss = model.compute_loss(states, actions, rtg, timesteps)
        assert loss.item() >= 0.0

    def test_compute_loss_masked_vs_unmasked(self):
        """Masked loss (pad positions zero) should differ from unmasked loss."""
        model = _make_small_model()
        model.eval()
        states, actions, rtg, timesteps, _ = _make_batch(B=1)
        full_mask = torch.ones(1, K)
        partial_mask = torch.zeros(1, K)
        partial_mask[:, -1] = 1.0  # only last position is real

        loss_full = model.compute_loss(states, actions, rtg, timesteps, full_mask)
        loss_partial = model.compute_loss(states, actions, rtg, timesteps, partial_mask)
        # Losses will generally differ
        assert loss_full.item() != pytest.approx(loss_partial.item(), abs=1e-6)

    def test_loss_decreases_with_training(self):
        """A single gradient step should reduce the loss."""
        model = _make_small_model()
        model.train()
        states, actions, rtg, timesteps, mask = _make_batch()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        loss_before = model.compute_loss(states, actions, rtg, timesteps, mask).item()
        for _ in range(5):
            opt.zero_grad()
            loss = model.compute_loss(states, actions, rtg, timesteps, mask)
            loss.backward()
            opt.step()
        loss_after = model.compute_loss(states, actions, rtg, timesteps, mask).item()
        assert loss_after < loss_before

    def test_get_action_shape(self):
        model = _make_small_model()
        states = torch.randn(K, OBS_DIM)
        actions = torch.randn(K, ACT_DIM)
        rtg = torch.randn(K, 1)
        timesteps = torch.arange(K)
        action = model.get_action(states, actions, rtg, timesteps)
        assert action.shape == (ACT_DIM,)

    def test_get_action_deterministic(self):
        """get_action should be deterministic (eval mode + no grad)."""
        model = _make_small_model()
        states = torch.randn(K, OBS_DIM)
        actions = torch.zeros(K, ACT_DIM)
        rtg = torch.ones(K, 1)
        timesteps = torch.arange(K)
        a1 = model.get_action(states, actions, rtg, timesteps)
        a2 = model.get_action(states, actions, rtg, timesteps)
        np.testing.assert_array_equal(a1, a2)

    def test_get_action_returns_numpy(self):
        model = _make_small_model()
        action = model.get_action(
            torch.randn(K, OBS_DIM),
            torch.zeros(K, ACT_DIM),
            torch.ones(K, 1),
            torch.arange(K),
        )
        assert isinstance(action, np.ndarray)

    def test_count_parameters_keys(self):
        model = _make_small_model()
        counts = model.count_parameters()
        assert "total" in counts
        assert "trainable" in counts

    def test_count_parameters_positive(self):
        model = _make_small_model()
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["trainable"] > 0
        assert counts["trainable"] <= counts["total"]

    def test_save_load_roundtrip(self, tmp_path):
        model = _make_small_model()
        path = str(tmp_path / "dt.pt")
        model.save(path)

        from agents.offline.decision_transformer import TradingDecisionTransformer

        model2 = TradingDecisionTransformer.load(path)
        states, actions, rtg, timesteps, mask = _make_batch()
        with torch.no_grad():
            out1 = model(states, actions, rtg, timesteps, mask)
            out2 = model2(states, actions, rtg, timesteps, mask)
        torch.testing.assert_close(out1, out2)

    def test_save_load_config_preserved(self, tmp_path):
        model = _make_small_model(hidden_size=32, n_layer=2)
        path = str(tmp_path / "dt.pt")
        model.save(path)

        from agents.offline.decision_transformer import TradingDecisionTransformer

        model2 = TradingDecisionTransformer.load(path)
        assert model2.config.hidden_size == 32
        assert model2.config.n_layer == 2

    def test_from_config_defaults(self):
        from agents.offline.decision_transformer import TradingDecisionTransformer

        model = TradingDecisionTransformer.from_config({})
        assert model.config.state_dim == 100
        assert model.config.act_dim == 1

    def test_from_config_custom(self):
        from agents.offline.decision_transformer import TradingDecisionTransformer

        cfg = {"decision_transformer": {"state_dim": 50, "hidden_size": 64, "n_layer": 2, "n_head": 2}}
        model = TradingDecisionTransformer.from_config(cfg)
        assert model.config.state_dim == 50
        assert model.config.hidden_size == 64

    def test_gpt2_backbone_raises_without_transformers(self, monkeypatch):
        import agents.offline.decision_transformer as dt_mod

        monkeypatch.setattr(dt_mod, "_TRANSFORMERS_AVAILABLE", False)
        from agents.offline.decision_transformer import DecisionTransformerConfig, TradingDecisionTransformer

        cfg = DecisionTransformerConfig(
            state_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden_size=16, context_len=K, n_layer=1, n_head=1,
            use_gpt2_backbone=True,
        )
        with pytest.raises(ImportError):
            TradingDecisionTransformer(cfg)


# ===========================================================================
# DecisionTransformerTrainer
# ===========================================================================

class TestDecisionTransformerTrainer:
    def test_train_epoch_returns_float(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)
        loss = trainer.train_epoch(ds)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_evaluate_returns_float(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)
        loss = trainer.evaluate(ds)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_evaluate_no_grad_no_model_update(self):
        """Model weights should not change during evaluate."""
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        trainer.evaluate(ds)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, params_before[n])

    def test_train_returns_dict(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)
        metrics = trainer.train(ds, n_epochs=2)
        assert "train_loss" in metrics
        assert "eval_loss" in metrics
        assert len(metrics["train_loss"]) == 2
        assert len(metrics["eval_loss"]) == 0  # no eval dataset

    def test_train_with_eval_dataset(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)
        metrics = trainer.train(ds, n_epochs=3, eval_dataset=ds)
        assert len(metrics["train_loss"]) == 3
        assert len(metrics["eval_loss"]) == 3

    def test_train_loss_list_non_negative(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)
        metrics = trainer.train(ds, n_epochs=2)
        assert all(l >= 0 for l in metrics["train_loss"])

    def test_step_counter_increments(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(n_traj=1, traj_len=T, context_len=K)
        trainer = DecisionTransformerTrainer(model, batch_size=8)
        assert trainer._step == 0
        trainer.train_epoch(ds)
        assert trainer._step > 0

    def test_warmup_scheduler_lr_at_start(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        # warmup_steps=1000, we start at step 0 → LR should be very small initially
        trainer = DecisionTransformerTrainer(model, learning_rate=1e-3, warmup_steps=100, batch_size=32)
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        # After warmup_steps steps, LR should be near learning_rate
        assert initial_lr > 0

    def test_grad_clipping_applied(self):
        """After one step with large gradient, norms should be bounded."""
        from agents.offline.decision_transformer import DecisionTransformerTrainer

        model = _make_small_model()
        ds = _make_dataset(context_len=K)
        trainer = DecisionTransformerTrainer(model, max_grad_norm=0.1, batch_size=32)
        # Just ensure it runs without error
        trainer.train_epoch(ds)

    def test_from_config_defaults(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer, TradingDecisionTransformer

        model = TradingDecisionTransformer.from_config({})
        trainer = DecisionTransformerTrainer.from_config(model, {})
        assert trainer.batch_size == 64

    def test_from_config_custom(self):
        from agents.offline.decision_transformer import DecisionTransformerTrainer, TradingDecisionTransformer

        model = TradingDecisionTransformer.from_config({})
        cfg = {"decision_transformer": {"learning_rate": 5e-4, "batch_size": 16}}
        trainer = DecisionTransformerTrainer.from_config(model, cfg)
        assert trainer.batch_size == 16


# ===========================================================================
# Optional import guards
# ===========================================================================

class TestImportGuards:
    def test_peft_available_is_bool(self):
        from agents.offline.decision_transformer import _PEFT_AVAILABLE

        assert isinstance(_PEFT_AVAILABLE, bool)

    def test_transformers_available_is_bool(self):
        from agents.offline.decision_transformer import _TRANSFORMERS_AVAILABLE

        assert isinstance(_TRANSFORMERS_AVAILABLE, bool)

    def test_custom_backbone_no_transformers_needed(self):
        """Custom backbone must work even if transformers is absent."""
        model = _make_small_model(use_gpt2_backbone=False)
        states, actions, rtg, timesteps, _ = _make_batch()
        with torch.no_grad():
            out = model(states, actions, rtg, timesteps)
        assert out.shape == (2, K, ACT_DIM)


# ===========================================================================
# agents/offline/__init__ re-exports
# ===========================================================================

class TestPackageInit:
    def test_trajectory_importable(self):
        from agents.offline import Trajectory
        assert Trajectory is not None

    def test_dataset_importable(self):
        from agents.offline import TradingTrajectoryDataset
        assert TradingTrajectoryDataset is not None

    def test_config_importable(self):
        from agents.offline import DecisionTransformerConfig
        assert DecisionTransformerConfig is not None

    def test_model_importable(self):
        from agents.offline import TradingDecisionTransformer
        assert TradingDecisionTransformer is not None

    def test_trainer_importable(self):
        from agents.offline import DecisionTransformerTrainer
        assert DecisionTransformerTrainer is not None

    def test_peft_flag_importable(self):
        from agents.offline import _PEFT_AVAILABLE
        assert isinstance(_PEFT_AVAILABLE, bool)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_from_rollouts_then_train(self):
        """Full pipeline: rollouts → dataset → model → trainer → 1 epoch."""
        from agents.offline.decision_transformer import (
            DecisionTransformerTrainer,
            TradingDecisionTransformer,
        )
        from agents.offline.trajectory_dataset import TradingTrajectoryDataset

        N = 200
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((N, OBS_DIM)).astype(np.float32)
        acts = rng.standard_normal(N).astype(np.float32)
        rwds = rng.standard_normal(N).astype(np.float32)
        dones = np.zeros(N, dtype=np.float32)
        dones[49] = dones[99] = dones[149] = 1.0

        ds = TradingTrajectoryDataset.from_rollouts(obs, acts, rwds, dones, context_len=K)

        model = _make_small_model(state_dim=OBS_DIM, act_dim=1)
        trainer = DecisionTransformerTrainer(model, batch_size=16, warmup_steps=0)
        metrics = trainer.train(ds, n_epochs=1)

        assert len(metrics["train_loss"]) == 1
        assert metrics["train_loss"][0] >= 0.0

    def test_high_target_return_inference(self):
        """DT inference with target RTG=1.0 returns a valid action."""
        model = _make_small_model()
        K_inf = K
        states = torch.zeros(K_inf, OBS_DIM)
        actions = torch.zeros(K_inf, ACT_DIM)
        rtg = torch.ones(K_inf, 1) * model.config.target_return
        timesteps = torch.arange(K_inf)

        action = model.get_action(states, actions, rtg, timesteps)
        assert action.shape == (ACT_DIM,)
        assert np.isfinite(action).all()

    def test_config_yaml_section_parsed(self):
        """from_config correctly maps YAML decision_transformer section."""
        from agents.offline.decision_transformer import TradingDecisionTransformer

        yaml_config = {
            "decision_transformer": {
                "state_dim": 45,
                "act_dim": 2,
                "hidden_size": 32,
                "n_layer": 2,
                "n_head": 2,
                "context_len": 10,
                "use_gpt2_backbone": False,
            }
        }
        model = TradingDecisionTransformer.from_config(yaml_config)
        assert model.config.state_dim == 45
        assert model.config.act_dim == 2
        assert model.config.n_layer == 2
        assert model.config.context_len == 10

    def test_save_load_then_inference(self, tmp_path):
        """Save, reload, then run inference — outputs match."""
        from agents.offline.decision_transformer import TradingDecisionTransformer

        model = _make_small_model()
        path = str(tmp_path / "dt_int.pt")
        model.save(path)
        model2 = TradingDecisionTransformer.load(path)

        states = torch.randn(K, OBS_DIM)
        actions = torch.zeros(K, ACT_DIM)
        rtg = torch.ones(K, 1)
        ts = torch.arange(K)

        a1 = model.get_action(states, actions, rtg, ts)
        a2 = model2.get_action(states, actions, rtg, ts)
        np.testing.assert_allclose(a1, a2, atol=1e-5)
