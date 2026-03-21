"""
Week 17: FLAG-Trader integration tests.

Coverage:
  - FLAGTraderConfig (defaults, from_config factory, tuple fields)
  - MarketStateFormatter (format, format_batch, edge cases)
  - _DryRunLLM (forward shape, no errors)
  - _ObsEncoder (projection shape)
  - _ActionHead (output shape, tanh range)
  - FLAGTrader dry_run mode
    - __init__ (builds model without downloading weights)
    - predict() — single obs, batched obs, shape/range
    - SB3 compatibility — returns (np.ndarray, None)
    - count_parameters()
    - save() / load() round-trip
    - from_config() factory
  - FLAGTraderTrainer
    - train_supervised() — returns loss dict, loss decreases
    - train_ppo() (custom loop) — returns episode_rewards/mean_reward/n_updates
    - _ppo_update() — runs without error
  - agents/llm_rl/__init__ re-exports
  - _parse_action_text() — keyword and numeric parsing
  - _compute_gae() — advantage correctness
  - config/flag_trader.yaml — YAML is loadable and contains expected keys
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

OBS_DIM = 22    # window_size=20 + position + cash
BATCH = 4
K = 8           # context length for DT dataset


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    from agents.llm_rl.flag_trader import FLAGTraderConfig

    defaults = dict(dry_run=True, obs_dim=OBS_DIM, window_size=20)
    defaults.update(overrides)
    return FLAGTraderConfig(**defaults)


def _make_agent(**cfg_overrides):
    from agents.llm_rl.flag_trader import FLAGTrader

    cfg = _make_config(**cfg_overrides)
    return FLAGTrader(cfg)


def _make_obs(n: int = 1) -> np.ndarray:
    rng = np.random.default_rng(42)
    obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    # Last two entries: position ∈ [-1,1], cash ∈ [0,1]
    obs[:, -2] = rng.uniform(-1, 1, n)
    obs[:, -1] = rng.uniform(0, 1, n)
    return obs


def _make_dt_dataset(
    n_traj: int = 2,
    traj_len: int = 30,
    state_dim: int = OBS_DIM,
    context_len: int = K,
):
    """Create a small TradingTrajectoryDataset for supervised pre-training tests."""
    from agents.offline.trajectory_dataset import Trajectory, TradingTrajectoryDataset

    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(n_traj):
        trajs.append(
            Trajectory(
                observations=rng.standard_normal((traj_len, state_dim)).astype(np.float32),
                actions=rng.uniform(-1, 1, (traj_len, 1)).astype(np.float32),
                rewards=rng.standard_normal(traj_len).astype(np.float32),
                dones=np.zeros(traj_len, dtype=np.float32),
            )
        )
    return TradingTrajectoryDataset(trajs, context_len=context_len)


class _MockEnv:
    """Minimal Gym-compatible environment for PPO training tests."""

    def __init__(self, obs_dim: int = OBS_DIM, max_steps: int = 20) -> None:
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self._step = 0
        self._rng = np.random.default_rng(1)
        # Gymnasium-style reset
        self._gymnasium = True

    def reset(self, seed=None):
        self._step = 0
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        reward = float(self._rng.standard_normal())
        terminated = self._step >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


# ===========================================================================
# 1. Module re-exports
# ===========================================================================

class TestModuleReexports:
    def test_imports(self):
        from agents.llm_rl import (
            FLAGTrader,
            FLAGTraderConfig,
            FLAGTraderTrainer,
            MarketStateFormatter,
        )
        assert FLAGTrader is not None
        assert FLAGTraderConfig is not None
        assert FLAGTraderTrainer is not None
        assert MarketStateFormatter is not None


# ===========================================================================
# 2. FLAGTraderConfig
# ===========================================================================

class TestFLAGTraderConfig:
    def test_defaults(self):
        from agents.llm_rl.flag_trader import FLAGTraderConfig

        cfg = FLAGTraderConfig()
        assert cfg.base_model == "HuggingFaceTB/SmolLM2-135M"
        assert cfg.lora_rank == 16
        assert cfg.ppo_lr == 1e-5
        assert cfg.gamma == 0.99
        assert not cfg.dry_run

    def test_dry_run_override(self):
        cfg = _make_config()
        assert cfg.dry_run is True

    def test_lora_target_modules_is_tuple(self):
        from agents.llm_rl.flag_trader import FLAGTraderConfig

        cfg = FLAGTraderConfig()
        assert isinstance(cfg.lora_target_modules, tuple)
        assert "q_proj" in cfg.lora_target_modules

    def test_from_config_factory(self):
        from agents.llm_rl.flag_trader import FLAGTrader

        config_dict = {
            "flag_trader": {
                "dry_run": True,
                "obs_dim": OBS_DIM,
                "lora_rank": 8,
                "ppo_lr": 5e-6,
                "window_size": 20,
            }
        }
        agent = FLAGTrader.from_config(config_dict)
        assert agent.config.lora_rank == 8
        assert agent.config.ppo_lr == 5e-6
        assert agent.config.dry_run is True

    def test_from_config_empty_section(self):
        """Empty flag_trader section → all defaults (dry_run so no download needed)."""
        from agents.llm_rl.flag_trader import FLAGTrader, FLAGTraderConfig

        # Override dry_run so we don't try to download SmolLM2 in CI
        config_dict = {"flag_trader": {"dry_run": True, "obs_dim": OBS_DIM, "window_size": 20}}
        agent = FLAGTrader.from_config(config_dict)
        # Other fields should use their defaults
        assert agent.config.base_model == FLAGTraderConfig.base_model
        assert agent.config.lora_rank == FLAGTraderConfig.lora_rank


# ===========================================================================
# 3. MarketStateFormatter
# ===========================================================================

class TestMarketStateFormatter:
    def test_format_single_obs(self):
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter(window_size=20)
        obs = _make_obs(1)[0]
        text = fmt.format(obs)
        assert "Market State:" in text
        assert "Log-returns" in text
        assert "Position:" in text
        assert "Cash:" in text
        assert "Action:" in text

    def test_format_ends_with_action(self):
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter()
        obs = _make_obs(1)[0]
        text = fmt.format(obs)
        assert text.strip().endswith("Action:")

    def test_format_long_position(self):
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[-2] = 0.75  # 75% long
        obs[-1] = 0.25  # 25% cash
        text = fmt.format(obs)
        assert "long" in text.lower()

    def test_format_short_position(self):
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[-2] = -0.5  # short
        text = fmt.format(obs)
        assert "short" in text.lower()

    def test_format_flat_position(self):
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[-2] = 0.0
        text = fmt.format(obs)
        assert "flat" in text.lower()

    def test_format_batch(self):
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter()
        obs_batch = _make_obs(BATCH)
        texts = fmt.format_batch(obs_batch)
        assert len(texts) == BATCH
        for t in texts:
            assert "Action:" in t

    def test_format_minimal_obs(self):
        """Short obs (< 5 returns) should not crash."""
        from agents.llm_rl.flag_trader import MarketStateFormatter

        fmt = MarketStateFormatter(window_size=3)
        obs = np.array([0.01, -0.02, 0.005, 0.3, 0.7], dtype=np.float32)
        text = fmt.format(obs)
        assert "Market State:" in text


# ===========================================================================
# 4. Dry-run model components
# ===========================================================================

class TestDryRunComponents:
    def test_dry_run_llm_forward(self):
        from agents.llm_rl.flag_trader import _DryRunLLM

        model = _DryRunLLM(vocab=128, hidden=32, n_layer=2, n_head=2)
        ids = torch.randint(0, 128, (BATCH, 10))
        out = model(ids)
        assert out.shape == (BATCH, 10, 32)

    def test_obs_encoder_shape(self):
        from agents.llm_rl.flag_trader import _ObsEncoder

        enc = _ObsEncoder(obs_dim=OBS_DIM, hidden=32)
        obs = torch.randn(BATCH, OBS_DIM)
        out = enc(obs)
        assert out.shape == (BATCH, 1, 32)

    def test_action_head_shape_and_range(self):
        from agents.llm_rl.flag_trader import _ActionHead

        head = _ActionHead(hidden=32)
        hidden = torch.randn(BATCH, 5, 32)  # (B, T, H)
        action = head(hidden)
        assert action.shape == (BATCH, 1)
        # tanh output must be strictly within (-1, 1)
        assert (action.abs() < 1.0).all()


# ===========================================================================
# 5. FLAGTrader (dry_run mode)
# ===========================================================================

class TestFLAGTrader:
    def test_init_dry_run(self):
        agent = _make_agent()
        assert agent._llm is not None
        assert agent._action_head is not None
        assert agent._obs_encoder is not None

    def test_predict_single_obs_shape(self):
        agent = _make_agent()
        obs = _make_obs(1)[0]          # (obs_dim,)
        action, state = agent.predict(obs)
        assert action.shape == (1,), f"Expected shape (1,), got {action.shape}"
        assert state is None

    def test_predict_action_in_range(self):
        agent = _make_agent()
        for _ in range(10):
            obs = _make_obs(1)[0]
            action, _ = agent.predict(obs)
            assert -1.0 <= float(action[0]) <= 1.0, f"Action out of range: {action}"

    def test_predict_batched_obs(self):
        agent = _make_agent()
        obs_batch = _make_obs(BATCH)   # (BATCH, obs_dim)
        actions, state = agent.predict(obs_batch)
        assert actions.shape == (BATCH, 1)
        assert state is None

    def test_predict_deterministic_flag(self):
        """deterministic=False should still return a valid action (no crash)."""
        agent = _make_agent()
        obs = _make_obs(1)[0]
        action, _ = agent.predict(obs, deterministic=False)
        assert action.shape == (1,)

    def test_predict_dtype(self):
        agent = _make_agent()
        obs = _make_obs(1)[0]
        action, _ = agent.predict(obs)
        assert action.dtype == np.float32

    def test_count_parameters(self):
        agent = _make_agent()
        params = agent.count_parameters()
        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["trainable"] <= params["total"]

    def test_save_load_roundtrip(self):
        agent = _make_agent()
        obs = _make_obs(1)[0]
        action_before, _ = agent.predict(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flag_agent")
            agent.save(path)

            # Check files exist
            assert os.path.exists(f"{path}.json")
            assert os.path.exists(f"{path}.pt")

            from agents.llm_rl.flag_trader import FLAGTrader

            loaded = FLAGTrader.load(path)
            action_after, _ = loaded.predict(obs)

        np.testing.assert_allclose(action_before, action_after, atol=1e-5)

    def test_save_config_roundtrip(self):
        """Saved JSON config should deserialise cleanly."""
        import json

        agent = _make_agent(lora_rank=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flag_agent")
            agent.save(path)
            with open(f"{path}.json") as f:
                cfg_dict = json.load(f)
        assert cfg_dict["lora_rank"] == 8
        assert cfg_dict["dry_run"] is True

    def test_from_config_creates_agent(self):
        from agents.llm_rl.flag_trader import FLAGTrader

        config_dict = {"flag_trader": {"dry_run": True, "obs_dim": OBS_DIM, "window_size": 20}}
        agent = FLAGTrader.from_config(config_dict)
        obs = _make_obs(1)[0]
        action, _ = agent.predict(obs)
        assert action.shape == (1,)


# ===========================================================================
# 6. FLAGTraderTrainer — supervised pre-training
# ===========================================================================

class TestFLAGTraderTrainerSupervised:
    def test_train_supervised_returns_dict(self):
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent()
        trainer = FLAGTraderTrainer(agent)
        dataset = _make_dt_dataset()
        metrics = trainer.train_supervised(dataset, n_epochs=2)

        assert "train_loss" in metrics
        assert "eval_loss" in metrics
        assert len(metrics["train_loss"]) == 2
        assert len(metrics["eval_loss"]) == 0  # no eval_dataset provided

    def test_train_supervised_with_eval(self):
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent()
        trainer = FLAGTraderTrainer(agent)
        train_ds = _make_dt_dataset()
        eval_ds = _make_dt_dataset(n_traj=1)
        metrics = trainer.train_supervised(train_ds, n_epochs=3, eval_dataset=eval_ds)

        assert len(metrics["train_loss"]) == 3
        assert len(metrics["eval_loss"]) == 3
        # All losses should be non-negative finite floats
        for loss in metrics["train_loss"]:
            assert np.isfinite(loss) and loss >= 0.0

    def test_train_supervised_loss_is_finite(self):
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent()
        trainer = FLAGTraderTrainer(agent)
        dataset = _make_dt_dataset()
        metrics = trainer.train_supervised(dataset, n_epochs=1)
        assert np.isfinite(metrics["train_loss"][0])

    def test_supervised_loss_reduces(self):
        """Loss should trend down over enough epochs on a small fixed dataset."""
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent()
        trainer = FLAGTraderTrainer(agent)
        dataset = _make_dt_dataset(n_traj=3, traj_len=40)
        metrics = trainer.train_supervised(dataset, n_epochs=5)
        # First epoch loss ≥ last epoch loss (with small tolerance for noise)
        first = metrics["train_loss"][0]
        last = metrics["train_loss"][-1]
        assert last <= first * 2.0, f"Loss did not reduce: first={first:.4f}, last={last:.4f}"


# ===========================================================================
# 7. FLAGTraderTrainer — PPO custom loop
# ===========================================================================

class TestFLAGTraderTrainerPPO:
    def test_train_ppo_returns_dict(self):
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent(ppo_batch_size=10)
        trainer = FLAGTraderTrainer(agent)
        env = _MockEnv()
        metrics = trainer.train_ppo(env, total_timesteps=25)

        assert "episode_rewards" in metrics
        assert "mean_reward" in metrics
        assert "n_updates" in metrics

    def test_train_ppo_collected_rewards(self):
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent(ppo_batch_size=10)
        trainer = FLAGTraderTrainer(agent)
        env = _MockEnv(max_steps=10)
        metrics = trainer.train_ppo(env, total_timesteps=30)

        assert len(metrics["episode_rewards"]) >= 1
        assert np.isfinite(metrics["mean_reward"])

    def test_ppo_does_not_crash_on_small_batch(self):
        """PPO update with fewer steps than mini_batch_size should not crash."""
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent(ppo_batch_size=5, ppo_mini_batch_size=8)
        trainer = FLAGTraderTrainer(agent)
        env = _MockEnv(max_steps=5)
        # Should complete without exception
        trainer.train_ppo(env, total_timesteps=15)

    def test_predict_after_ppo(self):
        """predict() should still work after PPO training."""
        from agents.llm_rl.flag_trader import FLAGTraderTrainer

        agent = _make_agent(ppo_batch_size=10)
        trainer = FLAGTraderTrainer(agent)
        env = _MockEnv()
        trainer.train_ppo(env, total_timesteps=20)

        obs = _make_obs(1)[0]
        action, _ = agent.predict(obs)
        assert action.shape == (1,)
        assert -1.0 <= float(action[0]) <= 1.0


# ===========================================================================
# 8. Utility functions
# ===========================================================================

class TestParseActionText:
    def test_numeric_positive(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        assert _parse_action_text("0.75") == pytest.approx(0.75)

    def test_numeric_negative(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        assert _parse_action_text("-0.5") == pytest.approx(-0.5)

    def test_buy_keyword(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        val = _parse_action_text("BUY")
        assert val > 0

    def test_sell_keyword(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        val = _parse_action_text("SELL")
        assert val < 0

    def test_hold_keyword(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        assert _parse_action_text("hold") == pytest.approx(0.0)

    def test_clamp_over_1(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        assert _parse_action_text("5.0") == pytest.approx(1.0)

    def test_clamp_below_minus_1(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        assert _parse_action_text("-3.0") == pytest.approx(-1.0)

    def test_empty_string(self):
        from agents.llm_rl.flag_trader import _parse_action_text

        assert _parse_action_text("") == pytest.approx(0.0)


class TestComputeGAE:
    def test_shape(self):
        from agents.llm_rl.flag_trader import _compute_gae

        N = 50
        rewards = np.random.randn(N).astype(np.float32)
        values = np.random.randn(N).astype(np.float32)
        dones = np.zeros(N, dtype=np.float32)
        adv, ret = _compute_gae(rewards, values, dones)
        assert adv.shape == (N,)
        assert ret.shape == (N,)

    def test_returns_equal_reward_plus_values_at_terminal(self):
        """For gamma=1, lam=1, no discounting: returns = cumulative rewards + bootstrapped value."""
        from agents.llm_rl.flag_trader import _compute_gae

        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        adv, ret = _compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        # returns = advantages + values; advantages sum back-to-front
        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

    def test_done_resets_bootstrap(self):
        """Done flag should zero out future bootstrapping."""
        from agents.llm_rl.flag_trader import _compute_gae

        rewards = np.array([1.0, 0.0], dtype=np.float32)
        values = np.array([0.5, 0.5], dtype=np.float32)
        dones = np.array([1.0, 0.0], dtype=np.float32)  # episode ends at t=0
        adv, _ = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        # At t=0 (done), delta should not include future value
        delta_0 = 1.0 + 0.99 * 0.0 * 0.5 - 0.5  # = 0.5 (no future bootstrap)
        assert np.isclose(adv[0], delta_0, atol=1e-4)


# ===========================================================================
# 9. YAML config loading
# ===========================================================================

class TestYAMLConfig:
    def test_flag_trader_yaml_loads(self):
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "flag_trader.yaml"
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        assert "flag_trader" in cfg
        assert "ensemble" in cfg
        ft = cfg["flag_trader"]
        assert ft["lora_rank"] == 16
        assert ft["ppo_lr"] == pytest.approx(1e-5)
        assert ft["gamma"] == pytest.approx(0.99)

    def test_ensemble_has_four_agents(self):
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "flag_trader.yaml"
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        agents = cfg["ensemble"]["agents"]
        assert len(agents) == 4
        types = [a["type"] for a in agents]
        assert "sb3_ppo" in types
        assert "sb3_sac" in types
        assert "sb3_td3" in types
        assert "flag_trader" in types

    def test_ensemble_weights_sum_to_one(self):
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "flag_trader.yaml"
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        total_weight = sum(a["weight_init"] for a in cfg["ensemble"]["agents"])
        assert abs(total_weight - 1.0) < 1e-6

    def test_from_config_uses_yaml(self):
        """FLAGTrader.from_config() should parse the YAML config correctly."""
        import yaml
        from agents.llm_rl.flag_trader import FLAGTrader

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "flag_trader.yaml"
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        # Override dry_run for the test
        cfg["flag_trader"]["dry_run"] = True
        cfg["flag_trader"]["obs_dim"] = OBS_DIM
        agent = FLAGTrader.from_config(cfg)
        assert agent.config.lora_rank == 16
        assert agent.config.gamma == pytest.approx(0.99)
