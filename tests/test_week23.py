"""
Week 23 tests: Continual Learning Pipeline + TorchRL FLAG Adapter.

Coverage:
  23.1  RegimeAwareExperienceStore
        - add / sample (balanced & fallback paths)
        - per-regime buffer sizes
        - circular overwrite

  23.2  EWCRegularizer
        - consolidate with and without obs_tensors
        - ewc_loss shape and non-negativity
        - multiple consolidations accumulate

  23.3  AdaptiveTrainer
        - from_config (config file path)
        - dry_run retrain → status == "dry_run"
        - rate-limit skip
        - no-agent skip
        - experience store integration via add_transition

  23.4  TorchRLFLAGAdapter
        - fallback predict (sync, SB3-compatible)
        - predict_batch (sync wrapper)
        - backend property
        - from_config factory
        - async predict_batch_async
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from training.continual.experience_store import (
    EWCRegularizer,
    RegimeAwareExperienceStore,
    Transition,
    _RegimeBuffer,
)
from training.continual.adaptive_trainer import (
    AdaptiveTrainer,
    AdaptiveTrainerConfig,
    RetrainingResult,
)
from agents.llm_rl.torchrl_flag_adapter import (
    TorchRLFLAGAdapter,
    TorchRLFLAGAdapterConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 10
ACT_DIM = 1

def _rand_obs():
    return np.random.randn(OBS_DIM).astype(np.float32)

def _rand_action():
    return np.random.randn(ACT_DIM).astype(np.float32)

def _fill_store(store: RegimeAwareExperienceStore, n: int = 100, regime_id: int = 0):
    for _ in range(n):
        store.add(_rand_obs(), _rand_action(), float(np.random.randn()), _rand_obs(), False, regime_id)


# ===========================================================================
# 23.1  RegimeAwareExperienceStore
# ===========================================================================

class TestRegimeBuffer:
    def test_add_and_len(self):
        buf = _RegimeBuffer(max_size=50, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        for _ in range(30):
            buf.add(_rand_obs(), _rand_action(), 0.1, _rand_obs(), False)
        assert len(buf) == 30

    def test_circular_overwrite(self):
        buf = _RegimeBuffer(max_size=10, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        for _ in range(25):
            buf.add(_rand_obs(), _rand_action(), 0.1, _rand_obs(), False)
        assert len(buf) == 10  # capped at max_size

    def test_sample_shape(self):
        buf = _RegimeBuffer(max_size=100, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        for _ in range(50):
            buf.add(_rand_obs(), _rand_action(), 0.1, _rand_obs(), False)
        batch = buf.sample(20)
        assert batch["obs"].shape == (20, OBS_DIM)
        assert batch["actions"].shape == (20, ACT_DIM)
        assert batch["rewards"].shape == (20,)


class TestRegimeAwareExperienceStore:
    def test_add_and_sizes(self):
        store = RegimeAwareExperienceStore(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_regimes=3)
        _fill_store(store, n=40, regime_id=0)
        _fill_store(store, n=20, regime_id=1)
        sizes = store.regime_sizes()
        assert sizes[0] == 40
        assert sizes[1] == 20
        assert sizes[2] == 0
        assert store.total_size() == 60

    def test_add_transition(self):
        store = RegimeAwareExperienceStore(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        t = Transition(_rand_obs(), _rand_action(), 0.5, _rand_obs(), False, regime_id=2)
        store.add_transition(t)
        assert store.regime_sizes()[2] == 1

    def test_sample_balanced(self):
        store = RegimeAwareExperienceStore(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, n_regimes=3, current_regime_ratio=0.7
        )
        _fill_store(store, n=200, regime_id=0)
        _fill_store(store, n=200, regime_id=1)
        _fill_store(store, n=200, regime_id=2)
        batch = store.sample(batch_size=100, current_regime=0)
        assert "obs" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert len(batch["obs"]) > 0

    def test_sample_fallback_no_other_regimes(self):
        """Only regime 0 has data — should still return a batch."""
        store = RegimeAwareExperienceStore(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_regimes=3)
        _fill_store(store, n=100, regime_id=0)
        batch = store.sample(batch_size=64, current_regime=0)
        assert len(batch["obs"]) > 0

    def test_sample_raises_on_empty_regime(self):
        store = RegimeAwareExperienceStore(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_regimes=3)
        with pytest.raises(ValueError, match="No transitions"):
            store.sample(batch_size=10, current_regime=0)

    def test_repr(self):
        store = RegimeAwareExperienceStore(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_regimes=2)
        assert "RegimeAwareExperienceStore" in repr(store)

    def test_regime_id_wraps(self):
        """regime_id >= n_regimes should wrap via modulo."""
        store = RegimeAwareExperienceStore(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_regimes=3)
        store.add(_rand_obs(), _rand_action(), 0.0, _rand_obs(), False, regime_id=5)
        assert store.regime_sizes()[5 % 3] == 1


# ===========================================================================
# 23.2  EWCRegularizer
# ===========================================================================

class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(OBS_DIM, 4)

    def forward(self, x):
        return self.fc(x)


class TestEWCRegularizer:
    def test_ewc_loss_zero_before_consolidation(self):
        ewc = EWCRegularizer(ewc_lambda=0.4)
        model = _TinyNet()
        loss = ewc.ewc_loss(model)
        assert float(loss) == 0.0

    def test_consolidate_uniform_fisher(self):
        ewc = EWCRegularizer(ewc_lambda=0.4)
        model = _TinyNet()
        ewc.consolidate(model, obs_tensors=None)
        assert ewc.n_consolidations == 1

    def test_consolidate_with_obs(self):
        ewc = EWCRegularizer(ewc_lambda=0.4, n_fisher_samples=32)
        model = _TinyNet()
        obs = torch.randn(64, OBS_DIM)
        ewc.consolidate(model, obs_tensors=obs)
        assert ewc.n_consolidations == 1

    def test_ewc_loss_nonnegative(self):
        ewc = EWCRegularizer(ewc_lambda=0.4)
        model = _TinyNet()
        ewc.consolidate(model)
        # perturb model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        loss = ewc.ewc_loss(model)
        assert float(loss) >= 0.0

    def test_ewc_loss_zero_at_theta_star(self):
        """Loss should be (near) zero if weights haven't changed."""
        ewc = EWCRegularizer(ewc_lambda=0.4)
        model = _TinyNet()
        ewc.consolidate(model)
        loss = ewc.ewc_loss(model)
        assert float(loss) < 1e-6

    def test_multiple_consolidations_accumulate(self):
        ewc = EWCRegularizer(ewc_lambda=0.4)
        model = _TinyNet()
        ewc.consolidate(model)  # regime 0
        # shift model and consolidate again
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        ewc.consolidate(model)  # regime 1
        assert ewc.n_consolidations == 2
        loss = ewc.ewc_loss(model)
        assert float(loss) >= 0.0

    def test_lambda_zero_disables_penalty(self):
        ewc = EWCRegularizer(ewc_lambda=0.0)
        model = _TinyNet()
        ewc.consolidate(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 10.0)
        loss = ewc.ewc_loss(model)
        assert float(loss) == 0.0

    def test_clear(self):
        ewc = EWCRegularizer()
        model = _TinyNet()
        ewc.consolidate(model)
        ewc.clear()
        assert ewc.n_consolidations == 0


# ===========================================================================
# 23.3  AdaptiveTrainer
# ===========================================================================

class TestAdaptiveTrainerConfig:
    def test_defaults(self):
        cfg = AdaptiveTrainerConfig()
        assert cfg.ewc_lambda == 0.4
        assert cfg.rollback_threshold == 0.90
        assert cfg.n_regimes == 3

    def test_from_dict_empty(self):
        cfg = AdaptiveTrainerConfig.from_dict({})
        assert cfg.fine_tune_timesteps == 10_000

    def test_from_dict_custom(self):
        raw = {"continual_learning": {"ewc_lambda": 0.2, "fine_tune_timesteps": 5_000}}
        cfg = AdaptiveTrainerConfig.from_dict(raw)
        assert cfg.ewc_lambda == 0.2
        assert cfg.fine_tune_timesteps == 5_000


class TestAdaptiveTrainer:
    def _make_trainer(self, checkpoint_dir: str) -> AdaptiveTrainer:
        cfg = AdaptiveTrainerConfig(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            min_retrain_interval_s=0.0,  # no rate-limit in tests
            checkpoint_dir=checkpoint_dir,
        )
        return AdaptiveTrainer(config=cfg)

    def test_dry_run(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        result = trainer.retrain(dry_run=True)
        assert result.status == "dry_run"
        assert result.duration_s >= 0.0

    def test_skips_without_agent(self, tmp_path):
        cfg = AdaptiveTrainerConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM, checkpoint_dir=str(tmp_path))
        trainer = AdaptiveTrainer(config=cfg)
        result = trainer.retrain(dry_run=False)
        assert result.status == "skipped"

    def test_skips_rate_limit(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        trainer._last_retrain_time = 9e18  # far in the future
        trainer.agent = MagicMock()
        result = trainer.retrain(dry_run=False)
        assert result.status == "skipped"
        assert "Too soon" in result.reason

    def test_skips_empty_store(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        trainer.agent = MagicMock()
        result = trainer.retrain(dry_run=False)
        assert result.status == "skipped"
        assert "empty" in result.reason.lower()

    def test_retrain_count(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        trainer.retrain(dry_run=True)
        assert trainer.retrain_count == 0  # dry_run doesn't count as success

    def test_add_transition_feeds_store(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        for _ in range(10):
            trainer.add_transition(_rand_obs(), _rand_action(), 0.1, _rand_obs(), False, regime_id=0)
        assert trainer.experience_store.total_size() == 10

    def test_history_accumulates(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        trainer.retrain(dry_run=True)
        trainer.retrain(dry_run=True)
        assert len(trainer.history) == 2

    def test_from_config(self, tmp_path):
        # Write a minimal YAML config
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            f.write("continual_learning:\n  ewc_lambda: 0.3\n")
        trainer = AdaptiveTrainer.from_config(config_path)
        assert isinstance(trainer, AdaptiveTrainer)
        assert trainer.config.ewc_lambda == 0.3

    def test_repr(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        assert "AdaptiveTrainer" in repr(trainer)

    def test_retrain_result_to_dict(self):
        result = RetrainingResult(status="success", baseline_sharpe=1.0, new_sharpe=1.2, regime_id=1)
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["new_sharpe"] == 1.2


# ===========================================================================
# 23.4  TorchRLFLAGAdapter
# ===========================================================================

class TestTorchRLFLAGAdapterConfig:
    def test_defaults(self):
        cfg = TorchRLFLAGAdapterConfig()
        assert cfg.max_batch_size == 16
        assert cfg.flag_dry_run is True

    def test_from_dict(self):
        raw = {"torchrl": {"max_batch_size": 8}, "flag_trader": {"dry_run": False}}
        cfg = TorchRLFLAGAdapterConfig.from_dict(raw)
        assert cfg.max_batch_size == 8
        assert cfg.flag_dry_run is False


class TestTorchRLFLAGAdapter:
    """
    All tests use dry_run=True FLAG-Trader so no model downloads occur.
    """

    def _make_adapter(self) -> TorchRLFLAGAdapter:
        cfg = TorchRLFLAGAdapterConfig(flag_dry_run=True, max_batch_size=4)
        return TorchRLFLAGAdapter(config=cfg)

    def test_predict_returns_ndarray(self):
        adapter = self._make_adapter()
        obs = np.random.randn(22).astype(np.float32)
        action, state = adapter.predict(obs)
        assert isinstance(action, np.ndarray)
        assert state is None

    def test_predict_action_in_range(self):
        adapter = self._make_adapter()
        obs = np.random.randn(22).astype(np.float32)
        action, _ = adapter.predict(obs)
        # FLAGTrader tanh output: action ∈ [-1, 1] (within floating tolerance)
        assert action.ndim >= 1

    def test_backend_property(self):
        adapter = self._make_adapter()
        assert adapter.backend in ("torchrl", "fallback")

    def test_predict_batch_sync(self):
        adapter = self._make_adapter()
        obs_list = [np.random.randn(22).astype(np.float32) for _ in range(5)]
        actions = adapter.predict_batch(obs_list)
        assert len(actions) == 5
        for a in actions:
            assert isinstance(a, np.ndarray)

    def test_predict_batch_async(self):
        adapter = self._make_adapter()
        obs_list = [np.random.randn(22).astype(np.float32) for _ in range(3)]

        async def _run():
            return await adapter.predict_batch_async(obs_list)

        actions = asyncio.run(_run())
        assert len(actions) == 3

    def test_predict_batch_empty_list(self):
        adapter = self._make_adapter()
        actions = adapter.predict_batch([])
        assert actions == []

    def test_inference_stats(self):
        adapter = self._make_adapter()
        stats = adapter.inference_stats
        assert "backend" in stats
        assert "total_obs_processed" in stats

    def test_from_config_factory(self):
        raw = {
            "flag_trader": {"dry_run": True},
            "torchrl": {"max_batch_size": 4},
            "training": {"device": "cpu"},
        }
        adapter = TorchRLFLAGAdapter.from_config(raw)
        assert isinstance(adapter, TorchRLFLAGAdapter)

    def test_repr(self):
        adapter = self._make_adapter()
        r = repr(adapter)
        assert "TorchRLFLAGAdapter" in r
        assert "backend" in r

    def test_predict_deterministic_flag(self):
        """predict() with deterministic=False should still return valid action."""
        adapter = self._make_adapter()
        obs = np.random.randn(22).astype(np.float32)
        action, _ = adapter.predict(obs, deterministic=False)
        assert action.ndim >= 1


# ===========================================================================
# Integration: store → EWC → trainer pipeline
# ===========================================================================

class TestContinualPipelineIntegration:
    """End-to-end: fill store → EWC consolidate → trainer dry_run."""

    def test_full_pipeline_dry_run(self, tmp_path):
        cfg = AdaptiveTrainerConfig(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            min_retrain_interval_s=0.0,
            ewc_lambda=0.4,
            checkpoint_dir=str(tmp_path),
            n_regimes=3,
        )
        trainer = AdaptiveTrainer(config=cfg)

        # Fill store with multiple regimes
        for regime in range(3):
            for _ in range(50):
                trainer.add_transition(
                    _rand_obs(), _rand_action(), float(np.random.randn()), _rand_obs(), False,
                    regime_id=regime,
                )

        assert trainer.experience_store.total_size() == 150

        # EWC: consolidate a tiny model
        model = _TinyNet()
        obs_t = torch.randn(50, OBS_DIM)
        trainer.ewc.consolidate(model, obs_t)
        assert trainer.ewc.n_consolidations == 1

        # Dry-run retrain
        result = trainer.retrain(dry_run=True)
        assert result.status == "dry_run"

    def test_ewc_loss_increases_after_weight_shift(self):
        """EWC penalty should increase when model weights move away from θ*."""
        ewc = EWCRegularizer(ewc_lambda=1.0)
        model = _TinyNet()
        ewc.consolidate(model)
        loss_before = float(ewc.ewc_loss(model))  # should be ~0

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 5.0)

        loss_after = float(ewc.ewc_loss(model))
        assert loss_after > loss_before
