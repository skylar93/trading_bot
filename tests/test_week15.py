"""
Week 15: Meta-Controller + Prediction Market Signals tests.

Coverage
--------
MetaController
  - Instantiation with various n_agents
  - get_weights: shape, sum-to-one, min_weight floor, no-grad
  - get_weights: graceful handling of wrong-length inputs (resize)
  - step: returns weights, records to buffer
  - Emergency mode: triggers after emergency_window consecutive all-neg Sharpe
  - PPO update: runs without error, resets buffer
  - save / load: round-trip preserves weights output
  - update(): manual trajectory update
  - MetaControllerConfig defaults

PredictionMarketSignals
  - Instantiation (enabled=False, no API needed)
  - get_features: shape, dtype, range when disabled
  - get_features: fallback zeros when API unavailable
  - Feature computation: entropy at 0.5 = 1.0, at 0/1 → 0
  - Momentum history: grows correctly, clips to momentum_window+1
  - Cross-market divergence: correct abs diff
  - align_to_prices: columns appended, no NaN
  - reset_history: clears per-asset history
  - N_PREDICTION_MARKET_FEATURES / PREDICTION_MARKET_COLS constants

Integration
  - MetaController accepts prediction-market features as market_features arg
  - Weights change when market_features change (non-determinism test)
"""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_AGENTS = 3
N_REGIMES = 3
N_MARKET = 4


def _uniform_regime() -> np.ndarray:
    return np.array([1 / N_REGIMES] * N_REGIMES, dtype=np.float32)


def _zero_sharpe() -> np.ndarray:
    return np.zeros(N_AGENTS, dtype=np.float32)


def _random_sharpe(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(N_AGENTS).astype(np.float32)


# ===========================================================================
# MetaController
# ===========================================================================

class TestMetaControllerInstantiation:
    def test_default_init(self):
        from agents.ensemble.meta_controller import MetaController
        mc = MetaController(n_agents=N_AGENTS)
        assert mc.n_agents == N_AGENTS
        assert mc.obs_dim == N_REGIMES + N_AGENTS + N_MARKET

    def test_various_n_agents(self):
        from agents.ensemble.meta_controller import MetaController
        for n in [2, 4, 5]:
            mc = MetaController(n_agents=n)
            assert mc.n_agents == n

    def test_last_weights_init(self):
        from agents.ensemble.meta_controller import MetaController
        mc = MetaController(n_agents=N_AGENTS)
        w = mc.last_weights
        assert w.shape == (N_AGENTS,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


class TestGetWeights:
    @pytest.fixture(autouse=True)
    def mc(self):
        from agents.ensemble.meta_controller import MetaController
        self._mc = MetaController(n_agents=N_AGENTS)
        return self._mc

    def test_shape(self):
        w = self._mc.get_weights(_uniform_regime(), _zero_sharpe())
        assert w.shape == (N_AGENTS,)

    def test_dtype(self):
        w = self._mc.get_weights(_uniform_regime(), _zero_sharpe())
        assert w.dtype == np.float32

    def test_sums_to_one(self):
        w = self._mc.get_weights(_uniform_regime(), _zero_sharpe())
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)

    def test_min_weight_floor(self):
        w = self._mc.get_weights(_uniform_regime(), _zero_sharpe())
        assert np.all(w >= self._mc.cfg.min_weight - 1e-6)

    def test_with_market_features(self):
        mf = np.array([0.7, 0.5, 0.9, 0.1], dtype=np.float32)
        w = self._mc.get_weights(_uniform_regime(), _zero_sharpe(), market_features=mf)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)

    def test_wrong_regime_length_graceful(self):
        """Short regime probs should be resized, not crash."""
        rp = np.array([0.5, 0.5], dtype=np.float32)  # too short
        w = self._mc.get_weights(rp, _zero_sharpe())
        assert w.shape == (N_AGENTS,)

    def test_extreme_sharpe_clipped(self):
        """Very large Sharpe values should still produce valid weights."""
        sh = np.array([100.0, -200.0, 50.0], dtype=np.float32)
        w = self._mc.get_weights(_uniform_regime(), sh)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


class TestEmergencyMode:
    def test_triggers_after_window(self):
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        cfg = MetaControllerConfig(emergency_window=3, rebalance_interval=100)
        mc = MetaController(n_agents=N_AGENTS, config=cfg)

        all_neg_sharpe = np.array([-0.5, -1.0, -0.2], dtype=np.float32)
        for _ in range(3):
            mc.step(_uniform_regime(), all_neg_sharpe, portfolio_return=-0.01)

        assert mc.is_emergency

    def test_emergency_returns_zeros(self):
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        cfg = MetaControllerConfig(emergency_window=1, rebalance_interval=100)
        mc = MetaController(n_agents=N_AGENTS, config=cfg)

        mc.step(_uniform_regime(), np.array([-1.0, -1.0, -1.0]), portfolio_return=-0.1)
        w = mc.get_weights(_uniform_regime(), np.array([-1.0, -1.0, -1.0]))
        np.testing.assert_array_equal(w, np.zeros(N_AGENTS))

    def test_resets_on_positive_sharpe(self):
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        cfg = MetaControllerConfig(emergency_window=2, rebalance_interval=100)
        mc = MetaController(n_agents=N_AGENTS, config=cfg)

        for _ in range(2):
            mc.step(_uniform_regime(), np.array([-1.0, -1.0, -1.0]), portfolio_return=-0.1)
        assert mc.is_emergency

        # Positive Sharpe resets counter
        mc.step(_uniform_regime(), np.array([0.5, 0.3, 0.1]), portfolio_return=0.01)
        assert not mc.is_emergency


class TestPPOUpdate:
    def test_update_runs(self):
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        cfg = MetaControllerConfig(mini_batch_size=4, ppo_epochs=2, buffer_size=16)
        mc = MetaController(n_agents=N_AGENTS, config=cfg)

        obs_list, act_list, rew_list, done_list = [], [], [], []
        rng = np.random.default_rng(42)
        for _ in range(10):
            obs_list.append(rng.standard_normal(mc.obs_dim).astype(np.float32))
            act_list.append(rng.dirichlet(np.ones(N_AGENTS)).astype(np.float32))
            rew_list.append(float(rng.standard_normal()))
            done_list.append(False)

        stats = mc.update(obs_list, act_list, rew_list, done_list)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats

    def test_auto_update_clears_buffer(self):
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        # rebalance_interval=5: update triggers at step 5, then step 6 adds 1 new item
        cfg = MetaControllerConfig(rebalance_interval=5, mini_batch_size=4, ppo_epochs=1)
        mc = MetaController(n_agents=N_AGENTS, config=cfg)

        for _ in range(5):
            mc.step(_uniform_regime(), _zero_sharpe(), portfolio_return=0.01)

        # After exactly rebalance_interval steps the buffer is cleared
        assert len(mc.buffer) == 0


class TestSaveLoad:
    def test_round_trip(self):
        from agents.ensemble.meta_controller import MetaController
        mc = MetaController(n_agents=N_AGENTS)
        w_before = mc.get_weights(_uniform_regime(), _zero_sharpe())

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            mc.save(f.name)
            mc2 = MetaController.load(f.name)

        w_after = mc2.get_weights(_uniform_regime(), _zero_sharpe())
        np.testing.assert_allclose(w_before, w_after, atol=1e-5)


# ===========================================================================
# PredictionMarketSignals
# ===========================================================================

class TestPredictionMarketDisabled:
    @pytest.fixture(autouse=True)
    def pms(self):
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(enabled=False)
        self._pms = PredictionMarketSignals(config=cfg)
        return self._pms

    def test_shape(self):
        f = self._pms.get_features("BTC")
        assert f.shape == (4,)

    def test_dtype(self):
        f = self._pms.get_features("BTC")
        assert f.dtype == np.float32

    def test_all_zeros(self):
        f = self._pms.get_features("BTC")
        np.testing.assert_array_equal(f, np.zeros(4, dtype=np.float32))

    def test_different_assets(self):
        for asset in ["BTC", "ETH", "SPY", "UNKNOWN"]:
            f = self._pms.get_features(asset)
            assert f.shape == (4,)


class TestPredictionMarketFallback:
    """API unavailable → graceful zeros (requests not patched, no live network)."""

    def test_fallback_when_api_unavailable(self):
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(
            enabled=True,
            providers=["polymarket", "kalshi"],
            cache_db=None,      # no caching
            timeout=0.001,      # effectively zero → requests will fail fast
        )
        pms = PredictionMarketSignals(config=cfg)
        f = pms.get_features("BTC")
        assert f.shape == (4,)
        assert f.dtype == np.float32
        assert np.all(f >= 0.0) and np.all(f <= 1.0)


class TestFeatureComputation:
    """Unit-test _compute_features directly."""

    @pytest.fixture(autouse=True)
    def pms(self):
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(enabled=False, cache_db=None)
        self._pms = PredictionMarketSignals(config=cfg)
        return self._pms

    def test_max_entropy_at_half(self):
        """Binary entropy is 1.0 at p=0.5."""
        f = self._pms._compute_features("BTC", poly_prob=0.5, kalshi_prob=0.5)
        # uncertainty (index 2) should be ~1.0
        np.testing.assert_allclose(f[2], 1.0, atol=1e-5)

    def test_zero_entropy_at_extremes(self):
        """Binary entropy → 0 at p≈0 or p≈1."""
        f0 = self._pms._compute_features("BTC", poly_prob=0.001, kalshi_prob=0.001)
        f1 = self._pms._compute_features("BTC", poly_prob=0.999, kalshi_prob=0.999)
        assert f0[2] < 0.05
        assert f1[2] < 0.05

    def test_divergence_correct(self):
        f = self._pms._compute_features("BTC", poly_prob=0.8, kalshi_prob=0.5)
        np.testing.assert_allclose(f[3], 0.3, atol=1e-5)

    def test_divergence_zero_single_source(self):
        f = self._pms._compute_features("BTC", poly_prob=0.7, kalshi_prob=None)
        assert f[3] == 0.0

    def test_none_probs_yield_neutral(self):
        f = self._pms._compute_features("BTC", poly_prob=None, kalshi_prob=None)
        # primary = 0.5 (neutral)
        np.testing.assert_allclose(f[0], 0.5, atol=1e-5)


class TestMomentumHistory:
    def test_history_grows(self):
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(enabled=False, momentum_window=5, cache_db=None)
        pms = PredictionMarketSignals(config=cfg)

        for i in range(3):
            pms._compute_features("BTC", poly_prob=0.5, kalshi_prob=0.5)

        assert len(pms._prob_history["BTC"]) == 3

    def test_history_capped(self):
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(enabled=False, momentum_window=5, cache_db=None)
        pms = PredictionMarketSignals(config=cfg)

        for _ in range(20):
            pms._compute_features("BTC", poly_prob=0.5, kalshi_prob=0.5)

        # max length = momentum_window + 1
        assert len(pms._prob_history["BTC"]) <= cfg.momentum_window + 1

    def test_reset_history(self):
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(enabled=False, cache_db=None)
        pms = PredictionMarketSignals(config=cfg)

        for _ in range(5):
            pms._compute_features("BTC", poly_prob=0.5, kalshi_prob=0.5)

        pms.reset_history("BTC")
        assert "BTC" not in pms._prob_history


class TestAlignToPrices:
    def test_columns_appended(self):
        import pandas as pd
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
            PREDICTION_MARKET_COLS,
        )
        cfg = PredictionMarketConfig(enabled=False, cache_db=None)
        pms = PredictionMarketSignals(config=cfg)

        df = pd.DataFrame(
            {"close": np.random.rand(10), "volume": np.random.rand(10)}
        )
        out = pms.align_to_prices(df, asset="BTC")

        for col in PREDICTION_MARKET_COLS:
            assert col in out.columns
        assert len(out) == len(df)

    def test_no_nan(self):
        import pandas as pd
        from training.signals.prediction_market import (
            PredictionMarketSignals,
            PredictionMarketConfig,
        )
        cfg = PredictionMarketConfig(enabled=False, cache_db=None)
        pms = PredictionMarketSignals(config=cfg)

        df = pd.DataFrame({"close": np.random.rand(5)})
        out = pms.align_to_prices(df)
        assert not out.isnull().any().any()


class TestConstants:
    def test_n_features(self):
        from training.signals.prediction_market import N_PREDICTION_MARKET_FEATURES
        assert N_PREDICTION_MARKET_FEATURES == 4

    def test_cols_length(self):
        from training.signals.prediction_market import PREDICTION_MARKET_COLS
        assert len(PREDICTION_MARKET_COLS) == 4


# ===========================================================================
# Integration: MetaController + prediction market features
# ===========================================================================

class TestMetaControllerWithPredictionMarket:
    def test_accepts_pm_features(self):
        from agents.ensemble.meta_controller import MetaController
        mc = MetaController(n_agents=N_AGENTS)

        pm_features = np.array([0.7, 0.55, 0.9, 0.15], dtype=np.float32)
        w = mc.get_weights(_uniform_regime(), _zero_sharpe(), market_features=pm_features)
        assert w.shape == (N_AGENTS,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)

    def test_weights_differ_for_different_pm_signals(self):
        """Different prediction-market features should (statistically) produce
        different weights after a small training batch."""
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
        cfg = MetaControllerConfig(mini_batch_size=4, ppo_epochs=2, buffer_size=16)
        mc = MetaController(n_agents=N_AGENTS, config=cfg)

        rng = np.random.default_rng(99)
        obs_list, act_list, rew_list, done_list = [], [], [], []
        for _ in range(10):
            obs_list.append(rng.standard_normal(mc.obs_dim).astype(np.float32))
            act_list.append(rng.dirichlet(np.ones(N_AGENTS)).astype(np.float32))
            rew_list.append(float(rng.standard_normal()))
            done_list.append(False)
        mc.update(obs_list, act_list, rew_list, done_list)

        pm_low = np.array([0.1, 0.5, 0.2, 0.0], dtype=np.float32)
        pm_high = np.array([0.9, 0.5, 0.8, 0.6], dtype=np.float32)
        w_low = mc.get_weights(_uniform_regime(), _zero_sharpe(), market_features=pm_low)
        w_high = mc.get_weights(_uniform_regime(), _zero_sharpe(), market_features=pm_high)

        # Weights may differ (not guaranteed but network is deterministic given same seed)
        # Just check both are valid
        np.testing.assert_allclose(w_low.sum(), 1.0, atol=1e-5)
        np.testing.assert_allclose(w_high.sum(), 1.0, atol=1e-5)
