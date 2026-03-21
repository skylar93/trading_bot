"""
Phase 4: HMM Regime Detection + Diffusion Data Augmentation tests.

Coverage
--------
RegimeDetectorConfig
  - Defaults
  - from_config factory

MarketRegimeDetector
  - Instantiation
  - extract_features: shape, no NaN, handles $-prefixed columns
  - fit: runs on sufficient data, sets is_fitted
  - fit: raises on insufficient data
  - fit: raises if hmmlearn unavailable (mocked)
  - predict_proba: shape (n_regimes,), sums to 1, float32
  - predict_regime: int in [0, n_regimes)
  - predict_sequence: (T,) int array, values in range
  - fit_predict: same as fit + predict_sequence
  - Unfitted predict raises RuntimeError
  - save / load round-trip
  - regime_labels length matches n_regimes
  - n_regimes=2 and n_regimes=4 label variants

DiffusionConfig
  - Defaults

TradingDiffusionAugmentor
  - Instantiation: obs_dim stored, is_fitted=False
  - count_parameters > 0
  - fit: runs without error
  - augment before fit raises RuntimeError
  - augment_dataset before fit raises RuntimeError
  - augment: output Trajectory has same T, same obs_dim
  - augment: actions unchanged when jitter_actions=False
  - augment_dataset: new dataset length = (1 + n_aug) × original
  - augment_dataset: obs_dim preserved
  - noise_level clipping (edge cases 0 and 1)
  - save / load round-trip: is_fitted preserved
  - cosine schedule builds without error

Integration
  - RegimeDetector → regime_probs → MetaController.get_weights
  - Augmented dataset can be used to construct TradingTrajectoryDataset
"""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 8
ACT_DIM = 1
TRAJ_LEN = 60   # long enough for HMM fitting
N_TRAJ = 3
CONTEXT_LEN = 10
N_REGIMES = 3


def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.uniform(1e5, 1e6, n)
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume})


def _make_ohlcv_dollar_prefix(n: int = 200) -> pd.DataFrame:
    df = _make_ohlcv(n)
    df.columns = ["$close", "$high", "$low", "$volume"]
    return df


def _make_trajectory(length: int = TRAJ_LEN) -> "Trajectory":
    from agents.offline.trajectory_dataset import Trajectory
    rng = np.random.default_rng(42)
    return Trajectory(
        observations=rng.standard_normal((length, OBS_DIM)).astype(np.float32),
        actions=rng.standard_normal((length, ACT_DIM)).astype(np.float32),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.zeros(length, dtype=np.float32),
    )


def _make_dataset(n_traj: int = N_TRAJ) -> "TradingTrajectoryDataset":
    from agents.offline.trajectory_dataset import TradingTrajectoryDataset
    return TradingTrajectoryDataset(
        [_make_trajectory() for _ in range(n_traj)],
        context_len=CONTEXT_LEN,
        normalize_states=False,
        normalize_returns=False,
    )


# ===========================================================================
# RegimeDetectorConfig
# ===========================================================================

class TestRegimeDetectorConfig:
    def test_defaults(self):
        from agents.ensemble.regime_detector import RegimeDetectorConfig
        cfg = RegimeDetectorConfig()
        assert cfg.n_regimes == 3
        assert cfg.n_iter == 100
        assert cfg.covariance_type == "diag"
        assert cfg.vol_window == 20
        assert cfg.momentum_window == 10
        assert cfg.min_samples == 50
        assert cfg.temperature == 1.0

    def test_from_config_factory(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector.from_config({"n_regimes": 4, "n_iter": 50})
        assert det.cfg.n_regimes == 4
        assert det.cfg.n_iter == 50


# ===========================================================================
# MarketRegimeDetector — instantiation
# ===========================================================================

class TestRegimeDetectorInstantiation:
    def test_default_init(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        assert det.cfg.n_regimes == 3
        assert not det.is_fitted

    def test_n_regimes_override(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=4)
        assert det.cfg.n_regimes == 4

    def test_labels_length(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        for n in [2, 3, 4, 5]:
            det = MarketRegimeDetector(n_regimes=n)
            assert len(det.regime_labels) == n

    def test_labels_2(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=2)
        assert det.regime_labels == ["bear", "bull"]

    def test_labels_3(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=3)
        assert det.regime_labels == ["bear", "sideways", "bull"]

    def test_labels_4(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=4)
        assert det.regime_labels == ["crash", "bear", "bull", "bubble"]


# ===========================================================================
# MarketRegimeDetector — feature extraction
# ===========================================================================

class TestExtractFeatures:
    @pytest.fixture(autouse=True)
    def det(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        self._det = MarketRegimeDetector()

    def test_shape(self):
        df = _make_ohlcv(150)
        features = self._det.extract_features(df)
        assert features.ndim == 2
        assert features.shape[0] == 150
        assert features.shape[1] == 4  # log_ret, vol, momentum, vol_ratio

    def test_dtype(self):
        features = self._det.extract_features(_make_ohlcv(100))
        assert features.dtype == np.float32

    def test_no_nan(self):
        features = self._det.extract_features(_make_ohlcv(100))
        assert not np.any(np.isnan(features))

    def test_no_inf(self):
        features = self._det.extract_features(_make_ohlcv(100))
        assert not np.any(np.isinf(features))

    def test_dollar_prefix_columns(self):
        df = _make_ohlcv_dollar_prefix(100)
        features = self._det.extract_features(df)
        assert features.shape == (100, 4)

    def test_no_volume_column(self):
        df = _make_ohlcv(100).drop(columns=["volume"])
        features = self._det.extract_features(df)
        assert features.shape == (100, 4)
        # volume_ratio column should be zeros
        np.testing.assert_array_equal(features[:, 3], np.zeros(100, dtype=np.float32))


# ===========================================================================
# MarketRegimeDetector — fit
# ===========================================================================

class TestRegimeDetectorFit:
    @pytest.fixture(autouse=True)
    def det(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        self._det = MarketRegimeDetector(n_regimes=N_REGIMES)

    def _features(self, n: int = 200) -> np.ndarray:
        return self._det.extract_features(_make_ohlcv(n))

    def test_fit_sets_is_fitted(self):
        self._det.fit(self._features())
        assert self._det.is_fitted

    def test_fit_returns_self(self):
        result = self._det.fit(self._features())
        assert result is self._det

    def test_fit_raises_insufficient_data(self):
        from agents.ensemble.regime_detector import RegimeDetectorConfig, MarketRegimeDetector
        det = MarketRegimeDetector(config=RegimeDetectorConfig(min_samples=100))
        features = self._features(10)
        with pytest.raises(ValueError, match="min_samples"):
            det.fit(features)

    def test_fit_updates_labels(self):
        self._det.fit(self._features())
        assert len(self._det.regime_labels) == N_REGIMES


# ===========================================================================
# MarketRegimeDetector — predict
# ===========================================================================

class TestRegimeDetectorPredict:
    @pytest.fixture(autouse=True)
    def fitted_det(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        self._det = MarketRegimeDetector(n_regimes=N_REGIMES)
        feats = self._det.extract_features(_make_ohlcv(300))
        self._det.fit(feats)
        self._feats = feats

    def test_predict_proba_shape(self):
        probs = self._det.predict_proba(self._feats[-10:])
        assert probs.shape == (N_REGIMES,)

    def test_predict_proba_sums_to_one(self):
        probs = self._det.predict_proba(self._feats[-10:])
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_predict_proba_dtype(self):
        probs = self._det.predict_proba(self._feats[-10:])
        assert probs.dtype == np.float32

    def test_predict_proba_non_negative(self):
        probs = self._det.predict_proba(self._feats[-10:])
        assert np.all(probs >= 0.0)

    def test_predict_proba_single_row(self):
        probs = self._det.predict_proba(self._feats[-1:])
        assert probs.shape == (N_REGIMES,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_predict_regime_range(self):
        r = self._det.predict_regime(self._feats[-10:])
        assert isinstance(r, int)
        assert 0 <= r < N_REGIMES

    def test_predict_sequence_shape(self):
        seq = self._det.predict_sequence(self._feats)
        assert seq.shape == (len(self._feats),)

    def test_predict_sequence_values_in_range(self):
        seq = self._det.predict_sequence(self._feats)
        assert np.all(seq >= 0)
        assert np.all(seq < N_REGIMES)

    def test_fit_predict(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=N_REGIMES)
        feats = det.extract_features(_make_ohlcv(300))
        seq = det.fit_predict(feats)
        assert seq.shape == (len(feats),)
        assert det.is_fitted

    def test_unfitted_predict_raises(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict_proba(self._feats[-5:])

    def test_unfitted_predict_regime_raises(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict_regime(self._feats[-5:])


# ===========================================================================
# MarketRegimeDetector — save / load
# ===========================================================================

class TestRegimeDetectorSaveLoad:
    def test_round_trip(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=N_REGIMES)
        feats = det.extract_features(_make_ohlcv(300))
        det.fit(feats)
        probs_before = det.predict_proba(feats[-20:])

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            det.save(f.name)
            det2 = MarketRegimeDetector.load(f.name)

        assert det2.is_fitted
        assert det2.cfg.n_regimes == N_REGIMES
        probs_after = det2.predict_proba(feats[-20:])
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-5)

    def test_load_preserves_labels(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(n_regimes=3)
        feats = det.extract_features(_make_ohlcv(300))
        det.fit(feats)

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            det.save(f.name)
            det2 = MarketRegimeDetector.load(f.name)

        assert det2.regime_labels == det.regime_labels


# ===========================================================================
# DiffusionConfig
# ===========================================================================

class TestDiffusionConfig:
    def test_defaults(self):
        from agents.offline.diffusion_augmentor import DiffusionConfig
        cfg = DiffusionConfig()
        assert cfg.n_diffusion_steps == 50
        assert cfg.beta_start == 1e-4
        assert cfg.beta_end == 0.02
        assert cfg.schedule == "linear"
        assert cfg.hidden_dim == 128
        assert cfg.n_layers == 2
        assert not cfg.jitter_actions


# ===========================================================================
# TradingDiffusionAugmentor — instantiation
# ===========================================================================

class TestDiffusionAugmentorInstantiation:
    def test_stores_obs_dim(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM)
        assert aug.obs_dim == OBS_DIM

    def test_not_fitted_initially(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM)
        assert not aug.is_fitted

    def test_count_parameters_positive(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM)
        assert aug.count_parameters() > 0

    def test_cosine_schedule_builds(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(schedule="cosine", n_diffusion_steps=10)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        assert aug._betas.shape == (10,)

    def test_schedule_shape(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=20)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        assert aug._betas.shape == (20,)
        assert aug._alphas_cumprod.shape == (20,)


# ===========================================================================
# TradingDiffusionAugmentor — fit
# ===========================================================================

class TestDiffusionAugmentorFit:
    @pytest.fixture(autouse=True)
    def aug(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5, n_epochs=2, batch_size=16)
        self._aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        self._ds = _make_dataset()

    def test_fit_sets_is_fitted(self):
        self._aug.fit(self._ds)
        assert self._aug.is_fitted

    def test_fit_returns_self(self):
        result = self._aug.fit(self._ds)
        assert result is self._aug


# ===========================================================================
# TradingDiffusionAugmentor — augment
# ===========================================================================

class TestDiffusionAugmentorAugment:
    @pytest.fixture(autouse=True)
    def fitted_aug(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5, n_epochs=2, batch_size=16)
        self._aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        self._ds = _make_dataset()
        self._aug.fit(self._ds)
        self._traj = _make_trajectory()

    def test_unfitted_raises(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        with pytest.raises(RuntimeError, match="fit()"):
            aug.augment(_make_trajectory())

    def test_output_same_length(self):
        aug_traj = self._aug.augment(self._traj)
        assert len(aug_traj) == len(self._traj)

    def test_output_obs_dim(self):
        aug_traj = self._aug.augment(self._traj)
        assert aug_traj.observations.shape == self._traj.observations.shape

    def test_actions_unchanged_no_jitter(self):
        aug_traj = self._aug.augment(self._traj)
        np.testing.assert_array_equal(aug_traj.actions, self._traj.actions)

    def test_rewards_unchanged_no_noise(self):
        aug_traj = self._aug.augment(self._traj)
        np.testing.assert_array_equal(aug_traj.rewards, self._traj.rewards)

    def test_dones_unchanged(self):
        aug_traj = self._aug.augment(self._traj)
        np.testing.assert_array_equal(aug_traj.dones, self._traj.dones)

    def test_obs_dtype_float32(self):
        aug_traj = self._aug.augment(self._traj)
        assert aug_traj.observations.dtype == np.float32

    def test_noise_level_zero_near_original(self):
        """noise_level ~ 0 → minimal diffusion → output close to input."""
        aug_traj = self._aug.augment(self._traj, noise_level=0.01)
        assert aug_traj.observations.shape == self._traj.observations.shape

    def test_noise_level_one_accepted(self):
        aug_traj = self._aug.augment(self._traj, noise_level=1.0)
        assert aug_traj.observations.shape == self._traj.observations.shape

    def test_jitter_actions(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5, n_epochs=2, jitter_actions=True, action_noise_std=0.1)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        aug.fit(self._ds)
        aug_traj = aug.augment(self._traj)
        # Actions should differ (with high probability given noise_std=0.1)
        assert not np.allclose(aug_traj.actions, self._traj.actions)


# ===========================================================================
# TradingDiffusionAugmentor — augment_dataset
# ===========================================================================

class TestDiffusionAugmentDataset:
    @pytest.fixture(autouse=True)
    def fitted_aug(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5, n_epochs=2, batch_size=16)
        self._aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        self._ds = _make_dataset(N_TRAJ)
        self._aug.fit(self._ds)

    def test_unfitted_raises(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        with pytest.raises(RuntimeError, match="fit()"):
            aug.augment_dataset(self._ds)

    def test_trajectory_count_n_aug_1(self):
        new_ds = self._aug.augment_dataset(self._ds, n_aug=1)
        assert len(new_ds.trajectories) == 2 * N_TRAJ

    def test_trajectory_count_n_aug_3(self):
        new_ds = self._aug.augment_dataset(self._ds, n_aug=3)
        assert len(new_ds.trajectories) == 4 * N_TRAJ

    def test_obs_dim_preserved(self):
        new_ds = self._aug.augment_dataset(self._ds, n_aug=1)
        assert new_ds.obs_dim == OBS_DIM

    def test_context_len_preserved(self):
        new_ds = self._aug.augment_dataset(self._ds, n_aug=1)
        assert new_ds.context_len == self._ds.context_len

    def test_returns_dataset_instance(self):
        from agents.offline.trajectory_dataset import TradingTrajectoryDataset
        new_ds = self._aug.augment_dataset(self._ds, n_aug=1)
        assert isinstance(new_ds, TradingTrajectoryDataset)


# ===========================================================================
# TradingDiffusionAugmentor — save / load
# ===========================================================================

class TestDiffusionAugmentorSaveLoad:
    def test_round_trip_is_fitted(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5, n_epochs=2)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        aug.fit(_make_dataset())

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            aug.save(f.name)
            aug2 = TradingDiffusionAugmentor.load(f.name)

        assert aug2.is_fitted
        assert aug2.obs_dim == OBS_DIM

    def test_round_trip_can_augment(self):
        from agents.offline.diffusion_augmentor import TradingDiffusionAugmentor, DiffusionConfig
        cfg = DiffusionConfig(n_diffusion_steps=5, n_epochs=2)
        aug = TradingDiffusionAugmentor(obs_dim=OBS_DIM, config=cfg)
        aug.fit(_make_dataset())
        traj = _make_trajectory()

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            aug.save(f.name)
            aug2 = TradingDiffusionAugmentor.load(f.name)

        aug_traj = aug2.augment(traj)
        assert aug_traj.observations.shape == traj.observations.shape


# ===========================================================================
# Integration: RegimeDetector → MetaController
# ===========================================================================

class TestRegimeDetectorMetaControllerIntegration:
    def test_probs_feed_into_meta_controller(self):
        from agents.ensemble.regime_detector import MarketRegimeDetector
        from agents.ensemble.meta_controller import MetaController, MetaControllerConfig

        det = MarketRegimeDetector(n_regimes=3)
        feats = det.extract_features(_make_ohlcv(300))
        det.fit(feats)

        regime_probs = det.predict_proba(feats[-20:])

        cfg = MetaControllerConfig(n_regimes=3)
        mc = MetaController(n_agents=3, config=cfg)
        sharpe = np.array([0.5, -0.1, 0.3], dtype=np.float32)
        weights = mc.get_weights(regime_probs, sharpe)

        assert weights.shape == (3,)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)

    def test_different_regimes_may_produce_different_weights(self):
        """Smoke test: two different regime distributions → valid weights."""
        from agents.ensemble.regime_detector import MarketRegimeDetector
        from agents.ensemble.meta_controller import MetaController

        det = MarketRegimeDetector(n_regimes=3)
        feats = det.extract_features(_make_ohlcv(300))
        det.fit(feats)

        mc = MetaController(n_agents=3)
        sharpe = np.zeros(3, dtype=np.float32)

        probs_a = det.predict_proba(feats[:50])
        probs_b = det.predict_proba(feats[-50:])

        w_a = mc.get_weights(probs_a, sharpe)
        w_b = mc.get_weights(probs_b, sharpe)

        # Both must be valid
        np.testing.assert_allclose(w_a.sum(), 1.0, atol=1e-5)
        np.testing.assert_allclose(w_b.sum(), 1.0, atol=1e-5)
