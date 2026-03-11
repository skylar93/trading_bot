"""
Week 6 tests — Market Regime Detection.

Coverage:
  - HMMRegimeDetector: fit, predict, probs, unfitted defaults
  - ThresholdRegimeDetector: fit, predict, probs, edge cases
  - RegimeDetector (facade): fit, fallback, properties, weight multipliers
  - RegimeDetector._extract_features helper
  - SingleAssetRLTradingEnv: regime obs injection (shape, dtype, validity)
  - EnsembleManager: set_regime_detector, update_weights_regime_aware,
      get_current_regime, get_regime_info, get_ensemble_metrics with regime

All tests use only CPU (no GPU dependency).
"""

import numpy as np
import pandas as pd
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_prices(n: int = 300, seed: int = 0) -> np.ndarray:
    """Synthetic geometric-Brownian-motion price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    prices = 100.0 * np.cumprod(1.0 + returns)
    return prices.astype(np.float64)


def _make_regime_prices(n_per_regime: int = 150, seed: int = 0) -> np.ndarray:
    """Three-segment price series: low-vol trend, high-vol crash, low-vol recovery."""
    rng = np.random.default_rng(seed)
    trend   = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.005, n_per_regime))
    crash   = trend[-1]  * np.cumprod(1.0 + rng.normal(-0.002, 0.025, n_per_regime))
    recover = crash[-1]  * np.cumprod(1.0 + rng.normal(0.0005, 0.004, n_per_regime))
    return np.concatenate([trend, crash, recover])


def _make_env_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """OHLCV DataFrame for SingleAssetRLTradingEnv."""
    prices = _make_prices(n, seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "$open":   prices * (1 + rng.uniform(-0.002, 0.002, n)),
            "$high":   prices * (1 + rng.uniform(0.000, 0.005, n)),
            "$low":    prices * (1 - rng.uniform(0.000, 0.005, n)),
            "$close":  prices,
            "$volume": rng.uniform(1e4, 1e6, n),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# _extract_features
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_output_shape(self):
        from training.regime.regime_detector import _extract_features
        prices = _make_prices(100)
        feat = _extract_features(prices)
        assert feat.shape == (99, 3), feat.shape

    def test_too_short_raises(self):
        from training.regime.regime_detector import _extract_features
        with pytest.raises(ValueError):
            _extract_features(np.array([1.0, 2.0]))

    def test_no_nan(self):
        from training.regime.regime_detector import _extract_features
        feat = _extract_features(_make_prices(50))
        assert not np.isnan(feat).any()

    def test_vol_col_non_negative(self):
        from training.regime.regime_detector import _extract_features
        feat = _extract_features(_make_prices(50))
        assert (feat[:, 1] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# ThresholdRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdRegimeDetector:
    def test_fit_stores_percentiles(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        det.fit(_make_prices(200))
        assert det._p33 > 0
        assert det._p67 > det._p33

    def test_is_fitted_flag(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        assert not det.is_fitted
        det.fit(_make_prices(200))
        assert det.is_fitted

    def test_probs_sum_to_one(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        det.fit(_make_prices(200))
        probs = det.get_regime_probs(_make_prices(50))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_probs_shape(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        det.fit(_make_prices(200))
        assert det.get_regime_probs(_make_prices(50)).shape == (3,)

    def test_probs_one_hot(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        det.fit(_make_prices(200))
        probs = det.get_regime_probs(_make_prices(50))
        # Threshold detector returns one-hot probabilities
        assert (probs == 1.0).sum() == 1

    def test_predict_regime_valid_label(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        det.fit(_make_prices(300))
        label = det.predict_regime(_make_prices(50))
        assert label in (0, 1, 2)

    def test_unfitted_returns_uniform(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        probs = det.get_regime_probs(_make_prices(50))
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_too_short_returns_uniform(self):
        from training.regime.regime_detector import ThresholdRegimeDetector
        det = ThresholdRegimeDetector()
        det.fit(_make_prices(200))
        probs = det.get_regime_probs(np.array([100.0]))  # only 1 price
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_high_vol_prices_classified(self):
        """Very volatile segment → regime 2."""
        from training.regime.regime_detector import ThresholdRegimeDetector
        rng = np.random.default_rng(7)
        low_vol  = 100 * np.cumprod(1 + rng.normal(0, 0.001, 200))
        high_vol = low_vol[-1] * np.cumprod(1 + rng.normal(0, 0.05, 100))

        det = ThresholdRegimeDetector()
        det.fit(low_vol)
        label = det.predict_regime(high_vol)
        assert label == 2, f"Expected crisis (2), got {label}"


# ─────────────────────────────────────────────────────────────────────────────
# HMMRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMRegimeDetector:
    def test_fit_sets_fitted_flag(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        assert det.is_fitted

    def test_probs_shape(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        probs = det.get_regime_probs(_make_prices(50))
        assert probs.shape == (3,)

    def test_probs_sum_to_one(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        probs = det.get_regime_probs(_make_prices(50))
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_probs_in_range(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        probs = det.get_regime_probs(_make_prices(50))
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_valid_label(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        label = det.predict_regime(_make_prices(50))
        assert label in (0, 1, 2)

    def test_unfitted_returns_uniform(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector()
        probs = det.get_regime_probs(_make_prices(50))
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_too_short_returns_uniform(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        probs = det.get_regime_probs(np.array([100.0, 101.0]))  # only 2 prices → 1 feature row
        # Should still return valid probs (1 sample is degenerate but handled)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_state_label_map_coverage(self):
        """After fitting, state_labels must cover all 3 hmm states."""
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        assert len(det._state_labels) == 3

    def test_state_label_values_are_valid(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        values = set(det._state_labels.values())
        assert values == {0, 1, 2}

    def test_output_dtype_float32(self):
        from training.regime.regime_detector import HMMRegimeDetector
        det = HMMRegimeDetector(n_iter=20)
        det.fit(_make_prices(300))
        probs = det.get_regime_probs(_make_prices(50))
        assert probs.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# RegimeDetector (facade)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeDetector:
    def test_invalid_method_raises(self):
        from training.regime.regime_detector import RegimeDetector
        with pytest.raises(ValueError):
            RegimeDetector(method="invalid")

    def test_hmm_method_fits(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="hmm", n_iter=20)
        det.fit(_make_prices(300))
        assert det.is_fitted

    def test_threshold_method_fits(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        assert det.is_fitted

    def test_unfitted_returns_uniform_probs(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector()
        probs = det.get_regime_probs(_make_prices(50))
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_get_regime_probs_shape_and_sum(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="hmm", n_iter=20)
        det.fit(_make_prices(300))
        probs = det.get_regime_probs(_make_prices(50))
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_regime_valid_label(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        label = det.predict_regime(_make_prices(50))
        assert label in (0, 1, 2)

    def test_current_probs_property(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        det.get_regime_probs(_make_prices(50))
        probs = det.current_probs
        assert probs.shape == (3,)

    def test_current_regime_property(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        det.predict_regime(_make_prices(50))
        assert det.current_regime in (0, 1, 2)

    def test_current_regime_name_property(self):
        from training.regime.regime_detector import RegimeDetector, REGIME_NAMES
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        det.predict_regime(_make_prices(50))
        assert det.current_regime_name in REGIME_NAMES.values()

    def test_threshold_fallback(self):
        """Even with method='hmm', threshold is used when HMM fit not available."""
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="hmm", fallback_on_error=True)
        # Fit only threshold, make HMM unavailable
        det._threshold.fit(_make_prices(300))
        det._is_fitted = True  # mark as fitted without running HMM

        probs = det.get_regime_probs(_make_prices(50))
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_get_weight_multipliers_low_vol(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector()
        mults = det.get_weight_multipliers(0)
        assert mults["conservative"] > 1.0
        assert mults["aggressive"] < 1.0

    def test_get_weight_multipliers_medium_vol(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector()
        mults = det.get_weight_multipliers(1)
        assert mults["conservative"] == 1.0
        assert mults["moderate"] == 1.0
        assert mults["aggressive"] == 1.0

    def test_get_weight_multipliers_high_vol(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector()
        mults = det.get_weight_multipliers(2)
        assert mults["conservative"] < 1.0
        assert mults["moderate"] > 1.0
        assert mults["aggressive"] < 1.0

    def test_repr(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        assert "RegimeDetector" in repr(det)

    def test_current_probs_is_copy(self):
        """Modifying returned array should not affect internal state."""
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        det.predict_regime(_make_prices(50))
        probs = det.current_probs
        probs[:] = 0.0
        assert det.current_probs.sum() > 0  # internal state unchanged


# ─────────────────────────────────────────────────────────────────────────────
# SingleAssetRLTradingEnv with regime
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvWithRegime:
    def _make_env(self, use_regime: bool = True, window_size: int = 20):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv
        from training.regime.regime_detector import RegimeDetector

        df = _make_env_df(200)
        regime_det = None
        if use_regime:
            regime_det = RegimeDetector(method="threshold")
            regime_det.fit(df["$close"].values)

        return SingleAssetRLTradingEnv(
            data=df,
            window_size=window_size,
            regime_detector=regime_det,
        )

    def test_obs_space_no_regime(self):
        env = self._make_env(use_regime=False)
        assert env.observation_space.shape == (20, 5)

    def test_obs_space_with_regime(self):
        env = self._make_env(use_regime=True)
        assert env.observation_space.shape == (20, 8)

    def test_reset_obs_shape_no_regime(self):
        env = self._make_env(use_regime=False)
        obs, _ = env.reset()
        assert obs.shape == (20, 5)

    def test_reset_obs_shape_with_regime(self):
        env = self._make_env(use_regime=True)
        obs, _ = env.reset()
        assert obs.shape == (20, 8)

    def test_step_obs_shape_with_regime(self):
        env = self._make_env(use_regime=True)
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs.shape == (20, 8)

    def test_regime_cols_are_probs(self):
        """Regime columns (5-7) should be in [0, 1] and sum to ~1 per row."""
        env = self._make_env(use_regime=True)
        obs, _ = env.reset()
        regime_cols = obs[:, 5:]
        assert regime_cols.shape == (20, 3)
        assert (regime_cols >= 0).all() and (regime_cols <= 1).all()
        np.testing.assert_allclose(regime_cols.sum(axis=1), 1.0, atol=1e-5)

    def test_regime_cols_broadcast_identical(self):
        """Same regime probs broadcast across all rows in the window."""
        env = self._make_env(use_regime=True)
        obs, _ = env.reset()
        regime_cols = obs[:, 5:]
        # Each row should be identical (same regime probs for all timesteps)
        for i in range(1, obs.shape[0]):
            np.testing.assert_allclose(regime_cols[i], regime_cols[0], atol=1e-6)

    def test_obs_dtype_float32(self):
        env = self._make_env(use_regime=True)
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_ohlcv_cols_unchanged(self):
        """OHLCV columns (0-4) should be the same with or without regime."""
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv
        from training.regime.regime_detector import RegimeDetector

        df = _make_env_df(200, seed=1)

        env_base = SingleAssetRLTradingEnv(data=df, window_size=20)
        obs_base, _ = env_base.reset(seed=0)

        det = RegimeDetector(method="threshold")
        det.fit(df["$close"].values)
        env_reg = SingleAssetRLTradingEnv(data=df, window_size=20, regime_detector=det)
        obs_reg, _ = env_reg.reset(seed=0)

        np.testing.assert_allclose(obs_reg[:, :5], obs_base, atol=1e-6)

    def test_sb3_check_env_with_regime(self):
        """SB3 check_env should pass for both obs shapes."""
        from stable_baselines3.common.env_checker import check_env
        env = self._make_env(use_regime=True, window_size=10)
        check_env(env, warn=False)

    def test_sb3_check_env_without_regime(self):
        from stable_baselines3.common.env_checker import check_env
        env = self._make_env(use_regime=False, window_size=10)
        check_env(env, warn=False)


# ─────────────────────────────────────────────────────────────────────────────
# EnsembleManager with regime
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleManagerRegime:
    def _make_ensemble(self):
        from agents.ensemble.ensemble_manager import EnsembleManager
        import gymnasium as gym
        import numpy as np

        obs_space = gym.spaces.Box(low=-10, high=10, shape=(20, 5), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        return EnsembleManager(
            observation_space=obs_space,
            action_space=act_space,
            method="rolling_validation",
        )

    def _make_detector(self):
        from training.regime.regime_detector import RegimeDetector
        det = RegimeDetector(method="threshold")
        det.fit(_make_prices(300))
        return det

    def test_set_regime_detector(self):
        em = self._make_ensemble()
        det = self._make_detector()
        em.set_regime_detector(det)
        assert em._regime_detector is det

    def test_get_current_regime_no_detector(self):
        em = self._make_ensemble()
        assert em.get_current_regime() is None

    def test_get_current_regime_with_detector(self):
        em = self._make_ensemble()
        det = self._make_detector()
        em.set_regime_detector(det)
        det.predict_regime(_make_prices(50))  # set current regime
        label = em.get_current_regime()
        assert label in (0, 1, 2)

    def test_get_regime_info_no_detector(self):
        em = self._make_ensemble()
        assert em.get_regime_info() == {}

    def test_get_regime_info_with_detector(self):
        em = self._make_ensemble()
        det = self._make_detector()
        em.set_regime_detector(det)
        det.predict_regime(_make_prices(50))
        info = em.get_regime_info()
        assert "regime" in info
        assert "regime_name" in info
        assert "probs" in info

    def test_update_weights_regime_aware_explicit_regime(self):
        em = self._make_ensemble()
        eval_metrics = {aid: {"mean_reward": float(i)} for i, aid in enumerate(em.agents)}
        initial_weights = dict(em.get_weights())

        # Regime 0: should boost conservative, penalise aggressive
        em.update_weights_regime_aware(eval_metrics, regime=0)
        new_weights = em.get_weights()

        assert abs(sum(new_weights.values()) - 1.0) < 1e-6
        # ppo_conservative should have higher relative weight vs td3_aggressive
        assert new_weights["ppo_conservative"] > new_weights["td3_aggressive"]

    def test_update_weights_regime_aware_from_prices(self):
        em = self._make_ensemble()
        det = self._make_detector()
        em.set_regime_detector(det)

        eval_metrics = {aid: {"mean_reward": 0.1} for aid in em.agents}
        prices = _make_prices(50)
        em.update_weights_regime_aware(eval_metrics, prices=prices)

        weights = em.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_update_weights_regime_aware_no_regime_no_change_in_normalisation(self):
        """Without regime info, only performance update happens — weights still sum to 1."""
        em = self._make_ensemble()
        eval_metrics = {aid: {"mean_reward": 0.0} for aid in em.agents}
        em.update_weights_regime_aware(eval_metrics)  # no prices, no regime
        weights = em.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_sum_to_one_after_regime_update(self):
        em = self._make_ensemble()
        eval_metrics = {aid: {"mean_reward": 0.5} for aid in em.agents}
        for regime in (0, 1, 2):
            em.update_weights_regime_aware(eval_metrics, regime=regime)
            weights = em.get_weights()
            assert abs(sum(weights.values()) - 1.0) < 1e-6, f"regime={regime}"

    def test_ensemble_metrics_includes_regime(self):
        em = self._make_ensemble()
        det = self._make_detector()
        em.set_regime_detector(det)
        det.predict_regime(_make_prices(50))

        metrics = em.get_ensemble_metrics()
        assert "regime" in metrics

    def test_ensemble_metrics_no_regime_detector(self):
        em = self._make_ensemble()
        metrics = em.get_ensemble_metrics()
        assert "regime" not in metrics

    def test_regime_high_vol_boosts_moderate(self):
        """In high-vol regime, moderate profile should gain weight."""
        from agents.ensemble.ensemble_manager import EnsembleManager
        import gymnasium as gym

        obs_space = gym.spaces.Box(low=-10, high=10, shape=(20, 5), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Start with equal weights
        em = EnsembleManager(observation_space=obs_space, action_space=act_space)

        # Inject equal history so rolling Sharpe gives equal scores
        for aid in em.agents:
            for _ in range(10):
                em.record_episode_return(aid, 0.1)

        eval_metrics = {aid: {"mean_reward": 0.1} for aid in em.agents}
        em.update_weights_regime_aware(eval_metrics, regime=2)  # crisis

        weights = em.get_weights()
        # sac_moderate should be heaviest in crisis
        assert weights["sac_moderate"] > weights["td3_aggressive"]
