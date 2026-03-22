"""
Week 25 tests: Extended Technical Indicators + Cross-Asset Correlation + SHAP Analysis.

Coverage:
  25.1  Extended FeatureConfig / FeatureEngineer (10 new indicators)
        - FEATURE_COLS, EXTENDED_FEATURE_COLS, ALL_FEATURE_COLS constants
        - FeatureConfig defaults (backward compat) & with_extended() factory
        - Each new indicator column: shape, finite, range [-1, 1]
        - compute_features adds all enabled columns without NaN
        - get_feature_matrix shape with extended config
        - n_features() counts correctly

  25.2  CrossAssetFeatureEngineer
        - Config feature_names() list
        - compute_features: correlation/beta/relstr columns added
        - All values finite, correlation ∈ [-1, 1]
        - VIX fear gauge column present and bounded
        - make_cross_asset_config convenience factory
        - n_features / get_feature_matrix shape
        - Alignment: different-length aux series

  25.3  SHAPAnalyzer
        - SHAPResult.ranking() order (highest |shap| first)
        - SHAPResult.importance_dict() keys match feature_names
        - SHAPResult.top_k() length
        - SHAPAnalyzer.explain() shape matches input
        - from_sb3_policy factory (mock policy)
        - compute_feature_importance convenience function
        - reset_explainer clears cache
        - explain_from_obs_buffer 3-D input flattening
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with $ prefix column names."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    close = np.maximum(close, 1.0)
    spread = rng.uniform(0.1, 2.0, n)
    high = close + spread
    low = np.maximum(close - spread, 0.5)
    open_ = close + rng.standard_normal(n) * 0.3
    open_ = np.maximum(open_, 0.5)
    volume = rng.uniform(1e4, 1e6, n)
    return pd.DataFrame({
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
    })


def _make_close_series(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 50.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.Series(np.maximum(prices, 1.0))


# ===========================================================================
# 25.1  Extended FeatureConfig / FeatureEngineer
# ===========================================================================

class TestExtendedFeatureCols:
    def test_feature_cols_unchanged(self):
        from training.data.feature_engineering import FEATURE_COLS
        assert FEATURE_COLS == ["rsi", "macd", "bb_width", "atr", "obv", "vwap_dev"]

    def test_extended_feature_cols_length(self):
        from training.data.feature_engineering import EXTENDED_FEATURE_COLS
        assert len(EXTENDED_FEATURE_COLS) == 10

    def test_extended_feature_cols_names(self):
        from training.data.feature_engineering import EXTENDED_FEATURE_COLS
        expected = {
            "adx", "stoch_k", "stoch_d", "cci", "williams_r",
            "mfi", "cmf", "aroon", "ema_ratio", "keltner",
        }
        assert set(EXTENDED_FEATURE_COLS) == expected

    def test_all_feature_cols_length(self):
        from training.data.feature_engineering import ALL_FEATURE_COLS
        assert len(ALL_FEATURE_COLS) == 16

    def test_all_feature_cols_is_union(self):
        from training.data.feature_engineering import (
            ALL_FEATURE_COLS, FEATURE_COLS, EXTENDED_FEATURE_COLS,
        )
        assert ALL_FEATURE_COLS == FEATURE_COLS + EXTENDED_FEATURE_COLS


class TestFeatureConfigBackwardCompat:
    def test_default_enabled_features_unchanged(self):
        from training.data.feature_engineering import FeatureConfig, FEATURE_COLS
        cfg = FeatureConfig()
        assert cfg.enabled_features == FEATURE_COLS

    def test_default_extended_flags_off(self):
        from training.data.feature_engineering import FeatureConfig
        cfg = FeatureConfig()
        for attr in [
            "use_adx", "use_stochastic", "use_cci", "use_williams_r",
            "use_mfi", "use_cmf", "use_aroon", "use_ema_ratio", "use_keltner",
        ]:
            assert getattr(cfg, attr) is False, f"{attr} should default to False"

    def test_with_extended_enables_all(self):
        from training.data.feature_engineering import FeatureConfig, ALL_FEATURE_COLS
        cfg = FeatureConfig.with_extended()
        assert cfg.use_adx is True
        assert cfg.use_stochastic is True
        assert cfg.use_cci is True
        assert cfg.use_williams_r is True
        assert cfg.use_mfi is True
        assert cfg.use_cmf is True
        assert cfg.use_aroon is True
        assert cfg.use_ema_ratio is True
        assert cfg.use_keltner is True
        assert cfg.enabled_features == ALL_FEATURE_COLS

    def test_n_features_default(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        fe = FeatureEngineer(FeatureConfig())
        assert fe.n_features() == 6

    def test_n_features_extended(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        fe = FeatureEngineer(FeatureConfig.with_extended())
        assert fe.n_features() == 16


class TestExtendedIndicators:
    """Each new indicator: column present, all-finite, values in (-1, 1)."""

    @pytest.fixture(scope="class")
    def df_ext(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig.with_extended()
        fe = FeatureEngineer(cfg)
        return fe.compute_features(_make_ohlcv(300))

    def _check_col(self, df, col):
        assert col in df.columns, f"Column '{col}' not found"
        vals = df[col].values
        assert np.all(np.isfinite(vals)), f"Non-finite values in '{col}'"
        assert vals.min() >= -1.0 - 1e-6, f"'{col}' min {vals.min()} < -1"
        assert vals.max() <= 1.0 + 1e-6, f"'{col}' max {vals.max()} > 1"

    def test_adx(self, df_ext):
        self._check_col(df_ext, "adx")

    def test_stoch_k(self, df_ext):
        self._check_col(df_ext, "stoch_k")

    def test_stoch_d(self, df_ext):
        self._check_col(df_ext, "stoch_d")

    def test_cci(self, df_ext):
        self._check_col(df_ext, "cci")

    def test_williams_r(self, df_ext):
        self._check_col(df_ext, "williams_r")

    def test_mfi(self, df_ext):
        self._check_col(df_ext, "mfi")

    def test_cmf(self, df_ext):
        self._check_col(df_ext, "cmf")

    def test_aroon(self, df_ext):
        self._check_col(df_ext, "aroon")

    def test_ema_ratio(self, df_ext):
        self._check_col(df_ext, "ema_ratio")

    def test_keltner(self, df_ext):
        self._check_col(df_ext, "keltner")

    def test_original_cols_still_present(self, df_ext):
        from training.data.feature_engineering import FEATURE_COLS
        for col in FEATURE_COLS:
            assert col in df_ext.columns

    def test_no_nan_after_compute(self, df_ext):
        from training.data.feature_engineering import ALL_FEATURE_COLS
        for col in ALL_FEATURE_COLS:
            if col in df_ext.columns:
                assert not df_ext[col].isna().any(), f"NaN found in '{col}'"

    def test_feature_matrix_shape(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig.with_extended()
        fe = FeatureEngineer(cfg)
        df = _make_ohlcv(200)
        df_feat = fe.compute_features(df)
        mat = fe.get_feature_matrix(df_feat)
        assert mat.shape == (200, 16)
        assert mat.dtype == np.float32


class TestSingleIndicatorToggle:
    """Test enabling individual extended indicators."""

    def _run(self, flag: str, expected_col: str):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(**{flag: True, "enabled_features": [expected_col]})
        fe = FeatureEngineer(cfg)
        df = fe.compute_features(_make_ohlcv(200))
        assert expected_col in df.columns
        assert np.all(np.isfinite(df[expected_col].values))

    def test_adx_only(self):
        self._run("use_adx", "adx")

    def test_stoch_only(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_stochastic=True, enabled_features=["stoch_k", "stoch_d"])
        fe = FeatureEngineer(cfg)
        df = fe.compute_features(_make_ohlcv(200))
        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns

    def test_cci_only(self):
        self._run("use_cci", "cci")

    def test_williams_r_only(self):
        self._run("use_williams_r", "williams_r")

    def test_mfi_only(self):
        self._run("use_mfi", "mfi")

    def test_cmf_only(self):
        self._run("use_cmf", "cmf")

    def test_aroon_only(self):
        self._run("use_aroon", "aroon")

    def test_ema_ratio_only(self):
        self._run("use_ema_ratio", "ema_ratio")

    def test_keltner_only(self):
        self._run("use_keltner", "keltner")


# ===========================================================================
# 25.2  CrossAssetFeatureEngineer
# ===========================================================================

class TestCrossAssetConfig:
    def test_feature_names_corr_beta_relstr(self):
        from training.data.cross_asset_features import CrossAssetConfig
        spy_df = pd.DataFrame({"$close": _make_close_series(300)})
        cfg = CrossAssetConfig(aux_assets={"spy": spy_df})
        names = cfg.feature_names()
        assert "spy_corr" in names
        assert "spy_beta" in names
        assert "spy_relstr" in names

    def test_feature_names_with_vix(self):
        from training.data.cross_asset_features import CrossAssetConfig
        spy_df = pd.DataFrame({"$close": _make_close_series(300)})
        cfg = CrossAssetConfig(aux_assets={"spy": spy_df}, vix_asset="spy")
        names = cfg.feature_names()
        assert "vix_norm" in names

    def test_feature_names_empty(self):
        from training.data.cross_asset_features import CrossAssetConfig
        cfg = CrossAssetConfig()
        assert cfg.feature_names() == []

    def test_feature_names_multiple_assets(self):
        from training.data.cross_asset_features import CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={
                "spy": pd.DataFrame({"$close": _make_close_series(100)}),
                "btc": pd.DataFrame({"$close": _make_close_series(100, seed=1)}),
            }
        )
        names = cfg.feature_names()
        assert len(names) == 6  # 3 features × 2 assets


class TestCrossAssetFeatureEngineer:
    @pytest.fixture(scope="class")
    def primary_df(self):
        return _make_ohlcv(300)

    @pytest.fixture(scope="class")
    def spy_series(self):
        return _make_close_series(300, seed=10)

    @pytest.fixture(scope="class")
    def vix_series(self):
        # VIX-like: oscillates 10-40
        rng = np.random.default_rng(99)
        return pd.Series(15.0 + rng.standard_normal(300) * 5 + 5)

    def test_compute_features_adds_cols(self, primary_df, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"spy": pd.DataFrame({"$close": spy_series})},
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        assert "spy_corr" in out.columns
        assert "spy_beta" in out.columns
        assert "spy_relstr" in out.columns

    def test_correlation_bounded(self, primary_df, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"spy": pd.DataFrame({"$close": spy_series})},
            min_periods=5,
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        corr = out["spy_corr"].values
        assert np.all(np.isfinite(corr))
        assert corr.min() >= -1.0 - 1e-6
        assert corr.max() <= 1.0 + 1e-6

    def test_beta_finite(self, primary_df, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"spy": pd.DataFrame({"$close": spy_series})},
            min_periods=5,
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        beta = out["spy_beta"].values
        assert np.all(np.isfinite(beta))

    def test_relstr_bounded(self, primary_df, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"spy": pd.DataFrame({"$close": spy_series})},
            min_periods=5,
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        rs = out["spy_relstr"].values
        assert np.all(np.isfinite(rs))
        assert rs.min() >= -1.0 - 1e-6
        assert rs.max() <= 1.0 + 1e-6

    def test_vix_column_present(self, primary_df, vix_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"vix": pd.DataFrame({"$close": vix_series})},
            vix_asset="vix",
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        assert "vix_norm" in out.columns
        vix_vals = out["vix_norm"].values
        assert np.all(np.isfinite(vix_vals))
        assert vix_vals.min() >= -1.0 - 1e-6
        assert vix_vals.max() <= 1.0 + 1e-6

    def test_multiple_assets(self, primary_df, spy_series, vix_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={
                "spy": pd.DataFrame({"$close": spy_series}),
                "vix": pd.DataFrame({"$close": vix_series}),
            },
            vix_asset="vix",
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        for col in ["spy_corr", "spy_beta", "spy_relstr", "vix_corr", "vix_norm"]:
            assert col in out.columns

    def test_no_nan_after_compute(self, primary_df, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"spy": pd.DataFrame({"$close": spy_series})},
            min_periods=5,
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        for col in ["spy_corr", "spy_beta", "spy_relstr"]:
            assert not out[col].isna().any(), f"NaN in {col}"

    def test_feature_matrix_shape(self, primary_df, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"spy": pd.DataFrame({"$close": spy_series})},
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)
        mat = ce.get_feature_matrix(out)
        assert mat.shape == (300, 3)  # corr, beta, relstr
        assert mat.dtype == np.float32

    def test_n_features(self, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={
                "spy": pd.DataFrame({"$close": spy_series}),
            },
            vix_asset=None,
        )
        ce = CrossAssetFeatureEngineer(cfg)
        assert ce.n_features() == 3

    def test_n_features_with_vix(self, spy_series):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"vix": pd.DataFrame({"$close": spy_series})},
            vix_asset="vix",
        )
        ce = CrossAssetFeatureEngineer(cfg)
        assert ce.n_features() == 4  # corr + beta + relstr + vix_norm

    def test_missing_close_col_skipped(self, primary_df):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        cfg = CrossAssetConfig(
            aux_assets={"bad": pd.DataFrame({"price": [1, 2, 3]})},
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary_df)  # should not raise
        assert "bad_corr" not in out.columns

    def test_primary_missing_close_raises(self):
        from training.data.cross_asset_features import CrossAssetFeatureEngineer
        ce = CrossAssetFeatureEngineer()
        with pytest.raises(ValueError, match="\$close"):
            ce.compute_features(pd.DataFrame({"price": [1, 2, 3]}))


class TestMakeCrossAssetConfig:
    def test_factory_creates_config(self):
        from training.data.cross_asset_features import make_cross_asset_config
        spy = _make_close_series(100)
        cfg = make_cross_asset_config({"spy": spy})
        assert "spy" in cfg.aux_assets
        assert "$close" in cfg.aux_assets["spy"].columns

    def test_factory_vix_name(self):
        from training.data.cross_asset_features import make_cross_asset_config
        vix = _make_close_series(100)
        cfg = make_cross_asset_config({"vix": vix}, vix_name="vix")
        assert cfg.vix_asset == "vix"

    def test_factory_default_windows(self):
        from training.data.cross_asset_features import make_cross_asset_config
        cfg = make_cross_asset_config({})
        assert cfg.correlation_window == 60
        assert cfg.beta_window == 60
        assert cfg.relstr_window == 20


class TestAlignSeries:
    def test_align_shorter_aux(self):
        """Aux series shorter than primary should be padded with NaN → ffill → 0."""
        from training.data.cross_asset_features import CrossAssetFeatureEngineer, CrossAssetConfig
        primary = _make_ohlcv(200)
        short_aux = _make_close_series(100)
        cfg = CrossAssetConfig(
            aux_assets={"short": pd.DataFrame({"$close": short_aux})},
            min_periods=5,
        )
        ce = CrossAssetFeatureEngineer(cfg)
        out = ce.compute_features(primary)
        assert len(out) == 200
        assert not out["short_corr"].isna().any()


# ===========================================================================
# 25.3  SHAPAnalyzer
# ===========================================================================

def _make_tabular_data(n: int = 200, n_features: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_features)).astype(np.float32)


def _simple_model_fn(obs: np.ndarray) -> np.ndarray:
    """Weighted sum: feature 0 has weight 3, feature 1 has weight 1."""
    return (obs[:, 0] * 3.0 + obs[:, 1] * 1.0).astype(np.float32)


class TestSHAPResult:
    @pytest.fixture(scope="class")
    def result(self):
        from training.analysis.shap_analysis import SHAPAnalyzer, SHAPConfig
        n_feat = 6
        feat_names = [f"feat_{i}" for i in range(n_feat)]
        bg = _make_tabular_data(50, n_feat, seed=0)
        test = _make_tabular_data(20, n_feat, seed=1)
        cfg = SHAPConfig(n_background=20, n_explain=20)
        analyzer = SHAPAnalyzer(
            model_fn=_simple_model_fn,
            feature_names=feat_names,
            background_data=bg,
            config=cfg,
        )
        return analyzer.explain(test)

    def test_shap_values_shape(self, result):
        assert result.shap_values.shape[1] == 6

    def test_shap_values_finite(self, result):
        assert np.all(np.isfinite(result.shap_values))

    def test_ranking_length(self, result):
        ranking = result.ranking()
        assert len(ranking) == 6

    def test_ranking_sorted_descending(self, result):
        ranking = result.ranking()
        scores = [v for _, v in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_feat0_most_important(self, result):
        """feat_0 has weight 3 → should have highest mean |shap|."""
        ranking = result.ranking()
        top_name, _ = ranking[0]
        assert top_name == "feat_0"

    def test_importance_dict_keys(self, result):
        d = result.importance_dict()
        for i in range(6):
            assert f"feat_{i}" in d

    def test_top_k(self, result):
        top3 = result.top_k(3)
        assert len(top3) == 3

    def test_top_k_exceeds_n_features(self, result):
        top20 = result.top_k(20)
        assert len(top20) == 6  # capped at n_features

    def test_explainer_type(self, result):
        assert result.explainer_type == "kernel"


class TestSHAPAnalyzer:
    def _make_analyzer(self, n_feat=8, n_bg=30):
        from training.analysis.shap_analysis import SHAPAnalyzer, SHAPConfig
        feat_names = [f"f{i}" for i in range(n_feat)]
        bg = _make_tabular_data(n_bg, n_feat, seed=0)
        cfg = SHAPConfig(n_background=n_bg, n_explain=30)
        return SHAPAnalyzer(
            model_fn=_simple_model_fn,
            feature_names=feat_names,
            background_data=bg,
            config=cfg,
        )

    def test_explain_shape(self):
        analyzer = self._make_analyzer(n_feat=6)
        test = _make_tabular_data(20, 6, seed=2)
        result = analyzer.explain(test)
        assert result.shap_values.shape == (20, 6)

    def test_explain_n_explain_limit(self):
        """n_explain should cap the number of samples explained."""
        from training.analysis.shap_analysis import SHAPAnalyzer, SHAPConfig
        cfg = SHAPConfig(n_background=20, n_explain=10)
        bg = _make_tabular_data(20, 4, seed=0)
        analyzer = SHAPAnalyzer(
            model_fn=_simple_model_fn,
            feature_names=["a", "b", "c", "d"],
            background_data=bg,
            config=cfg,
        )
        test = _make_tabular_data(50, 4, seed=5)
        result = analyzer.explain(test)
        assert result.shap_values.shape[0] <= 10

    def test_explainer_cached(self):
        analyzer = self._make_analyzer()
        test = _make_tabular_data(10, 8, seed=3)
        analyzer.explain(test)
        explainer_1 = analyzer._explainer
        analyzer.explain(test)
        explainer_2 = analyzer._explainer
        assert explainer_1 is explainer_2

    def test_reset_explainer(self):
        analyzer = self._make_analyzer()
        test = _make_tabular_data(10, 8, seed=4)
        analyzer.explain(test)
        assert analyzer._explainer is not None
        analyzer.reset_explainer()
        assert analyzer._explainer is None

    def test_explain_from_obs_buffer_3d(self):
        """3-D obs buffer should be flattened and explained.

        Background data must be provided already flattened (T, W*n_feat)
        so it matches the flat model input.
        """
        from training.analysis.shap_analysis import SHAPAnalyzer, SHAPConfig
        n_feat = 4
        W = 3  # window
        # Background already flat: shape (30, W*n_feat)
        rng = np.random.default_rng(0)
        bg_flat = rng.standard_normal((30, W * n_feat)).astype(np.float32)

        def flat_model(obs):
            # Accepts (n, W*n_feat) flat input
            return obs.mean(axis=1)

        cfg = SHAPConfig(n_background=20, n_explain=15)
        analyzer = SHAPAnalyzer(
            model_fn=flat_model,
            feature_names=[f"f{i}" for i in range(n_feat)],
            background_data=bg_flat,
            config=cfg,
        )
        obs_3d = np.random.randn(20, W, n_feat).astype(np.float32)
        result = analyzer.explain_from_obs_buffer(obs_3d, flatten=True)
        assert result.shap_values.shape == (min(15, 20), W * n_feat)
        assert len(result.feature_names) == W * n_feat
        assert "f0_t0" in result.feature_names

    def test_from_sb3_policy_factory(self):
        """from_sb3_policy with a mock policy should produce valid results."""
        from training.analysis.shap_analysis import SHAPAnalyzer, SHAPConfig

        mock_policy = MagicMock()
        mock_policy.predict.return_value = (np.array([[0.5]]), None)

        feat_names = [f"obs_{i}" for i in range(5)]
        bg = _make_tabular_data(30, 5, seed=0)
        cfg = SHAPConfig(n_background=20, n_explain=10)

        analyzer = SHAPAnalyzer.from_sb3_policy(
            policy=mock_policy,
            feature_names=feat_names,
            background_data=bg,
            config=cfg,
            use_value_fn=False,
        )
        test = _make_tabular_data(10, 5, seed=9)
        result = analyzer.explain(test)
        assert result.shap_values.shape[1] == 5
        assert result.explainer_type == "kernel"

    def test_feature_names_preserved(self):
        analyzer = self._make_analyzer(n_feat=4)
        test = _make_tabular_data(10, 4, seed=6)
        result = analyzer.explain(test)
        assert result.feature_names == ["f0", "f1", "f2", "f3"]


class TestComputeFeatureImportance:
    def test_returns_sorted_list(self):
        from training.analysis.shap_analysis import compute_feature_importance
        n_feat = 6
        bg = _make_tabular_data(40, n_feat, seed=0)
        test = _make_tabular_data(30, n_feat, seed=1)
        names = [f"feat_{i}" for i in range(n_feat)]

        ranking = compute_feature_importance(
            model_fn=_simple_model_fn,
            feature_names=names,
            background_data=bg,
            explain_data=test,
            n_background=20,
            n_explain=20,
        )
        assert len(ranking) == n_feat
        # Sorted descending
        scores = [v for _, v in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_feat0_ranks_first(self):
        from training.analysis.shap_analysis import compute_feature_importance
        n_feat = 6
        bg = _make_tabular_data(50, n_feat, seed=0)
        test = _make_tabular_data(40, n_feat, seed=2)
        names = [f"feat_{i}" for i in range(n_feat)]

        ranking = compute_feature_importance(
            model_fn=_simple_model_fn,
            feature_names=names,
            background_data=bg,
            explain_data=test,
            n_background=30,
            n_explain=30,
        )
        top_name, _ = ranking[0]
        assert top_name == "feat_0"


# ===========================================================================
# Integration: extended features + cross-asset + SHAP pipeline
# ===========================================================================

class TestWeek25Integration:
    """End-to-end: compute 16 indicators + cross-asset → flat obs → SHAP ranking."""

    def test_full_pipeline(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        from training.data.cross_asset_features import (
            CrossAssetFeatureEngineer, CrossAssetConfig,
        )
        from training.analysis.shap_analysis import SHAPAnalyzer, SHAPConfig

        n = 300
        primary = _make_ohlcv(n)
        spy = _make_close_series(n, seed=10)
        vix = pd.Series(15.0 + np.random.default_rng(77).standard_normal(n) * 4)

        # Step 1: compute 16 technical indicators
        fe_cfg = FeatureConfig.with_extended()
        fe = FeatureEngineer(fe_cfg)
        df_feat = fe.compute_features(primary)
        feat_mat = fe.get_feature_matrix(df_feat)  # (300, 16)
        assert feat_mat.shape == (n, 16)

        # Step 2: compute cross-asset features (spy + vix)
        cross_cfg = CrossAssetConfig(
            aux_assets={
                "spy": pd.DataFrame({"$close": spy}),
                "vix": pd.DataFrame({"$close": vix}),
            },
            vix_asset="vix",
            min_periods=5,
        )
        ce = CrossAssetFeatureEngineer(cross_cfg)
        df_cross = ce.compute_features(primary)
        cross_mat = ce.get_feature_matrix(df_cross)  # (300, 7): spy×3 + vix×3 + vix_norm
        assert cross_mat.shape == (n, 7)

        # Step 3: concatenate into full observation matrix
        obs = np.concatenate([feat_mat, cross_mat], axis=1)  # (300, 23)
        assert obs.shape == (n, 23)
        assert np.all(np.isfinite(obs))

        # Step 4: SHAP analysis with a simple mock model
        feature_names = fe_cfg.enabled_features + ce.config.feature_names()
        assert len(feature_names) == 23

        def mock_model(x: np.ndarray) -> np.ndarray:
            # Weighted sum: rsi most important
            return (x[:, 0] * 2.0 + x[:, 1] * 0.5).astype(np.float32)

        cfg = SHAPConfig(n_background=30, n_explain=30)
        analyzer = SHAPAnalyzer(
            model_fn=mock_model,
            feature_names=feature_names,
            background_data=obs[:100],
            config=cfg,
        )
        result = analyzer.explain(obs[100:200])
        ranking = result.ranking()

        assert len(ranking) == 23
        # rsi (index 0) should rank first since weight=2.0 is highest
        assert ranking[0][0] == "rsi"
        # All SHAP values finite
        assert np.all(np.isfinite(result.shap_values))
