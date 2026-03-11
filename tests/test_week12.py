"""
Week 12 Tests – Data Pipeline Enhancement.

Coverage targets:
- training/data/feature_engineering.py  (FeatureConfig, FeatureEngineer)
- training/env_factory.py additions     (validate_data, split_data, create_env with validate/features)
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with $ prefix column names."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    close = np.maximum(close, 1.0)  # keep positive
    spread = rng.uniform(0.1, 2.0, n)
    high = close + spread
    low = close - spread
    open_ = close + rng.standard_normal(n) * 0.5
    open_ = np.maximum(open_, 0.5)
    volume = rng.uniform(1e4, 1e6, n)
    return pd.DataFrame({
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
    })


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------

class TestFeatureConfig:
    def test_defaults(self):
        from training.data.feature_engineering import FeatureConfig, FEATURE_COLS
        cfg = FeatureConfig()
        assert cfg.use_rsi is True
        assert cfg.use_macd is True
        assert cfg.use_bollinger is True
        assert cfg.use_atr is True
        assert cfg.use_obv is True
        assert cfg.use_vwap is True
        assert cfg.rsi_period == 14
        assert cfg.macd_fast == 12
        assert cfg.macd_slow == 26
        assert cfg.macd_signal == 9
        assert cfg.bb_period == 20
        assert cfg.atr_period == 14
        assert cfg.enabled_features == FEATURE_COLS

    def test_custom_params(self):
        from training.data.feature_engineering import FeatureConfig
        cfg = FeatureConfig(rsi_period=21, macd_fast=8, macd_slow=21)
        assert cfg.rsi_period == 21
        assert cfg.macd_fast == 8
        assert cfg.macd_slow == 21

    def test_partial_disable(self):
        from training.data.feature_engineering import FeatureConfig
        cfg = FeatureConfig(use_obv=False, use_vwap=False)
        assert cfg.use_obv is False
        assert cfg.use_vwap is False
        assert cfg.use_rsi is True

    def test_enabled_features_custom(self):
        from training.data.feature_engineering import FeatureConfig
        cfg = FeatureConfig(enabled_features=["rsi", "atr"])
        assert cfg.enabled_features == ["rsi", "atr"]


# ---------------------------------------------------------------------------
# FeatureEngineer – compute_features
# ---------------------------------------------------------------------------

class TestFeatureEngineerCompute:
    def test_returns_copy(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        assert out is not df  # copy

    def test_original_columns_preserved(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        for col in ["$open", "$high", "$low", "$close", "$volume"]:
            assert col in out.columns

    def test_feature_columns_present(self):
        from training.data.feature_engineering import FeatureEngineer, FEATURE_COLS
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        for col in FEATURE_COLS:
            assert col in out.columns, f"Missing feature column: {col}"

    def test_no_nan_in_output(self):
        from training.data.feature_engineering import FeatureEngineer, FEATURE_COLS
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        for col in FEATURE_COLS:
            assert not out[col].isna().any(), f"NaN found in {col}"

    def test_range_minus1_to_1(self):
        from training.data.feature_engineering import FeatureEngineer, FEATURE_COLS
        df = _make_ohlcv(300)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        for col in FEATURE_COLS:
            assert out[col].min() >= -1.0 - 1e-6, f"{col} below -1"
            assert out[col].max() <= 1.0 + 1e-6, f"{col} above 1"

    def test_missing_ohlcv_raises(self):
        from training.data.feature_engineering import FeatureEngineer
        bad = pd.DataFrame({"$close": [1, 2, 3]})
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="missing required columns"):
            fe.compute_features(bad)

    def test_disable_rsi_no_rsi_col(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_rsi=False)
        fe = FeatureEngineer(cfg)
        out = fe.compute_features(_make_ohlcv(200))
        assert "rsi" not in out.columns

    def test_disable_macd_no_macd_col(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_macd=False)
        fe = FeatureEngineer(cfg)
        out = fe.compute_features(_make_ohlcv(200))
        assert "macd" not in out.columns

    def test_disable_bollinger_no_bb_col(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_bollinger=False)
        fe = FeatureEngineer(cfg)
        out = fe.compute_features(_make_ohlcv(200))
        assert "bb_width" not in out.columns

    def test_disable_atr_no_atr_col(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_atr=False)
        fe = FeatureEngineer(cfg)
        out = fe.compute_features(_make_ohlcv(200))
        assert "atr" not in out.columns

    def test_disable_obv_no_obv_col(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_obv=False)
        fe = FeatureEngineer(cfg)
        out = fe.compute_features(_make_ohlcv(200))
        assert "obv" not in out.columns

    def test_disable_vwap_no_vwap_col(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(use_vwap=False)
        fe = FeatureEngineer(cfg)
        out = fe.compute_features(_make_ohlcv(200))
        assert "vwap_dev" not in out.columns

    def test_row_count_preserved(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(150)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        assert len(out) == len(df)

    def test_small_dataset(self):
        """Should work on short datasets (>= indicator window)."""
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(30)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        assert len(out) == 30
        assert not out["rsi"].isna().any()

    def test_rsi_values_are_tanh_of_centred(self):
        """RSI values should lie strictly in (-1, 1)."""
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(300)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        assert (out["rsi"] > -1.0).all()
        assert (out["rsi"] < 1.0).all()

    def test_atr_nonnegative(self):
        """ATR is always non-negative (before tanh), so tanh(atr*10) >= 0."""
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        assert (out["atr"] >= 0.0).all()

    def test_deterministic(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out1 = fe.compute_features(df)
        out2 = fe.compute_features(df)
        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# FeatureEngineer – get_feature_matrix / n_features
# ---------------------------------------------------------------------------

class TestFeatureMatrix:
    def test_shape(self):
        from training.data.feature_engineering import FeatureEngineer, FEATURE_COLS
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        mat = fe.get_feature_matrix(out)
        assert mat.shape == (200, len(FEATURE_COLS))

    def test_dtype_float32(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        mat = fe.get_feature_matrix(out)
        assert mat.dtype == np.float32

    def test_no_nan(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        mat = fe.get_feature_matrix(out)
        assert not np.isnan(mat).any()

    def test_n_features_default(self):
        from training.data.feature_engineering import FeatureEngineer, FEATURE_COLS
        fe = FeatureEngineer()
        assert fe.n_features() == len(FEATURE_COLS)

    def test_n_features_custom(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(enabled_features=["rsi", "atr"])
        fe = FeatureEngineer(cfg)
        assert fe.n_features() == 2

    def test_subset_feature_matrix(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(enabled_features=["rsi", "atr"])
        fe = FeatureEngineer(cfg)
        df = _make_ohlcv(200)
        out = fe.compute_features(df)
        mat = fe.get_feature_matrix(out)
        assert mat.shape == (200, 2)

    def test_empty_features_returns_zeros(self):
        from training.data.feature_engineering import FeatureEngineer, FeatureConfig
        cfg = FeatureConfig(enabled_features=[])
        fe = FeatureEngineer(cfg)
        df = _make_ohlcv(200)
        out = fe.compute_features(df)
        mat = fe.get_feature_matrix(out)
        assert mat.shape == (200, 0)

    def test_range_minus1_to_1(self):
        from training.data.feature_engineering import FeatureEngineer
        df = _make_ohlcv(300)
        fe = FeatureEngineer()
        out = fe.compute_features(df)
        mat = fe.get_feature_matrix(out)
        assert mat.min() >= -1.0 - 1e-6
        assert mat.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# DataValidationResult + validate_data
# ---------------------------------------------------------------------------

class TestDataValidationResult:
    def test_summary_valid(self):
        from training.env_factory import DataValidationResult
        r = DataValidationResult(is_valid=True)
        assert "valid=True" in r.summary()

    def test_summary_errors(self):
        from training.env_factory import DataValidationResult
        r = DataValidationResult(is_valid=False, errors=["err1", "err2"])
        s = r.summary()
        assert "ERRORS" in s
        assert "err1" in s

    def test_summary_warnings(self):
        from training.env_factory import DataValidationResult
        r = DataValidationResult(is_valid=True, warnings=["warn1"])
        assert "WARNINGS" in r.summary()


class TestValidateData:
    def test_valid_df(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        result = validate_data(df)
        assert result.is_valid
        assert result.errors == []

    def test_too_few_rows(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(10)
        result = validate_data(df, min_rows=50)
        assert not result.is_valid
        assert any("Too few rows" in e for e in result.errors)

    def test_nan_in_close(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[10, "$close"] = float("nan")
        result = validate_data(df)
        assert not result.is_valid
        assert any("$close" in e for e in result.errors)

    def test_nan_in_volume(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[5, "$volume"] = float("nan")
        result = validate_data(df)
        assert not result.is_valid

    def test_negative_price(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[0, "$close"] = -5.0
        result = validate_data(df)
        assert not result.is_valid
        assert any("$close" in e for e in result.errors)

    def test_zero_price(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[0, "$open"] = 0.0
        result = validate_data(df)
        assert not result.is_valid

    def test_high_less_than_low(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[5, "$high"] = df.loc[5, "$low"] - 1.0
        result = validate_data(df)
        assert not result.is_valid
        assert any("High < Low" in e for e in result.errors)

    def test_zero_volume_is_warning_not_error(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[3, "$volume"] = 0.0
        result = validate_data(df)
        # zero volume → warning, not error (df is otherwise valid)
        assert result.is_valid
        assert any("Zero volume" in w for w in result.warnings)

    def test_missing_column_is_error(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200).drop(columns=["$volume"])
        result = validate_data(df)
        assert not result.is_valid
        assert any("Missing columns" in e for e in result.errors)

    def test_stats_populated(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        result = validate_data(df)
        assert "n_rows" in result.stats
        assert result.stats["n_rows"] == 200
        assert "price_stats" in result.stats
        assert "nan_counts" in result.stats

    def test_multiple_errors_all_reported(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(200)
        df.loc[0, "$close"] = float("nan")
        df.loc[1, "$open"] = -1.0
        result = validate_data(df)
        assert len(result.errors) >= 2

    def test_min_rows_exact(self):
        from training.env_factory import validate_data
        df = _make_ohlcv(50)
        result = validate_data(df, min_rows=50)
        assert result.is_valid


# ---------------------------------------------------------------------------
# log_data_quality_report
# ---------------------------------------------------------------------------

class TestLogDataQualityReport:
    def test_calls_mlflow_on_valid(self):
        from training.env_factory import validate_data, log_data_quality_report
        df = _make_ohlcv(200)
        result = validate_data(df)

        logged = {}

        class FakeMlflow:
            def log_metrics(self, d):
                logged.update(d)

        log_data_quality_report(result, mlflow_manager=FakeMlflow())
        assert "data/validation_passed" in logged
        assert logged["data/validation_passed"] == 1

    def test_calls_mlflow_on_invalid(self):
        from training.env_factory import DataValidationResult, log_data_quality_report
        result = DataValidationResult(is_valid=False, errors=["err"])

        logged = {}

        class FakeMlflow:
            def log_metrics(self, d):
                logged.update(d)

        log_data_quality_report(result, mlflow_manager=FakeMlflow())
        assert logged["data/validation_passed"] == 0
        assert logged["data/n_errors"] == 1

    def test_no_mlflow_no_error(self):
        from training.env_factory import validate_data, log_data_quality_report
        df = _make_ohlcv(200)
        result = validate_data(df)
        log_data_quality_report(result, mlflow_manager=None)  # should not raise


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_basic_split(self):
        from training.env_factory import split_data
        df = _make_ohlcv(1000)
        train, val, test = split_data(df, 0.7, 0.15, 0.15)
        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_total_rows_preserved(self):
        from training.env_factory import split_data
        df = _make_ohlcv(200)
        train, val, test = split_data(df, 0.7, 0.15, 0.15)
        assert len(train) + len(val) + len(test) == len(df)

    def test_chronological_order(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        df["idx"] = range(100)
        train, val, test = split_data(df, 0.7, 0.15, 0.15)
        assert list(train["idx"]) == list(range(0, 70))
        assert list(val["idx"]) == list(range(70, 85))
        assert list(test["idx"]) == list(range(85, 100))

    def test_index_reset(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        train, val, test = split_data(df, 0.7, 0.15, 0.15)
        assert train.index[0] == 0
        assert val.index[0] == 0
        assert test.index[0] == 0

    def test_ratios_not_summing_to_one_raises(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match="must equal 1.0"):
            split_data(df, 0.6, 0.2, 0.3)

    def test_zero_val_ratio(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        train, val, test = split_data(df, 0.8, 0.0, 0.2)
        assert len(val) == 0
        assert len(train) == 80
        assert len(test) == 20

    def test_zero_test_ratio(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        train, val, test = split_data(df, 0.8, 0.2, 0.0)
        assert len(test) == 0

    def test_invalid_train_ratio_zero_raises(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        with pytest.raises(ValueError):
            split_data(df, 0.0, 0.5, 0.5)

    def test_returns_dataframes(self):
        from training.env_factory import split_data
        df = _make_ohlcv(100)
        train, val, test = split_data(df, 0.7, 0.15, 0.15)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)


# ---------------------------------------------------------------------------
# split_data_from_config
# ---------------------------------------------------------------------------

class TestSplitDataFromConfig:
    def test_reads_ratios_from_config(self):
        from training.env_factory import split_data_from_config
        df = _make_ohlcv(100)
        config = {"data": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}}
        train, val, test = split_data_from_config(df, config)
        assert len(train) == 70

    def test_defaults_when_config_missing(self):
        from training.env_factory import split_data_from_config
        df = _make_ohlcv(100)
        train, val, test = split_data_from_config(df, {})
        assert len(train) + len(val) + len(test) == 100


# ---------------------------------------------------------------------------
# create_env with validate / apply_features
# ---------------------------------------------------------------------------

class TestCreateEnvValidate:
    def _base_config(self):
        return {
            "env": {
                "type": "single_asset_rl",
                "window_size": 20,
                "initial_balance": 10000.0,
                "trading_fee": 0.001,
                "max_position_size": 1.0,
                "apply_slippage": False,
                "partial_fills": False,
            },
            "data": {"min_rows": 50},
        }

    def test_valid_data_creates_env(self):
        from training.env_factory import create_env
        df = _make_ohlcv(200)
        env = create_env(self._base_config(), data=df, validate=True)
        assert env is not None
        env.close()

    def test_invalid_data_raises(self):
        from training.env_factory import create_env
        df = _make_ohlcv(200)
        df.loc[0, "$close"] = -5.0
        with pytest.raises(ValueError, match="Data validation failed"):
            create_env(self._base_config(), data=df, validate=True)

    def test_validate_false_skips_check(self):
        from training.env_factory import create_env
        df = _make_ohlcv(200)
        df.loc[0, "$close"] = -5.0
        # Should NOT raise when validate=False
        env = create_env(self._base_config(), data=df, validate=False)
        assert env is not None
        env.close()

    def test_apply_features_adds_columns(self):
        from training.env_factory import create_env
        from training.data.feature_engineering import FEATURE_COLS
        df = _make_ohlcv(200)
        # We just test that feature computation runs without error;
        # the env ignores extra columns
        env = create_env(
            self._base_config(), data=df,
            validate=True, apply_features=True
        )
        assert env is not None
        env.close()

    def test_create_env_with_mlflow_manager(self):
        from training.env_factory import create_env
        df = _make_ohlcv(200)
        logged = {}

        class FakeMlflow:
            def log_metrics(self, d):
                logged.update(d)

        env = create_env(
            self._base_config(), data=df,
            validate=True, mlflow_manager=FakeMlflow()
        )
        assert "data/validation_passed" in logged
        env.close()


# ---------------------------------------------------------------------------
# Integration: feature_engineering + env_factory together
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_feature_engineer_output_feeds_env(self):
        """FeatureEngineer output (extra columns) should not break env creation."""
        from training.data.feature_engineering import FeatureEngineer
        from training.env_factory import validate_data, create_env

        df = _make_ohlcv(200)
        fe = FeatureEngineer()
        df_feat = fe.compute_features(df)

        result = validate_data(df_feat)
        assert result.is_valid

        config = {
            "env": {
                "type": "single_asset_rl",
                "window_size": 20,
                "initial_balance": 10000.0,
                "trading_fee": 0.001,
                "max_position_size": 1.0,
                "apply_slippage": False,
                "partial_fills": False,
            }
        }
        env = create_env(config, data=df_feat, validate=False)
        assert env is not None
        env.close()

    def test_split_then_validate_all_splits(self):
        from training.env_factory import split_data, validate_data
        df = _make_ohlcv(300)
        train, val, test = split_data(df, 0.7, 0.15, 0.15)
        for split_df in [train, val, test]:
            result = validate_data(split_df, min_rows=10)
            assert result.is_valid, result.errors

    def test_full_pipeline(self):
        """validate → feature_eng → split → create_env for train/val."""
        from training.data.feature_engineering import FeatureEngineer
        from training.env_factory import validate_data, split_data, create_env

        df = _make_ohlcv(500)
        assert validate_data(df).is_valid

        fe = FeatureEngineer()
        df_feat = fe.compute_features(df)

        train, val, _ = split_data(df_feat, 0.7, 0.15, 0.15)

        config = {
            "env": {
                "type": "single_asset_rl",
                "window_size": 20,
                "initial_balance": 10000.0,
                "trading_fee": 0.001,
                "max_position_size": 1.0,
                "apply_slippage": False,
                "partial_fills": False,
            }
        }
        train_env = create_env(config, data=train, validate=True)
        val_env = create_env(config, data=val, validate=True)

        obs, _ = train_env.reset()
        assert obs is not None
        train_env.close()
        val_env.close()
