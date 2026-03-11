"""
Week 13 tests: LLM Sentiment Signal Integration

Covers:
- SentimentConfig / SentimentFeatures dataclasses
- SentimentEngine: score_text, score_batch, compute_features, align_to_prices
- SQLite cache
- from_config factory
- SingleAssetRLTradingEnv observation space extension
- End-to-end alignment → env pipeline
"""

import os
import sqlite3
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from training.signals.sentiment_engine import (
    N_SENTIMENT_FEATURES,
    SENTIMENT_COLS,
    SentimentConfig,
    SentimentEngine,
    SentimentFeatures,
    _TRANSFORMERS_AVAILABLE,
)
from envs.single_asset_rl_env import SingleAssetRLTradingEnv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline(positive: float = 0.7, negative: float = 0.2, neutral: float = 0.1):
    """Return a mock FinBERT pipeline function with fixed scores."""
    def _pipeline(texts, **kwargs):
        return [
            [
                {"label": "positive", "score": positive},
                {"label": "negative", "score": negative},
                {"label": "neutral",  "score": neutral},
            ]
            for _ in texts
        ]
    return _pipeline


def _alternating_pipeline():
    """Pipeline that alternates positive/negative on successive calls."""
    call_idx = [0]

    def _pipeline(texts, **kwargs):
        results = []
        for _ in texts:
            if call_idx[0] % 2 == 0:
                results.append([
                    {"label": "positive", "score": 0.9},
                    {"label": "negative", "score": 0.05},
                    {"label": "neutral",  "score": 0.05},
                ])
            else:
                results.append([
                    {"label": "positive", "score": 0.05},
                    {"label": "negative", "score": 0.9},
                    {"label": "neutral",  "score": 0.05},
                ])
            call_idx[0] += 1
        return results
    return _pipeline


@pytest.fixture
def tiny_df():
    """50-row OHLCV DataFrame with integer index."""
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "$open":   rng.uniform(100, 200, n).astype(np.float32),
            "$high":   rng.uniform(200, 300, n).astype(np.float32),
            "$low":    rng.uniform(50,  100, n).astype(np.float32),
            "$close":  rng.uniform(100, 200, n).astype(np.float32),
            "$volume": rng.uniform(1e3, 1e4, n).astype(np.float32),
        }
    )


@pytest.fixture
def mock_engine():
    """SentimentEngine with mocked pipeline (pos=0.7, neg=0.2, neu=0.1)."""
    engine = SentimentEngine()
    engine._pipeline = _make_pipeline()
    return engine


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "cache.sqlite")


@pytest.fixture
def cached_engine(tmp_db):
    cfg = SentimentConfig(cache_db=tmp_db)
    engine = SentimentEngine(cfg)
    engine._pipeline = _make_pipeline()
    return engine


def _zero_sentiment(n: int) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((n, 4), dtype=np.float32), columns=SENTIMENT_COLS)


def _const_sentiment(n: int, value: float = 0.5) -> pd.DataFrame:
    data = np.full((n, 4), value, dtype=np.float32)
    return pd.DataFrame(data, columns=SENTIMENT_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SentimentConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentConfig:
    def test_default_model_name(self):
        assert SentimentConfig().model_name == "ProsusAI/finbert"

    def test_default_batch_size(self):
        assert SentimentConfig().batch_size == 32

    def test_default_max_length(self):
        assert SentimentConfig().max_length == 128

    def test_default_cache_db_is_none(self):
        assert SentimentConfig().cache_db is None

    def test_default_device(self):
        assert SentimentConfig().device == "cpu"

    def test_default_extreme_threshold(self):
        assert SentimentConfig().extreme_threshold == 0.7

    def test_custom_values(self):
        cfg = SentimentConfig(model_name="my/model", batch_size=8, max_length=64)
        assert cfg.model_name == "my/model"
        assert cfg.batch_size == 8
        assert cfg.max_length == 64

    def test_custom_cache_db(self):
        cfg = SentimentConfig(cache_db="/tmp/test.sqlite")
        assert cfg.cache_db == "/tmp/test.sqlite"

    def test_custom_device(self):
        cfg = SentimentConfig(device="cuda:0")
        assert cfg.device == "cuda:0"

    def test_custom_extreme_threshold(self):
        cfg = SentimentConfig(extreme_threshold=0.9)
        assert cfg.extreme_threshold == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 2. SentimentFeatures
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentFeatures:
    def test_creation(self):
        f = SentimentFeatures(0.5, 0.2, 0.1, -0.3)
        assert f.mean_sentiment == 0.5
        assert f.dispersion == 0.2
        assert f.extreme_count == 0.1
        assert f.momentum == -0.3

    def test_to_array_shape(self):
        arr = SentimentFeatures(0.1, 0.2, 0.3, 0.4).to_array()
        assert arr.shape == (4,)

    def test_to_array_dtype(self):
        arr = SentimentFeatures(0.1, 0.2, 0.3, 0.4).to_array()
        assert arr.dtype == np.float32

    def test_to_array_values(self):
        f = SentimentFeatures(0.1, 0.2, 0.3, 0.4)
        np.testing.assert_allclose(f.to_array(), [0.1, 0.2, 0.3, 0.4], atol=1e-6)

    def test_neutral_features_sum_zero(self):
        assert SentimentFeatures(0.0, 0.0, 0.0, 0.0).to_array().sum() == 0.0

    def test_n_sentiment_features_constant(self):
        assert N_SENTIMENT_FEATURES == 4

    def test_sentiment_cols_length(self):
        assert len(SENTIMENT_COLS) == 4

    def test_sentiment_cols_names(self):
        assert "mean_sentiment"    in SENTIMENT_COLS
        assert "dispersion"        in SENTIMENT_COLS
        assert "extreme_count"     in SENTIMENT_COLS
        assert "sentiment_momentum" in SENTIMENT_COLS


# ─────────────────────────────────────────────────────────────────────────────
# 3. SentimentEngine — initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentEngineInit:
    def test_default_config(self):
        engine = SentimentEngine()
        assert isinstance(engine.config, SentimentConfig)

    def test_custom_config(self):
        cfg = SentimentConfig(batch_size=4)
        engine = SentimentEngine(cfg)
        assert engine.config.batch_size == 4

    def test_pipeline_none_initially(self):
        assert SentimentEngine()._pipeline is None

    def test_db_conn_none_initially(self):
        assert SentimentEngine()._db_conn is None

    def test_ensure_model_raises_without_transformers(self, monkeypatch):
        monkeypatch.setattr(
            "training.signals.sentiment_engine._TRANSFORMERS_AVAILABLE", False
        )
        engine = SentimentEngine()
        with pytest.raises(ImportError, match="transformers"):
            engine._ensure_model()


# ─────────────────────────────────────────────────────────────────────────────
# 4. score_text
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreText:
    def test_returns_dict(self, mock_engine):
        assert isinstance(mock_engine.score_text("Some headline"), dict)

    def test_has_three_keys(self, mock_engine):
        result = mock_engine.score_text("Some headline")
        assert set(result.keys()) == {"positive", "negative", "neutral"}

    def test_values_are_float(self, mock_engine):
        for v in mock_engine.score_text("text").values():
            assert isinstance(v, float)

    def test_positive_score(self, mock_engine):
        assert mock_engine.score_text("Good news")["positive"] == pytest.approx(0.7)

    def test_negative_score(self, mock_engine):
        assert mock_engine.score_text("Bad news")["negative"] == pytest.approx(0.2)

    def test_neutral_score(self, mock_engine):
        assert mock_engine.score_text("Neutral news")["neutral"] == pytest.approx(0.1)

    def test_cache_writes_entry(self, cached_engine, tmp_db):
        cached_engine.score_text("Hello market")
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]
        conn.close()
        assert count == 1

    def test_cache_hit_skips_pipeline(self, cached_engine):
        cached_engine.score_text("Same text once")
        # Replace pipeline with something that would fail
        def _bad_pipe(texts, **kw):
            raise RuntimeError("pipeline should not be called on cache hit")
        cached_engine._pipeline = _bad_pipe
        # Second call should read from cache, not call pipeline
        result = cached_engine.score_text("Same text once")
        assert "positive" in result

    def test_cache_idempotent(self, cached_engine, tmp_db):
        cached_engine.score_text("Repeated text")
        cached_engine.score_text("Repeated text")
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]
        conn.close()
        assert count == 1

    def test_no_cache_db_keeps_conn_none(self, mock_engine):
        mock_engine.score_text("text")
        assert mock_engine._db_conn is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. score_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreBatch:
    def test_empty_list(self, mock_engine):
        assert mock_engine.score_batch([]) == []

    def test_single_item(self, mock_engine):
        result = mock_engine.score_batch(["text"])
        assert len(result) == 1
        assert "positive" in result[0]

    def test_multiple_items(self, mock_engine):
        result = mock_engine.score_batch(["a", "b", "c"])
        assert len(result) == 3

    def test_each_item_has_three_keys(self, mock_engine):
        for item in mock_engine.score_batch(["x", "y"]):
            assert set(item.keys()) == {"positive", "negative", "neutral"}

    def test_returns_list(self, mock_engine):
        assert isinstance(mock_engine.score_batch(["a"]), list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. compute_features
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_empty_headlines_returns_neutral(self, mock_engine):
        f = mock_engine.compute_features([])
        assert f.mean_sentiment == 0.0
        assert f.dispersion == 0.0
        assert f.extreme_count == 0.0
        assert f.momentum == 0.0

    def test_returns_sentiment_features_instance(self, mock_engine):
        assert isinstance(mock_engine.compute_features(["text"]), SentimentFeatures)

    def test_mean_sentiment_in_range(self, mock_engine):
        f = mock_engine.compute_features(["a", "b"])
        assert -1.0 <= f.mean_sentiment <= 1.0

    def test_dispersion_in_range(self, mock_engine):
        f = mock_engine.compute_features(["a", "b"])
        assert 0.0 <= f.dispersion <= 1.0

    def test_extreme_count_in_range(self, mock_engine):
        f = mock_engine.compute_features(["text"])
        assert 0.0 <= f.extreme_count <= 1.0

    def test_momentum_in_range(self, mock_engine):
        f = mock_engine.compute_features(["text"], prev_mean=0.0)
        assert -1.0 <= f.momentum <= 1.0

    def test_momentum_zero_when_no_prev(self, mock_engine):
        assert mock_engine.compute_features(["text"], prev_mean=None).momentum == 0.0

    def test_dispersion_zero_for_single(self, mock_engine):
        # Single headline: no spread possible
        assert mock_engine.compute_features(["one"]).dispersion == 0.0

    def test_dispersion_positive_for_mixed(self):
        engine = SentimentEngine()
        engine._pipeline = _alternating_pipeline()
        f = engine.compute_features(["pos headline", "neg headline"])
        assert f.dispersion > 0.0

    def test_positive_sentiment_gives_positive_mean(self):
        engine = SentimentEngine()
        engine._pipeline = _make_pipeline(positive=0.9, negative=0.05, neutral=0.05)
        assert engine.compute_features(["great news"]).mean_sentiment > 0.0

    def test_negative_sentiment_gives_negative_mean(self):
        engine = SentimentEngine()
        engine._pipeline = _make_pipeline(positive=0.05, negative=0.9, neutral=0.05)
        assert engine.compute_features(["terrible news"]).mean_sentiment < 0.0

    def test_extreme_count_above_threshold(self):
        # pos=0.95, neg=0.03 → sentiment=0.92 > threshold=0.7
        engine = SentimentEngine(SentimentConfig(extreme_threshold=0.7))
        engine._pipeline = _make_pipeline(positive=0.95, negative=0.03, neutral=0.02)
        assert engine.compute_features(["extreme"]).extreme_count == 1.0

    def test_extreme_count_below_threshold(self):
        # pos=0.6, neg=0.35 → sentiment=0.25 < threshold=0.7
        engine = SentimentEngine(SentimentConfig(extreme_threshold=0.7))
        engine._pipeline = _make_pipeline(positive=0.6, negative=0.35, neutral=0.05)
        assert engine.compute_features(["mild"]).extreme_count == 0.0

    def test_momentum_positive_when_increasing(self):
        engine = SentimentEngine()
        engine._pipeline = _make_pipeline(positive=0.9, negative=0.05)
        # prev_mean = 0.0 → current ≈ 0.85 → momentum = tanh(0.85) > 0
        f = engine.compute_features(["good"], prev_mean=0.0)
        assert f.momentum > 0.0

    def test_momentum_negative_when_decreasing(self):
        engine = SentimentEngine()
        engine._pipeline = _make_pipeline(positive=0.05, negative=0.9)
        # prev_mean = 0.5 → current ≈ -0.85 → momentum = tanh(-1.35) < 0
        f = engine.compute_features(["bad"], prev_mean=0.5)
        assert f.momentum < 0.0

    def test_multiple_headlines_mean_correct(self):
        engine = SentimentEngine()
        # All same score: pos=0.7, neg=0.2 → sentiment=0.5 per headline
        engine._pipeline = _make_pipeline(positive=0.7, negative=0.2)
        f = engine.compute_features(["a", "b", "c"])
        assert f.mean_sentiment == pytest.approx(0.5, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 7. align_to_prices
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignToPrices:
    def test_empty_news_returns_zero_columns(self, tiny_df, mock_engine):
        empty = pd.DataFrame(columns=["timestamp", "headline"])
        result = mock_engine.align_to_prices(empty, tiny_df)
        for col in SENTIMENT_COLS:
            assert col in result.columns
            assert (result[col] == 0.0).all()

    def test_none_news_returns_zero_columns(self, tiny_df, mock_engine):
        result = mock_engine.align_to_prices(None, tiny_df)
        for col in SENTIMENT_COLS:
            assert (result[col] == 0.0).all()

    def test_adds_exactly_four_sentiment_columns(self, tiny_df, mock_engine):
        empty = pd.DataFrame(columns=["timestamp", "headline"])
        result = mock_engine.align_to_prices(empty, tiny_df)
        for col in SENTIMENT_COLS:
            assert col in result.columns

    def test_original_ohlcv_columns_preserved(self, tiny_df, mock_engine):
        empty = pd.DataFrame(columns=["timestamp", "headline"])
        result = mock_engine.align_to_prices(empty, tiny_df)
        for col in ["$open", "$high", "$low", "$close", "$volume"]:
            assert col in result.columns

    def test_result_same_length_as_prices(self, tiny_df, mock_engine):
        news = pd.DataFrame({"timestamp": [5, 20], "headline": ["A", "B"]})
        result = mock_engine.align_to_prices(news, tiny_df)
        assert len(result) == len(tiny_df)

    def test_matching_step_gets_nonzero_sentiment(self, tiny_df, mock_engine):
        # mock gives pos=0.7, neg=0.2 → mean_sentiment=0.5
        news = pd.DataFrame({"timestamp": [5], "headline": ["Market moves"]})
        result = mock_engine.align_to_prices(news, tiny_df)
        assert result.loc[5, "mean_sentiment"] == pytest.approx(0.5, abs=1e-4)

    def test_forward_fill_propagates_values(self, tiny_df, mock_engine):
        news = pd.DataFrame({"timestamp": [5], "headline": ["News at 5"]})
        result = mock_engine.align_to_prices(news, tiny_df)
        # Steps 6–49 must equal step 5
        assert result.loc[10, "mean_sentiment"] == pytest.approx(
            result.loc[5, "mean_sentiment"], abs=1e-6
        )

    def test_steps_before_news_are_zero(self, tiny_df, mock_engine):
        news = pd.DataFrame({"timestamp": [30], "headline": ["Late headline"]})
        result = mock_engine.align_to_prices(news, tiny_df)
        assert result.loc[0, "mean_sentiment"] == 0.0

    def test_custom_column_names(self, tiny_df, mock_engine):
        news = pd.DataFrame({"ts": [5], "text": ["Headline"]})
        result = mock_engine.align_to_prices(
            news, tiny_df, timestamp_col="ts", headline_col="text"
        )
        assert "mean_sentiment" in result.columns

    def test_multiple_headlines_same_step(self, tiny_df, mock_engine):
        news = pd.DataFrame(
            {
                "timestamp": [5, 5, 5],
                "headline": ["A", "B", "C"],
            }
        )
        result = mock_engine.align_to_prices(news, tiny_df)
        # All three have same mock score (0.5 each) → mean = 0.5
        assert result.loc[5, "mean_sentiment"] == pytest.approx(0.5, abs=1e-4)

    def test_skip_empty_headlines(self, tiny_df, mock_engine):
        news = pd.DataFrame(
            {"timestamp": [5, 5], "headline": ["Real headline", ""]}
        )
        result = mock_engine.align_to_prices(news, tiny_df)
        # Should process only non-empty headline without error
        assert "mean_sentiment" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# 8. SQLite cache
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLiteCache:
    def test_table_created_on_first_call(self, cached_engine, tmp_db):
        cached_engine.score_text("init")
        conn = sqlite3.connect(tmp_db)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        assert "sentiment" in tables

    def test_two_different_texts_two_rows(self, cached_engine, tmp_db):
        cached_engine.score_text("text alpha")
        cached_engine.score_text("text beta")
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]
        conn.close()
        assert count == 2

    def test_same_text_one_row(self, cached_engine, tmp_db):
        cached_engine.score_text("same")
        cached_engine.score_text("same")
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]
        conn.close()
        assert count == 1

    def test_cached_values_correct(self, cached_engine, tmp_db):
        result = cached_engine.score_text("verify cached")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT positive, negative, neutral FROM sentiment"
        ).fetchone()
        conn.close()
        assert row[0] == pytest.approx(result["positive"], abs=1e-6)

    def test_cache_disabled_no_db_conn(self, mock_engine):
        mock_engine.score_text("no cache")
        assert mock_engine._db_conn is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. from_config
# ─────────────────────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_empty_config_uses_defaults(self):
        engine = SentimentEngine.from_config({})
        assert engine.config.model_name == "ProsusAI/finbert"
        assert engine.config.batch_size == 32

    def test_custom_model_name(self):
        engine = SentimentEngine.from_config({"sentiment": {"model_name": "my/bert"}})
        assert engine.config.model_name == "my/bert"

    def test_custom_batch_size(self):
        engine = SentimentEngine.from_config({"sentiment": {"batch_size": 16}})
        assert engine.config.batch_size == 16

    def test_custom_cache_db(self, tmp_db):
        engine = SentimentEngine.from_config({"sentiment": {"cache_db": tmp_db}})
        assert engine.config.cache_db == tmp_db

    def test_custom_device(self):
        engine = SentimentEngine.from_config({"sentiment": {"device": "cuda"}})
        assert engine.config.device == "cuda"

    def test_custom_extreme_threshold(self):
        engine = SentimentEngine.from_config(
            {"sentiment": {"extreme_threshold": 0.85}}
        )
        assert engine.config.extreme_threshold == 0.85

    def test_returns_sentiment_engine(self):
        assert isinstance(SentimentEngine.from_config({}), SentimentEngine)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Env observation space — backward compat + extension
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvObsSpace:
    def test_obs_shape_without_sentiment(self, tiny_df):
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10)
        assert env.observation_space.shape == (10, 5)

    def test_obs_shape_with_sentiment(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10, sentiment_data=sent)
        assert env.observation_space.shape == (10, 9)

    def test_n_features_without_sentiment(self, tiny_df):
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10)
        assert env._n_features == 5

    def test_n_features_with_sentiment(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10, sentiment_data=sent)
        assert env._n_features == 9

    def test_n_sentiment_zero_without(self, tiny_df):
        assert SingleAssetRLTradingEnv(data=tiny_df)._n_sentiment == 0

    def test_n_sentiment_four_with(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, sentiment_data=sent)
        assert env._n_sentiment == 4

    def test_sentiment_length_mismatch_raises(self, tiny_df):
        wrong = _zero_sentiment(10)  # 10 != 50
        with pytest.raises(ValueError, match="sentiment_data length"):
            SingleAssetRLTradingEnv(data=tiny_df, sentiment_data=wrong)

    def test_reset_shape_without_sentiment(self, tiny_df):
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10)
        obs, _ = env.reset()
        assert obs.shape == (10, 5)

    def test_reset_shape_with_sentiment(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10, sentiment_data=sent)
        obs, _ = env.reset()
        assert obs.shape == (10, 9)

    def test_reset_dtype_float32(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5, sentiment_data=sent)
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_shape_with_sentiment(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5, sentiment_data=sent)
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs.shape == (5, 9)

    def test_step_shape_without_sentiment(self, tiny_df):
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5)
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs.shape == (5, 5)

    def test_sentiment_cols_in_obs_correct_values(self, tiny_df):
        # Set mean_sentiment = 0.5 for all rows; check obs columns 5–8
        sent = _const_sentiment(len(tiny_df), 0.5)
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5, sentiment_data=sent)
        obs, _ = env.reset()
        np.testing.assert_allclose(obs[:, 5], 0.5, atol=1e-5)

    def test_sentiment_zeros_when_all_zero(self, tiny_df):
        sent = _zero_sentiment(len(tiny_df))
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5, sentiment_data=sent)
        obs, _ = env.reset()
        np.testing.assert_allclose(obs[:, 5:], 0.0, atol=1e-5)

    def test_ohlcv_cols_unchanged_when_sentiment_added(self, tiny_df):
        env_no_sent = SingleAssetRLTradingEnv(data=tiny_df, window_size=5)
        obs_no, _ = env_no_sent.reset(seed=0)

        sent = _zero_sentiment(len(tiny_df))
        env_sent = SingleAssetRLTradingEnv(
            data=tiny_df, window_size=5, sentiment_data=sent
        )
        obs_with, _ = env_sent.reset(seed=0)

        # First 5 columns should be identical
        np.testing.assert_allclose(obs_no, obs_with[:, :5], atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_align_then_env_pipeline(self, tiny_df, mock_engine):
        """End-to-end: news → align → env reset → correct obs shape."""
        news = pd.DataFrame(
            {"timestamp": [5, 15, 25], "headline": ["A", "B", "C"]}
        )
        full_df = mock_engine.align_to_prices(news, tiny_df)
        sent_data = full_df[SENTIMENT_COLS].copy()
        env = SingleAssetRLTradingEnv(
            data=tiny_df, window_size=10, sentiment_data=sent_data
        )
        obs, _ = env.reset()
        assert obs.shape == (10, 9)

    def test_multi_step_rollout_with_sentiment(self, tiny_df):
        sent = _const_sentiment(len(tiny_df), 0.3)
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5, sentiment_data=sent)
        env.reset()
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(
                np.array([0.0], dtype=np.float32)
            )
            assert obs.shape == (5, 9)
            if terminated or truncated:
                break

    def test_backward_compat_no_sentiment_param(self, tiny_df):
        """Existing code that does NOT pass sentiment_data should still work."""
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=10)
        obs, _ = env.reset()
        assert obs.shape == (10, 5)
        obs2, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs2.shape == (10, 5)

    def test_sentiment_stored_with_reset_index(self, tiny_df):
        # Pass sentiment with a non-default index; env should reset it
        sent = _zero_sentiment(len(tiny_df))
        sent.index = range(100, 100 + len(tiny_df))  # non-zero start
        env = SingleAssetRLTradingEnv(data=tiny_df, window_size=5, sentiment_data=sent)
        obs, _ = env.reset()
        assert obs.shape == (5, 9)  # should work fine

    def test_score_batch_consistent_with_score_text(self, mock_engine):
        texts = ["A", "B", "C"]
        individual = [mock_engine.score_text(t) for t in texts]
        batch = mock_engine.score_batch(texts)
        for ind, bat in zip(individual, batch):
            assert ind == bat
