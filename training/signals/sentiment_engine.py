"""
FinBERT-based sentiment signal engine for financial news.

Pipeline:
    1. Score news headlines via ProsusAI/finbert (HuggingFace)
    2. Aggregate per-timestep: mean sentiment, dispersion,
       extreme count, momentum
    3. Output: 4 features per timestep, all in [-1, 1] / [0, 1]

Caching:
    Scores are cached in SQLite to avoid redundant inference.
    Set ``SentimentConfig.cache_db = None`` to disable.

Graceful degradation:
    The environment accepts ``sentiment_data=None``; in that case the
    observation space stays at (window_size, 5).  This module only
    needs ``transformers`` when actually running inference.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import transformers  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Number of sentiment features added to each timestep
N_SENTIMENT_FEATURES = 4

# Column names produced by align_to_prices()
SENTIMENT_COLS = [
    "mean_sentiment",
    "dispersion",
    "extreme_count",
    "sentiment_momentum",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentConfig:
    """Configuration for the FinBERT sentiment engine."""
    model_name: str = "ProsusAI/finbert"
    max_length: int = 128          # token limit passed to the pipeline
    batch_size: int = 32           # headline batch size for inference
    cache_db: Optional[str] = None  # path to SQLite cache; None = disabled
    device: str = "cpu"            # "cpu", "cuda", "cuda:0", etc.
    extreme_threshold: float = 0.7 # |pos - neg| above this → extreme headline


@dataclass
class SentimentFeatures:
    """Four sentiment features aggregated from a set of headlines."""
    mean_sentiment: float   # mean(pos - neg), clipped to [-1, 1]
    dispersion: float       # std(pos - neg) across headlines, clipped to [0, 1]
    extreme_count: float    # fraction of extreme headlines, in [0, 1]
    momentum: float         # tanh(Δmean_sentiment vs prev window), in [-1, 1]

    def to_array(self) -> np.ndarray:
        """Return a float32 array of shape (4,)."""
        return np.array(
            [self.mean_sentiment, self.dispersion, self.extreme_count, self.momentum],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SentimentEngine:
    """
    FinBERT-based financial sentiment analyser.

    Usage::

        engine = SentimentEngine(SentimentConfig(cache_db="cache.sqlite"))
        scores  = engine.score_text("Apple beats earnings estimate")
        # {'positive': 0.91, 'negative': 0.04, 'neutral': 0.05}

        # Aggregate multiple headlines for one timestep
        features = engine.compute_features(["Headline A", "Headline B"])
        # SentimentFeatures(mean_sentiment=0.7, ...)

        # Align news DataFrame to price data
        df_with_sentiment = engine.align_to_prices(news_df, prices_df)
    """

    def __init__(self, config: Optional[SentimentConfig] = None) -> None:
        self.config = config or SentimentConfig()
        self._pipeline = None          # lazy-loaded HuggingFace pipeline
        self._db_conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the FinBERT pipeline on first call (lazy initialisation)."""
        if self._pipeline is not None:
            return
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for sentiment inference. "
                "Install it with: pip install transformers"
            )
        from transformers import pipeline as hf_pipeline  # local import

        logger.info("Loading sentiment model '%s' …", self.config.model_name)
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.config.model_name,
            device=self.config.device,
            top_k=None,   # return all three label scores
        )

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------

    def _get_db(self) -> Optional[sqlite3.Connection]:
        """Return the cache connection, creating the DB on first access."""
        if self.config.cache_db is None:
            return None
        if self._db_conn is None:
            db_dir = os.path.dirname(self.config.cache_db)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            conn = sqlite3.connect(self.config.cache_db)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment (
                    text_hash TEXT PRIMARY KEY,
                    positive  REAL NOT NULL,
                    negative  REAL NOT NULL,
                    neutral   REAL NOT NULL
                )
                """
            )
            conn.commit()
            self._db_conn = conn
        return self._db_conn

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single headline.

        Returns a dict with keys ``positive``, ``negative``, ``neutral``
        (each a float in [0, 1]).  Results are cached in SQLite when
        ``cache_db`` is configured.
        """
        text_hash = self._hash(text)
        db = self._get_db()

        # Cache hit
        if db is not None:
            row = db.execute(
                "SELECT positive, negative, neutral FROM sentiment WHERE text_hash = ?",
                (text_hash,),
            ).fetchone()
            if row is not None:
                return {"positive": row[0], "negative": row[1], "neutral": row[2]}

        # Inference
        self._ensure_model()
        # Rough char limit (≈6 chars/token)
        truncated = text[: self.config.max_length * 6]
        raw = self._pipeline([truncated])   # list(1) → list-of-lists
        label_scores = raw[0]              # first (only) input
        scores: Dict[str, float] = {
            r["label"].lower(): float(r["score"]) for r in label_scores
        }
        out = {
            "positive": scores.get("positive", 0.0),
            "negative": scores.get("negative", 0.0),
            "neutral": scores.get(
                "neutral",
                max(0.0, 1.0 - scores.get("positive", 0.0) - scores.get("negative", 0.0)),
            ),
        }

        # Cache write
        if db is not None:
            db.execute(
                "INSERT OR REPLACE INTO sentiment VALUES (?, ?, ?, ?)",
                (text_hash, out["positive"], out["negative"], out["neutral"]),
            )
            db.commit()

        return out

    def score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Score a list of headlines.

        Calls :meth:`score_text` per item so each result is individually
        cached.
        """
        return [self.score_text(t) for t in texts]

    def compute_features(
        self,
        headlines: List[str],
        prev_mean: Optional[float] = None,
    ) -> SentimentFeatures:
        """
        Aggregate a list of headlines into four scalar features.

        Args:
            headlines:  List of headline strings (may be empty).
            prev_mean:  Mean sentiment from the previous window, used to
                        compute momentum.  Pass ``None`` for the first step.

        Returns:
            :class:`SentimentFeatures` with values in [-1, 1] / [0, 1].
        """
        if not headlines:
            return SentimentFeatures(0.0, 0.0, 0.0, 0.0)

        scores = self.score_batch(headlines)
        sentiments = np.array(
            [s["positive"] - s["negative"] for s in scores], dtype=np.float64
        )

        mean_sent = float(np.mean(sentiments))
        dispersion = (
            float(np.std(sentiments, ddof=0)) if len(sentiments) > 1 else 0.0
        )
        extreme_frac = float(
            np.mean(np.abs(sentiments) > self.config.extreme_threshold)
        )

        if prev_mean is not None:
            momentum = float(np.tanh(mean_sent - prev_mean))
        else:
            momentum = 0.0

        return SentimentFeatures(
            mean_sentiment=float(np.clip(mean_sent, -1.0, 1.0)),
            dispersion=float(np.clip(dispersion, 0.0, 1.0)),
            extreme_count=float(np.clip(extreme_frac, 0.0, 1.0)),
            momentum=float(np.clip(momentum, -1.0, 1.0)),
        )

    def align_to_prices(
        self,
        news_df: Optional[pd.DataFrame],
        prices_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        headline_col: str = "headline",
    ) -> pd.DataFrame:
        """
        Align news sentiment features to price data.

        Groups headlines by ``timestamp_col`` value and computes
        :meth:`compute_features` for each group.  The result is then
        forward-filled so every row in ``prices_df`` has a sentiment value.

        Args:
            news_df:       DataFrame with at least ``timestamp_col`` and
                           ``headline_col`` columns.  Pass ``None`` or an
                           empty DataFrame to get all-zero sentiment.
            prices_df:     Price DataFrame (any index).
            timestamp_col: Column in ``news_df`` whose values match
                           ``prices_df.index`` entries.
            headline_col:  Column in ``news_df`` with headline text.

        Returns:
            Copy of ``prices_df`` with four additional columns:
            ``mean_sentiment``, ``dispersion``, ``extreme_count``,
            ``sentiment_momentum``.
        """
        result = prices_df.copy()
        for col in SENTIMENT_COLS:
            result[col] = np.nan  # NaN so that ffill later propagates correctly

        if news_df is None or len(news_df) == 0:
            for col in SENTIMENT_COLS:
                result[col] = 0.0
            return result

        # ── Group headlines by timestamp ──────────────────────────────
        grouped: Dict[Any, List[str]] = {}
        for _, row in news_df.iterrows():
            ts = row[timestamp_col]
            text = row.get(headline_col) if hasattr(row, "get") else row[headline_col]
            # Skip empty / NaN headlines
            if text is None or (not isinstance(text, str) and pd.isna(text)):
                continue
            text = str(text).strip()
            if not text:
                continue
            if ts not in grouped:
                grouped[ts] = []
            grouped[ts].append(text)

        # ── Compute features and assign to matching rows ───────────────
        prev_mean: Optional[float] = None
        for ts_key in sorted(grouped.keys()):
            headlines = grouped[ts_key]
            features = self.compute_features(headlines, prev_mean=prev_mean)
            if ts_key in result.index:
                result.loc[ts_key, "mean_sentiment"]    = features.mean_sentiment
                result.loc[ts_key, "dispersion"]        = features.dispersion
                result.loc[ts_key, "extreme_count"]     = features.extreme_count
                result.loc[ts_key, "sentiment_momentum"] = features.momentum
            prev_mean = features.mean_sentiment

        # ── Forward-fill so every bar has a value ─────────────────────
        for col in SENTIMENT_COLS:
            result[col] = result[col].ffill().fillna(0.0)

        return result

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SentimentEngine":
        """
        Create a :class:`SentimentEngine` from a training config dict.

        Reads the ``sentiment`` sub-dict; falls back to defaults for
        any missing key.
        """
        sent_cfg = config.get("sentiment", {})
        return cls(
            SentimentConfig(
                model_name=sent_cfg.get("model_name", "ProsusAI/finbert"),
                max_length=sent_cfg.get("max_length", 128),
                batch_size=sent_cfg.get("batch_size", 32),
                cache_db=sent_cfg.get("cache_db", None),
                device=sent_cfg.get("device", "cpu"),
                extreme_threshold=sent_cfg.get("extreme_threshold", 0.7),
            )
        )
