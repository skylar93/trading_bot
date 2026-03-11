"""
Sentiment and alternative data signals for trading environments.
"""

from training.signals.sentiment_engine import (
    SentimentConfig,
    SentimentFeatures,
    SentimentEngine,
    N_SENTIMENT_FEATURES,
    SENTIMENT_COLS,
)

__all__ = [
    "SentimentConfig",
    "SentimentFeatures",
    "SentimentEngine",
    "N_SENTIMENT_FEATURES",
    "SENTIMENT_COLS",
]
