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
from training.signals.prediction_market import (
    PredictionMarketConfig,
    PredictionMarketSignals,
    N_PREDICTION_MARKET_FEATURES,
    PREDICTION_MARKET_COLS,
)

__all__ = [
    "SentimentConfig",
    "SentimentFeatures",
    "SentimentEngine",
    "N_SENTIMENT_FEATURES",
    "SENTIMENT_COLS",
    "PredictionMarketConfig",
    "PredictionMarketSignals",
    "N_PREDICTION_MARKET_FEATURES",
    "PREDICTION_MARKET_COLS",
]
