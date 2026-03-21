"""
Prediction Market Signal Integration.

Fetches event-contract prices from Polymarket and Kalshi and converts
them into 4 normalised probability features for the RL observation space.

Features per timestep
---------------------
1. primary_event_prob     — probability of the most-relevant event contract
2. event_prob_momentum    — change in probability over last ``momentum_window`` steps
3. event_uncertainty      — entropy of probability distribution across tracked contracts
4. cross_market_divergence — absolute disagreement between Polymarket and Kalshi
                             (0 if only one provider available)

All 4 values are clipped to [0, 1].

Caching
-------
API responses are cached in an SQLite database (``cache_db``).  Set to
``None`` to disable.  TTL is controlled by ``cache_ttl`` (seconds).

Graceful degradation
--------------------
If no API is reachable the module returns zero-filled features and logs a
warning.  The environment remains fully functional; it just loses the
prediction-market signal.

Usage
-----
    pms = PredictionMarketSignals()
    features = pms.get_features("BTC")  # shape: (4,)

    # For alignment with a price DataFrame (backtesting):
    signal_df = pms.align_to_prices(price_df, asset="BTC")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Number of prediction-market features
N_PREDICTION_MARKET_FEATURES = 4

PREDICTION_MARKET_COLS = [
    "primary_event_prob",
    "event_prob_momentum",
    "event_uncertainty",
    "cross_market_divergence",
]

# ---------------------------------------------------------------------------
# Optional dependencies — only needed for live API calls
# ---------------------------------------------------------------------------

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PredictionMarketConfig:
    """Configuration for PredictionMarketSignals."""

    enabled: bool = False
    providers: List[str] = field(default_factory=lambda: ["polymarket", "kalshi"])
    cache_db: Optional[str] = "data_cache/prediction_market_cache.sqlite"
    cache_ttl: int = 300          # seconds
    momentum_window: int = 5      # steps for prob_momentum
    timeout: float = 5.0          # HTTP request timeout (seconds)

    # Asset → list of relevant event keywords for contract search
    asset_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "BTC":  ["bitcoin", "btc", "crypto"],
        "ETH":  ["ethereum", "eth", "crypto"],
        "SPY":  ["fed", "rate", "recession", "s&p"],
        "GLD":  ["gold", "inflation", "fed"],
        "DEFAULT": ["recession", "fed rate"],
    })

    # Polymarket public API base URL
    polymarket_base_url: str = "https://clob.polymarket.com"
    # Kalshi public API base URL
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class _Cache:
    """Simple SQLite-backed key-value cache with TTL."""

    def __init__(self, db_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        if time.time() > row[1]:
            self._conn.execute("DELETE FROM cache WHERE key=?", (key,))
            self._conn.commit()
            return None
        return json.loads(row[0])

    def set(self, key: str, value: dict, ttl: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?,?,?)",
            (key, json.dumps(value), time.time() + ttl),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _cache_key(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provider fetch functions
# ---------------------------------------------------------------------------

def _fetch_polymarket(asset: str, keywords: List[str], cfg: PredictionMarketConfig) -> Optional[float]:
    """
    Fetch the probability of the most-relevant event on Polymarket.

    Returns the mid-price (probability) of the best-matching active market,
    or None if the request fails.
    """
    if not _REQUESTS_AVAILABLE:
        return None

    url = f"{cfg.polymarket_base_url}/markets"
    params = {"active": "true", "closed": "false", "limit": 100}

    try:
        resp = _requests.get(url, params=params, timeout=cfg.timeout)
        resp.raise_for_status()
        markets = resp.json().get("data", [])
    except Exception as exc:
        logger.debug("Polymarket fetch failed: %s", exc)
        return None

    best_price: Optional[float] = None
    for market in markets:
        question = (market.get("question", "") or "").lower()
        if any(kw in question for kw in keywords):
            # outcomePrices is a JSON string like '["0.72", "0.28"]'
            try:
                prices_raw = market.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                if prices:
                    best_price = float(prices[0])
                    break
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    return best_price


def _fetch_kalshi(asset: str, keywords: List[str], cfg: PredictionMarketConfig) -> Optional[float]:
    """
    Fetch the probability of the most-relevant event on Kalshi.

    Returns the yes-price of the best-matching active market, or None.
    """
    if not _REQUESTS_AVAILABLE:
        return None

    url = f"{cfg.kalshi_base_url}/markets"
    params = {"status": "open", "limit": 100}

    try:
        resp = _requests.get(url, params=params, timeout=cfg.timeout)
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
    except Exception as exc:
        logger.debug("Kalshi fetch failed: %s", exc)
        return None

    for market in markets:
        title = (market.get("title", "") or "").lower()
        if any(kw in title for kw in keywords):
            try:
                yes_price = market.get("yes_ask") or market.get("last_price")
                if yes_price is not None:
                    return float(yes_price)
            except (ValueError, TypeError):
                continue

    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PredictionMarketSignals:
    """
    Fetches event-contract prices from Polymarket/Kalshi and converts them
    into 4 normalised features for the RL observation space.

    Parameters
    ----------
    config : PredictionMarketConfig, optional

    Example
    -------
    >>> pms = PredictionMarketSignals()
    >>> features = pms.get_features("BTC")
    >>> assert features.shape == (4,)
    >>> assert np.all(features >= 0) and np.all(features <= 1)
    """

    def __init__(self, config: Optional[PredictionMarketConfig] = None) -> None:
        self.cfg = config or PredictionMarketConfig()
        self._cache: Optional[_Cache] = None
        if self.cfg.cache_db:
            try:
                self._cache = _Cache(self.cfg.cache_db)
            except Exception as exc:
                logger.warning("Could not open prediction-market cache: %s", exc)

        # Rolling history for momentum calculation per asset
        self._prob_history: Dict[str, List[float]] = {}

        logger.info(
            "PredictionMarketSignals initialised (enabled=%s, providers=%s)",
            self.cfg.enabled,
            self.cfg.providers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self, asset: str = "BTC") -> np.ndarray:
        """
        Return the 4 prediction-market features for the given asset.

        Always returns a valid (4,) float32 array — falls back to zeros if
        the API is unavailable or ``enabled=False``.

        Parameters
        ----------
        asset : str
            Asset ticker (e.g. "BTC", "ETH", "SPY").

        Returns
        -------
        features : np.ndarray, shape (4,), dtype float32
            [primary_event_prob, event_prob_momentum,
             event_uncertainty, cross_market_divergence]
        """
        if not self.cfg.enabled:
            return np.zeros(N_PREDICTION_MARKET_FEATURES, dtype=np.float32)

        poly_prob, kalshi_prob = self._fetch_both(asset)
        features = self._compute_features(asset, poly_prob, kalshi_prob)
        return features

    def align_to_prices(self, price_df, asset: str = "BTC"):
        """
        Attach prediction-market features to a price DataFrame.

        Calls ``get_features`` once per row (intended for backtesting with
        historical snapshots where each row represents a different timestamp).
        For large historical DataFrames consider caching at a coarser
        granularity.

        Parameters
        ----------
        price_df : pd.DataFrame
            Must have a DatetimeIndex or integer index.
        asset : str

        Returns
        -------
        pd.DataFrame
            price_df with 4 new columns appended.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas is required for align_to_prices()")

        result = price_df.copy()
        features_rows = []

        for _ in range(len(price_df)):
            features_rows.append(self.get_features(asset))

        feat_df = pd.DataFrame(
            features_rows,
            columns=PREDICTION_MARKET_COLS,
            index=price_df.index,
        )
        return pd.concat([result, feat_df], axis=1)

    def reset_history(self, asset: Optional[str] = None) -> None:
        """Clear rolling probability history (useful between episodes)."""
        if asset:
            self._prob_history.pop(asset, None)
        else:
            self._prob_history.clear()

    def close(self) -> None:
        if self._cache:
            self._cache.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_both(self, asset: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch probs from all configured providers, with caching."""
        keywords = self.cfg.asset_keywords.get(
            asset.upper(), self.cfg.asset_keywords.get("DEFAULT", ["market"])
        )

        poly_prob: Optional[float] = None
        kalshi_prob: Optional[float] = None

        if "polymarket" in self.cfg.providers:
            poly_prob = self._cached_fetch("polymarket", asset, keywords)

        if "kalshi" in self.cfg.providers:
            kalshi_prob = self._cached_fetch("kalshi", asset, keywords)

        return poly_prob, kalshi_prob

    def _cached_fetch(
        self, provider: str, asset: str, keywords: List[str]
    ) -> Optional[float]:
        key = _cache_key(provider, asset, sorted(keywords))

        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.get("prob")

        if provider == "polymarket":
            prob = _fetch_polymarket(asset, keywords, self.cfg)
        elif provider == "kalshi":
            prob = _fetch_kalshi(asset, keywords, self.cfg)
        else:
            prob = None

        if self._cache and prob is not None:
            self._cache.set(key, {"prob": prob}, ttl=self.cfg.cache_ttl)

        return prob

    def _compute_features(
        self,
        asset: str,
        poly_prob: Optional[float],
        kalshi_prob: Optional[float],
    ) -> np.ndarray:
        """Compute the 4 features from raw provider probabilities."""

        # 1. primary_event_prob — average of available probs, else 0.5 (max uncertainty)
        available = [p for p in [poly_prob, kalshi_prob] if p is not None]
        if available:
            primary = float(np.clip(np.mean(available), 0.0, 1.0))
        else:
            primary = 0.5  # neutral / unknown

        # 2. event_prob_momentum — change over last momentum_window steps
        history = self._prob_history.setdefault(asset, [])
        history.append(primary)
        if len(history) > self.cfg.momentum_window + 1:
            history.pop(0)

        if len(history) >= 2:
            momentum = float(np.clip(primary - history[-min(len(history), self.cfg.momentum_window + 1)], -1.0, 1.0))
            # rescale to [0, 1]
            momentum = (momentum + 1.0) / 2.0
        else:
            momentum = 0.5  # no history → neutral

        # 3. event_uncertainty — entropy of [primary, 1-primary] normalised to [0,1]
        p = np.clip(primary, 1e-7, 1 - 1e-7)
        binary_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        uncertainty = float(np.clip(binary_entropy, 0.0, 1.0))  # max = 1 at p=0.5

        # 4. cross_market_divergence — |poly_prob - kalshi_prob| normalised to [0,1]
        if poly_prob is not None and kalshi_prob is not None:
            divergence = float(np.clip(abs(poly_prob - kalshi_prob), 0.0, 1.0))
        else:
            divergence = 0.0  # unknown / single source

        features = np.array(
            [primary, momentum, uncertainty, divergence], dtype=np.float32
        )
        return features
