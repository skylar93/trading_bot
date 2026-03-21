"""
HMM-based Market Regime Detector.

Architecture
------------
Fits a Gaussian HMM on market features extracted from OHLCV data.
Each hidden state corresponds to a market regime (e.g. bull, bear, sideways).

Feature extraction (from raw OHLCV or pre-computed returns):
  - log_return      : ln(close_t / close_{t-1})
  - volatility      : rolling std of log_return (window=vol_window)
  - volume_ratio    : volume / rolling_mean(volume, vol_window)
  - price_momentum  : close / rolling_mean(close, momentum_window) - 1

Output
------
predict_proba() returns a softmax-normalised (n_regimes,) probability vector
that can be fed directly into MetaController.get_weights().

Usage
-----
    detector = MarketRegimeDetector(n_regimes=3)
    features = detector.extract_features(ohlcv_df)
    detector.fit(features)

    probs = detector.predict_proba(features[-1:])   # (3,)
    mc.get_weights(regime_probs=probs, ...)

    detector.save("regime_detector.pkl")
    detector2 = MarketRegimeDetector.load("regime_detector.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional hmmlearn import
# ---------------------------------------------------------------------------

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GaussianHMM = None
    _HMMLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeDetectorConfig:
    """Hyper-parameters for MarketRegimeDetector."""

    # HMM
    n_regimes: int = 3                # number of hidden market states
    n_iter: int = 100                 # Baum-Welch iterations
    covariance_type: str = "diag"     # "full" | "diag" | "tied" | "spherical"
    tol: float = 1e-4                 # convergence tolerance
    random_state: int = 42

    # Feature extraction
    vol_window: int = 20              # rolling window for volatility & volume ratio
    momentum_window: int = 10         # rolling window for price momentum
    min_samples: int = 50             # minimum rows needed to fit

    # Inference
    temperature: float = 1.0         # softmax temperature (lower → sharper)
    smoothing_alpha: float = 1e-6    # Laplace smoothing for regime probs


# ---------------------------------------------------------------------------
# MarketRegimeDetector
# ---------------------------------------------------------------------------

class MarketRegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Parameters
    ----------
    n_regimes : int, optional
        Number of market regimes (overrides config.n_regimes).
    config : RegimeDetectorConfig, optional
        Full configuration; defaults to ``RegimeDetectorConfig()``.

    Attributes
    ----------
    is_fitted : bool
        True after ``fit()`` has been called successfully.
    regime_labels : list[str]
        Human-readable labels assigned post-fit by sorting regimes
        by mean log-return (low → high): ["bear", "sideways", "bull"]
        for n_regimes=3; generic "regime_0" ... "regime_k" otherwise.
    """

    def __init__(
        self,
        n_regimes: Optional[int] = None,
        config: Optional[RegimeDetectorConfig] = None,
    ) -> None:
        self.cfg = config or RegimeDetectorConfig()
        if n_regimes is not None:
            self.cfg.n_regimes = n_regimes

        self._hmm: Optional[_GaussianHMM] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._sorted_states: Optional[np.ndarray] = None  # mapping: sorted → original
        self.is_fitted: bool = False
        self.regime_labels: List[str] = self._default_labels()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract (T, 4) feature matrix from OHLCV-like DataFrame.

        Expected columns (case-insensitive, accepts $ prefix):
            close / $close, high / $high, low / $low, volume / $volume

        Returns
        -------
        features : (T, n_features) float32 — NaN rows removed
        """
        df = _normalise_columns(data)
        close = df["close"].astype(np.float64)
        volume = df["volume"].astype(np.float64) if "volume" in df.columns else None

        w_vol = self.cfg.vol_window
        w_mom = self.cfg.momentum_window

        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        volatility = log_ret.rolling(w_vol, min_periods=1).std().fillna(0.0)
        momentum = (close / close.rolling(w_mom, min_periods=1).mean() - 1.0).fillna(0.0)

        if volume is not None:
            vol_mean = volume.rolling(w_vol, min_periods=1).mean().replace(0, 1)
            vol_ratio = np.log1p(volume / vol_mean).fillna(0.0)
        else:
            vol_ratio = pd.Series(np.zeros(len(close)), index=close.index)

        features = np.column_stack([
            log_ret.values,
            volatility.values,
            momentum.values,
            vol_ratio.values,
        ]).astype(np.float32)

        return features

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "MarketRegimeDetector":
        """
        Fit the HMM on feature matrix.

        Parameters
        ----------
        features : (T, n_features) float32
        """
        if not _HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for MarketRegimeDetector. "
                "Install with: pip install hmmlearn"
            )

        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if len(X) < self.cfg.min_samples:
            raise ValueError(
                f"Need >= {self.cfg.min_samples} samples (min_samples) to fit HMM, got {len(X)}."
            )

        # Z-score normalise
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._feature_mean) / self._feature_std

        self._hmm = _GaussianHMM(
            n_components=self.cfg.n_regimes,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
            random_state=self.cfg.random_state,
        )
        self._hmm.fit(X_norm)

        # Sort states by mean log-return (feature 0) → low to high
        means = self._hmm.means_[:, 0]  # mean of log_return per state
        self._sorted_states = np.argsort(means)  # ascending

        self.is_fitted = True
        self.regime_labels = self._assign_labels()

        logger.info(
            "MarketRegimeDetector fitted — %d regimes, labels: %s",
            self.cfg.n_regimes,
            self.regime_labels,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Posterior regime probabilities for the last timestep.

        Parameters
        ----------
        features : (T, n_features) or (n_features,) float32

        Returns
        -------
        probs : (n_regimes,) float32  — sums to 1.0, sorted bear→bull
        """
        self._check_fitted()
        X = self._prepare(features)

        # posteriors: shape (T, n_regimes)
        _, posteriors = self._hmm.decode(X, algorithm="viterbi")
        # Use forward-backward posteriors instead for soft probabilities
        log_prob, posteriors = self._hmm.score_samples(X)

        last_post = posteriors[-1]  # (n_regimes,) — HMM original ordering

        # Reorder to bear→bull
        reordered = last_post[self._sorted_states]

        # Laplace smoothing + softmax temperature
        reordered = reordered + self.cfg.smoothing_alpha
        reordered = reordered / reordered.sum()

        if self.cfg.temperature != 1.0:
            log_p = np.log(reordered + 1e-12) / self.cfg.temperature
            reordered = np.exp(log_p - log_p.max())
            reordered /= reordered.sum()

        return reordered.astype(np.float32)

    def predict_regime(self, features: np.ndarray) -> int:
        """
        Return the dominant regime index (0=bear, n-1=bull for default labels).

        Parameters
        ----------
        features : (T, n_features) or (n_features,) float32

        Returns
        -------
        int in [0, n_regimes)
        """
        probs = self.predict_proba(features)
        return int(np.argmax(probs))

    def predict_sequence(self, features: np.ndarray) -> np.ndarray:
        """
        Viterbi-decoded regime index for each timestep.

        Returns
        -------
        regimes : (T,) int — values in [0, n_regimes), bear-sorted
        """
        self._check_fitted()
        X = self._prepare(features)
        _, hidden_states = self._hmm.decode(X, algorithm="viterbi")

        # Remap original HMM state ids to sorted order
        inv_map = np.empty(self.cfg.n_regimes, dtype=int)
        for sorted_idx, orig_idx in enumerate(self._sorted_states):
            inv_map[orig_idx] = sorted_idx

        return inv_map[hidden_states]

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit and return full sequence of regime indices."""
        self.fit(features)
        return self.predict_sequence(features)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "cfg": self.cfg,
            "hmm": self._hmm,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "sorted_states": self._sorted_states,
            "is_fitted": self.is_fitted,
            "regime_labels": self.regime_labels,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("MarketRegimeDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MarketRegimeDetector":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        det = cls(config=payload["cfg"])
        det._hmm = payload["hmm"]
        det._feature_mean = payload["feature_mean"]
        det._feature_std = payload["feature_std"]
        det._sorted_states = payload["sorted_states"]
        det.is_fitted = payload["is_fitted"]
        det.regime_labels = payload["regime_labels"]
        logger.info("MarketRegimeDetector loaded from %s", path)
        return det

    @classmethod
    def from_config(cls, config_dict: dict) -> "MarketRegimeDetector":
        cfg = RegimeDetectorConfig(**{
            k: v for k, v in config_dict.items()
            if k in RegimeDetectorConfig.__dataclass_fields__
        })
        return cls(config=cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = (X - self._feature_mean) / self._feature_std
        return X_norm.astype(np.float64)  # hmmlearn expects float64

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "MarketRegimeDetector is not fitted. Call fit() first."
            )

    def _default_labels(self) -> List[str]:
        n = self.cfg.n_regimes
        if n == 3:
            return ["bear", "sideways", "bull"]
        if n == 4:
            return ["crash", "bear", "bull", "bubble"]
        if n == 2:
            return ["bear", "bull"]
        return [f"regime_{i}" for i in range(n)]

    def _assign_labels(self) -> List[str]:
        """Assign human-readable labels after sorting states."""
        n = self.cfg.n_regimes
        if n == 3:
            return ["bear", "sideways", "bull"]
        if n == 4:
            return ["crash", "bear", "bull", "bubble"]
        if n == 2:
            return ["bear", "bull"]
        return [f"regime_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip $ prefix and lowercase column names."""
    mapping = {c: c.lstrip("$").lower() for c in df.columns}
    return df.rename(columns=mapping)
