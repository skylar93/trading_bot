"""HMM-based Market Regime Detector.

Classifies market states into regimes (bull/bear/sideways) using a
Gaussian Hidden Markov Model fitted on returns, volatility, and volume.

Output is a probability vector over regimes that feeds directly into
the MetaController's ``regime_probs`` input.

Usage
-----
    detector = RegimeDetector(n_regimes=3)
    detector.fit(price_df)  # df with $close, $volume columns
    probs = detector.predict_proba(recent_window)  # (3,)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Gaussian HMM regime detector for financial time series.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states (default 3: bull, bear, sideways).
    lookback : int
        Rolling window size for feature computation.
    vol_window : int
        Window for realized volatility estimate.
    n_iter : int
        Max EM iterations for HMM fitting.
    random_state : int
        Seed for reproducibility.
    """

    # Canonical regime labels (sorted by mean return after fit)
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2

    def __init__(
        self,
        n_regimes: int = 3,
        lookback: int = 60,
        vol_window: int = 20,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.vol_window = vol_window
        self.n_iter = n_iter
        self.random_state = random_state

        self._model = None
        self._fitted = False
        self._regime_order: Optional[np.ndarray] = None

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features: log returns, realized vol, volume change.

        Parameters
        ----------
        df : DataFrame with '$close' and optionally '$volume' columns.

        Returns
        -------
        features : (n_samples, 3) array
        """
        close = df["$close"].values.astype(np.float64)
        log_ret = np.diff(np.log(close + 1e-8))

        # Realized volatility (rolling std of returns)
        vol = pd.Series(log_ret).rolling(self.vol_window, min_periods=1).std().values

        # Volume change rate (if available)
        if "$volume" in df.columns:
            volume = df["$volume"].values[1:].astype(np.float64)
            vol_change = np.diff(np.log(volume + 1.0), prepend=0.0)
        else:
            vol_change = np.zeros_like(log_ret)

        features = np.column_stack([log_ret, vol, vol_change])

        # Replace NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit HMM on historical data.

        Parameters
        ----------
        df : DataFrame with '$close' (and optionally '$volume').
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required for RegimeDetector. "
                "Install with: pip install hmmlearn"
            ) from e

        features = self._build_features(df)

        if len(features) < self.n_regimes * 5:
            raise ValueError(
                f"Not enough data ({len(features)} rows) for {self.n_regimes} regimes"
            )

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(features)

        # Sort regimes by mean return (so BEAR=0, SIDEWAYS=1, BULL=2)
        mean_returns = self._model.means_[:, 0]
        self._regime_order = np.argsort(mean_returns)

        self._fitted = True
        logger.info(
            "RegimeDetector fitted — %d regimes on %d observations. "
            "Mean returns by regime: %s",
            self.n_regimes,
            len(features),
            mean_returns[self._regime_order].round(6).tolist(),
        )
        return self

    def predict(self, df: pd.DataFrame) -> int:
        """Predict the most likely current regime.

        Parameters
        ----------
        df : Recent price data (at least `lookback` rows).

        Returns
        -------
        regime : int (0=bear, 1=sideways, 2=bull after sorting)
        """
        probs = self.predict_proba(df)
        return int(np.argmax(probs))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability distribution over regimes.

        Parameters
        ----------
        df : Recent price data.

        Returns
        -------
        probs : (n_regimes,) array summing to 1.0, ordered [bear, sideways, bull]
        """
        if not self._fitted:
            logger.warning("RegimeDetector not fitted — returning uniform probs")
            return np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=np.float32)

        features = self._build_features(df)
        if len(features) == 0:
            return np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=np.float32)

        # Use last `lookback` rows
        window = features[-self.lookback:]
        try:
            _, posteriors = self._model.score_samples(window)
            # Get the last timestep's posterior probabilities
            raw_probs = posteriors[-1]
        except Exception as e:
            logger.warning("HMM predict_proba failed: %s — returning uniform", e)
            return np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=np.float32)

        # Reorder to canonical (bear, sideways, bull)
        ordered_probs = raw_probs[self._regime_order]
        return ordered_probs.astype(np.float32)

    def get_regime_label(self, regime_idx: int) -> str:
        """Human-readable label for a regime index."""
        labels = {0: "bear", 1: "sideways", 2: "bull"}
        return labels.get(regime_idx, f"regime_{regime_idx}")

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "regime_order": self._regime_order,
                "config": {
                    "n_regimes": self.n_regimes,
                    "lookback": self.lookback,
                    "vol_window": self.vol_window,
                },
            }, f)
        logger.info("RegimeDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data["config"]
        detector = cls(
            n_regimes=cfg["n_regimes"],
            lookback=cfg["lookback"],
            vol_window=cfg["vol_window"],
        )
        detector._model = data["model"]
        detector._regime_order = data["regime_order"]
        detector._fitted = True
        logger.info("RegimeDetector loaded from %s", path)
        return detector
