"""
Regime Detector — Week 19 Implementation

Detects market regimes using Gaussian HMM (3-state) with a threshold-based
fallback when hmmlearn is not installed.

Regimes:
    0 — Low-vol trending   (trend-following strategies preferred)
    1 — Medium-vol ranging (balanced / neutral)
    2 — High-vol crisis    (defensive / mean-reversion preferred)

Features fed to HMM:
    [log_returns, rolling_vol_20d, MA50-MA200 spread (normalised)]

Usage:
    rd = RegimeDetector(method='hmm', n_regimes=3)
    rd.fit(price_series)              # 1-D array of prices (or returns)
    probs = rd.predict(window)        # → shape (3,), sums to 1.0
"""

import logging
import numpy as np
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Try to import hmmlearn; fall back gracefully
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.warning(
        "hmmlearn not installed. RegimeDetector will use threshold-based fallback. "
        "Install with: pip install hmmlearn>=0.3.0"
    )


def _compute_features(prices: np.ndarray) -> np.ndarray:
    """
    Compute HMM feature matrix from a 1-D price series.

    Returns shape (T-50, 3):
        col 0 — log returns (annualised, scaled)
        col 1 — rolling 20-day realised volatility (annualised)
        col 2 — (MA50 - MA200) / price  spread  [only available once len >= 200]
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)

    # --- log returns ---
    log_ret = np.diff(np.log(np.where(prices > 0, prices, 1e-10)))

    # --- rolling 20-day vol (std of log returns) ---
    vol = np.array([
        np.std(log_ret[max(0, i - 20):i + 1]) * np.sqrt(252)
        for i in range(len(log_ret))
    ])

    # --- MA50 – MA200 spread (relative to price) ---
    ma50 = np.array([
        np.mean(prices[max(0, i - 50):i + 1])
        for i in range(1, n)          # align with log_ret
    ])
    ma200 = np.array([
        np.mean(prices[max(0, i - 200):i + 1])
        for i in range(1, n)
    ])
    spread = (ma50 - ma200) / np.where(prices[1:] > 0, prices[1:], 1.0)

    feats = np.column_stack([log_ret, vol, spread])

    # Drop early rows where MA200 is not yet reliable (first 199)
    start = min(199, len(feats) - 1)
    feats = feats[start:]

    return feats


class RegimeDetector:
    """
    Market regime detector.

    Parameters
    ----------
    method : 'hmm' | 'threshold'
        Detection method.  'hmm' requires hmmlearn; falls back to 'threshold'
        automatically if not available.
    n_regimes : int
        Number of hidden states (default 3).
    n_iter : int
        EM iterations for HMM training (default 200).
    covariance_type : str
        HMM covariance type passed to GaussianHMM (default 'full').
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        method: Literal["hmm", "threshold"] = "hmm",
        n_regimes: int = 3,
        n_iter: int = 200,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._is_fitted = False
        self._hmm: Optional[object] = None

        # Resolve method
        if method == "hmm" and not _HMM_AVAILABLE:
            logger.warning("hmmlearn unavailable — switching to threshold method.")
            self.method = "threshold"
        else:
            self.method = method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> "RegimeDetector":
        """
        Fit the regime detector on historical price data.

        Parameters
        ----------
        data : np.ndarray
            1-D array of prices **or** log-returns.
            If values are large (> 5) they are treated as prices; otherwise
            as log-returns and prices are reconstructed for feature computation.
        """
        data = np.asarray(data, dtype=np.float64).ravel()

        if self.method == "hmm":
            self._fit_hmm(data)
        # threshold method requires no fitting
        self._is_fitted = True
        return self

    def predict(self, window: np.ndarray) -> np.ndarray:
        """
        Predict regime probability distribution for a given window.

        Parameters
        ----------
        window : np.ndarray
            Recent price / return series (1-D).

        Returns
        -------
        np.ndarray, shape (n_regimes,)
            Probability of each regime; sums to 1.0.
        """
        window = np.asarray(window, dtype=np.float64).ravel()

        if self.method == "hmm" and self._is_fitted and self._hmm is not None:
            probs = self._predict_hmm(window)
        else:
            probs = self._predict_threshold(window)

        # Safety: ensure valid probability vector
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total < 1e-9:
            probs = np.ones(self.n_regimes) / self.n_regimes
        else:
            probs = probs / total
        return probs

    def get_regime(self, window: np.ndarray) -> int:
        """Return the most likely regime index (0, 1, or 2)."""
        return int(np.argmax(self.predict(window)))

    # ------------------------------------------------------------------
    # HMM internals
    # ------------------------------------------------------------------

    def _prices_from_data(self, data: np.ndarray) -> np.ndarray:
        """Heuristically convert data to prices if needed."""
        if np.abs(data).max() > 5.0:
            # Looks like prices already
            return data
        # Treat as log-returns → reconstruct price index
        return np.exp(np.cumsum(np.concatenate([[0.0], data])))

    def _fit_hmm(self, data: np.ndarray) -> None:
        prices = self._prices_from_data(data)

        if len(prices) < 60:
            logger.warning(
                "Not enough data for HMM fit (%d rows); need ≥60. "
                "Falling back to threshold detection.",
                len(prices),
            )
            self.method = "threshold"
            return

        feats = _compute_features(prices)
        if len(feats) < 10:
            logger.warning("Feature matrix too short after trimming. Using threshold fallback.")
            self.method = "threshold"
            return

        # Standardise features for numerical stability
        self._feat_mean = feats.mean(axis=0)
        self._feat_std = np.where(feats.std(axis=0) < 1e-10, 1.0, feats.std(axis=0))
        feats_norm = (feats - self._feat_mean) / self._feat_std

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        try:
            model.fit(feats_norm)
            self._hmm = model
            logger.info(
                "HMM fitted: %d states, score=%.2f",
                self.n_regimes,
                model.score(feats_norm),
            )
            # Re-order states so regime 0 = lowest vol, 2 = highest vol
            self._reorder_states_by_volatility()
        except Exception as exc:
            logger.error("HMM fit failed (%s). Falling back to threshold.", exc)
            self._hmm = None
            self.method = "threshold"

    def _reorder_states_by_volatility(self) -> None:
        """
        Permute HMM states so that state index correlates with volatility
        (state 0 = low vol, state 2 = high vol).
        """
        if self._hmm is None:
            return
        # Use the variance of feature col 1 (rolling vol) per state as proxy
        means = self._hmm.means_  # shape (n_states, n_features)
        vol_means = means[:, 1]   # second feature is vol
        order = np.argsort(vol_means)          # ascending vol → states 0,1,2
        perm = np.argsort(order)               # inverse permutation

        # Permute model internals
        self._hmm.means_ = self._hmm.means_[order]
        if hasattr(self._hmm, "covars_"):
            self._hmm.covars_ = self._hmm.covars_[order]
        self._hmm.startprob_ = self._hmm.startprob_[order]
        self._hmm.transmat_ = self._hmm.transmat_[order][:, order]
        self._state_perm = perm

    def _predict_hmm(self, window: np.ndarray) -> np.ndarray:
        """Posterior state probabilities from HMM for the given window."""
        prices = self._prices_from_data(window)

        if len(prices) < 5:
            return self._predict_threshold(window)

        feats = _compute_features(prices)
        if len(feats) < 1:
            return self._predict_threshold(window)

        feats_norm = (feats - self._feat_mean) / self._feat_std

        try:
            # posteriors shape: (T, n_states)
            _, posteriors = self._hmm.score_samples(feats_norm)
            # Use the last timestep's posterior as the current regime estimate
            probs = posteriors[-1]
            return probs
        except Exception as exc:
            logger.warning("HMM predict failed (%s); using threshold fallback.", exc)
            return self._predict_threshold(window)

    # ------------------------------------------------------------------
    # Threshold-based fallback
    # ------------------------------------------------------------------

    def _predict_threshold(self, window: np.ndarray) -> np.ndarray:
        """
        Simple rule-based regime classification using rolling volatility.

        Thresholds (annualised):
            vol < 0.10  → regime 0 (low-vol trending)
            vol < 0.25  → regime 1 (medium-vol ranging)
            vol >= 0.25 → regime 2 (high-vol crisis)
        """
        window = np.asarray(window, dtype=np.float64).ravel()

        if len(window) < 2:
            return np.array([1.0 / self.n_regimes] * self.n_regimes)

        prices = self._prices_from_data(window)
        log_ret = np.diff(np.log(np.where(prices > 0, prices, 1e-10)))

        if len(log_ret) == 0:
            return np.array([1.0 / self.n_regimes] * self.n_regimes)

        vol = np.std(log_ret) * np.sqrt(252)

        probs = np.zeros(self.n_regimes)
        if self.n_regimes == 3:
            if vol < 0.10:
                probs[0] = 1.0
            elif vol < 0.25:
                probs[1] = 1.0
            else:
                probs[2] = 1.0
        else:
            # Generic: assign uniformly to the middle regime
            mid = self.n_regimes // 2
            probs[mid] = 1.0

        return probs
