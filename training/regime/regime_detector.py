"""
Market Regime Detector — Week 6.

Two detection approaches:
  HMMRegimeDetector       — GaussianHMM (3 hidden states), primary method.
  ThresholdRegimeDetector — Volatility-percentile classifier, simple fallback.

Both expose the same interface:
    detector.fit(prices)                      # train on price array
    probs = detector.get_regime_probs(prices) # → np.ndarray shape (3,)
    label = detector.predict_regime(prices)   # → int  0 | 1 | 2

Regime label semantics (after fitting):
    0 → low_vol  / trending   (trend-following agents favoured)
    1 → medium_vol / ranging  (balanced weights)
    2 → high_vol / crisis     (defensive / mean-reversion favoured)

RegimeDetector wraps both and provides:
    - automatic HMM→threshold fallback on errors
    - regime-aware ensemble weight multipliers per risk profile
    - stateful current_regime / current_probs properties

Usage::
    detector = RegimeDetector(method="hmm")
    detector.fit(training_prices)
    probs = detector.get_regime_probs(recent_prices[-50:])
    multipliers = detector.get_weight_multipliers(detector.current_regime)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Regime semantics ──────────────────────────────────────────────────────────
REGIME_NAMES: Dict[int, str] = {
    0: "low_vol",
    1: "medium_vol",
    2: "high_vol",
}

# Per-regime weight multipliers keyed by risk_profile string.
# Applied multiplicatively to current ensemble weights before re-normalising.
_REGIME_WEIGHT_MULTIPLIERS: Dict[int, Dict[str, float]] = {
    0: {"conservative": 1.5, "moderate": 1.0, "aggressive": 0.5},   # trending → boost conservative
    1: {"conservative": 1.0, "moderate": 1.0, "aggressive": 1.0},   # ranging  → balanced
    2: {"conservative": 0.8, "moderate": 1.5, "aggressive": 0.3},   # crisis   → boost moderate/defensive
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(prices: np.ndarray) -> np.ndarray:
    """
    Compute a 3-column feature matrix from a 1-D price array.

    Columns:
        0 — log returns                    (length = len(prices) - 1)
        1 — rolling 20-bar volatility      (std of log returns)
        2 — directional trend score        (cumulative log-return over window)

    Returns shape (N-1, 3).  Raises ValueError if fewer than 3 prices.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < 3:
        raise ValueError(f"Need at least 3 prices; got {len(prices)}")

    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    n = len(log_returns)

    # Rolling 20-bar vol (clipped at window boundaries)
    roll = min(20, n)
    vol = np.array(
        [np.std(log_returns[max(0, i - roll + 1): i + 1]) for i in range(n)],
        dtype=np.float64,
    )

    # Directional trend: cumulative return since window start, normalised by vol
    cum_ret = np.cumsum(log_returns)
    trend = cum_ret / (vol + 1e-10)

    return np.column_stack([log_returns, vol, trend])


# ─────────────────────────────────────────────────────────────────────────────
# HMMRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """
    GaussianHMM with 3 hidden states.

    States are relabelled after fitting by ascending rolling volatility so that
    state 0 ≡ low-vol, state 1 ≡ medium-vol, state 2 ≡ high-vol.

    Args:
        n_states:       Number of HMM hidden states (must be 3 for regime semantics).
        n_iter:         EM iterations for HMM fitting.
        covariance_type: GaussianHMM covariance type ("full" or "diag").
        random_state:   RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        covariance_type: str = "diag",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model = None
        self._state_labels: Optional[Dict[int, int]] = None  # hmm_state → regime_label
        self._is_fitted: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray) -> "HMMRegimeDetector":
        """Fit the HMM on the provided price series."""
        from hmmlearn import hmm as hmmlearn_hmm  # lazy import

        prices = np.asarray(prices, dtype=np.float64).ravel()
        features = _extract_features(prices)

        model = hmmlearn_hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(features)

        # Label states by mean volatility (feature column 1)
        mean_vols = model.means_[:, 1]
        sorted_states = np.argsort(mean_vols)  # ascending vol
        self._state_labels = {
            int(sorted_states[i]): i for i in range(self.n_states)
        }
        self._model = model
        self._is_fitted = True
        logger.info(
            "HMMRegimeDetector fitted. State label map (hmm→regime): %s",
            self._state_labels,
        )
        return self

    def get_regime_probs(self, prices: np.ndarray) -> np.ndarray:
        """
        Return posterior regime probabilities [P(low), P(med), P(high)] for
        the most recent observation in *prices*.
        """
        if not self._is_fitted:
            return np.full(self.n_states, 1.0 / self.n_states)

        prices = np.asarray(prices, dtype=np.float64).ravel()
        try:
            features = _extract_features(prices)
        except ValueError:
            return np.full(self.n_states, 1.0 / self.n_states)

        try:
            _, posteriors = self._model.score_samples(features)
        except Exception as exc:
            logger.warning("HMM score_samples failed: %s", exc)
            return np.full(self.n_states, 1.0 / self.n_states)

        last_post = posteriors[-1]  # shape (n_states,)
        regime_probs = np.zeros(self.n_states, dtype=np.float64)
        for hmm_state, regime_label in self._state_labels.items():
            regime_probs[regime_label] += last_post[hmm_state]

        # Numerical safety
        total = regime_probs.sum()
        if total > 0:
            regime_probs /= total
        else:
            regime_probs[:] = 1.0 / self.n_states

        return regime_probs.astype(np.float32)

    def predict_regime(self, prices: np.ndarray) -> int:
        """Return the most probable regime label (0, 1, or 2)."""
        return int(np.argmax(self.get_regime_probs(prices)))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# ThresholdRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdRegimeDetector:
    """
    Volatility-percentile classifier (simple, always interpretable).

    Fit stores 33rd and 67th percentiles of 20-bar rolling vol on training data.
    At inference the current 20-bar vol is compared against those thresholds:
        vol < p33  → regime 0 (low-vol)
        vol < p67  → regime 1 (medium-vol)
        otherwise  → regime 2 (high-vol)

    Args:
        vol_window: Number of bars for rolling volatility (default 20).
    """

    def __init__(self, vol_window: int = 20) -> None:
        self.vol_window = vol_window
        self._p33: float = 0.0
        self._p67: float = 0.0
        self._is_fitted: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray) -> "ThresholdRegimeDetector":
        """Compute volatility percentiles from training prices."""
        prices = np.asarray(prices, dtype=np.float64).ravel()
        if len(prices) < self.vol_window + 1:
            raise ValueError(
                f"Need at least {self.vol_window + 1} prices; got {len(prices)}"
            )

        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        vols = np.array(
            [
                np.std(log_returns[max(0, i - self.vol_window + 1): i + 1])
                for i in range(len(log_returns))
            ],
            dtype=np.float64,
        )

        self._p33 = float(np.percentile(vols, 33))
        self._p67 = float(np.percentile(vols, 67))
        self._is_fitted = True
        logger.info(
            "ThresholdRegimeDetector fitted: p33=%.6f, p67=%.6f",
            self._p33, self._p67,
        )
        return self

    def get_regime_probs(self, prices: np.ndarray) -> np.ndarray:
        """Return one-hot regime probability vector of shape (3,)."""
        prices = np.asarray(prices, dtype=np.float64).ravel()

        if not self._is_fitted or len(prices) < 2:
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        window = log_returns[-self.vol_window:]
        current_vol = float(np.std(window)) if len(window) > 1 else 0.0

        probs = np.zeros(3, dtype=np.float32)
        if current_vol < self._p33:
            probs[0] = 1.0
        elif current_vol < self._p67:
            probs[1] = 1.0
        else:
            probs[2] = 1.0

        return probs

    def predict_regime(self, prices: np.ndarray) -> int:
        """Return regime label (0, 1, or 2)."""
        return int(np.argmax(self.get_regime_probs(prices)))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# RegimeDetector — main facade
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Facade that wraps HMMRegimeDetector (primary) and ThresholdRegimeDetector
    (fallback), exposing a unified interface.

    Args:
        method:            "hmm" (default) or "threshold".
        n_states:          HMM hidden states (passed to HMMRegimeDetector).
        vol_window:        Rolling vol window (passed to ThresholdRegimeDetector).
        n_iter:            HMM EM iterations.
        covariance_type:   HMM covariance structure ("diag" or "full").
        fallback_on_error: If True, fall back to threshold detector when HMM
                           raises at inference time.
        random_state:      RNG seed for HMM.
    """

    def __init__(
        self,
        method: str = "hmm",
        n_states: int = 3,
        vol_window: int = 20,
        n_iter: int = 100,
        covariance_type: str = "diag",
        fallback_on_error: bool = True,
        random_state: int = 42,
    ) -> None:
        if method not in ("hmm", "threshold"):
            raise ValueError(f"method must be 'hmm' or 'threshold'; got '{method}'")

        self.method = method
        self.n_states = n_states
        self.fallback_on_error = fallback_on_error

        self._hmm = HMMRegimeDetector(
            n_states=n_states,
            n_iter=n_iter,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self._threshold = ThresholdRegimeDetector(vol_window=vol_window)
        self._is_fitted: bool = False
        self._current_probs: np.ndarray = np.full(n_states, 1.0 / n_states, dtype=np.float32)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray) -> "RegimeDetector":
        """
        Fit both HMM and threshold detector on *prices*.

        A fitting failure in either sub-detector is logged as a warning but
        does not abort the fit — the other detector still becomes available.
        """
        prices = np.asarray(prices, dtype=np.float64).ravel()
        hmm_ok = False
        thr_ok = False

        try:
            self._hmm.fit(prices)
            hmm_ok = True
        except Exception as exc:
            logger.warning("HMM fit failed (%s); only threshold detector available.", exc)

        try:
            self._threshold.fit(prices)
            thr_ok = True
        except Exception as exc:
            logger.warning("Threshold detector fit failed (%s).", exc)

        if not hmm_ok and not thr_ok:
            raise RuntimeError("Both HMM and threshold fitting failed.")

        self._is_fitted = True
        logger.info(
            "RegimeDetector fitted (method=%s, hmm_ok=%s, threshold_ok=%s)",
            self.method, hmm_ok, thr_ok,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_regime_probs(self, prices: np.ndarray) -> np.ndarray:
        """
        Return regime probability vector of shape (3,) for the most recent
        observation in *prices*.  Values are in [0, 1] and sum to 1.
        """
        prices = np.asarray(prices, dtype=np.float64).ravel()

        if not self._is_fitted:
            return np.full(self.n_states, 1.0 / self.n_states, dtype=np.float32)

        if self.method == "threshold":
            probs = self._threshold.get_regime_probs(prices)
        else:
            if self._hmm.is_fitted:
                try:
                    probs = self._hmm.get_regime_probs(prices)
                except Exception as exc:
                    logger.warning("HMM inference failed (%s); using threshold fallback.", exc)
                    probs = (
                        self._threshold.get_regime_probs(prices)
                        if self.fallback_on_error and self._threshold.is_fitted
                        else np.full(self.n_states, 1.0 / self.n_states, dtype=np.float32)
                    )
            elif self._threshold.is_fitted:
                probs = self._threshold.get_regime_probs(prices)
            else:
                probs = np.full(self.n_states, 1.0 / self.n_states, dtype=np.float32)

        self._current_probs = probs
        return probs

    def predict_regime(self, prices: np.ndarray) -> int:
        """Return most probable regime label (0, 1, or 2)."""
        return int(np.argmax(self.get_regime_probs(prices)))

    # ── Ensemble integration helpers ──────────────────────────────────────────

    def get_weight_multipliers(self, regime: int) -> Dict[str, float]:
        """
        Return per-risk-profile weight multipliers for the given regime label.

        These are *multiplicative* adjustments (before re-normalisation).

        Returns:
            {"conservative": float, "moderate": float, "aggressive": float}
        """
        return dict(_REGIME_WEIGHT_MULTIPLIERS.get(regime, {k: 1.0 for k in ("conservative", "moderate", "aggressive")}))

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def current_probs(self) -> np.ndarray:
        """Last computed regime probabilities (shape (3,))."""
        return self._current_probs.copy()

    @property
    def current_regime(self) -> int:
        """Most probable regime from the last get_regime_probs() call."""
        return int(np.argmax(self._current_probs))

    @property
    def current_regime_name(self) -> str:
        """Human-readable name for current_regime."""
        return REGIME_NAMES.get(self.current_regime, "unknown")

    def __repr__(self) -> str:
        return (
            f"RegimeDetector(method={self.method}, fitted={self._is_fitted}, "
            f"regime={self.current_regime_name})"
        )
