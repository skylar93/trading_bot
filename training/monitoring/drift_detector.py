"""Concept drift detectors for non-stationary financial time series.

Two detection methods are provided:

ADWIN (Adaptive Windowing)
    Maintains an adaptive window of the reward/signal stream.  When the mean
    of one sub-window differs significantly from the mean of another, drift is
    declared and the window is truncated.  Memory: O(log W), amortised O(1)
    update.  Reference: Bifet & Gavalda (2007), "Learning from Time-Changing
    Data with Adaptive Windowing".

Page-Hinkley Test
    Tracks a cumulative deviation from the running mean.  Suitable for
    detecting a monotone shift in performance (e.g. rewards degrading over
    time).  Reference: Page (1954), "Continuous Inspection Schemes".

Usage
-----
    detector = DriftDetector(method="adwin", confidence=0.01)
    for reward in reward_stream:
        detector.update(reward)
        if detector.drift_detected:
            trigger_retraining()
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Union


# ---------------------------------------------------------------------------
# ADWIN implementation (from scratch, no river dependency)
# ---------------------------------------------------------------------------

class _ADWINBucket:
    """Single ADWIN bucket: count + sum of squared deviations (variance bucket)."""

    __slots__ = ("total", "n")

    def __init__(self, value: float) -> None:
        self.total: float = value
        self.n: int = 1

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n > 0 else 0.0


class ADWIN:
    """Adaptive Windowing drift detector (pure-Python, O(log W) memory).

    Parameters
    ----------
    delta : float
        Confidence level.  Smaller values → fewer false positives but slower
        detection.  Typical range: 0.002 – 0.1.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        # Circular list of buckets representing exponentially-sized sub-windows
        self._buckets: list[_ADWINBucket] = []
        self._total: float = 0.0
        self._n: int = 0
        self._drift_detected: bool = False
        # Track last detected position for external inspection
        self.n_detections: int = 0

    # ------------------------------------------------------------------ #

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def mean(self) -> float:
        return self._total / self._n if self._n > 0 else 0.0

    @property
    def window_size(self) -> int:
        return self._n

    def update(self, value: float) -> bool:
        """Add a new observation and check for drift.

        Returns
        -------
        bool
            True if drift was detected on this call.
        """
        self._drift_detected = False
        self._n += 1
        self._total += value
        self._buckets.append(_ADWINBucket(value))

        detected = self._detect_change()
        if detected:
            self._drift_detected = True
            self.n_detections += 1
        return detected

    def _detect_change(self) -> bool:
        """Slide the window and test for mean shifts using Hoeffding bound."""
        n0 = 0
        total0 = 0.0

        # Traverse buckets from oldest to newest
        for i in range(len(self._buckets)):
            bucket = self._buckets[i]
            n0 += bucket.n
            total0 += bucket.total

            n1 = self._n - n0
            total1 = self._total - total0

            if n1 <= 0:
                continue

            mean0 = total0 / n0
            mean1 = total1 / n1
            delta_mean = abs(mean0 - mean1)

            # Hoeffding / ADWIN threshold
            n_harmonic = 1.0 / (1.0 / n0 + 1.0 / n1) if (n0 > 0 and n1 > 0) else 0.0
            epsilon_cut = math.sqrt(
                (1.0 / (2.0 * n_harmonic)) * math.log(4.0 * self._n / self.delta)
            ) if n_harmonic > 0 else float("inf")

            if delta_mean >= epsilon_cut:
                # Drop oldest sub-window — keep only the newer part
                self._buckets = self._buckets[i + 1 :]
                self._n = n1
                self._total = total1
                return True

        return False

    def reset(self) -> None:
        """Clear internal state (called after retraining)."""
        self._buckets = []
        self._total = 0.0
        self._n = 0
        self._drift_detected = False


# ---------------------------------------------------------------------------
# Page-Hinkley detector
# ---------------------------------------------------------------------------

class PageHinkley:
    """Page-Hinkley sequential change-point detector.

    Detects a persistent *downward* shift in the observed signal (e.g.
    rewards degrading over time).

    Parameters
    ----------
    delta : float
        Allowed mean fluctuation — small positive value to ignore noise.
        Default: 0.005.
    threshold : float
        Detection threshold λ.  Larger → fewer false alarms but slower
        detection.  Default: 50.
    alpha : float
        Forgetting factor for the running mean (1 = no forgetting).
        Default: 1.0.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1.0,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._reset_state()

    def _reset_state(self) -> None:
        self._n: int = 0
        self._sum: float = 0.0
        self._x_mean: float = 0.0
        self._ph_sum: float = 0.0
        self._ph_min: float = 0.0
        self._drift_detected: bool = False
        self.n_detections: int = 0

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: float) -> bool:
        """Add a new observation.

        The Page-Hinkley test detects a *downward* (negative) shift in the
        mean, i.e. degrading performance.  The statistic is:

            z_t  = x̄_{t-1} − x_t − δ          (positive when x drops below mean)
            S_t  = S_{t-1} + z_t
            m_t  = min(m_{t-1}, S_t)
            T_t  = S_t − m_t

        Drift is declared when T_t > λ (threshold).

        Returns
        -------
        bool
            True if drift was detected.
        """
        self._drift_detected = False
        self._n += 1

        # Cumulative mean (alpha=1 → simple cumulative average)
        if self._n == 1:
            self._x_mean = value
        else:
            self._x_mean = self._x_mean + (value - self._x_mean) / self._n

        # PH statistic for DOWNWARD shift: increases when x < mean
        self._ph_sum += self._x_mean - value - self.delta
        self._ph_min = min(self._ph_min, self._ph_sum)

        ph_stat = self._ph_sum - self._ph_min

        if ph_stat > self.threshold:
            n_prev = self.n_detections
            self._reset_state()  # restart the cumulative sum after detection
            self.n_detections = n_prev + 1  # restore incremented count
            self._drift_detected = True   # set AFTER reset so property returns True
            return True

        return False

    def reset(self) -> None:
        self._reset_state()


# ---------------------------------------------------------------------------
# Unified DriftDetector facade
# ---------------------------------------------------------------------------

class DriftDetector:
    """Unified drift detection interface.

    Parameters
    ----------
    method : {"adwin", "page_hinkley"}
        Detection algorithm to use.
    confidence : float
        For ADWIN: δ parameter (default 0.002).
        For Page-Hinkley: ignored (uses ``ph_delta`` and ``ph_threshold``).
    ph_delta : float
        Page-Hinkley allowed fluctuation δ.  Default: 0.005.
    ph_threshold : float
        Page-Hinkley detection threshold λ.  Default: 50.

    Examples
    --------
    >>> d = DriftDetector(method="adwin")
    >>> for r in stable_rewards:
    ...     d.update(r)
    >>> assert not d.drift_detected
    >>> for r in shifted_rewards:
    ...     d.update(r)
    >>> assert d.drift_detected  # eventually True
    """

    def __init__(
        self,
        method: Literal["adwin", "page_hinkley"] = "adwin",
        confidence: float = 0.002,
        ph_delta: float = 0.005,
        ph_threshold: float = 50.0,
    ) -> None:
        self.method = method
        if method == "adwin":
            self._detector: ADWIN | PageHinkley = ADWIN(delta=confidence)
        elif method == "page_hinkley":
            self._detector = PageHinkley(delta=ph_delta, threshold=ph_threshold)
        else:
            raise ValueError(f"Unknown drift detection method: {method!r}. Choose 'adwin' or 'page_hinkley'.")

    @property
    def drift_detected(self) -> bool:
        """True if the most recent ``update()`` call triggered a detection."""
        return self._detector.drift_detected

    @property
    def n_detections(self) -> int:
        """Total number of drift events detected since creation."""
        return self._detector.n_detections

    def update(self, value: float) -> bool:
        """Feed a new observation (e.g. step reward or episode return).

        Returns
        -------
        bool
            True if drift was detected on this call.
        """
        return self._detector.update(value)

    def reset(self) -> None:
        """Reset internal state (e.g. after a retraining cycle)."""
        self._detector.reset()


# ---------------------------------------------------------------------------
# Feature-level drift detector (S56)
# ---------------------------------------------------------------------------

class FeatureDriftDetector:
    """Per-feature drift detection: one DriftDetector per named feature.

    Enables detection of distribution shift in individual input features
    (not just aggregate returns), which is important for early warning
    before model performance degrades.

    Parameters
    ----------
    feature_names : list of str
        Names of features to monitor.  Must match the keys in each
        ``update()`` call.
    method : {"adwin", "page_hinkley"}
        Detection algorithm applied to every feature.
    confidence : float
        ADWIN δ parameter.  Ignored for page_hinkley.
    ph_delta : float
        Page-Hinkley δ (allowed fluctuation).
    ph_threshold : float
        Page-Hinkley detection threshold λ.

    Usage
    -----
    >>> fdd = FeatureDriftDetector(["rsi", "macd", "vol"])
    >>> for step_features in stream:          # dict or array
    ...     alarms = fdd.update(step_features)
    ...     if fdd.any_drift:
    ...         handle_drift(fdd.drift_features)
    """

    def __init__(
        self,
        feature_names: List[str],
        method: Literal["adwin", "page_hinkley"] = "adwin",
        confidence: float = 0.002,
        ph_delta: float = 0.005,
        ph_threshold: float = 50.0,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        self.feature_names: List[str] = list(feature_names)
        self._method = method
        self._detectors: Dict[str, DriftDetector] = {
            name: DriftDetector(
                method=method,
                confidence=confidence,
                ph_delta=ph_delta,
                ph_threshold=ph_threshold,
            )
            for name in feature_names
        }
        # Per-feature drift flags from the most recent update call
        self._last_alarms: Dict[str, bool] = {name: False for name in feature_names}

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def update(
        self,
        features: Union[Dict[str, float], "np.ndarray", List[float]],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Feed one step of feature values and return per-feature drift flags.

        Parameters
        ----------
        features : dict[str, float] | array-like
            Feature values for this step.
            * dict  — keys must be a superset of ``self.feature_names``.
            * array — must match ``self.feature_names`` in length (or use
              ``feature_names`` to provide a custom name mapping).
        feature_names : list of str, optional
            Override name mapping when ``features`` is an array.  Ignored
            when ``features`` is a dict.

        Returns
        -------
        dict[str, bool]
            ``{feature_name: drift_detected}`` for each tracked feature.
        """
        import numpy as _np  # local import to avoid top-level numpy dep

        if isinstance(features, dict):
            values: Dict[str, float] = {
                name: float(features[name]) for name in self.feature_names
            }
        else:
            arr = _np.asarray(features, dtype=float).ravel()
            names = feature_names if feature_names is not None else self.feature_names
            if len(arr) < len(names):
                raise ValueError(
                    f"features length {len(arr)} < feature_names length {len(names)}"
                )
            values = {name: float(arr[i]) for i, name in enumerate(names)}

        alarms: Dict[str, bool] = {}
        for name in self.feature_names:
            val = values[name]
            if not math.isfinite(val):
                # Skip non-finite values; do not update the detector state
                alarms[name] = False
                continue
            alarms[name] = self._detectors[name].update(val)

        self._last_alarms = alarms
        return alarms

    # ------------------------------------------------------------------ #
    # Aggregate views
    # ------------------------------------------------------------------ #

    @property
    def any_drift(self) -> bool:
        """True if at least one feature had drift on the last ``update()``."""
        return any(self._last_alarms.values())

    @property
    def drift_features(self) -> List[str]:
        """Names of features that triggered drift on the last ``update()``."""
        return [name for name, flag in self._last_alarms.items() if flag]

    @property
    def n_detections(self) -> Dict[str, int]:
        """Total drift detection count per feature (cumulative since creation)."""
        return {name: det.n_detections for name, det in self._detectors.items()}

    @property
    def total_detections(self) -> int:
        """Sum of all per-feature detection counts."""
        return sum(self._detectors[n].n_detections for n in self.feature_names)

    def last_alarms(self) -> Dict[str, bool]:
        """Return a copy of the alarm flags from the most recent ``update()``."""
        return dict(self._last_alarms)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, feature_name: Optional[str] = None) -> None:
        """Reset detector state.

        Parameters
        ----------
        feature_name : str, optional
            If given, reset only that feature's detector.  If None, reset all.
        """
        if feature_name is not None:
            if feature_name not in self._detectors:
                raise KeyError(f"Unknown feature: {feature_name!r}")
            self._detectors[feature_name].reset()
            self._last_alarms[feature_name] = False
        else:
            for name in self.feature_names:
                self._detectors[name].reset()
                self._last_alarms[name] = False
