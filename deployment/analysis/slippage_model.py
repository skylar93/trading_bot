"""
Slippage Model Calibration — Week 74 (F15).

Estimates expected slippage from observed fill data using a linear regression
model over execution features.  Calibrated from real exchange fill records;
used in paper mode to inject realistic slippage into order simulation.

Features:
    log_volume   — log(1 + bar_volume): larger volume → smaller slippage
    realized_vol — recent price volatility: higher vol → larger slippage
    side_enc     — 0=buy, 1=sell (direction asymmetry)
    size_frac    — order_size / bar_volume: market-impact proxy

Target:
    slippage_frac — |fill_price - expected_price| / expected_price

Usage:
    model = SlippageModel()
    model.fit(observations)                  # train on List[SlippageObservation]
    frac = model.predict(volume=1e6, realized_vol=0.02, side="buy", size=0.01)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SlippageObservation:
    """One recorded fill event with execution context."""
    side: str               # "buy" | "sell"
    order_size: float       # in base currency
    fill_price: float       # actual fill price
    expected_price: float   # mid/last price at order submission
    bar_volume: float       # volume of the bar at submission (0 if unknown)
    realized_vol: float     # annualised-or-unit realised vol at submission (0 if unknown)

    @property
    def slippage_frac(self) -> float:
        if self.expected_price <= 0:
            return 0.0
        return abs(self.fill_price - self.expected_price) / self.expected_price

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SlippageModel:
    """
    Linear regression slippage predictor.

    Coefficients are fitted via ordinary least squares (closed-form).
    Prediction is clipped to [0, max_slippage_frac] to prevent unbounded
    estimates on extrapolation.

    Parameters
    ----------
    max_slippage_frac : float
        Upper cap on predicted slippage (default 2%, or 0.02).
    min_observations : int
        Minimum training samples before prediction is enabled (default 10).
    """

    def __init__(
        self,
        max_slippage_frac: float = 0.02,
        min_observations: int = 10,
    ) -> None:
        self._max_slip = max_slippage_frac
        self._min_obs = min_observations
        self._coeffs: Optional[np.ndarray] = None    # [intercept, b_log_vol, b_vol, b_side, b_size]
        self._observations: List[SlippageObservation] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, obs: SlippageObservation) -> None:
        """Append one observation.  Call fit() to retrain."""
        self._observations.append(obs)

    def fit(self, observations: Optional[Sequence[SlippageObservation]] = None) -> Dict[str, Any]:
        """
        Train linear regression on observations.

        Parameters
        ----------
        observations :
            If provided, replaces internal observation buffer.

        Returns
        -------
        dict with 'n_samples', 'r2', 'coeffs' keys.
        """
        if observations is not None:
            self._observations = list(observations)

        n = len(self._observations)
        if n < self._min_obs:
            logger.warning(
                "SlippageModel: only %d observations (need %d); model not fitted.",
                n, self._min_obs,
            )
            return {"n_samples": n, "r2": None, "coeffs": None, "fitted": False}

        X, y = self._build_design_matrix(self._observations)

        # OLS: β = (XᵀX)⁻¹ Xᵀy  (regularised with small ridge for stability)
        ridge = 1e-6 * np.eye(X.shape[1])
        try:
            self._coeffs = np.linalg.solve(X.T @ X + ridge, X.T @ y)
        except np.linalg.LinAlgError:
            logger.error("SlippageModel: singular matrix; falling back to zeros.")
            self._coeffs = np.zeros(X.shape[1])

        self._fitted = True

        # R² for diagnostics
        y_pred = X @ self._coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        logger.info(
            "SlippageModel fitted | n=%d R²=%.4f intercept=%.6f",
            n, r2, float(self._coeffs[0]),
        )
        return {
            "n_samples": n,
            "r2": round(r2, 6),
            "coeffs": self._coeffs.tolist(),
            "fitted": True,
        }

    def predict(
        self,
        volume: float,
        realized_vol: float,
        side: str,
        size: float,
    ) -> float:
        """
        Predict slippage fraction for an order.

        Returns 0.0 if model is not yet fitted (safe default).
        """
        if not self._fitted or self._coeffs is None:
            return 0.0

        x = self._featurise(volume, realized_vol, side, size)
        raw = float(x @ self._coeffs)
        return float(np.clip(raw, 0.0, self._max_slip))

    def summary(self) -> Dict[str, Any]:
        """Return model summary dict (coefficients + observation stats)."""
        if not self._fitted or self._coeffs is None:
            return {"fitted": False, "n_observations": len(self._observations)}
        slips = [o.slippage_frac for o in self._observations]
        return {
            "fitted": True,
            "n_observations": len(self._observations),
            "mean_slippage_frac": float(np.mean(slips)),
            "median_slippage_frac": float(np.median(slips)),
            "p95_slippage_frac": float(np.percentile(slips, 95)),
            "coefficients": {
                "intercept": float(self._coeffs[0]),
                "log_volume": float(self._coeffs[1]),
                "realized_vol": float(self._coeffs[2]),
                "side_enc": float(self._coeffs[3]),
                "size_frac": float(self._coeffs[4]),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _featurise(
        self,
        volume: float,
        realized_vol: float,
        side: str,
        size: float,
    ) -> np.ndarray:
        log_vol = np.log1p(max(volume, 0.0))
        side_enc = 1.0 if side == "sell" else 0.0
        size_frac = size / max(volume, 1.0)
        return np.array([1.0, log_vol, max(realized_vol, 0.0), side_enc, size_frac])

    def _build_design_matrix(
        self,
        obs: List[SlippageObservation],
    ):
        rows = [
            self._featurise(o.bar_volume, o.realized_vol, o.side, o.order_size)
            for o in obs
        ]
        X = np.array(rows, dtype=float)
        y = np.array([o.slippage_frac for o in obs], dtype=float)
        return X, y
