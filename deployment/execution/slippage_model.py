"""
Slippage Model — Week 82 (G2).

Calibrates observed execution slippage using a linear regression:

    slippage_bps = β₀ + β₁·vol + β₂·log(size) + β₃·spread_bps

Where:
    vol        — realised volatility of the bar (e.g. |close-open|/open)
    size       — order size in quote currency (USD notional)
    spread_bps — bid-ask spread in basis-points at submission time

The model is re-fit every 24 hours as sandbox data accumulates.
PnLAttributor can call predict() to separate expected slippage from
residual market_move in its attribution table.

Usage:
    from deployment.execution.slippage_model import SlippageModel, SlippageRecord

    model = SlippageModel()
    model.fit(records)          # records: List[SlippageRecord]
    bps = model.predict({"vol": 0.002, "size": 500.0, "spread_bps": 1.5})
    meta = model.metadata()     # last_fit_at, n_samples, r_squared
"""

from __future__ import annotations

import logging
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_REFIT_INTERVAL_SEC: float = 86_400.0   # 24 hours


@dataclass
class SlippageRecord:
    """One fill observation used for calibration.

    Attributes
    ----------
    expected_px  : mid-price or limit-price at submission
    fill_px      : actual fill price
    vol          : bar realised volatility (|close-open|/open)
    spread_bps   : bid-ask spread in bps at order submission
    size         : order notional in quote currency (e.g. USD)
    """
    expected_px: float
    fill_px: float
    vol: float
    spread_bps: float
    size: float

    @property
    def slippage_bps(self) -> float:
        """Observed slippage in basis-points (always ≥ 0)."""
        if self.expected_px <= 0:
            return 0.0
        return abs(self.fill_px - self.expected_px) / self.expected_px * 10_000.0


class SlippageModel:
    """
    Linear slippage calibration model.

    Parameters
    ----------
    refit_interval_sec : float
        How often to re-fit from accumulated records (default 86 400 = 24 h).
    min_samples : int
        Minimum number of records required before fitting (default 10).
    """

    def __init__(
        self,
        refit_interval_sec: float = _REFIT_INTERVAL_SEC,
        min_samples: int = 10,
    ) -> None:
        self._refit_interval = refit_interval_sec
        self._min_samples = min_samples
        self._lock = threading.Lock()

        # Model coefficients [intercept, vol, log_size, spread_bps]
        self._coef: Optional[np.ndarray] = None
        self._r_squared: float = 0.0
        self._n_samples: int = 0
        self._last_fit_at: float = 0.0

        logger.info("SlippageModel initialised | refit_interval=%.0fs", refit_interval_sec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, records: List[SlippageRecord]) -> None:
        """
        Fit the linear model from a list of SlippageRecords.

        Model: slippage_bps ~ 1 + vol + log(size) + spread_bps

        Parameters
        ----------
        records : List[SlippageRecord]
            Observations to fit.  Must have len ≥ min_samples.

        Raises
        ------
        ValueError if records has fewer than min_samples items.
        """
        if len(records) < self._min_samples:
            raise ValueError(
                f"SlippageModel.fit requires ≥ {self._min_samples} records "
                f"(got {len(records)})"
            )

        y = np.array([r.slippage_bps for r in records], dtype=float)
        X = np.column_stack([
            np.ones(len(records)),                                  # intercept
            np.array([r.vol for r in records], dtype=float),       # vol
            np.log(np.maximum([r.size for r in records], 1e-6)),    # log(size)
            np.array([r.spread_bps for r in records], dtype=float), # spread_bps
        ])

        # OLS via numpy lstsq
        coef, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)

        # R²
        y_pred = X @ coef
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        with self._lock:
            self._coef = coef
            self._r_squared = r2
            self._n_samples = len(records)
            self._last_fit_at = time.time()

        logger.info(
            "SlippageModel.fit | n=%d R²=%.4f coef=%s",
            len(records), r2, np.round(coef, 6).tolist(),
        )

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict expected slippage in basis-points for a given order.

        Parameters
        ----------
        features : dict with keys:
            vol        — realised volatility (fraction, e.g. 0.002 = 0.2%)
            size       — notional in quote currency
            spread_bps — bid-ask spread in basis-points

        Returns
        -------
        float — predicted slippage in bps (clamped to ≥ 0).
        """
        with self._lock:
            coef = self._coef

        if coef is None:
            # No fit yet — return 0 (caller falls back to raw slippage_records)
            return 0.0

        vol = float(features.get("vol", 0.0))
        size = float(features.get("size", 1.0))
        spread = float(features.get("spread_bps", 0.0))

        x = np.array([1.0, vol, math.log(max(size, 1e-6)), spread])
        pred = float(np.dot(coef, x))
        return max(pred, 0.0)

    def metadata(self) -> Dict[str, Any]:
        """Return last_fit_at (epoch), n_samples, r_squared."""
        with self._lock:
            return {
                "last_fit_at": self._last_fit_at,
                "n_samples": self._n_samples,
                "r_squared": self._r_squared,
                "needs_refit": self.needs_refit(),
                "coef": self._coef.tolist() if self._coef is not None else None,
            }

    def needs_refit(self) -> bool:
        """Return True if the refit interval has elapsed since the last fit."""
        return (time.time() - self._last_fit_at) >= self._refit_interval
