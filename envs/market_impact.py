"""
Almgren-Chriss Market Impact Model.

Provides realistic execution cost estimation that accounts for:
  - Temporary price impact (linear in trade rate)
  - Permanent price impact (linear in total trade size)
  - Square-root model (empirically validated for equity/crypto markets)

Reference:
  Almgren & Chriss, "Optimal execution of portfolio transactions" (2001)
  Grinold & Kahn, "Active Portfolio Management" (1999) — square-root model

Week 21 implementation.
"""

import math
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class AlmgrenChrissImpact:
    """Market impact model for realistic slippage estimation.

    Two models are supported:

    **Linear model** (classical Almgren-Chriss):
        temporary_impact = eta * (|shares| / T)
        permanent_impact = gamma * |shares|
        total_cost_fraction = (temporary + permanent) / price

    **Square-root model** (empirically preferred for equities / crypto):
        impact = sigma * sqrt(|shares| / daily_volume) * kappa

    The returned value is a dimensionless cost fraction in [0, max_impact_cap],
    representing how much the effective execution price deviates from the
    mid-market price.  The environment applies it directionally:
        executed_price = mid_price * (1 + direction * impact_fraction)

    Args:
        model: "linear" or "sqrt".
        eta: Temporary impact coefficient (linear model only).
        gamma: Permanent impact coefficient (linear model only).
        sigma: Intraday volatility as fraction (e.g. 0.02 = 2 %).
               Used by the sqrt model.
        daily_volume: Estimated daily traded volume in the same unit as
                      the ``shares`` argument of :meth:`compute`.
                      Defaults to 1 000 000 share-equivalents.
        kappa: Scaling constant for the square-root model.
               Typical empirical values: 0.3 – 1.0.
        max_impact_cap: Hard cap on the returned fraction (default 5 %).
    """

    def __init__(
        self,
        model: Literal["linear", "sqrt"] = "sqrt",
        eta: float = 0.01,
        gamma: float = 0.001,
        sigma: float = 0.02,
        daily_volume: float = 1_000_000.0,
        kappa: float = 0.5,
        max_impact_cap: float = 0.05,
    ) -> None:
        if model not in ("linear", "sqrt"):
            raise ValueError(f"model must be 'linear' or 'sqrt', got '{model}'")
        if daily_volume <= 0:
            raise ValueError("daily_volume must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")

        self.model = model
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.daily_volume = daily_volume
        self.kappa = kappa
        self.max_impact_cap = max_impact_cap

    def compute(
        self,
        shares: float,
        price: float = 1.0,
        daily_volume: float | None = None,
        T: int = 1,
    ) -> float:
        """Compute market impact as a fraction of price.

        Args:
            shares: Trade size. Can be in any consistent unit (shares,
                    dollar-normalised position change, etc.) as long as
                    ``daily_volume`` uses the same unit.
            price:  Current mid-market price. Used only by the linear model
                    to convert absolute impact into a fraction.
            daily_volume: Override the instance-level daily volume estimate.
                          Pass the current bar's volume when available for
                          a dynamic, data-driven estimate.
            T:      Execution horizon in steps (linear model only).

        Returns:
            Non-negative impact fraction in [0, max_impact_cap].
        """
        if shares == 0.0:
            return 0.0

        abs_shares = abs(shares)
        vol = daily_volume if daily_volume is not None else self.daily_volume
        vol = max(vol, 1e-8)  # guard against zero volume

        if self.model == "linear":
            if price <= 0:
                logger.warning("price <= 0 in linear impact model; returning 0")
                return 0.0
            temporary = self.eta * (abs_shares / max(T, 1))
            permanent = self.gamma * abs_shares
            impact = (temporary + permanent) / price

        else:  # sqrt
            # σ * sqrt(|Q| / V) * κ
            impact = self.sigma * math.sqrt(abs_shares / vol) * self.kappa

        impact = float(max(0.0, min(impact, self.max_impact_cap)))
        logger.debug(
            "MarketImpact[%s]: shares=%.4f, vol=%.0f → impact=%.6f",
            self.model, shares, vol, impact,
        )
        return impact

    def compute_from_trade_value(
        self,
        trade_value: float,
        price: float,
        bar_volume: float | None = None,
    ) -> float:
        """Convenience wrapper when trade size is expressed in dollar value.

        Converts trade_value / price → shares, then delegates to :meth:`compute`.
        Uses bar_volume * price as daily dollar volume (proxy) when bar_volume is given.

        Args:
            trade_value: Absolute dollar value of the trade.
            price: Current execution price.
            bar_volume: Raw volume from the OHLCV bar (share-equivalent or
                        unit-less). Used to build a dynamic daily_volume proxy.
        """
        if price <= 0:
            return 0.0
        shares = trade_value / price
        daily_vol = bar_volume if bar_volume is not None else self.daily_volume
        return self.compute(shares=shares, price=price, daily_volume=daily_vol)
