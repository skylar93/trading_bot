"""
Fee Model — Week 74 (F16).

Computes exchange trading fees that reflect:
  - Maker vs taker distinction
  - VIP tier discounts (volume-based)
  - BNB / native-token fee discount
  - Daily tier refresh from exchange API

Binance fee schedule (as of 2026):
    VIP 0: maker 0.10%, taker 0.10%
    VIP 1: maker 0.09%, taker 0.10%  (≥ 1M BUSD/30d)
    VIP 2: maker 0.08%, taker 0.10%  (≥ 5M BUSD/30d)
    VIP 3: maker 0.07%, taker 0.08%  (≥ 20M BUSD/30d)
    BNB discount: 25% off when paid with BNB

Usage:
    model = FeeModel()                       # default Binance VIP0 schedule
    fee = model.compute_fee(qty=0.1, price=30_000.0, is_maker=False)
    # → 3.0  (0.1% × 0.1 BTC × 30000)

    # With daily refresh from exchange:
    model = FeeModel.from_exchange(exchange)
    fee = model.compute_fee(qty=0.1, price=30_000.0, is_maker=True)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VIP tier definitions (Binance spot, 2026-04-19)
# Maker/taker rates in basis-points (1 bps = 0.01%).
# ---------------------------------------------------------------------------

@dataclass
class VipTier:
    tier: int
    min_volume_30d_usdt: float   # 30-day trading volume threshold
    maker_bps: float             # maker fee in basis-points
    taker_bps: float             # taker fee in basis-points


_BINANCE_VIP_SCHEDULE: List[VipTier] = [
    VipTier(tier=0, min_volume_30d_usdt=0,          maker_bps=10.0, taker_bps=10.0),
    VipTier(tier=1, min_volume_30d_usdt=1_000_000,  maker_bps=9.0,  taker_bps=10.0),
    VipTier(tier=2, min_volume_30d_usdt=5_000_000,  maker_bps=8.0,  taker_bps=10.0),
    VipTier(tier=3, min_volume_30d_usdt=20_000_000, maker_bps=7.0,  taker_bps=8.0),
    VipTier(tier=4, min_volume_30d_usdt=40_000_000, maker_bps=6.0,  taker_bps=8.0),
    VipTier(tier=5, min_volume_30d_usdt=80_000_000, maker_bps=5.0,  taker_bps=7.0),
]

_BNB_DISCOUNT_FRACTION = 0.25   # 25% discount when paying with BNB


class FeeModel:
    """
    Exchange fee model with VIP tier and BNB discount support.

    Parameters
    ----------
    maker_bps : float
        Maker fee in basis-points.  Overrides tier schedule when set explicitly.
    taker_bps : float
        Taker fee in basis-points.
    bnb_discount : bool
        Whether to apply BNB discount (default False).
    vip_tier : int
        VIP tier index into the schedule (0-5 for Binance).  Used when
        maker_bps/taker_bps are not explicitly overridden.
    schedule : list of VipTier
        Custom tier schedule.  Defaults to Binance spot schedule.
    refresh_interval_sec : float
        How often to re-fetch tier from the exchange API (default 86400 = 1 day).
    """

    def __init__(
        self,
        maker_bps: Optional[float] = None,
        taker_bps: Optional[float] = None,
        bnb_discount: bool = False,
        vip_tier: int = 0,
        schedule: Optional[List[VipTier]] = None,
        refresh_interval_sec: float = 86_400.0,
    ) -> None:
        self._schedule = schedule or _BINANCE_VIP_SCHEDULE
        self._vip_tier: int = max(0, min(vip_tier, len(self._schedule) - 1))
        self._bnb_discount = bnb_discount
        self._refresh_interval = refresh_interval_sec
        self._last_refresh_at: float = 0.0
        self._lock = threading.Lock()

        # Explicit override takes priority over tier schedule
        tier = self._schedule[self._vip_tier]
        self._maker_bps: float = maker_bps if maker_bps is not None else tier.maker_bps
        self._taker_bps: float = taker_bps if taker_bps is not None else tier.taker_bps
        self._maker_override = maker_bps is not None
        self._taker_override = taker_bps is not None

        logger.info(
            "FeeModel | tier=VIP%d maker=%.2fbps taker=%.2fbps bnb=%s",
            self._vip_tier, self._maker_bps, self._taker_bps, bnb_discount,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fee(
        self,
        quantity: float,
        price: float,
        is_maker: bool = False,
        use_bnb: Optional[bool] = None,
    ) -> float:
        """
        Compute trading fee for one order.

        Parameters
        ----------
        quantity : float  — order size in base currency
        price    : float  — fill price in quote currency
        is_maker : bool   — True for limit orders that rest on the book
        use_bnb  : bool   — override BNB discount flag for this order

        Returns
        -------
        float — fee in quote currency
        """
        with self._lock:
            bps = self._maker_bps if is_maker else self._taker_bps

        notional = quantity * price
        fee = notional * bps / 10_000.0

        apply_bnb = use_bnb if use_bnb is not None else self._bnb_discount
        if apply_bnb:
            fee *= (1.0 - _BNB_DISCOUNT_FRACTION)

        return fee

    def effective_rate(self, is_maker: bool = False, use_bnb: Optional[bool] = None) -> float:
        """Return effective fee rate as a fraction (e.g., 0.001 = 0.1%)."""
        with self._lock:
            bps = self._maker_bps if is_maker else self._taker_bps
        rate = bps / 10_000.0
        apply_bnb = use_bnb if use_bnb is not None else self._bnb_discount
        if apply_bnb:
            rate *= (1.0 - _BNB_DISCOUNT_FRACTION)
        return rate

    def set_vip_tier(self, tier: int) -> None:
        """Switch to a different VIP tier (updates maker/taker bps from schedule)."""
        tier = max(0, min(tier, len(self._schedule) - 1))
        with self._lock:
            self._vip_tier = tier
            if not self._maker_override:
                self._maker_bps = self._schedule[tier].maker_bps
            if not self._taker_override:
                self._taker_bps = self._schedule[tier].taker_bps
        logger.info(
            "FeeModel: tier updated to VIP%d | maker=%.2fbps taker=%.2fbps",
            tier, self._maker_bps, self._taker_bps,
        )

    def refresh_from_exchange(self, exchange: Any, symbol: Optional[str] = None) -> bool:
        """
        Fetch the current fee schedule from the exchange via CCXT and update rates.

        Parameters
        ----------
        exchange : CCXT exchange instance
        symbol   : trading pair to query (e.g. "BTC/USDT")

        Returns
        -------
        bool — True if rates were updated, False on failure.
        """
        try:
            fees = exchange.fetch_trading_fees()
            if symbol and symbol in fees:
                raw = fees[symbol]
            else:
                # Use first available symbol or top-level keys
                raw = next(iter(fees.values())) if fees else {}

            maker_rate = raw.get("maker")
            taker_rate = raw.get("taker")

            with self._lock:
                if maker_rate is not None and not self._maker_override:
                    self._maker_bps = float(maker_rate) * 10_000.0
                if taker_rate is not None and not self._taker_override:
                    self._taker_bps = float(taker_rate) * 10_000.0
                self._last_refresh_at = time.monotonic()

            logger.info(
                "FeeModel: refreshed from exchange | maker=%.2fbps taker=%.2fbps",
                self._maker_bps, self._taker_bps,
            )
            return True
        except Exception as e:
            logger.warning("FeeModel.refresh_from_exchange failed: %s", e)
            return False

    def needs_refresh(self) -> bool:
        """Return True if the refresh interval has elapsed since last fetch."""
        return (time.monotonic() - self._last_refresh_at) >= self._refresh_interval

    def summary(self) -> Dict[str, Any]:
        """Return current fee configuration as a plain dict."""
        with self._lock:
            return {
                "vip_tier": self._vip_tier,
                "maker_bps": self._maker_bps,
                "taker_bps": self._taker_bps,
                "maker_rate": self._maker_bps / 10_000.0,
                "taker_rate": self._taker_bps / 10_000.0,
                "bnb_discount": self._bnb_discount,
                "bnb_discount_fraction": _BNB_DISCOUNT_FRACTION if self._bnb_discount else 0.0,
                "last_refresh_age_sec": time.monotonic() - self._last_refresh_at,
                "refresh_interval_sec": self._refresh_interval,
            }

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_exchange(
        cls,
        exchange: Any,
        symbol: Optional[str] = None,
        bnb_discount: bool = False,
        vip_tier: int = 0,
        refresh_interval_sec: float = 86_400.0,
    ) -> "FeeModel":
        """Create a FeeModel and immediately fetch rates from the exchange."""
        model = cls(
            bnb_discount=bnb_discount,
            vip_tier=vip_tier,
            refresh_interval_sec=refresh_interval_sec,
        )
        model.refresh_from_exchange(exchange, symbol=symbol)
        return model

    @classmethod
    def flat(cls, rate_bps: float, bnb_discount: bool = False) -> "FeeModel":
        """Create a FeeModel with identical maker and taker rates."""
        return cls(
            maker_bps=rate_bps,
            taker_bps=rate_bps,
            bnb_discount=bnb_discount,
        )
