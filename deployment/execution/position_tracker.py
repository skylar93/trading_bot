"""
PositionTracker: thread-safe, single source of truth for position state.

Consolidates the 5 position-related variables that were previously scattered
across OrderManager and PaperTrader:
    - position      (units held)
    - entry_price
    - cash
    - current_price
    - peak_value

All mutations go through a single RLock, making concurrent reads and writes safe
when a live feed thread updates prices while the trading loop executes orders.

Usage:
    tracker = PositionTracker(initial_cash=10_000.0)
    tracker.update_price(50_000.0)
    tracker.apply_buy(quantity=0.01, price=50_000.0, fee=0.5)
    pnl = tracker.apply_sell(quantity=0.01, price=51_000.0, fee=0.51)
    snap = tracker.snapshot()   # dict, safe to serialise
    tracker.restore(snap)       # restore from checkpoint
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Thread-safe container for all live position state.

    Parameters
    ----------
    initial_cash : float
        Starting cash balance.
    """

    def __init__(self, initial_cash: float) -> None:
        self._lock = threading.RLock()
        self._position: float = 0.0
        self._entry_price: float = 0.0
        self._cash: float = float(initial_cash)
        self._current_price: float = 0.0
        self._peak_value: float = float(initial_cash)

    # ------------------------------------------------------------------
    # Read-only properties (each acquires lock independently)
    # ------------------------------------------------------------------

    @property
    def position(self) -> float:
        with self._lock:
            return self._position

    @property
    def entry_price(self) -> float:
        with self._lock:
            return self._entry_price

    @property
    def cash(self) -> float:
        with self._lock:
            return self._cash

    @property
    def current_price(self) -> float:
        with self._lock:
            return self._current_price

    @property
    def portfolio_value(self) -> float:
        with self._lock:
            return self._cash + self._position * self._current_price

    @property
    def peak_value(self) -> float:
        with self._lock:
            return self._peak_value

    @property
    def drawdown(self) -> float:
        """Current drawdown fraction from peak (0.0 – 1.0)."""
        with self._lock:
            if self._peak_value <= 0:
                return 0.0
            pv = self._cash + self._position * self._current_price
            return max(0.0, (self._peak_value - pv) / self._peak_value)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update_price(self, price: float) -> None:
        """Record latest market price and update peak portfolio value."""
        with self._lock:
            self._current_price = price
            pv = self._cash + self._position * price
            if pv > self._peak_value:
                self._peak_value = pv

    def apply_buy(self, quantity: float, price: float, fee: float) -> None:
        """
        Record a completed buy.

        Parameters
        ----------
        quantity : float
            Units purchased.
        price : float
            Fill price per unit.
        fee : float
            Total transaction fee (already included in cost).
        """
        with self._lock:
            cost = quantity * price + fee
            self._cash -= cost
            self._position += quantity
            if self._entry_price == 0.0:
                self._entry_price = price

    def apply_sell(self, quantity: float, price: float, fee: float) -> float:
        """
        Record a completed sell.

        Returns
        -------
        float
            Realised P&L for this sell (positive = profit).
        """
        with self._lock:
            if quantity > self._position + 1e-8:
                logger.warning(
                    "Sell quantity %.6f exceeds position %.6f; clamping.",
                    quantity, self._position,
                )
                quantity = self._position
            if quantity < 1e-8:
                return 0.0
            proceeds = quantity * price - fee
            pnl = (price - self._entry_price) * quantity if self._entry_price else 0.0
            self._cash += proceeds
            self._position -= quantity
            if self._position < 1e-8:
                self._position = 0.0
                self._entry_price = 0.0
            return pnl

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        """Return a serialisable copy of all position state."""
        with self._lock:
            return {
                "position": self._position,
                "entry_price": self._entry_price,
                "cash": self._cash,
                "current_price": self._current_price,
                "peak_value": self._peak_value,
            }

    def restore(self, data: Dict[str, float]) -> None:
        """Restore state from a snapshot dict (e.g. loaded from disk)."""
        with self._lock:
            self._position = float(data.get("position", 0.0))
            self._entry_price = float(data.get("entry_price", 0.0))
            self._cash = float(data.get("cash", 0.0))
            self._current_price = float(data.get("current_price", 0.0))
            self._peak_value = float(data.get("peak_value", self._cash))

    def reset(self, initial_cash: float) -> None:
        """Reset to a clean state (e.g. start of a new episode)."""
        with self._lock:
            self._position = 0.0
            self._entry_price = 0.0
            self._cash = float(initial_cash)
            self._current_price = 0.0
            self._peak_value = float(initial_cash)
