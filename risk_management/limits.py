"""Pre-trade compliance enforcement.

Week 76 (Track G — Governance & Go-Live Gate):
  G6 — Position limits per symbol / portfolio
  G7 — Self-trade prevention
  G8 — Notional cap per unit time (hourly / daily)
  G9 — Wash trade guard (same symbol + direction cooldown)

All ``check_*`` methods return ``(allowed: bool, reason: str)``.
``reason`` is an empty string when ``allowed`` is True.

Usage
-----
checker = PreTradeComplianceChecker(ComplianceConfig(
    per_symbol_notional_max=50_000,
    hourly_notional_cap=200_000,
    wash_trade_cooldown_sec=5.0,
))
ok, reason = checker.check_all(
    symbol="BTC/USDT", side="buy", order_notional=1_000,
    limit_price=None,
)
if not ok:
    reject_order(reason)
else:
    checker.record_order("BTC/USDT", "buy", notional=1_000)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ComplianceConfig:
    """Configuration for pre-trade compliance checks.

    All monetary limits are in the same currency unit as order notional
    (typically quote currency, e.g. USD).
    """

    # G6: position limits
    per_symbol_notional_max: float = float("inf")
    """Maximum allowed open notional per symbol after this order."""

    portfolio_notional_max: float = float("inf")
    """Maximum allowed total portfolio notional after this order."""

    leverage_max: float = float("inf")
    """Maximum leverage multiple (order_notional / portfolio_equity)."""

    # G8: notional caps per unit time
    hourly_notional_cap: float = float("inf")
    """Maximum total accepted order notional in any rolling 60-minute window."""

    daily_notional_cap: float = float("inf")
    """Maximum total accepted order notional in any rolling 24-hour window."""

    # G9: wash trade cooldown
    wash_trade_cooldown_sec: float = 0.0
    """Minimum seconds between two same-direction orders on the same symbol.
    Set to 0 to disable the guard."""

    # G7: self-trade prevention
    self_trade_prevention: bool = True
    """When True, reject orders that would immediately cross an open resting order."""


class PreTradeComplianceChecker:
    """Thread-safe pre-trade compliance enforcement (G6-G9).

    Instances are meant to be shared across threads (e.g. passed to
    ``OrderManager``). All public methods are guarded by a single lock.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None) -> None:
        self._cfg = config or ComplianceConfig()
        self._lock = threading.Lock()

        # G8: sliding-window notional accounting — each entry: (unix_ts, notional)
        self._hourly_window: deque = deque()
        self._daily_window: deque = deque()

        # G9: per (symbol, side) → unix timestamp of last accepted order
        self._wash_guard: Dict[str, Dict[str, float]] = {}

        # G7: open resting orders: symbol → {rounded_price: side}
        self._open_orders: Dict[str, Dict[float, str]] = {}

    # ------------------------------------------------------------------
    # G6: Position limits
    # ------------------------------------------------------------------

    def check_position_limits(
        self,
        symbol: str,
        order_notional: float,
        current_symbol_notional: float = 0.0,
        current_portfolio_notional: float = 0.0,
        leverage: float = 1.0,
    ) -> Tuple[bool, str]:
        """Enforce per-symbol notional, portfolio notional, and leverage limits."""
        cfg = self._cfg

        projected_symbol = current_symbol_notional + order_notional
        if projected_symbol > cfg.per_symbol_notional_max:
            return False, (
                f"position_limit:per_symbol: {symbol} would reach "
                f"{projected_symbol:.2f} > max {cfg.per_symbol_notional_max:.2f}"
            )

        projected_portfolio = current_portfolio_notional + order_notional
        if projected_portfolio > cfg.portfolio_notional_max:
            return False, (
                f"position_limit:portfolio: would reach "
                f"{projected_portfolio:.2f} > max {cfg.portfolio_notional_max:.2f}"
            )

        if leverage > cfg.leverage_max:
            return False, (
                f"position_limit:leverage: {leverage:.3f}x > max {cfg.leverage_max:.3f}x"
            )

        return True, ""

    # ------------------------------------------------------------------
    # G7: Self-trade prevention
    # ------------------------------------------------------------------

    def register_open_order(self, symbol: str, price: float, side: str) -> None:
        """Register a resting limit order so future checks can detect crossing."""
        if not self._cfg.self_trade_prevention:
            return
        with self._lock:
            self._open_orders.setdefault(symbol, {})[round(price, 8)] = side

    def deregister_open_order(self, symbol: str, price: float) -> None:
        """Remove a filled or cancelled order from self-trade tracking."""
        with self._lock:
            self._open_orders.get(symbol, {}).pop(round(price, 8), None)

    def check_self_trade(
        self, symbol: str, price: float, side: str
    ) -> Tuple[bool, str]:
        """Reject if a resting order at the same price sits on the opposite side."""
        if not self._cfg.self_trade_prevention:
            return True, ""
        opposite = "sell" if side == "buy" else "buy"
        with self._lock:
            existing_side = self._open_orders.get(symbol, {}).get(round(price, 8))
        if existing_side == opposite:
            return False, (
                f"self_trade: open {opposite} at {price} would cross new {side} on {symbol}"
            )
        return True, ""

    # ------------------------------------------------------------------
    # G8: Notional cap per unit time
    # ------------------------------------------------------------------

    def check_notional_cap(self, order_notional: float) -> Tuple[bool, str]:
        """Reject if adding this order would breach the rolling hourly or daily cap."""
        cfg = self._cfg
        now = time.time()

        with self._lock:
            # Evict entries that have fallen outside each window
            hour_cutoff = now - 3600.0
            day_cutoff = now - 86400.0
            while self._hourly_window and self._hourly_window[0][0] < hour_cutoff:
                self._hourly_window.popleft()
            while self._daily_window and self._daily_window[0][0] < day_cutoff:
                self._daily_window.popleft()
            hourly_used = sum(n for _, n in self._hourly_window)
            daily_used = sum(n for _, n in self._daily_window)

        if hourly_used + order_notional > cfg.hourly_notional_cap:
            return False, (
                f"notional_cap:hourly: {hourly_used + order_notional:.2f} "
                f"> cap {cfg.hourly_notional_cap:.2f}"
            )

        if daily_used + order_notional > cfg.daily_notional_cap:
            return False, (
                f"notional_cap:daily: {daily_used + order_notional:.2f} "
                f"> cap {cfg.daily_notional_cap:.2f}"
            )

        return True, ""

    def _record_notional(self, notional: float) -> None:
        """Commit a notional amount into the sliding windows after order acceptance."""
        now = time.time()
        with self._lock:
            self._hourly_window.append((now, notional))
            self._daily_window.append((now, notional))

    # ------------------------------------------------------------------
    # G9: Wash trade guard
    # ------------------------------------------------------------------

    def check_wash_trade(self, symbol: str, side: str) -> Tuple[bool, str]:
        """Reject if the same symbol+direction was accepted within the cooldown window."""
        cooldown = self._cfg.wash_trade_cooldown_sec
        if cooldown <= 0.0:
            return True, ""
        now = time.time()
        with self._lock:
            last_ts = self._wash_guard.get(symbol, {}).get(side, 0.0)
        elapsed = now - last_ts
        if elapsed < cooldown:
            return False, (
                f"wash_trade: {side} {symbol} last seen {elapsed:.3f}s ago "
                f"< cooldown {cooldown:.1f}s"
            )
        return True, ""

    def _record_order_timestamp(self, symbol: str, side: str) -> None:
        """Stamp the current time as the last accepted order for this symbol+side."""
        if self._cfg.wash_trade_cooldown_sec <= 0.0:
            return
        now = time.time()
        with self._lock:
            self._wash_guard.setdefault(symbol, {})[side] = now

    # ------------------------------------------------------------------
    # Combined API
    # ------------------------------------------------------------------

    def check_all(
        self,
        symbol: str,
        side: str,
        order_notional: float,
        limit_price: Optional[float] = None,
        current_symbol_notional: float = 0.0,
        current_portfolio_notional: float = 0.0,
        leverage: float = 1.0,
    ) -> Tuple[bool, str]:
        """Run G6 → G7 → G8 → G9 in sequence.

        Returns the first failure as ``(False, reason)`` or ``(True, "")`` when
        all checks pass.
        """
        ok, reason = self.check_position_limits(
            symbol, order_notional,
            current_symbol_notional, current_portfolio_notional, leverage,
        )
        if not ok:
            return ok, reason

        if limit_price is not None:
            ok, reason = self.check_self_trade(symbol, limit_price, side)
            if not ok:
                return ok, reason

        ok, reason = self.check_notional_cap(order_notional)
        if not ok:
            return ok, reason

        ok, reason = self.check_wash_trade(symbol, side)
        if not ok:
            return ok, reason

        return True, ""

    def record_order(self, symbol: str, side: str, notional: float) -> None:
        """Commit state updates after a compliant order is accepted.

        Must be called exactly once per accepted order so that G8 and G9
        accumulators stay accurate.
        """
        self._record_notional(notional)
        self._record_order_timestamp(symbol, side)
