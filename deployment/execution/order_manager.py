"""
Order Execution Manager: Paper trading과 live 모두 지원하는 Exchange API 래퍼.

Paper mode (기본값)에서는 in-memory 시뮬레이션만 실행하며 실제 API 호출 없음.
Live mode에서는 ccxt 라이브러리가 필요 (선택적 의존성).

Usage
-----
    from deployment.execution.order_manager import OrderManager

    # Paper mode
    mgr = OrderManager({"daily_loss_limit": -500.0}, paper_mode=True)
    order_id = mgr.submit_order("buy", amount=0.01)
    status = mgr.check_order(order_id)    # "filled" in paper mode

    # Reconcile
    info = mgr.reconcile()
    print(info["position"])
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple token-bucket rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter (thread-safe)."""

    def __init__(self, max_calls: int = 10, period: float = 1.0) -> None:
        self.max_calls = max_calls
        self.period = period
        self._timestamps: List[float] = []
        self._lock = Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            # Remove timestamps outside the window
            self._timestamps = [t for t in self._timestamps if now - t < self.period]
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self._timestamps = self._timestamps[1:]
            self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Order record
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    side: str                   # "buy" | "sell"
    amount: float
    order_type: str = "market"
    status: str = "pending"     # pending | filled | partial | failed | cancelled
    filled_amount: float = 0.0
    fill_price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: str = ""


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """Exchange order management with paper and live mode.

    Parameters
    ----------
    exchange_config : dict
        Configuration dict.  Key ``daily_loss_limit`` (float, default -500.0)
        sets the daily loss threshold.  In live mode ``api_key`` and
        ``api_secret`` are also required.
    paper_mode : bool
        If ``True`` (default) all orders are simulated in-memory.
    """

    def __init__(
        self,
        exchange_config: Optional[Dict] = None,
        paper_mode: bool = True,
    ) -> None:
        exchange_config = exchange_config or {}
        self.paper_mode = paper_mode
        self.daily_loss_limit: float = float(
            exchange_config.get("daily_loss_limit", -500.0)
        )
        self.max_order_size: float = float(
            exchange_config.get("max_order_size", 1.0)
        )
        self.rate_limiter = RateLimiter(
            max_calls=int(exchange_config.get("rate_limit_calls", 10)),
            period=float(exchange_config.get("rate_limit_period", 1.0)),
        )

        # Internal state
        self.daily_pnl: float = 0.0
        self._orders: Dict[str, Order] = {}
        self._position: float = 0.0          # units held
        self._last_price: float = 0.0        # used for paper PnL calc
        self._lock = Lock()

        if not paper_mode:
            self._init_live(exchange_config)
        else:
            self._exchange = None
            logger.info("OrderManager initialised in PAPER mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_order(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> str:
        """Submit a buy or sell order.

        Parameters
        ----------
        side : str
            ``"buy"`` or ``"sell"``.
        amount : float
            Order size in base currency units.
        order_type : str
            ``"market"`` (default) or ``"limit"``.
        price : float, optional
            Required for limit orders.

        Returns
        -------
        str
            Unique order ID.

        Raises
        ------
        RuntimeError
            If daily loss limit has been breached.
        """
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

        if self.daily_pnl < self.daily_loss_limit:
            raise RuntimeError(
                f"Daily loss limit breached ({self.daily_pnl:.2f} < "
                f"{self.daily_loss_limit:.2f}) — trading halted."
            )

        amount = min(float(amount), self.max_order_size)
        order_id = str(uuid.uuid4())[:8]

        self.rate_limiter.acquire()

        if self.paper_mode:
            order = self._paper_fill(order_id, side, amount, order_type, price)
        else:
            order = self._live_submit(order_id, side, amount, order_type, price)

        with self._lock:
            self._orders[order_id] = order

        logger.debug(
            "Order %s submitted: %s %s @ %s → %s",
            order_id, side, amount, order_type, order.status,
        )
        return order_id

    def check_order(self, order_id: str) -> str:
        """Return the current status of an order.

        Returns
        -------
        str
            One of ``"pending"``, ``"filled"``, ``"partial"``,
            ``"failed"``, ``"cancelled"``.
        """
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("check_order: unknown order_id %s", order_id)
            return "unknown"
        if not self.paper_mode and order.status == "pending":
            self._refresh_live_order(order)
        return order.status

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Returns
        -------
        bool
            ``True`` if successfully cancelled.
        """
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("cancel_order: unknown order_id %s", order_id)
            return False
        if order.status != "pending":
            logger.warning(
                "cancel_order: order %s is already '%s'", order_id, order.status
            )
            return False

        if self.paper_mode:
            order.status = "cancelled"
            return True
        else:
            return self._live_cancel(order)

    def reconcile(self) -> Dict:
        """Compare internal position state vs exchange and warn on mismatch.

        Returns
        -------
        dict
            ``{"position": float, "daily_pnl": float, "open_orders": int,
               "mismatch": bool}``
        """
        with self._lock:
            open_orders = sum(
                1 for o in self._orders.values() if o.status == "pending"
            )
            internal = {
                "position": self._position,
                "daily_pnl": self.daily_pnl,
                "open_orders": open_orders,
                "mismatch": False,
            }

        if not self.paper_mode and self._exchange is not None:
            try:
                exchange_pos = self._fetch_exchange_position()
                if abs(exchange_pos - self._position) > 1e-6:
                    logger.warning(
                        "Position mismatch: internal=%.6f, exchange=%.6f",
                        self._position, exchange_pos,
                    )
                    internal["mismatch"] = True
            except Exception as exc:
                logger.error("reconcile: exchange query failed: %s", exc)

        return internal

    # ------------------------------------------------------------------
    # Paper mode helpers
    # ------------------------------------------------------------------

    def _paper_fill(
        self,
        order_id: str,
        side: str,
        amount: float,
        order_type: str,
        price: Optional[float],
    ) -> Order:
        fill_price = price if price is not None else self._last_price
        if fill_price <= 0:
            fill_price = 100.0   # fallback for tests with no live price

        order = Order(
            order_id=order_id,
            side=side,
            amount=amount,
            order_type=order_type,
            status="filled",
            filled_amount=amount,
            fill_price=fill_price,
        )

        # Update internal position
        with self._lock:
            if side == "buy":
                self._position += amount
                self.daily_pnl -= amount * fill_price   # cash out
            else:
                pnl = (fill_price - self._last_price) * min(amount, self._position)
                self._position = max(0.0, self._position - amount)
                self.daily_pnl += pnl

        return order

    # ------------------------------------------------------------------
    # Live mode helpers (ccxt)
    # ------------------------------------------------------------------

    def _init_live(self, config: Dict) -> None:
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is required for live trading: pip install ccxt"
            )
        symbol_map = {
            "binance": ccxt.binance,
        }
        exchange_name = config.get("exchange", "binance").lower()
        cls = symbol_map.get(exchange_name)
        if cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        self._exchange = cls(
            {
                "apiKey": config.get("api_key", ""),
                "secret": config.get("api_secret", ""),
                "enableRateLimit": True,
            }
        )
        logger.info("OrderManager initialised in LIVE mode (%s)", exchange_name)

    def _live_submit(
        self,
        order_id: str,
        side: str,
        amount: float,
        order_type: str,
        price: Optional[float],
    ) -> Order:
        symbol = "BTC/USDT"   # TODO: make configurable
        for attempt in range(3):
            try:
                if order_type == "market":
                    resp = self._exchange.create_market_order(symbol, side, amount)
                else:
                    resp = self._exchange.create_limit_order(symbol, side, amount, price)
                return Order(
                    order_id=resp.get("id", order_id),
                    side=side,
                    amount=amount,
                    order_type=order_type,
                    status="pending",
                )
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Live order attempt %d failed: %s — retrying in %ds",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
        return Order(
            order_id=order_id, side=side, amount=amount,
            order_type=order_type, status="failed", error="max retries exceeded"
        )

    def _refresh_live_order(self, order: Order) -> None:
        try:
            resp = self._exchange.fetch_order(order.order_id)
            order.status = resp.get("status", order.status)
            order.filled_amount = float(resp.get("filled", order.filled_amount))
        except Exception as exc:
            logger.error("refresh_live_order failed: %s", exc)

    def _live_cancel(self, order: Order) -> bool:
        try:
            self._exchange.cancel_order(order.order_id)
            order.status = "cancelled"
            return True
        except Exception as exc:
            logger.error("live_cancel failed: %s", exc)
            return False

    def _fetch_exchange_position(self) -> float:
        try:
            balance = self._exchange.fetch_balance()
            btc = balance.get("BTC", {}).get("total", 0.0)
            return float(btc)
        except Exception:
            return self._position
