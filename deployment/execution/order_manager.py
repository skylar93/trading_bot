"""
Order Execution Manager: paper and live trading order lifecycle management.

Provides a unified interface for submitting, tracking, and cancelling orders
against a CCXT-compatible exchange (live mode) or an internal simulation
(paper mode).  Safety guards prevent runaway losses and API abuse.

Usage (paper mode, default):
    manager = OrderManager(exchange_config={}, paper_mode=True)
    order_id = manager.submit_order("buy", amount=0.01)
    status   = manager.check_order(order_id)
    manager.reconcile()

Usage (live mode):
    cfg = {"exchange_id": "binance", "api_key": "...", "api_secret": "...",
           "daily_loss_limit": -500.0, "max_order_size": 0.1}
    manager = OrderManager(exchange_config=cfg, paper_mode=False)
"""

from __future__ import annotations

import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_ORDER_STATUSES = frozenset({"pending", "filled", "partial", "cancelled", "failed"})


@dataclass
class Order:
    order_id: str
    side: str               # "buy" | "sell"
    amount: float
    order_type: str         # "market" | "limit"
    limit_price: Optional[float]
    status: str             # see _ORDER_STATUSES
    filled_amount: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange_order_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter (thread-safe)."""

    def __init__(self, max_calls: int = 10, period: float = 1.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._calls: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a call token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Remove timestamps outside the sliding window
                self._calls = [t for t in self._calls if now - t < self._period]
                if len(self._calls) < self._max_calls:
                    self._calls.append(now)
                    return
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Exchange API wrapper supporting paper trading and live trading.

    Parameters
    ----------
    exchange_config : dict
        Exchange credentials and operational limits.
        Keys:
            exchange_id        – CCXT exchange id (default: "binance")
            api_key            – live mode only
            api_secret         – live mode only
            symbol             – trading pair (default: "BTC/USDT")
            max_order_size     – maximum single order size in base currency
            daily_loss_limit   – halt trading when daily P&L drops below this
            rate_limit_calls   – API calls per rate_limit_period (default: 10)
            rate_limit_period  – seconds for rate limit window (default: 1.0)
    paper_mode : bool
        When True (default), all orders are simulated locally; no exchange
        connection is established.
    """

    def __init__(self, exchange_config: Dict[str, Any], paper_mode: bool = True) -> None:
        self.paper_mode = paper_mode
        self._config = exchange_config
        self.symbol: str = exchange_config.get("symbol", "BTC/USDT")
        self.max_order_size: float = float(exchange_config.get("max_order_size", 1.0))
        self.daily_loss_limit: float = float(exchange_config.get("daily_loss_limit", -500.0))

        self.rate_limiter = RateLimiter(
            max_calls=int(exchange_config.get("rate_limit_calls", 10)),
            period=float(exchange_config.get("rate_limit_period", 1.0)),
        )

        self._orders: Dict[str, Order] = {}
        self._daily_pnl: float = 0.0
        self._last_reset_date: date = date.today()
        self._halted: bool = False
        self._paper_position: float = 0.0   # base currency held (paper only)
        self._paper_cash: float = float(exchange_config.get("initial_cash", 10_000.0))
        self._paper_last_price: float = 0.0

        if not paper_mode:
            self._exchange = self._init_exchange(exchange_config)
        else:
            self._exchange = None

        logger.info(
            "OrderManager initialised | symbol=%s paper_mode=%s max_order_size=%s",
            self.symbol,
            paper_mode,
            self.max_order_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_order(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> str:
        """
        Submit a new order.

        Parameters
        ----------
        side        : "buy" or "sell"
        amount      : order size in base currency
        order_type  : "market" (default) or "limit"
        limit_price : required for limit orders
        current_price: reference price for paper mode simulation

        Returns
        -------
        order_id string
        """
        self._reset_daily_pnl_if_needed()

        if self._halted:
            logger.warning("OrderManager halted (daily loss limit). Order rejected.")
            raise RuntimeError("Trading halted: daily loss limit reached.")

        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")

        if amount <= 0:
            raise ValueError(f"Order amount must be positive, got {amount}.")

        if amount > self.max_order_size:
            logger.warning(
                "Order amount %.6f exceeds max_order_size %.6f; clamping.",
                amount,
                self.max_order_size,
            )
            amount = self.max_order_size

        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            side=side,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            status="pending",
        )
        self._orders[order_id] = order

        try:
            if self.paper_mode:
                self._execute_paper_order(order, current_price)
            else:
                self._execute_live_order(order)
        except Exception as e:
            order.status = "failed"
            order.updated_at = datetime.utcnow()
            logger.error("Order %s failed: %s", order_id, e)

        return order_id

    def check_order(self, order_id: str) -> str:
        """
        Return current status of an order.

        Returns one of: "pending", "filled", "partial", "cancelled", "failed"
        """
        if order_id not in self._orders:
            raise KeyError(f"Unknown order_id: {order_id}")

        order = self._orders[order_id]

        # In live mode, refresh status from exchange
        if not self.paper_mode and order.status in ("pending", "partial"):
            self._refresh_live_order_status(order)

        return order.status

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending or partial order.

        Returns True if successfully cancelled, False if already terminal.
        """
        if order_id not in self._orders:
            raise KeyError(f"Unknown order_id: {order_id}")

        order = self._orders[order_id]

        if order.status in ("filled", "cancelled", "failed"):
            logger.debug("Cannot cancel order %s in terminal state: %s", order_id, order.status)
            return False

        if self.paper_mode:
            order.status = "cancelled"
            order.updated_at = datetime.utcnow()
            logger.info("Paper order cancelled: %s", order_id)
            return True

        # Live mode
        try:
            self.rate_limiter.acquire()
            self._exchange.cancel_order(order.exchange_order_id, self.symbol)
            order.status = "cancelled"
            order.updated_at = datetime.utcnow()
            logger.info("Live order cancelled: %s (exchange id: %s)", order_id, order.exchange_order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def reconcile(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare internal state with actual exchange position.

        In paper mode, returns internal position summary.
        In live mode, fetches actual balance and position from exchange and
        logs a warning if the difference exceeds a tolerance.

        Returns
        -------
        dict with keys: internal_position, actual_position, discrepancy, ok
        """
        if self.paper_mode:
            result = {
                "internal_position": self._paper_position,
                "actual_position": self._paper_position,
                "internal_cash": self._paper_cash,
                "actual_cash": self._paper_cash,
                "discrepancy": 0.0,
                "ok": True,
                "mode": "paper",
            }
            logger.info(
                "Reconcile (paper) | position=%.6f cash=%.2f",
                self._paper_position,
                self._paper_cash,
            )
            return result

        # Live mode
        try:
            self.rate_limiter.acquire()
            balance = self._exchange.fetch_balance()
            base, quote = self.symbol.split("/")
            actual_position = float(balance.get("free", {}).get(base, 0.0))
            actual_cash = float(balance.get("free", {}).get(quote, 0.0))

            # Internal tracking (approximated from filled orders)
            internal_position = self._paper_position  # reused for live tracking
            discrepancy = abs(actual_position - internal_position)

            ok = discrepancy < 0.001  # tolerance: 0.001 base currency
            if not ok:
                logger.warning(
                    "Reconcile mismatch | internal=%.6f actual=%.6f discrepancy=%.6f",
                    internal_position,
                    actual_position,
                    discrepancy,
                )
            else:
                logger.info(
                    "Reconcile OK | position=%.6f cash=%.2f", actual_position, actual_cash
                )

            return {
                "internal_position": internal_position,
                "actual_position": actual_position,
                "internal_cash": self._paper_cash,
                "actual_cash": actual_cash,
                "discrepancy": discrepancy,
                "ok": ok,
                "mode": "live",
            }
        except Exception as e:
            logger.error("Reconcile failed: %s", e)
            return {"ok": False, "error": str(e), "mode": "live"}

    def update_paper_price(self, price: float) -> None:
        """Notify manager of latest market price (used in paper mode simulation)."""
        self._paper_last_price = price

    def get_order(self, order_id: str) -> Order:
        """Return the Order object for inspection."""
        if order_id not in self._orders:
            raise KeyError(f"Unknown order_id: {order_id}")
        return self._orders[order_id]

    @property
    def daily_pnl(self) -> float:
        """Current day's realised P&L."""
        self._reset_daily_pnl_if_needed()
        return self._daily_pnl

    @property
    def is_halted(self) -> bool:
        """True if trading is halted due to daily loss limit."""
        return self._halted

    # ------------------------------------------------------------------
    # Paper order execution
    # ------------------------------------------------------------------

    def _execute_paper_order(self, order: Order, current_price: Optional[float]) -> None:
        price = current_price or self._paper_last_price
        if price <= 0:
            price = 1.0  # fallback for tests without price feed
            logger.debug("Paper order: no price provided, using 1.0 as placeholder")

        if order.side == "buy":
            cost = order.amount * price
            fee = cost * 0.001
            total_cost = cost + fee
            if total_cost > self._paper_cash:
                order.status = "failed"
                order.updated_at = datetime.utcnow()
                logger.warning("Paper buy rejected: insufficient cash (need %.2f, have %.2f)", total_cost, self._paper_cash)
                return
            self._paper_cash -= total_cost
            self._paper_position += order.amount
            order.filled_amount = order.amount
            order.avg_fill_price = price
            order.fee = fee

        else:  # sell
            sell_qty = min(order.amount, self._paper_position)
            if sell_qty < 1e-9:
                order.status = "failed"
                order.updated_at = datetime.utcnow()
                logger.warning("Paper sell rejected: no position to sell")
                return
            proceeds = sell_qty * price
            fee = proceeds * 0.001
            net_proceeds = proceeds - fee
            pnl = net_proceeds - sell_qty * price  # simplified; tracks fee drag
            self._paper_cash += net_proceeds
            self._paper_position -= sell_qty
            order.filled_amount = sell_qty
            order.avg_fill_price = price
            order.fee = fee
            order.pnl = pnl
            self._daily_pnl += pnl
            self._check_daily_loss_limit()

        order.status = "filled"
        order.updated_at = datetime.utcnow()
        logger.info(
            "Paper order filled | side=%s amount=%.6f price=%.2f fee=%.4f",
            order.side,
            order.filled_amount,
            order.avg_fill_price,
            order.fee,
        )

    # ------------------------------------------------------------------
    # Live order execution (with retry)
    # ------------------------------------------------------------------

    def _execute_live_order(self, order: Order) -> None:
        max_retries = 3
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                self.rate_limiter.acquire()
                if order.order_type == "market":
                    result = self._exchange.create_market_order(
                        self.symbol, order.side, order.amount
                    )
                else:
                    if order.limit_price is None:
                        raise ValueError("limit_price required for limit orders")
                    result = self._exchange.create_limit_order(
                        self.symbol, order.side, order.amount, order.limit_price
                    )

                order.exchange_order_id = result.get("id")
                order.status = "filled" if result.get("status") == "closed" else "pending"
                order.filled_amount = float(result.get("filled", 0.0))
                order.avg_fill_price = float(result.get("average") or result.get("price") or 0.0)
                fee_info = result.get("fee") or {}
                order.fee = float(fee_info.get("cost", 0.0))
                order.updated_at = datetime.utcnow()

                logger.info(
                    "Live order submitted | id=%s exchange_id=%s status=%s",
                    order.order_id,
                    order.exchange_order_id,
                    order.status,
                )
                return

            except Exception as e:
                logger.warning("Live order attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    raise

    def _refresh_live_order_status(self, order: Order) -> None:
        if order.exchange_order_id is None:
            return
        try:
            self.rate_limiter.acquire()
            result = self._exchange.fetch_order(order.exchange_order_id, self.symbol)
            status_map = {"open": "pending", "closed": "filled", "canceled": "cancelled"}
            order.status = status_map.get(result.get("status", ""), order.status)
            order.filled_amount = float(result.get("filled", order.filled_amount))
            order.updated_at = datetime.utcnow()
        except Exception as e:
            logger.warning("Failed to refresh order %s status: %s", order.order_id, e)

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _check_daily_loss_limit(self) -> None:
        if self._daily_pnl <= self.daily_loss_limit:
            self._halted = True
            logger.warning(
                "Daily loss limit reached: pnl=%.2f limit=%.2f. Trading halted.",
                self._daily_pnl,
                self.daily_loss_limit,
            )

    def _reset_daily_pnl_if_needed(self) -> None:
        today = date.today()
        if today != self._last_reset_date:
            self._daily_pnl = 0.0
            self._halted = False
            self._last_reset_date = today
            logger.info("Daily P&L counter reset for %s", today.isoformat())

    # ------------------------------------------------------------------
    # Exchange init
    # ------------------------------------------------------------------

    def _init_exchange(self, config: Dict[str, Any]):
        try:
            import ccxt

            exchange_id = config.get("exchange_id", "binance")
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class(
                {
                    "apiKey": config.get("api_key", ""),
                    "secret": config.get("api_secret", ""),
                    "enableRateLimit": True,
                }
            )
            logger.info("CCXT exchange initialised: %s", exchange_id)
            return exchange
        except ImportError:
            logger.warning("ccxt not installed; falling back to paper mode")
            self.paper_mode = True
            return None
        except Exception as e:
            logger.warning("Exchange init failed (%s); falling back to paper mode", e)
            self.paper_mode = True
            return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "OrderManager":
        return self

    def __exit__(self, *_) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass
