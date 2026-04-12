"""
Order Execution Manager: paper and live trading order lifecycle management.

Provides a unified interface for submitting, tracking, and cancelling orders
against a CCXT-compatible exchange (live mode) or an internal simulation
(paper mode).  Safety guards prevent runaway losses and API abuse.

Week 64 additions (Track C — Live Risk Enforcement):
  S41 — Correlation limit enforced at submission time via risk_manager
  S42 — FatFingerGuard: rejects abnormally large orders
  S43 — VolatilityCircuitBreaker: halts new orders on high volatility
  S44 — Idempotency key: prevents duplicate live orders on retry
  S45 — RateLimiter / ClockSync now in dedicated modules

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
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import numpy as np

from deployment.execution.position_tracker import PositionTracker
from deployment.execution.rate_limiter import RateLimiter
from deployment.execution.fat_finger_guard import FatFingerGuard
from deployment.execution.circuit_breaker import VolatilityCircuitBreaker
from deployment.execution.clock_sync import ClockSync

logger = logging.getLogger(__name__)

# AuditLogger is optional — imported lazily to avoid hard dep at module level.
try:
    from deployment.audit.audit_logger import AuditLogger as _AuditLogger
except ImportError:  # pragma: no cover
    _AuditLogger = None  # type: ignore[assignment,misc]

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
    idempotency_key: Optional[str] = None   # S44: duplicate-order prevention
    # S52: latency timestamps (submit → ack → fill)
    submitted_at: Optional[datetime] = None   # set in submit_order before execution
    acked_at: Optional[datetime] = None       # exchange ack (live) or immediate (paper)
    filled_at: Optional[datetime] = None      # fill completion


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
            exchange_id           – CCXT exchange id (default: "binance")
            api_key               – live mode only
            api_secret            – live mode only
            symbol                – trading pair (default: "BTC/USDT")
            max_order_size        – maximum single order size in base currency
            daily_loss_limit      – halt trading when daily P&L drops below this
            rate_limit_calls      – API calls per rate_limit_period (default: 10)
            rate_limit_period     – seconds for rate limit window (default: 1.0)
            correlation_threshold – correlation limit for S41 (default: 0.7)
            max_retries           – live order retry attempts (default: 3)
    paper_mode : bool
        When True (default), all orders are simulated locally.
    risk_manager : optional
        Any object with ``check_max_drawdown(peak, current) -> bool`` and
        optionally ``check_correlation(value, threshold) -> bool``.
        UnifiedRiskManager satisfies both.
    audit_logger : AuditLogger, optional
        Append-only audit trail receiver.
    fat_finger_guard : FatFingerGuard, optional
        Override the default guard (useful for testing).
    circuit_breaker : VolatilityCircuitBreaker, optional
        Override the default circuit breaker.
    clock_sync : ClockSync, optional
        Override the default clock sync checker.
    """

    def __init__(
        self,
        exchange_config: Optional[Dict[str, Any]] = None,
        paper_mode: bool = True,
        risk_manager=None,
        audit_logger=None,
        fat_finger_guard: Optional[FatFingerGuard] = None,
        circuit_breaker: Optional[VolatilityCircuitBreaker] = None,
        clock_sync: Optional[ClockSync] = None,
    ) -> None:
        exchange_config = exchange_config or {}
        self.paper_mode = paper_mode
        self._config = exchange_config
        self.symbol: str = exchange_config.get("symbol", "BTC/USDT")
        self.max_order_size: float = float(exchange_config.get("max_order_size", 1.0))
        self.daily_loss_limit: float = float(exchange_config.get("daily_loss_limit", -500.0))
        self._risk_manager = risk_manager
        self._audit_logger = audit_logger
        self._max_retries: int = int(exchange_config.get("max_retries", 3))

        # S41: correlation threshold (from config or default 0.7)
        self._correlation_threshold: float = float(
            exchange_config.get("correlation_threshold", 0.7)
        )
        self._current_correlation: Optional[float] = None

        self.rate_limiter = RateLimiter(
            max_calls=int(exchange_config.get("rate_limit_calls", 10)),
            period=float(exchange_config.get("rate_limit_period", 1.0)),
        )

        # S42: fat-finger guard
        self._fat_finger: FatFingerGuard = fat_finger_guard or FatFingerGuard(
            size_multiplier_limit=float(exchange_config.get("fat_finger_multiplier", 5.0)),
            hard_cap=float(exchange_config.get("fat_finger_hard_cap", 0.0)),
            lookback=int(exchange_config.get("fat_finger_lookback", 20)),
        )

        # S43: volatility circuit breaker
        self._circuit_breaker: VolatilityCircuitBreaker = circuit_breaker or VolatilityCircuitBreaker(
            vol_threshold=float(exchange_config.get("vol_threshold", 0.05)),
            window=int(exchange_config.get("vol_window", 20)),
            cooldown=float(exchange_config.get("vol_cooldown", 300.0)),
        )

        # S45: clock sync
        self._clock_sync: ClockSync = clock_sync or ClockSync(
            max_drift_sec=float(exchange_config.get("max_clock_drift_sec", 5.0)),
            halt_on_skew=bool(exchange_config.get("halt_on_clock_skew", False)),
        )

        self._lock = threading.RLock()
        self._orders: Dict[str, Order] = {}
        self._idempotency_map: Dict[str, str] = {}   # S44: key → order_id
        self._daily_pnl: float = 0.0
        self._last_reset_date: date = date.today()
        self._halted: bool = False
        # S52: latency tracking (submit-to-fill, ms)
        self._latency_samples: List[float] = []
        self._position_tracker = PositionTracker(
            initial_cash=float(exchange_config.get("initial_cash", 10_000.0))
        )

        if not paper_mode:
            self._exchange = self._init_exchange(exchange_config)
            self._clock_sync.set_exchange(self._exchange)
        else:
            self._exchange = None

        logger.info(
            "OrderManager initialised | symbol=%s paper_mode=%s max_order_size=%s",
            self.symbol, paper_mode, self.max_order_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_correlation(self, value: float) -> None:
        """Update the current inter-asset correlation used in S41 checks."""
        self._current_correlation = float(value)

    def submit_order(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        current_price: Optional[float] = None,
        price: Optional[float] = None,   # compat alias for current_price
        idempotency_key: Optional[str] = None,   # S44
    ) -> str:
        """Submit a new order.

        Returns order_id string.
        Raises RuntimeError if trading is halted.
        """
        self._reset_daily_pnl_if_needed()

        if self._halted:
            raise RuntimeError("Daily loss limit breached — trading halted.")

        # S45: clock skew halt
        if self._clock_sync.is_halted:
            raise RuntimeError("Trading halted due to excessive clock skew.")

        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")

        if amount <= 0:
            raise ValueError(f"Order amount must be positive, got {amount}.")

        # S44: idempotency check — return existing order if key already seen
        if idempotency_key is not None:
            with self._lock:
                existing_id = self._idempotency_map.get(idempotency_key)
                if existing_id is not None:
                    logger.info(
                        "submit_order: idempotency_key=%r already exists → returning %s",
                        idempotency_key, existing_id,
                    )
                    return existing_id

        # S41: correlation limit check
        if self._current_correlation is not None:
            corr_violated = self._check_correlation_limit(self._current_correlation)
            if corr_violated:
                order_id = self._reject_order(
                    side, amount, order_type, limit_price,
                    reason="correlation_limit",
                    idempotency_key=idempotency_key,
                )
                return order_id

        # S43: volatility circuit breaker — blocks new orders only
        if self._circuit_breaker.is_tripped():
            order_id = self._reject_order(
                side, amount, order_type, limit_price,
                reason="volatility_circuit_breaker",
                idempotency_key=idempotency_key,
            )
            return order_id

        # Pre-trade drawdown check (existing logic)
        if self._risk_manager is not None:
            tracker = self._position_tracker
            if tracker is not None:
                peak = tracker.peak_value
                current = tracker.portfolio_value
                if self._risk_manager.check_max_drawdown(peak, current):
                    order_id = self._reject_order(
                        side, amount, order_type, limit_price,
                        reason="max_drawdown",
                        idempotency_key=idempotency_key,
                    )
                    return order_id

        # S42: fat-finger guard
        ok, reason = self._fat_finger.check(amount)
        if not ok:
            order_id = self._reject_order(
                side, amount, order_type, limit_price,
                reason=f"fat_finger:{reason}",
                idempotency_key=idempotency_key,
            )
            return order_id

        if amount > self.max_order_size:
            logger.warning(
                "Order amount %.6f exceeds max_order_size %.6f; clamping.",
                amount, self.max_order_size,
            )
            amount = self.max_order_size

        # Accept `price` as alias for `current_price`
        if current_price is None and price is not None:
            current_price = price

        order_id = str(uuid.uuid4())[:8]
        _now = datetime.utcnow()
        order = Order(
            order_id=order_id,
            side=side,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            status="pending",
            idempotency_key=idempotency_key,
            submitted_at=_now,   # S52: latency start
        )
        with self._lock:
            self._orders[order_id] = order
            if idempotency_key is not None:
                self._idempotency_map[idempotency_key] = order_id

        # Audit: order submitted
        if self._audit_logger is not None:
            self._audit_logger.log_order(order)

        try:
            if self.paper_mode:
                self._execute_paper_order(order, current_price)
            else:
                self._execute_live_order(order)
        except Exception as e:
            order.status = "failed"
            order.updated_at = datetime.utcnow()
            logger.error("Order %s failed: %s", order_id, e)

        # S52: record fill latency (submit → fill)
        if order.filled_at is not None and order.submitted_at is not None:
            latency_ms = (
                (order.filled_at - order.submitted_at).total_seconds() * 1000.0
            )
            with self._lock:
                self._latency_samples.append(latency_ms)

        # Audit: fill (or failure) recorded after execution
        if self._audit_logger is not None:
            self._audit_logger.log_fill(order)

        # S42: record fill size for future fat-finger baselines
        if order.status == "filled" and order.filled_amount > 0:
            self._fat_finger.record_fill(order.filled_amount)

        return order_id

    def check_order(self, order_id: str) -> str:
        """Return current status of an order."""
        with self._lock:
            if order_id not in self._orders:
                logger.warning("check_order: unknown order_id %s", order_id)
                return "unknown"
            order = self._orders[order_id]
        if not self.paper_mode and order.status in ("pending", "partial"):
            self._refresh_live_order_status(order)
        return order.status

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""
        with self._lock:
            if order_id not in self._orders:
                logger.warning("cancel_order: unknown order_id %s", order_id)
                return False
            order = self._orders[order_id]
            if order.status in ("filled", "cancelled", "failed"):
                return False
            if self.paper_mode:
                order.status = "cancelled"
                order.updated_at = datetime.utcnow()
                return True
        try:
            self.rate_limiter.acquire()
            self._exchange.cancel_order(order.exchange_order_id, self.symbol)
            order.status = "cancelled"
            order.updated_at = datetime.utcnow()
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def reconcile(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """Compare internal state with exchange and return summary dict."""
        open_orders = sum(1 for o in self._orders.values() if o.status == "pending")
        internal_pos = self._position_tracker.position
        if self.paper_mode:
            return {
                "position": internal_pos,
                "daily_pnl": self._daily_pnl,
                "open_orders": open_orders,
                "internal_position": internal_pos,
                "actual_position": internal_pos,
                "discrepancy": 0.0,
                "ok": True,
                "mode": "paper",
            }
        try:
            self.rate_limiter.acquire()
            balance = self._exchange.fetch_balance()
            base, _ = self.symbol.split("/")
            actual_position = float(balance.get("free", {}).get(base, 0.0))
            discrepancy = abs(actual_position - internal_pos)
            ok = discrepancy < 0.001
            if not ok:
                logger.warning(
                    "Reconcile mismatch | internal=%.6f actual=%.6f",
                    internal_pos, actual_position,
                )
            return {
                "position": internal_pos,
                "daily_pnl": self._daily_pnl,
                "open_orders": open_orders,
                "internal_position": internal_pos,
                "actual_position": actual_position,
                "discrepancy": discrepancy,
                "ok": ok,
                "mode": "live",
            }
        except Exception as e:
            logger.error("Reconcile failed: %s", e)
            return {"ok": False, "error": str(e), "mode": "live",
                    "position": internal_pos, "daily_pnl": self._daily_pnl,
                    "open_orders": open_orders}

    def update_paper_price(self, price: float) -> None:
        """Notify manager of latest market price (paper mode).

        Also feeds the price to the circuit breaker for vol tracking (S43).
        """
        self._position_tracker.update_price(price)
        self._circuit_breaker.update(price)

    def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise KeyError(f"Unknown order_id: {order_id}")
        return self._orders[order_id]

    @property
    def daily_pnl(self) -> float:
        self._reset_daily_pnl_if_needed()
        return self._daily_pnl

    @property
    def is_halted(self) -> bool:
        return self._halted

    def compute_latency_percentiles(self) -> Dict[str, float]:
        """Return latency percentiles (p50/p95/p99) in milliseconds.

        Computed from all submit-to-fill samples collected since init.
        Returns zero values if no samples yet.
        """
        with self._lock:
            samples = list(self._latency_samples)
        if not samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0.0}
        return {
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)),
            "count": float(len(samples)),
        }

    @property
    def fat_finger_guard(self) -> FatFingerGuard:
        return self._fat_finger

    @property
    def circuit_breaker(self) -> VolatilityCircuitBreaker:
        return self._circuit_breaker

    @property
    def clock_sync(self) -> ClockSync:
        return self._clock_sync

    # ------------------------------------------------------------------
    # S41: Correlation limit helper
    # ------------------------------------------------------------------

    def _check_correlation_limit(self, correlation_value: float) -> bool:
        """
        Return True (= reject order) if correlation limit is breached.

        Uses risk_manager.check_correlation() if available, otherwise
        falls back to the stored threshold.
        """
        check_fn = getattr(self._risk_manager, "check_correlation", None)
        if callable(check_fn):
            breached = check_fn(correlation_value, self._correlation_threshold)
        else:
            breached = abs(correlation_value) > self._correlation_threshold

        if breached:
            logger.warning(
                "Order rejected: correlation %.4f exceeds threshold %.4f",
                correlation_value,
                self._correlation_threshold,
            )
            if self._audit_logger is not None:
                self._audit_logger.log_risk_event({
                    "type": "correlation_limit",
                    "correlation": correlation_value,
                    "threshold": self._correlation_threshold,
                })
        return breached

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reject_order(
        self,
        side: str,
        amount: float,
        order_type: str,
        limit_price: Optional[float],
        reason: str,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """Create a failed order record and log the rejection reason."""
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            order_id=order_id,
            side=side,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            status="failed",
            idempotency_key=idempotency_key,
        )
        with self._lock:
            self._orders[order_id] = order
            if idempotency_key is not None:
                self._idempotency_map[idempotency_key] = order_id

        logger.warning("Order %s rejected: %s", order_id, reason)
        if self._audit_logger is not None:
            self._audit_logger.log_risk_event({
                "type": "order_rejected",
                "order_id": order_id,
                "reason": reason,
                "side": side,
                "amount": amount,
            })
        return order_id

    # ------------------------------------------------------------------
    # Paper order execution
    # ------------------------------------------------------------------

    def _execute_paper_order(self, order: Order, current_price: Optional[float]) -> None:
        price = current_price or self._position_tracker.current_price
        if price <= 0:
            price = 1.0

        # S52: paper orders are instant — ack and fill happen together
        order.acked_at = datetime.utcnow()

        if order.side == "buy":
            fee = order.amount * price * 0.001
            self._position_tracker.apply_buy(
                quantity=order.amount, price=price, fee=fee
            )
            order.filled_amount = order.amount
            order.avg_fill_price = price
            order.fee = fee
        else:
            sell_qty = min(order.amount, self._position_tracker.position)
            if sell_qty < 1e-9:
                order.status = "failed"
                order.updated_at = datetime.utcnow()
                order.filled_at = order.updated_at
                return
            proceeds = sell_qty * price
            fee = proceeds * 0.001
            # Capture entry price BEFORE apply_sell (resets on full close)
            entry_price = self._position_tracker.entry_price
            self._position_tracker.apply_sell(
                quantity=sell_qty, price=price, fee=fee
            )
            order.filled_amount = sell_qty
            order.avg_fill_price = price
            order.fee = fee
            pnl = (price - entry_price) * sell_qty if entry_price > 0 else 0.0
            with self._lock:
                self._daily_pnl += (pnl - fee)
            self._check_daily_loss_limit()

        order.status = "filled"
        order.updated_at = datetime.utcnow()
        order.filled_at = order.updated_at   # S52

    # ------------------------------------------------------------------
    # Live order execution — S44: idempotency key + improved backoff
    # ------------------------------------------------------------------

    def _execute_live_order(self, order: Order) -> None:
        backoff = 1.0
        for attempt in range(1, self._max_retries + 1):
            try:
                self.rate_limiter.acquire()
                params: Dict[str, Any] = {}
                if order.idempotency_key is not None:
                    # Standard CCXT client order id field (exchange may ignore)
                    params["clientOrderId"] = order.idempotency_key

                if order.order_type == "market":
                    result = self._exchange.create_market_order(
                        self.symbol, order.side, order.amount, params=params
                    )
                else:
                    result = self._exchange.create_limit_order(
                        self.symbol, order.side, order.amount, order.limit_price,
                        params=params,
                    )
                order.exchange_order_id = result.get("id")
                order.acked_at = datetime.utcnow()   # S52: exchange has the order
                order.status = "filled" if result.get("status") == "closed" else "pending"
                order.filled_amount = float(result.get("filled", 0.0))
                order.avg_fill_price = float(result.get("average") or result.get("price") or 0.0)
                fee_info = result.get("fee") or {}
                order.fee = float(fee_info.get("cost", 0.0))
                order.updated_at = datetime.utcnow()
                if order.status == "filled":
                    order.filled_at = order.updated_at   # S52
                # Update position tracker on successful fill
                if order.status == "filled" and self._position_tracker is not None:
                    if order.side == "buy":
                        self._position_tracker.apply_buy(
                            quantity=order.filled_amount,
                            price=order.avg_fill_price,
                            fee=order.fee,
                        )
                    else:
                        self._position_tracker.apply_sell(
                            quantity=order.filled_amount,
                            price=order.avg_fill_price,
                            fee=order.fee,
                        )
                return
            except Exception as e:
                logger.warning(
                    "Live order attempt %d/%d failed: %s",
                    attempt, self._max_retries, e,
                )
                if attempt < self._max_retries:
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
            logger.warning("Failed to refresh order %s: %s", order.order_id, e)

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _check_daily_loss_limit(self) -> None:
        with self._lock:
            if self._daily_pnl <= self.daily_loss_limit:
                self._halted = True
                logger.warning(
                    "Daily loss limit reached: pnl=%.2f limit=%.2f. Trading halted.",
                    self._daily_pnl, self.daily_loss_limit,
                )

    def _reset_daily_pnl_if_needed(self) -> None:
        today = date.today()
        with self._lock:
            if today != self._last_reset_date:
                self._daily_pnl = 0.0
                self._halted = False
                self._last_reset_date = today

    # ------------------------------------------------------------------
    # Exchange init
    # ------------------------------------------------------------------

    def _init_exchange(self, config: Dict[str, Any]):
        try:
            import ccxt
            exchange_id = config.get("exchange_id", "binance")
            exchange_class = getattr(ccxt, exchange_id)
            return exchange_class({
                "apiKey": config.get("api_key", ""),
                "secret": config.get("api_secret", ""),
                "enableRateLimit": True,
            })
        except ImportError:
            if not self.paper_mode:
                raise RuntimeError(
                    "ccxt not installed but paper_mode=False. "
                    "Install ccxt or set paper_mode=True."
                )
            logger.info("ccxt not installed; using paper mode")
            return None
        except Exception as e:
            if not self.paper_mode:
                raise RuntimeError(f"Exchange init failed: {e}. Check API credentials.")
            logger.warning("Exchange init failed (%s); using paper mode", e)
            return None

    def __enter__(self) -> "OrderManager":
        return self

    def __exit__(self, *_) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass
