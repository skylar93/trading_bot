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

Week 74 additions (Track F — Execution Realism):
  F12 — Order types: limit, stop_loss_limit, take_profit (paper + live)
  F13 — Partial fill simulation (paper) + proper status mapping (live)
  F14 — Cancel-replace, per-order TTL with background expiry thread

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
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from deployment.execution.position_tracker import PositionTracker
from deployment.execution.rate_limiter import RateLimiter
from deployment.execution.fat_finger_guard import FatFingerGuard
from deployment.execution.circuit_breaker import VolatilityCircuitBreaker
from deployment.execution.clock_sync import ClockSync
from risk_management.limits import PreTradeComplianceChecker
from deployment.monitoring.tracing import start_span, record_order_latency

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

# F12: supported order types
_ORDER_TYPES = frozenset({"market", "limit", "stop_loss_limit", "take_profit"})


@dataclass
class Order:
    order_id: str
    side: str               # "buy" | "sell"
    amount: float
    order_type: str         # see _ORDER_TYPES
    limit_price: Optional[float]
    status: str             # see _ORDER_STATUSES
    stop_price: Optional[float] = None     # F12: trigger price for stop/take-profit orders
    filled_amount: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None  # F14: TTL expiry timestamp
    exchange_order_id: Optional[str] = None
    idempotency_key: Optional[str] = None   # S44: duplicate-order prevention
    # S52: latency timestamps (submit → ack → fill)
    submitted_at: Optional[datetime] = None   # set in submit_order before execution
    acked_at: Optional[datetime] = None       # exchange ack (live) or immediate (paper)
    filled_at: Optional[datetime] = None      # fill completion
    fills: List[Dict[str, Any]] = field(default_factory=list)  # F13: individual fill events


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
            exchange_mode         – "paper" | "sandbox" | "live"  (default: "paper")
                                    Overrides paper_mode when present.
                                    sandbox: connects to exchange testnet.
                                    live:    connects to real exchange.
            api_key               – required for sandbox/live mode
            api_secret            – required for sandbox/live mode
            symbol                – trading pair (default: "BTC/USDT")
            max_order_size        – maximum single order size in base currency
            daily_loss_limit      – halt trading when daily P&L drops below this
            rate_limit_calls      – API calls per rate_limit_period (default: 10)
            rate_limit_period     – seconds for rate limit window (default: 1.0)
            correlation_threshold – correlation limit for S41 (default: 0.7)
            max_retries           – live order retry attempts (default: 3)
            partial_fill_sim      – F13: enable partial fill simulation in paper mode
            partial_fill_min_ratio– F13: minimum fill ratio when simulation enabled (default: 0.3)
            order_ttl_sec         – F14: pending order TTL in seconds (0 = disabled)
            order_ttl_check_interval_sec – F14: expiry check interval (default: 10.0)
    paper_mode : bool
        When True (default), all orders are simulated locally.
    risk_manager : optional
        Any object with ``check_drawdown(peak, current) -> bool`` and
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
    fee_model : optional
        F16: FeeModel instance.  If provided, overrides hardcoded 0.1% paper fee.
    alerter : optional
        Trading alerter; used to emit cancel-failure alerts (F14).
    compliance_checker : PreTradeComplianceChecker, optional
        G6-G9: pre-trade compliance enforcement (position limits, self-trade
        prevention, notional caps, wash trade guard).
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
        fee_model=None,
        alerter=None,
        compliance_checker: Optional[PreTradeComplianceChecker] = None,
    ) -> None:
        exchange_config = exchange_config or {}
        # F3: exchange_mode ("paper" | "sandbox" | "live") overrides paper_mode bool.
        _mode = exchange_config.get("exchange_mode")
        if _mode is not None:
            paper_mode = (_mode == "paper")
        self._exchange_mode: str = _mode if _mode is not None else ("paper" if paper_mode else "live")
        self.paper_mode = paper_mode
        self._config = exchange_config
        self.symbol: str = exchange_config.get("symbol", "BTC/USDT")
        self.max_order_size: float = float(exchange_config.get("max_order_size", 1.0))
        self.daily_loss_limit: float = float(exchange_config.get("daily_loss_limit", -500.0))
        self._risk_manager = risk_manager
        self._audit_logger = audit_logger
        self._max_retries: int = int(exchange_config.get("max_retries", 3))
        self._fee_model = fee_model
        self._alerter = alerter
        # G6-G9: pre-trade compliance checker (Week 76)
        self._compliance_checker: Optional[PreTradeComplianceChecker] = compliance_checker

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

        # F13: partial fill simulation
        self._partial_fill_sim: bool = bool(exchange_config.get("partial_fill_sim", False))
        self._partial_fill_min_ratio: float = float(
            exchange_config.get("partial_fill_min_ratio", 0.3)
        )

        # F14: TTL-based order expiry
        self._order_ttl_sec: float = float(exchange_config.get("order_ttl_sec", 0.0))
        self._ttl_check_interval: float = float(
            exchange_config.get("order_ttl_check_interval_sec", 10.0)
        )

        self._lock = threading.RLock()
        self._orders: Dict[str, Order] = {}
        self._idempotency_map: Dict[str, str] = {}   # S44: key → order_id
        self._daily_pnl: float = 0.0
        self._last_reset_date: date = date.today()
        self._halted: bool = False
        # S52: latency tracking (submit-to-fill, ms)
        self._latency_samples: List[float] = []
        # F11: throttle clock-skew checks to avoid per-order exchange round trips
        self._clock_check_interval: float = float(
            exchange_config.get("clock_check_interval_sec", 30.0)
        )
        self._last_clock_check_at: float = 0.0
        self._position_tracker = PositionTracker(
            initial_cash=float(exchange_config.get("initial_cash", 10_000.0))
        )

        if not paper_mode:
            self._exchange = self._init_exchange(exchange_config)
            self._clock_sync.set_exchange(self._exchange)
        else:
            self._exchange = None

        # F14: background expiry thread
        self._stop_event = threading.Event()
        if self._order_ttl_sec > 0:
            self._expiry_thread = threading.Thread(
                target=self._order_expiry_worker, daemon=True, name="order-expiry"
            )
            self._expiry_thread.start()
        else:
            self._expiry_thread = None

        logger.info(
            "OrderManager initialised | symbol=%s mode=%s max_order_size=%s partial_fill_sim=%s ttl=%.0fs",
            self.symbol, self._exchange_mode, self.max_order_size,
            self._partial_fill_sim, self._order_ttl_sec,
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
        stop_price: Optional[float] = None,          # F12
        current_price: Optional[float] = None,
        price: Optional[float] = None,   # compat alias for current_price
        idempotency_key: Optional[str] = None,       # S44
        ttl_sec: Optional[float] = None,             # F14: per-order TTL override
    ) -> str:
        """Submit a new order.

        Returns order_id string.
        Raises RuntimeError if trading is halted.
        """
        _submit_start = time.monotonic()
        with start_span(
            "trading.order.submit",
            attributes={"symbol": self.symbol, "side": side, "amount": amount, "order_type": order_type},
        ) as _parent_span:
            return self._submit_order_inner(
                _parent_span, _submit_start,
                side=side, amount=amount, order_type=order_type,
                limit_price=limit_price, stop_price=stop_price,
                current_price=current_price, price=price,
                idempotency_key=idempotency_key, ttl_sec=ttl_sec,
            )

    def _submit_order_inner(
        self,
        _parent_span: Any,
        _submit_start: float,
        *,
        side: str,
        amount: float,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        current_price: Optional[float],
        price: Optional[float],
        idempotency_key: Optional[str],
        ttl_sec: Optional[float],
    ) -> str:
        """Inner implementation of submit_order — called inside the OTel parent span."""
        self._reset_daily_pnl_if_needed()

        if self._halted:
            raise RuntimeError("Daily loss limit breached — trading halted.")

        # F11: proactively measure clock drift (throttled to avoid per-order RTT).
        _now = time.monotonic()
        if not self.paper_mode and _now - self._last_clock_check_at >= self._clock_check_interval:
            self._clock_sync.check()
            self._last_clock_check_at = _now

        # S45: clock skew halt
        if self._clock_sync.is_halted:
            raise RuntimeError("Trading halted due to excessive clock skew.")

        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")

        if amount <= 0:
            raise ValueError(f"Order amount must be positive, got {amount}.")

        # F12: validate order type
        if order_type not in _ORDER_TYPES:
            raise ValueError(
                f"Invalid order_type: {order_type!r}. Must be one of {sorted(_ORDER_TYPES)}."
            )

        # S44: idempotency check — atomic get-or-reserve using setdefault.
        _pre_order_id = str(uuid.uuid4())[:8]
        with start_span("trading.order.idempotency_lookup") as _idem_span:
            if idempotency_key is not None:
                with self._lock:
                    registered_id = self._idempotency_map.setdefault(
                        idempotency_key, _pre_order_id
                    )
                    if registered_id != _pre_order_id:
                        _idem_span.set_attribute("idempotency.hit", True)
                        logger.info(
                            "submit_order: idempotency_key=%r already exists → returning %s",
                            idempotency_key, registered_id,
                        )
                        return registered_id
                _idem_span.set_attribute("idempotency.hit", False)

        with start_span("trading.order.risk_check") as _risk_span:
            # S41: correlation limit check
            if self._current_correlation is not None:
                corr_violated = self._check_correlation_limit(self._current_correlation)
                _risk_span.set_attribute("risk.correlation_check", not corr_violated)
                if corr_violated:
                    return self._reject_order(
                        side, amount, order_type, limit_price,
                        reason="correlation_limit",
                        idempotency_key=idempotency_key,
                    )

            # S43: volatility circuit breaker
            _cb_tripped = self._circuit_breaker.is_tripped()
            _risk_span.set_attribute("risk.circuit_breaker_tripped", _cb_tripped)
            if _cb_tripped:
                return self._reject_order(
                    side, amount, order_type, limit_price,
                    reason="volatility_circuit_breaker",
                    idempotency_key=idempotency_key,
                )

            # Pre-trade drawdown check
            if self._risk_manager is not None:
                tracker = self._position_tracker
                if tracker is not None:
                    peak = tracker.peak_value
                    current = tracker.portfolio_value
                    _dd_hit = self._risk_manager.check_drawdown(peak, current)
                    _risk_span.set_attribute("risk.drawdown_check_passed", not _dd_hit)
                    if _dd_hit:
                        return self._reject_order(
                            side, amount, order_type, limit_price,
                            reason="max_drawdown",
                            idempotency_key=idempotency_key,
                        )

            # S42: fat-finger guard
            ok, reason = self._fat_finger.check(amount)
            _risk_span.set_attribute("risk.fat_finger_passed", ok)
            if not ok:
                return self._reject_order(
                    side, amount, order_type, limit_price,
                    reason=f"fat_finger:{reason}",
                    idempotency_key=idempotency_key,
                )

        with start_span("trading.order.compliance_check") as _comp_span:
            # G6-G9: pre-trade compliance (Week 76)
            if self._compliance_checker is not None:
                _ref_price = current_price or price or limit_price or 1.0
                _order_notional = amount * _ref_price
                _tracker = self._position_tracker
                _sym_notional = _tracker.position * _ref_price
                _port_notional = _tracker.portfolio_value
                _comp_ok, _comp_reason = self._compliance_checker.check_all(
                    symbol=self.symbol,
                    side=side,
                    order_notional=_order_notional,
                    limit_price=limit_price,
                    current_symbol_notional=_sym_notional,
                    current_portfolio_notional=_port_notional,
                )
                _comp_span.set_attribute("compliance.passed", _comp_ok)
                if not _comp_ok:
                    _comp_span.set_attribute("compliance.reject_reason", _comp_reason)
                    return self._reject_order(
                        side, amount, order_type, limit_price,
                        reason=_comp_reason,
                        idempotency_key=idempotency_key,
                    )
            else:
                _comp_span.set_attribute("compliance.passed", True)

        if amount > self.max_order_size:
            logger.warning(
                "Order amount %.6f exceeds max_order_size %.6f; clamping.",
                amount, self.max_order_size,
            )
            amount = self.max_order_size

        if current_price is None and price is not None:
            current_price = price

        # F14: compute expiry timestamp
        effective_ttl = ttl_sec if ttl_sec is not None else self._order_ttl_sec
        expires_at: Optional[datetime] = None
        if effective_ttl > 0:
            expires_at = datetime.utcnow() + timedelta(seconds=effective_ttl)

        order_id = _pre_order_id
        _now_dt = datetime.utcnow()
        order = Order(
            order_id=order_id,
            side=side,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,       # F12
            status="pending",
            expires_at=expires_at,       # F14
            idempotency_key=idempotency_key,
            submitted_at=_now_dt,        # S52
        )
        with self._lock:
            self._orders[order_id] = order

        _parent_span.set_attribute("order.id", order_id)

        # Audit: order submitted
        if self._audit_logger is not None:
            self._audit_logger.log_order(order)

        # G8+G9: commit notional and timestamp for accepted order
        if self._compliance_checker is not None:
            _ref_price = current_price or price or limit_price or 1.0
            self._compliance_checker.record_order(
                self.symbol, side, amount * _ref_price
            )
            # G7: register pending limit orders so future submits can detect crossing
            if limit_price is not None and order_type in (
                "limit", "stop_loss_limit", "take_profit"
            ):
                self._compliance_checker.register_open_order(
                    self.symbol, limit_price, side
                )

        with start_span("trading.order.exchange_submit") as _exch_span:
            _exch_span.set_attribute("order.paper_mode", self.paper_mode)
            try:
                if self.paper_mode:
                    self._execute_paper_order(order, current_price)
                else:
                    self._execute_live_order(order)
            except Exception as e:
                order.status = "failed"
                order.updated_at = datetime.utcnow()
                logger.error("Order %s failed: %s", order_id, e)
            _exch_span.set_attribute("order.status", order.status)

        # S52: record fill latency (submit → fill)
        if order.filled_at is not None and order.submitted_at is not None:
            latency_ms = (
                (order.filled_at - order.submitted_at).total_seconds() * 1000.0
            )
            with self._lock:
                self._latency_samples.append(latency_ms)
            record_order_latency(_parent_span, latency_ms)

        # Audit: fill (or failure) recorded after execution
        if self._audit_logger is not None:
            self._audit_logger.log_fill(order)

        # S42: record fill size for future fat-finger baselines
        if order.status in ("filled", "partial") and order.filled_amount > 0:
            self._fat_finger.record_fill(order.filled_amount)

        _total_ms = (time.monotonic() - _submit_start) * 1000.0
        _parent_span.set_attribute("order.total_latency_ms", _total_ms)

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
        """Cancel a pending or partial order. Returns True if successful."""
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
                # G7: release from self-trade tracking
                if self._compliance_checker is not None and order.limit_price is not None:
                    self._compliance_checker.deregister_open_order(
                        self.symbol, order.limit_price
                    )
                return True
        try:
            self.rate_limiter.acquire()
            self._exchange.cancel_order(order.exchange_order_id, self.symbol)
            order.status = "cancelled"
            order.updated_at = datetime.utcnow()
            # G7: release from self-trade tracking
            if self._compliance_checker is not None and order.limit_price is not None:
                self._compliance_checker.deregister_open_order(
                    self.symbol, order.limit_price
                )
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open (pending/partial) orders. Returns count cancelled.

        G13: Called by kill switch to ensure a clean exit.
        """
        with self._lock:
            open_ids = [
                oid for oid, o in self._orders.items()
                if o.status in ("pending", "partial")
            ]
        cancelled = 0
        for oid in open_ids:
            if self.cancel_order(oid):
                cancelled += 1
            else:
                logger.warning("cancel_all_orders: could not cancel %s", oid)
        logger.info("cancel_all_orders: cancelled %d / %d open orders", cancelled, len(open_ids))
        return cancelled

    def cancel_replace_order(
        self,
        order_id: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_price: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        ttl_sec: Optional[float] = None,
    ) -> str:
        """Cancel an existing order and submit a replacement atomically.

        Raises RuntimeError if the original order cannot be cancelled.
        Returns the new order_id.
        """
        cancelled = self.cancel_order(order_id)
        if not cancelled:
            if self._audit_logger is not None:
                self._audit_logger.log_risk_event({
                    "type": "cancel_replace_failed",
                    "order_id": order_id,
                    "reason": "cancel_failed",
                })
            if self._alerter is not None:
                try:
                    self._alerter.notify_error(
                        f"cancel_replace_order: could not cancel {order_id}"
                    )
                except Exception:
                    pass
            raise RuntimeError(f"cancel_replace_order: could not cancel order {order_id}")
        return self.submit_order(
            side=side,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            current_price=current_price,
            idempotency_key=idempotency_key,
            ttl_sec=ttl_sec,
        )

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

        Also feeds the price to the circuit breaker for vol tracking (S43),
        and checks pending limit/stop orders against the new price (F12).
        """
        self._position_tracker.update_price(price)
        self._circuit_breaker.update(price)
        if self.paper_mode:
            self._check_pending_orders(price)

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
        """Return latency percentiles (p50/p95/p99) in milliseconds."""
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

    def close(self) -> None:
        """Shut down background threads and release exchange connection."""
        self._stop_event.set()
        if self._expiry_thread is not None and self._expiry_thread.is_alive():
            self._expiry_thread.join(timeout=2.0)
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # S41: Correlation limit helper
    # ------------------------------------------------------------------

    def _check_correlation_limit(self, correlation_value: float) -> bool:
        check_fn = getattr(self._risk_manager, "check_correlation", None)
        if callable(check_fn):
            breached = check_fn(correlation_value, self._correlation_threshold)
        else:
            breached = abs(correlation_value) > self._correlation_threshold

        if breached:
            logger.warning(
                "Order rejected: correlation %.4f exceeds threshold %.4f",
                correlation_value, self._correlation_threshold,
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
    # F12: Fill price resolution (paper mode)
    # ------------------------------------------------------------------

    def _resolve_paper_fill_price(self, order: Order, market_price: float) -> Optional[float]:
        """Return fill price if order condition is met, else None (stay pending)."""
        if order.order_type == "market":
            return market_price

        lp = order.limit_price
        sp = order.stop_price

        if order.order_type == "limit":
            if lp is None:
                logger.warning("Limit order %s missing limit_price; treating as market", order.order_id)
                return market_price
            # Buy limit: fill when market dips to or below limit
            if order.side == "buy":
                return lp if market_price <= lp else None
            # Sell limit: fill when market rises to or above limit
            return lp if market_price >= lp else None

        if order.order_type == "stop_loss_limit":
            if sp is None or lp is None:
                logger.warning("stop_loss_limit order %s missing stop_price or limit_price", order.order_id)
                return market_price
            # Buy stop: trigger when price breaks above stop, fill at limit
            if order.side == "buy":
                return lp if market_price >= sp else None
            # Sell stop-loss: trigger when price drops below stop, fill at limit
            return lp if market_price <= sp else None

        if order.order_type == "take_profit":
            if sp is None or lp is None:
                logger.warning("take_profit order %s missing stop_price or limit_price", order.order_id)
                return market_price
            # Take-profit sell: trigger when price rises above stop, fill at limit
            if order.side == "sell":
                return lp if market_price >= sp else None
            # Take-profit buy (uncommon): trigger when price dips below stop
            return lp if market_price <= sp else None

        logger.warning("Unknown order_type %r; treating as market", order.order_type)
        return market_price

    # ------------------------------------------------------------------
    # F13: Partial fill simulation (paper mode)
    # ------------------------------------------------------------------

    def _draw_partial_fill_ratio(self) -> float:
        """Return fill ratio in [min_ratio, 1.0]. Always 1.0 when sim disabled."""
        if not self._partial_fill_sim:
            return 1.0
        return float(np.random.uniform(self._partial_fill_min_ratio, 1.0))

    # ------------------------------------------------------------------
    # F16: Fee computation (paper mode)
    # ------------------------------------------------------------------

    def _compute_paper_fee(self, quantity: float, price: float, is_maker: bool = False) -> float:
        """Compute paper-mode fee using FeeModel if available, else 0.1% default."""
        if self._fee_model is not None:
            try:
                return float(self._fee_model.compute_fee(quantity, price, is_maker=is_maker))
            except Exception as e:
                logger.warning("FeeModel.compute_fee failed (%s); using default 0.1%%", e)
        return quantity * price * 0.001

    # ------------------------------------------------------------------
    # Paper order execution
    # ------------------------------------------------------------------

    def _execute_paper_order(self, order: Order, current_price: Optional[float]) -> None:
        price = current_price or self._position_tracker.current_price
        if price <= 0:
            price = 1.0

        order.acked_at = datetime.utcnow()

        # F12: resolve fill price (may return None for unmet conditions)
        fill_price = self._resolve_paper_fill_price(order, price)
        if fill_price is None:
            # Condition not met — leave as pending for later check
            order.status = "pending"
            order.updated_at = datetime.utcnow()
            return

        # F13: partial fill ratio
        fill_ratio = self._draw_partial_fill_ratio()
        is_maker = order.order_type in ("limit", "stop_loss_limit", "take_profit")

        if order.side == "buy":
            fill_qty = order.amount * fill_ratio
            fee = self._compute_paper_fee(fill_qty, fill_price, is_maker=is_maker)
            self._position_tracker.apply_buy(quantity=fill_qty, price=fill_price, fee=fee)
            order.filled_amount = fill_qty
            order.avg_fill_price = fill_price
            order.fee = fee
        else:
            available = self._position_tracker.position
            sell_qty = min(order.amount * fill_ratio, available)
            if sell_qty < 1e-9:
                order.status = "failed"
                order.updated_at = datetime.utcnow()
                order.filled_at = order.updated_at
                return
            fee = self._compute_paper_fee(sell_qty, fill_price, is_maker=is_maker)
            entry_price = self._position_tracker.entry_price
            self._position_tracker.apply_sell(quantity=sell_qty, price=fill_price, fee=fee)
            order.filled_amount = sell_qty
            order.avg_fill_price = fill_price
            order.fee = fee
            pnl = (fill_price - entry_price) * sell_qty if entry_price > 0 else 0.0
            with self._lock:
                self._daily_pnl += (pnl - fee)
            self._check_daily_loss_limit()

        # F13: record individual fill event in audit log
        fill_event: Dict[str, Any] = {
            "fill_id": str(uuid.uuid4())[:8],
            "order_id": order.order_id,
            "side": order.side,
            "qty": order.filled_amount,
            "price": fill_price,
            "fee": order.fee,
            "is_partial": fill_ratio < 1.0,
            "timestamp": datetime.utcnow().isoformat(),
        }
        order.fills.append(fill_event)
        if self._audit_logger is not None:
            self._audit_logger.log_fill(fill_event)

        # F13: determine partial vs full status
        remaining = order.amount - order.filled_amount
        if remaining > 1e-9 and fill_ratio < 1.0:
            order.status = "partial"
            # Leave order alive for future fills via _check_pending_orders
        else:
            order.status = "filled"
            order.filled_at = datetime.utcnow()   # S52
            # G7: release filled limit order from self-trade tracking
            if self._compliance_checker is not None and order.limit_price is not None:
                self._compliance_checker.deregister_open_order(
                    self.symbol, order.limit_price
                )
        order.updated_at = datetime.utcnow()

    def _check_pending_orders(self, price: float) -> None:
        """Try to fill pending limit/stop orders at the new market price (F12)."""
        with self._lock:
            candidates = [
                o for o in self._orders.values()
                if o.status in ("pending", "partial") and o.order_type != "market"
            ]
        for order in candidates:
            self._execute_paper_order(order, price)
            if self._audit_logger is not None and order.status in ("filled", "partial"):
                pass  # fill event already logged inside _execute_paper_order
            if order.status == "filled" and order.filled_amount > 0:
                self._fat_finger.record_fill(order.filled_amount)
            # S52: latency for orders filled via price update
            if order.filled_at is not None and order.submitted_at is not None:
                latency_ms = (
                    (order.filled_at - order.submitted_at).total_seconds() * 1000.0
                )
                with self._lock:
                    self._latency_samples.append(latency_ms)

    # ------------------------------------------------------------------
    # Live order execution — S44: idempotency key + F12: order types
    # ------------------------------------------------------------------

    def _execute_live_order(self, order: Order) -> None:
        backoff = 1.0
        for attempt in range(1, self._max_retries + 1):
            try:
                self.rate_limiter.acquire()
                params: Dict[str, Any] = {}
                if order.idempotency_key is not None:
                    params["clientOrderId"] = order.idempotency_key

                # F12: dispatch by order type
                if order.order_type == "market":
                    result = self._exchange.create_market_order(
                        self.symbol, order.side, order.amount, params=params
                    )
                elif order.order_type == "limit":
                    result = self._exchange.create_limit_order(
                        self.symbol, order.side, order.amount, order.limit_price, params=params
                    )
                elif order.order_type == "stop_loss_limit":
                    params["stopPrice"] = order.stop_price
                    result = self._exchange.create_order(
                        self.symbol, "stop_loss_limit", order.side,
                        order.amount, order.limit_price, params=params,
                    )
                elif order.order_type == "take_profit":
                    params["stopPrice"] = order.stop_price
                    result = self._exchange.create_order(
                        self.symbol, "take_profit_limit", order.side,
                        order.amount, order.limit_price, params=params,
                    )
                else:
                    result = self._exchange.create_market_order(
                        self.symbol, order.side, order.amount, params=params
                    )

                order.exchange_order_id = result.get("id")
                order.acked_at = datetime.utcnow()   # S52

                # F13: map CCXT status → internal status (including partial)
                status_raw = result.get("status", "")
                filled = float(result.get("filled", 0.0))
                remaining = float(result.get("remaining") or 0.0)
                if status_raw == "closed":
                    order.status = "filled"
                elif status_raw in ("canceled", "cancelled"):
                    order.status = "cancelled"
                elif status_raw == "open" and filled > 0 and remaining > 0:
                    order.status = "partial"
                else:
                    order.status = "pending"

                order.filled_amount = filled
                order.avg_fill_price = float(result.get("average") or result.get("price") or 0.0)
                fee_info = result.get("fee") or {}
                order.fee = float(fee_info.get("cost", 0.0))
                order.updated_at = datetime.utcnow()

                if order.status == "filled":
                    order.filled_at = order.updated_at   # S52

                # F13: record fill event in audit log
                if filled > 0:
                    fill_event: Dict[str, Any] = {
                        "fill_id": str(uuid.uuid4())[:8],
                        "order_id": order.order_id,
                        "exchange_order_id": order.exchange_order_id,
                        "side": order.side,
                        "qty": filled,
                        "remaining": remaining,
                        "price": order.avg_fill_price,
                        "fee": order.fee,
                        "is_partial": order.status == "partial",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    order.fills.append(fill_event)
                    if self._audit_logger is not None:
                        self._audit_logger.log_fill(fill_event)

                # Update position tracker on filled orders
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
            filled = float(result.get("filled", order.filled_amount))
            remaining = float(result.get("remaining") or 0.0)
            status_raw = result.get("status", "")
            if status_raw == "closed":
                order.status = "filled"
                order.filled_at = datetime.utcnow()
            elif status_raw in ("canceled", "cancelled"):
                order.status = "cancelled"
            elif status_raw == "open" and filled > 0 and remaining > 0:
                order.status = "partial"
            else:
                order.status = "pending"

            # F13: record new fills since last check
            prev_filled = order.filled_amount
            if filled > prev_filled + 1e-9:
                incremental_qty = filled - prev_filled
                fill_event: Dict[str, Any] = {
                    "fill_id": str(uuid.uuid4())[:8],
                    "order_id": order.order_id,
                    "exchange_order_id": order.exchange_order_id,
                    "side": order.side,
                    "qty": incremental_qty,
                    "remaining": remaining,
                    "price": float(result.get("average") or result.get("price") or 0.0),
                    "is_partial": order.status == "partial",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                order.fills.append(fill_event)
                if self._audit_logger is not None:
                    self._audit_logger.log_fill(fill_event)

            order.filled_amount = filled
            order.updated_at = datetime.utcnow()
        except Exception as e:
            logger.warning("Failed to refresh order %s: %s", order.order_id, e)

    # ------------------------------------------------------------------
    # F14: TTL expiry
    # ------------------------------------------------------------------

    def _expire_stale_orders(self) -> None:
        """Cancel pending/partial orders that have exceeded their TTL."""
        now = datetime.utcnow()
        with self._lock:
            candidates = [
                o for o in self._orders.values()
                if o.status in ("pending", "partial")
                and o.expires_at is not None
                and o.expires_at <= now
            ]
        for order in candidates:
            logger.info(
                "Order %s expired (ttl=%.0fs, type=%s)",
                order.order_id, self._order_ttl_sec, order.order_type,
            )
            cancelled = self.cancel_order(order.order_id)
            if not cancelled:
                logger.error("Failed to cancel expired order %s", order.order_id)
                if self._audit_logger is not None:
                    self._audit_logger.log_risk_event({
                        "type": "cancel_expired_failed",
                        "order_id": order.order_id,
                        "order_type": order.order_type,
                    })
                if self._alerter is not None:
                    try:
                        self._alerter.notify_error(
                            f"Failed to auto-cancel expired order {order.order_id}"
                        )
                    except Exception:
                        pass
            else:
                if self._audit_logger is not None:
                    self._audit_logger.log_risk_event({
                        "type": "order_expired",
                        "order_id": order.order_id,
                        "order_type": order.order_type,
                    })

    def _order_expiry_worker(self) -> None:
        """Background thread: periodically expire stale orders."""
        while not self._stop_event.is_set():
            try:
                self._expire_stale_orders()
            except Exception as e:
                logger.error("Order expiry worker error: %s", e)
            self._stop_event.wait(self._ttl_check_interval)

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
            exchange = exchange_class({
                "apiKey": config.get("api_key", ""),
                "secret": config.get("api_secret", ""),
                "enableRateLimit": True,
            })
            if self._exchange_mode == "sandbox":
                exchange.set_sandbox_mode(True)
                logger.info("OrderManager: sandbox (testnet) mode activated for %s", exchange_id)
            return exchange
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
        self.close()
