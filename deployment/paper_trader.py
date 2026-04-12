"""
Paper Trader: Production paper trading with real-time data and risk management.

PaperTrader connects to a CCXT exchange (or uses simulation mode) to run a
trained RL agent against live/historical market data. All decisions and P&L
are logged to MLflow. Risk management (position limits, drawdown shutdown) is
enforced at every step.

Usage (CLI):
    python -m deployment.paper_trader --config config/paper_trading.yaml --duration 3600

Usage (API):
    trader = PaperTrader(agent, config)
    trader.run(price_stream=prices)
    report = trader.generate_report()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from deployment.analysis.pnl_attribution import PnLAttributor
from deployment.audit.audit_logger import AuditLogger
from deployment.execution.order_manager import OrderManager
from deployment.execution.position_tracker import PositionTracker
from deployment.monitoring.alerter import TradingAlerter
from deployment.monitoring.metrics_exporter import MetricsExporter
from deployment.persistence.state_store import StateStore
from training.monitoring.drift_detector import DriftDetector, FeatureDriftDetector
from training.regime.regime_detector import RegimeDetector
from risk_management.risk_manager_base import RiskManagerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    timestamp: datetime
    side: str          # "buy" | "sell"
    price: float
    quantity: float
    fee: float
    pnl: float = 0.0   # realised PnL for closing trades


@dataclass
class TradingState:
    """
    Mutable trading state.

    Position-related fields (position, entry_price, cash/balance,
    current_price, peak_portfolio_value) are consolidated in
    `pos` (a PositionTracker).  All other bookkeeping lives here.
    """
    pos: PositionTracker
    trades: List[Trade] = field(default_factory=list)
    portfolio_history: deque = field(default_factory=lambda: deque(maxlen=100_000))
    step: int = 0
    shutdown_triggered: bool = False
    shutdown_reason: str = ""

    # ------------------------------------------------------------------
    # Convenience pass-throughs so callers keep working without changes
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        return self.pos.cash

    @balance.setter
    def balance(self, value: float) -> None:
        with self.pos._lock:
            self.pos._cash = float(value)

    @property
    def position(self) -> float:
        return self.pos.position

    @position.setter
    def position(self, value: float) -> None:
        with self.pos._lock:
            self.pos._position = float(value)

    @property
    def entry_price(self) -> float:
        return self.pos.entry_price

    @entry_price.setter
    def entry_price(self, value: float) -> None:
        with self.pos._lock:
            self.pos._entry_price = float(value)

    @property
    def peak_portfolio_value(self) -> float:
        return self.pos.peak_value

    @peak_portfolio_value.setter
    def peak_portfolio_value(self, value: float) -> None:
        with self.pos._lock:
            self.pos._peak_value = float(value)

    @property
    def _current_price(self) -> float:
        return self.pos.current_price

    @_current_price.setter
    def _current_price(self, value: float) -> None:
        with self.pos._lock:
            self.pos._current_price = float(value)

    @property
    def portfolio_value(self) -> float:
        return self.pos.portfolio_value

    # ------------------------------------------------------------------
    # Serialisation (Phase 6 Week 56 — S5)
    # ------------------------------------------------------------------

    def to_dict(self, symbol: str = "DEFAULT") -> Dict[str, Any]:
        """Serialise to a JSON-safe dict for StateStore.

        datetime → ISO string. NaN/inf is rejected at the StateStore layer.
        """
        import math as _math
        snap = self.pos.snapshot()
        for k, v in snap.items():
            if isinstance(v, float) and not _math.isfinite(v):
                raise ValueError(f"TradingState contains non-finite value '{k}': {v}")
        return {
            "symbol": symbol,
            "position": snap["position"],
            "entry_price": snap["entry_price"],
            "cash": snap["cash"],
            "current_price": snap["current_price"],
            "peak_value": snap["peak_value"],
            "equity": snap["cash"] + snap["position"] * snap["current_price"],
            "step": self.step,
            "shutdown_triggered": self.shutdown_triggered,
            "shutdown_reason": self.shutdown_reason,
            "portfolio_history": list(self.portfolio_history),
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "side": t.side,
                    "price": t.price,
                    "quantity": t.quantity,
                    "fee": t.fee,
                    "pnl": t.pnl,
                }
                for t in self.trades
            ],
            "orders": [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingState":
        """Reconstruct from a dict produced by ``to_dict``."""
        pos = PositionTracker(initial_cash=float(data.get("cash", 0.0)))
        pos.restore(
            {
                "position": float(data.get("position", 0.0)),
                "entry_price": float(data.get("entry_price", 0.0)),
                "cash": float(data.get("cash", 0.0)),
                "current_price": float(data.get("current_price", 0.0)),
                "peak_value": float(data.get("peak_value", data.get("cash", 0.0))),
            }
        )
        st = cls(pos=pos)
        st.step = int(data.get("step", 0))
        st.shutdown_triggered = bool(data.get("shutdown_triggered", False))
        st.shutdown_reason = str(data.get("shutdown_reason", ""))
        st.portfolio_history = deque(
            [float(v) for v in data.get("portfolio_history", [])],
            maxlen=100_000,
        )
        st.trades = [
            Trade(
                timestamp=datetime.fromisoformat(t["timestamp"]),
                side=t["side"],
                price=float(t["price"]),
                quantity=float(t["quantity"]),
                fee=float(t["fee"]),
                pnl=float(t.get("pnl", 0.0)),
            )
            for t in data.get("trades", [])
        ]
        return st


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Paper trading runner.

    Parameters
    ----------
    agent :
        Any object with a ``predict(obs, deterministic=True)`` method that
        returns ``(action, state)``.  Compatible with all SB3 agents.
    config : dict
        Paper trading configuration (see ``config/paper_trading.yaml``).
    mlflow_manager : optional
        ``MLflowManager`` instance; if provided, all trades and metrics are
        logged per step and a final report artifact is uploaded.
    simulation_mode : bool
        When True, no CCXT connection is made.  Market data must be supplied
        via ``run(price_stream=...)``.
    """

    def __init__(
        self,
        agent,
        config: Dict[str, Any],
        mlflow_manager=None,
        simulation_mode: bool = False,
        alerter: Optional[TradingAlerter] = None,
        drift_detector: Optional[DriftDetector] = None,
        order_manager: Optional[OrderManager] = None,
        risk_manager: Optional[RiskManagerBase] = None,
        state_store: Optional[StateStore] = None,
        audit_logger: Optional[AuditLogger] = None,
        data_source=None,
        regime_detector: Optional[RegimeDetector] = None,
        feature_drift_detector: Optional[FeatureDriftDetector] = None,
        on_regime_change: Optional[Callable[[int, int, np.ndarray], None]] = None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.simulation_mode = simulation_mode
        self.alerter = alerter
        self.drift_detector = drift_detector
        self.order_manager = order_manager
        monitoring_config = config.get("monitoring", {})
        self.metrics_exporter = MetricsExporter(monitoring_config)
        self.risk_manager = risk_manager

        pt = config.get("paper_trading", config)
        self.symbol: str = pt.get("symbol", "BTC/USDT")
        self.initial_balance: float = float(pt.get("initial_balance", 10_000.0))
        self.trading_fee: float = float(pt.get("trading_fee", 0.001))
        self.max_position_size: float = float(pt.get("max_position_size", 1.0))
        self.max_drawdown_threshold: float = float(
            pt.get("max_drawdown_threshold", 0.20)
        )
        self.window_size: int = int(pt.get("window_size", 20))
        self.daily_report_interval: int = int(
            pt.get("daily_report_interval", 86400)
        )  # seconds

        self._price_history: List[float] = []
        self._last_report_time: float = time.time()
        self._slippage_records: List[float] = []  # |fill_price - expected_price| / expected_price

        self.state = TradingState(
            pos=PositionTracker(initial_cash=self.initial_balance),
        )

        # Phase 6 Week 56 (S2/S6): optional state persistence.
        # Either pass a StateStore directly, or include a `persistence` block
        # in config: {enabled: bool, db_path: str, checkpoint_every_n_steps: int}.
        self.state_store = state_store
        persist_cfg = config.get("persistence", {}) or {}
        self._checkpoint_every_n_steps: int = int(
            persist_cfg.get("checkpoint_every_n_steps", 1)
        )
        if self.state_store is None and persist_cfg.get("enabled", False):
            db_path = persist_cfg.get("db_path", "./state/paper_trader.db")
            self.state_store = StateStore(db_path)

        # Phase 6 Week 65 (S47-S48): data pipeline safety.
        self.audit_logger = audit_logger
        self.data_source = data_source

        # Phase 6 Week 67 (S56-S57): drift & regime.
        self.feature_drift_detector = feature_drift_detector
        self.regime_detector = regime_detector
        self.on_regime_change = on_regime_change
        self._current_regime: int = -1

        pipeline_cfg = config.get("data_pipeline_safety", {}) or {}
        self._staleness_enabled: bool = bool(pipeline_cfg.get("staleness_enabled", True))
        self._max_staleness_sec: float = float(pipeline_cfg.get("max_staleness_sec", 60.0))
        self._nan_check_enabled: bool = bool(pipeline_cfg.get("nan_check_enabled", True))
        self._nan_halt_after_n: int = int(pipeline_cfg.get("nan_halt_after_n", 5))
        self._consecutive_nan_steps: int = 0

        if not simulation_mode:
            self._exchange = self._init_exchange(pt)
        else:
            self._exchange = None

        logger.info(
            "PaperTrader initialised | symbol=%s balance=%.2f simulation=%s",
            self.symbol,
            self.initial_balance,
            simulation_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        duration_seconds: Optional[float] = None,
        price_stream: Optional[Iterator[float]] = None,
    ) -> Dict[str, Any]:
        """
        Main trading loop.

        Parameters
        ----------
        duration_seconds :
            Stop after this many wall-clock seconds (None = run until stream
            exhausted or shutdown triggered).
        price_stream :
            Iterator of prices.  Required in simulation mode; in live mode
            prices are fetched from the exchange if omitted.

        Returns
        -------
        Final performance report dict.
        """
        start_time = time.time()
        step = self.state.step
        _restored_log_pending = getattr(self, "_restored_from_checkpoint", False)

        for price in self._price_iterator(price_stream):
            if self.state.shutdown_triggered:
                break
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break

            self._update_price(price)

            if _restored_log_pending:
                logger.info(
                    "restored: resuming PaperTrader at step=%d price=%.4f",
                    self.state.step, price,
                )
                _restored_log_pending = False
                self._restored_from_checkpoint = False

            # Phase 6 Week 65 (S47): feed staleness check.
            if self._check_feed_staleness():
                break

            obs = self._build_observation()
            if obs is None:
                step += 1
                continue

            # Phase 6 Week 65 (S48): NaN/inf in computed features.
            if self._check_obs_nan(obs, step):
                step += 1
                continue

            self._consecutive_nan_steps = 0  # reset on clean observation
            action, _ = self.agent.predict(obs, deterministic=True)
            self._execute_action(action, price)
            self._check_risk(price)
            self._check_drift()
            self._check_regime()
            self._maybe_daily_report()

            step += 1
            self.state.step = step

            self._log_step_metrics(price)

            # Phase 6 Week 56 (S2): crash-recovery checkpoint.
            if self.state_store is not None and (
                self._checkpoint_every_n_steps <= 1
                or step % self._checkpoint_every_n_steps == 0
            ):
                self._checkpoint()

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Return a dictionary with all performance metrics."""
        history = self.state.portfolio_history
        if not history:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0,
                "final_balance": self.state.balance,
                "final_portfolio_value": self.state.balance,
                "shutdown_triggered": self.state.shutdown_triggered,
                "shutdown_reason": self.state.shutdown_reason,
                "win_rate": 0.0,
                "avg_trade_pnl": 0.0,
                "total_fees": 0.0,
                "avg_fill_slippage": 0.0,
            }

        values = np.array(history, dtype=float)
        returns = np.diff(values) / np.where(values[:-1] != 0, values[:-1], 1e-8)

        total_return = (values[-1] - self.initial_balance) / self.initial_balance
        sharpe = self._compute_sharpe(returns)
        max_dd = self._compute_max_drawdown(values)

        closing_trades = [t for t in self.state.trades if t.side == "sell"]
        winning = [t for t in closing_trades if t.pnl > 0]
        win_rate = len(winning) / len(closing_trades) if closing_trades else 0.0
        avg_pnl = (
            np.mean([t.pnl for t in closing_trades]) if closing_trades else 0.0
        )
        total_fees = sum(t.fee for t in self.state.trades)

        report = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "num_trades": len(self.state.trades),
            "final_balance": float(self.state.balance),
            "final_portfolio_value": float(values[-1]),
            "shutdown_triggered": self.state.shutdown_triggered,
            "shutdown_reason": self.state.shutdown_reason,
            "win_rate": float(win_rate),
            "avg_trade_pnl": float(avg_pnl),
            "total_fees": float(total_fees),
            "avg_fill_slippage": (
                sum(self._slippage_records) / len(self._slippage_records)
                if self._slippage_records else 0.0
            ),
            "steps": self.state.step,
            "generated_at": datetime.utcnow().isoformat(),
        }

        if self.mlflow_manager:
            self._log_final_report(report)

        return report

    # ------------------------------------------------------------------
    # Phase 6 Week 56: StateStore-backed checkpoint / restore
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Persist current TradingState to the configured StateStore.

        Runs under the PositionTracker lock so the snapshot is consistent
        with concurrent price-feed mutations.
        """
        if self.state_store is None:
            return
        try:
            with self.state.pos._lock:
                snap = self.state.to_dict(symbol=self.symbol)
            self.state_store.save_snapshot(snap)
        except Exception as e:
            logger.warning("State checkpoint failed: %s", e)

    @classmethod
    def restore(
        cls,
        state_store: StateStore,
        agent,
        config: Dict[str, Any],
        **kwargs: Any,
    ) -> "PaperTrader":
        """Construct a PaperTrader and restore TradingState from ``state_store``.

        Any kwargs are forwarded to ``__init__``. If no snapshot is found, a
        fresh trader is returned (caller can detect via ``trader.state.step == 0``).
        """
        kwargs.pop("state_store", None)
        trader = cls(agent, config, state_store=state_store, **kwargs)
        snap = state_store.load_latest()
        if snap is None:
            logger.info("StateStore empty; starting fresh PaperTrader")
            return trader
        trader.state = TradingState.from_dict(snap)
        trader._restored_from_checkpoint = True
        # Rebuild auxiliary buffers used by _build_observation. We do not have
        # the raw price stream from before the crash, so we seed the price
        # history window with the last known price so the trader can resume.
        last_price = trader.state._current_price
        if last_price > 0:
            trader._price_history = [last_price] * trader.window_size
        logger.info(
            "PaperTrader restored from %s | step=%d cash=%.2f position=%.6f",
            state_store.db_path,
            trader.state.step,
            trader.state.balance,
            trader.state.position,
        )
        return trader

    def save_checkpoint(self, path: str) -> None:
        """Save current trading state to disk."""
        import json

        state_dict = {
            "position_tracker": self.state.pos.snapshot(),
            "step": self.state.step,
            "shutdown_triggered": self.state.shutdown_triggered,
            "shutdown_reason": self.state.shutdown_reason,
            "portfolio_history": list(self.state.portfolio_history),
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "side": t.side,
                    "price": t.price,
                    "quantity": t.quantity,
                    "fee": t.fee,
                    "pnl": t.pnl,
                }
                for t in self.state.trades
            ],
        }
        Path(path).write_text(json.dumps(state_dict, indent=2))
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Restore trading state from a checkpoint file."""
        import json

        data = json.loads(Path(path).read_text())
        self.state.pos.restore(data["position_tracker"])
        self.state.step = data["step"]
        self.state.shutdown_triggered = data["shutdown_triggered"]
        self.state.shutdown_reason = data["shutdown_reason"]
        self.state.portfolio_history = deque(data["portfolio_history"], maxlen=100_000)
        self.state.trades = [
            Trade(
                timestamp=datetime.fromisoformat(t["timestamp"]),
                side=t["side"],
                price=t["price"],
                quantity=t["quantity"],
                fee=t["fee"],
                pnl=t["pnl"],
            )
            for t in data["trades"]
        ]
        logger.info("Checkpoint loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_price(self, price: float) -> None:
        self._price_history.append(price)
        self.state.pos.update_price(price)   # updates current_price + peak_value atomically
        self.state.portfolio_history.append(self.state.pos.portfolio_value)

    def _build_observation(self) -> Optional[np.ndarray]:
        """Build observation vector from price history window."""
        if len(self._price_history) < self.window_size:
            return None
        window = np.array(self._price_history[-self.window_size :], dtype=np.float32)
        # log-returns
        log_returns = np.diff(np.log(np.maximum(window, 1e-8)))
        position_ratio = np.array(
            [self.state.position * self.state._current_price / self.initial_balance],
            dtype=np.float32,
        )
        cash_ratio = np.array(
            [self.state.balance / self.initial_balance], dtype=np.float32
        )
        obs = np.concatenate([log_returns, position_ratio, cash_ratio])
        return obs

    def _execute_action(self, action, price: float) -> None:
        """
        Translate continuous action in [-1, 1] to a trade.

        action > 0  → buy (size proportional to action)
        action < 0  → sell / reduce position
        action ≈ 0  → hold
        """
        if isinstance(action, np.ndarray):
            action = float(action.flat[0])

        DEADBAND = 0.05
        if abs(action) < DEADBAND:
            return

        if action > 0:
            self._execute_buy(action, price)
        else:
            self._execute_sell(abs(action), price)

    def _execute_buy(self, strength: float, price: float) -> None:
        max_spend = self.state.pos.cash * min(strength, self.max_position_size)
        if max_spend < price * 1e-6:
            return
        quantity = max_spend / price
        fee = max_spend * self.trading_fee
        cost = max_spend + fee
        if cost > self.state.pos.cash:
            return
        self.state.pos.apply_buy(quantity=quantity, price=price, fee=fee)
        self.state.trades.append(
            Trade(
                timestamp=datetime.utcnow(),
                side="buy",
                price=price,
                quantity=quantity,
                fee=fee,
            )
        )
        logger.debug("BUY qty=%.6f price=%.2f fee=%.4f", quantity, price, fee)

        if self.order_manager is not None:
            try:
                order_id = self.order_manager.submit_order("buy", quantity, current_price=price)
                if self.alerter is not None:
                    self.alerter.notify_trade("buy", quantity, price, order_id=order_id)
                # Record slippage (for reconciliation)
                try:
                    order = self.order_manager.check_order(order_id)
                    if order and order.avg_fill_price and order.avg_fill_price > 0:
                        slip = abs(order.avg_fill_price - price) / price
                        self._slippage_records.append(slip)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("OrderManager buy submission failed: %s", e)
                logger.error(
                    "STATE DESYNC: position updated (buy qty=%.6f) but order not tracked. "
                    "Manual reconciliation may be needed.", quantity,
                )

    def _execute_sell(self, strength: float, price: float) -> None:
        sell_qty = self.state.pos.position * min(strength, 1.0)
        if sell_qty < 1e-8:
            return
        fee = sell_qty * price * self.trading_fee
        pnl = self.state.pos.apply_sell(quantity=sell_qty, price=price, fee=fee)
        self.state.trades.append(
            Trade(
                timestamp=datetime.utcnow(),
                side="sell",
                price=price,
                quantity=sell_qty,
                fee=fee,
                pnl=pnl,
            )
        )
        logger.debug("SELL qty=%.6f price=%.2f pnl=%.4f", sell_qty, price, pnl)

        if self.order_manager is not None:
            try:
                order_id = self.order_manager.submit_order("sell", sell_qty, current_price=price)
                if self.alerter is not None:
                    self.alerter.notify_trade("sell", sell_qty, price, order_id=order_id)
                # Record slippage (for reconciliation)
                try:
                    order = self.order_manager.check_order(order_id)
                    if order and order.avg_fill_price and order.avg_fill_price > 0:
                        slip = abs(order.avg_fill_price - price) / price
                        self._slippage_records.append(slip)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("OrderManager sell submission failed: %s", e)
                logger.error(
                    "STATE DESYNC: position updated (sell qty=%.6f) but order not tracked. "
                    "Manual reconciliation may be needed.", sell_qty,
                )

    def _check_risk(self, price: float) -> None:
        """Enforce max drawdown shutdown and fire alerts."""
        if not self.state.portfolio_history:
            return
        current_pv = self.state.portfolio_history[-1]
        peak_pv = self.state.peak_portfolio_value

        if self.alerter is not None:
            self.alerter.check_drawdown(current=current_pv, peak=peak_pv)
            daily_pnl = current_pv - self.initial_balance
            self.alerter.check_daily_pnl(daily_pnl)

        # Delegate to risk_manager if available
        if self.risk_manager is not None:
            if self.risk_manager.check_max_drawdown(peak_pv, current_pv):
                self._trigger_shutdown(
                    f"RiskManager: max drawdown exceeded (peak={peak_pv:.2f}, current={current_pv:.2f})"
                )
                return

        # Fallback: original homebrew check
        if peak_pv > 0:
            drawdown = (peak_pv - current_pv) / peak_pv
            if drawdown >= self.max_drawdown_threshold:
                self._trigger_shutdown(
                    f"Max drawdown {drawdown:.1%} >= threshold "
                    f"{self.max_drawdown_threshold:.1%}"
                )

    def _check_drift(self) -> None:
        """Feed latest portfolio return into drift detector; alert if drift found.

        Also updates the FeatureDriftDetector (S56) when an observation is
        available, logging each per-feature alarm to the audit trail.
        """
        history = self.state.portfolio_history
        if len(history) >= 2:
            prev_pv = history[-2]
            if prev_pv > 0 and self.drift_detector is not None:
                step_return = (history[-1] - prev_pv) / prev_pv
                if self.drift_detector.update(step_return) and self.alerter is not None:
                    self.alerter.notify_drift(
                        detector=self.drift_detector.method,
                        signal_name="portfolio_return",
                    )

        # S56: per-feature drift check using latest raw observation
        if self.feature_drift_detector is not None:
            obs = self._build_observation()
            if obs is not None and np.all(np.isfinite(obs)):
                # Map obs array to detector's feature_names (positional)
                n = len(self.feature_drift_detector.feature_names)
                alarms = self.feature_drift_detector.update(obs[:n])
                fired = [name for name, flag in alarms.items() if flag]
                if fired:
                    if self.alerter is not None:
                        for name in fired:
                            self.alerter.notify_drift(
                                detector=self.feature_drift_detector._method,
                                signal_name=name,
                            )
                    if self.audit_logger is not None:
                        self.audit_logger.log_risk_event({
                            "type": "feature_drift_alarm",
                            "features": fired,
                            "step": self.state.step,
                            "total_detections": self.feature_drift_detector.total_detections,
                        })

    def _check_regime(self) -> None:
        """S57: Evaluate market regime at each step when regime_detector is set.

        Calls ``regime_detector.predict()`` on the current price history
        window.  When the argmax regime index changes from the previous step,
        the ``on_regime_change`` hook is invoked (default: log only) and the
        event is written to the audit trail.
        """
        if self.regime_detector is None:
            return
        if len(self._price_history) < 5:
            return

        window = np.array(self._price_history, dtype=float)
        try:
            probs = self.regime_detector.predict(window)
        except Exception as exc:
            logger.warning("RegimeDetector.predict failed: %s", exc)
            return

        new_regime = int(np.argmax(probs))

        if new_regime != self._current_regime:
            prev = self._current_regime
            self._current_regime = new_regime
            logger.info(
                "Regime change: %d → %d at step=%d (probs=%s)",
                prev,
                new_regime,
                self.state.step,
                np.round(probs, 3).tolist(),
            )
            if self.audit_logger is not None:
                self.audit_logger.log_risk_event({
                    "type": "regime_change",
                    "prev_regime": prev,
                    "new_regime": new_regime,
                    "probs": probs.tolist(),
                    "step": self.state.step,
                })
            # Invoke user-supplied hook (or default no-op)
            if self.on_regime_change is not None:
                try:
                    self.on_regime_change(prev, new_regime, probs)
                except Exception as exc:
                    logger.warning("on_regime_change hook raised: %s", exc)

    def _trigger_shutdown(self, reason: str) -> None:
        logger.warning("SHUTDOWN triggered: %s", reason)
        # Liquidate position at current price
        if self.state.position > 0 and self.state._current_price > 0:
            self._execute_sell(1.0, self.state._current_price)
        self.state.shutdown_triggered = True
        self.state.shutdown_reason = reason

    # ------------------------------------------------------------------
    # Phase 6 Week 65 helpers
    # ------------------------------------------------------------------

    def _check_feed_staleness(self) -> bool:
        """S47: Return True (and trigger shutdown) if the data source is stale.

        Only runs when *staleness_enabled* is True, *max_staleness_sec* > 0,
        and a live data_source has been attached to this trader.
        """
        if (
            not self._staleness_enabled
            or self._max_staleness_sec <= 0
            or self.data_source is None
        ):
            return False

        if self.data_source.is_stale(self._max_staleness_sec):
            reason = (
                f"data_feed_stale: no update for >{self._max_staleness_sec}s "
                f"(max_staleness_sec={self._max_staleness_sec})"
            )
            if self.audit_logger is not None:
                self.audit_logger.log_risk_event(
                    {
                        "type": "feed_staleness_halt",
                        "reason": reason,
                        "max_staleness_sec": self._max_staleness_sec,
                    }
                )
            self._trigger_shutdown(reason)
            return True
        return False

    def _check_obs_nan(self, obs: np.ndarray, step: int) -> bool:
        """S48: Check observation for NaN/inf; skip step and possibly halt.

        Returns True if the step should be skipped.
        """
        if not self._nan_check_enabled:
            return False

        if not np.all(np.isfinite(obs)):
            self._consecutive_nan_steps += 1
            nan_count = int(np.sum(~np.isfinite(obs)))
            logger.warning(
                "NaN/inf in observation at step=%d: %d bad feature(s), "
                "consecutive=%d",
                step,
                nan_count,
                self._consecutive_nan_steps,
            )
            if self.audit_logger is not None:
                self.audit_logger.log_risk_event(
                    {
                        "type": "nan_in_features",
                        "step": step,
                        "nan_count": nan_count,
                        "consecutive": self._consecutive_nan_steps,
                    }
                )

            if (
                self._nan_halt_after_n > 0
                and self._consecutive_nan_steps >= self._nan_halt_after_n
            ):
                reason = (
                    f"nan_in_features: {self._consecutive_nan_steps} consecutive "
                    f"steps with NaN/inf observations (threshold={self._nan_halt_after_n})"
                )
                if self.audit_logger is not None:
                    self.audit_logger.log_risk_event(
                        {"type": "nan_halt", "reason": reason, "step": step}
                    )
                self._trigger_shutdown(reason)
            return True
        return False

    def _maybe_daily_report(self) -> None:
        now = time.time()
        if now - self._last_report_time >= self.daily_report_interval:
            report = self.generate_report()
            logger.info(
                "Daily report | return=%.2f%% sharpe=%.3f drawdown=%.2f%%",
                report["total_return"] * 100,
                report["sharpe_ratio"],
                report["max_drawdown"] * 100,
            )
            self._last_report_time = now

    def _log_step_metrics(self, price: float) -> None:
        try:
            pv = self.state.portfolio_history[-1] if self.state.portfolio_history else self.state.balance
            self.mlflow_manager.log_metric("portfolio_value", pv, step=self.state.step)
            self.mlflow_manager.log_metric("price", price, step=self.state.step)
            self.mlflow_manager.log_metric("position", self.state.position, step=self.state.step)
            self.mlflow_manager.log_metric("balance", self.state.balance, step=self.state.step)
        except Exception as e:
            logger.debug("MLflow step logging failed: %s", e)

        # Export to metrics exporter
        if self.metrics_exporter is not None:
            pv = self.state.portfolio_history[-1] if self.state.portfolio_history else self.initial_balance
            peak = self.state.peak_portfolio_value
            dd = (peak - pv) / peak if peak > 0 else 0.0

            # S52: latency percentiles from order_manager
            latency_kwargs: dict = {}
            if self.order_manager is not None:
                lp = self.order_manager.compute_latency_percentiles()
                latency_kwargs = {
                    "latency_p50_ms": lp["p50"],
                    "latency_p95_ms": lp["p95"],
                    "latency_p99_ms": lp["p99"],
                }

            # S51: P&L attribution
            pnl_kwargs: dict = {}
            closing_trades = [t for t in self.state.trades if t.side == "sell"]
            if closing_trades:
                attributor = PnLAttributor()
                attributions = attributor.attribute(
                    self.state.trades,
                    slippage_records=self._slippage_records,
                )
                summary = attributor.summarise(attributions)
                pnl_kwargs = attributor.to_exporter_fields(summary)

            self.metrics_exporter.update(
                portfolio_value=pv,
                cash=self.state.balance,
                position=self.state.position,
                drawdown_pct=dd,
                num_trades=len(self.state.trades),
                drift_detected=self.drift_detector.drift_detected if self.drift_detector else False,
                alerts_fired=len(self.alerter.alert_history) if self.alerter else 0,
                # S57: current market regime
                current_regime=self._current_regime,
                # S56: cumulative feature drift alarms
                feature_drift_alarms=(
                    self.feature_drift_detector.total_detections
                    if self.feature_drift_detector else 0
                ),
                **latency_kwargs,
                **pnl_kwargs,
            )

            # S53: rolling Sharpe/Sortino (computed from accumulated history)
            r_sharpe = self.metrics_exporter.rolling_sharpe(window=20)
            r_sortino = self.metrics_exporter.rolling_sortino(window=20)
            if r_sharpe != 0.0 or r_sortino != 0.0:
                self.metrics_exporter.update(
                    rolling_sharpe=r_sharpe,
                    rolling_sortino=r_sortino,
                )

    def _log_final_report(self, report: Dict[str, Any]) -> None:
        try:
            import json, tempfile

            self.mlflow_manager.log_metric("total_return", report["total_return"])
            self.mlflow_manager.log_metric("sharpe_ratio", report["sharpe_ratio"])
            self.mlflow_manager.log_metric("max_drawdown", report["max_drawdown"])
            self.mlflow_manager.log_metric("num_trades", report["num_trades"])
            self.mlflow_manager.log_metric("win_rate", report["win_rate"])

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(report, f, indent=2)
                tmp_path = f.name
            self.mlflow_manager.log_artifact(tmp_path, "paper_trading_report.json")
        except Exception as e:
            logger.debug("MLflow report logging failed: %s", e)

    # ------------------------------------------------------------------
    # Exchange helpers
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
            logger.warning("ccxt not installed; falling back to simulation mode")
            self.simulation_mode = True
            return None
        except Exception as e:
            logger.warning("Exchange init failed (%s); using simulation mode", e)
            self.simulation_mode = True
            return None

    def _fetch_live_price(self) -> Optional[float]:
        if self._exchange is None:
            return None
        try:
            ticker = self._exchange.fetch_ticker(self.symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.warning("Failed to fetch price: %s", e)
            return None

    def _price_iterator(
        self, price_stream: Optional[Iterator[float]]
    ) -> Iterator[float]:
        if price_stream is not None:
            yield from price_stream
        elif self.simulation_mode:
            # No stream + simulation mode: empty (caller should pass stream)
            return
        else:
            # Live mode: poll exchange
            poll_interval = self.config.get("paper_trading", {}).get(
                "poll_interval_seconds", 5.0
            )
            while True:
                price = self._fetch_live_price()
                if price is not None:
                    yield price
                time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, annualize: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(annualize))

    @staticmethod
    def _compute_max_drawdown(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        peak = np.maximum.accumulate(values)
        drawdowns = np.where(peak > 0, (peak - values) / peak, 0.0)
        return float(np.max(drawdowns))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PaperTrader":
        return self

    def __exit__(self, *_) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Paper Trader CLI")
    parser.add_argument("--config", required=True, help="Path to paper_trading.yaml")
    parser.add_argument("--duration", type=float, default=None, help="Run duration (seconds)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load agent from checkpoint
    agent_cfg = config.get("agent", {})
    checkpoint = agent_cfg.get("checkpoint")
    algo = agent_cfg.get("algo", "PPO").upper()

    if checkpoint:
        try:
            from stable_baselines3 import PPO, SAC, TD3
            algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
            agent = algo_map[algo].load(checkpoint)
        except Exception as e:
            logger.error("Failed to load agent from %s: %s", checkpoint, e)
            raise
    else:
        raise ValueError("agent.checkpoint must be set in config")

    trader = PaperTrader(agent, config, simulation_mode=False)
    report = trader.run(duration_seconds=args.duration)
    import json
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
