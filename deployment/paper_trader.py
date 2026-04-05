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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from deployment.execution.order_manager import OrderManager
from deployment.execution.position_tracker import PositionTracker
from deployment.monitoring.alerter import TradingAlerter
from training.monitoring.drift_detector import DriftDetector

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
        risk_manager=None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.simulation_mode = simulation_mode
        self.alerter = alerter
        self.drift_detector = drift_detector
        self.order_manager = order_manager
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

        self.state = TradingState(
            pos=PositionTracker(initial_cash=self.initial_balance),
        )

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
        step = 0

        for price in self._price_iterator(price_stream):
            if self.state.shutdown_triggered:
                break
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break

            self._update_price(price)

            obs = self._build_observation()
            if obs is None:
                step += 1
                continue

            action, _ = self.agent.predict(obs, deterministic=True)
            self._execute_action(action, price)
            self._check_risk(price)
            self._check_drift()
            self._maybe_daily_report()

            step += 1
            self.state.step = step

            if self.mlflow_manager:
                self._log_step_metrics(price)

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
            "steps": self.state.step,
            "generated_at": datetime.utcnow().isoformat(),
        }

        if self.mlflow_manager:
            self._log_final_report(report)

        return report

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
        """Feed latest portfolio return into drift detector; alert if drift found."""
        if self.drift_detector is None:
            return
        history = self.state.portfolio_history
        if len(history) < 2:
            return
        prev_pv = history[-2]
        if prev_pv <= 0:
            return
        step_return = (history[-1] - prev_pv) / prev_pv
        if self.drift_detector.update(step_return) and self.alerter is not None:
            self.alerter.notify_drift(
                detector=self.drift_detector.method,
                signal_name="portfolio_return",
            )

    def _trigger_shutdown(self, reason: str) -> None:
        logger.warning("SHUTDOWN triggered: %s", reason)
        # Liquidate position at current price
        if self.state.position > 0 and self.state._current_price > 0:
            self._execute_sell(1.0, self.state._current_price)
        self.state.shutdown_triggered = True
        self.state.shutdown_reason = reason

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
