"""
Backtesting ↔ Live Reconciliation.

Compares backtest report with paper/live trading report to identify
systematic discrepancies (slippage, fee model, timing differences).

Usage:
    from training.analysis.reconciliation import ReconciliationReport

    report = ReconciliationReport.from_reports(backtest_report, live_report)
    print(report.summary())
    report.to_json("reconciliation.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OrderDivergence:
    """Per-order slippage decomposition (S54).

    Represents the difference between the expected execution price
    (the market price at decision time) and the actual fill price.
    """
    order_id: str
    expected_price: float       # market price when order was submitted
    fill_price: float           # actual avg fill price
    quantity: float
    slippage: float             # |fill_price - expected_price| / expected_price
    slippage_cost: float        # slippage * quantity * expected_price (sign: negative = cost)
    side: str = ""              # "buy" | "sell"


@dataclass
class NormalisedMetrics:
    """Normalised metric set that both backtester and paper trader can produce."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    total_fees: float = 0.0
    avg_fill_slippage: float = 0.0  # avg |actual_price - expected_price| / expected_price
    final_portfolio_value: float = 0.0
    # Metadata
    source: str = ""  # "backtest" or "live" or "paper"
    period_start: str = ""
    period_end: str = ""


@dataclass
class ReconciliationReport:
    """Side-by-side comparison of backtest vs live metrics."""
    backtest: NormalisedMetrics
    live: NormalisedMetrics
    deltas: Dict[str, float] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    by_order: List[OrderDivergence] = field(default_factory=list)  # S54

    @classmethod
    def from_reports(
        cls,
        backtest_report: Dict[str, Any],
        live_report: Dict[str, Any],
        orders: Optional[Sequence] = None,      # S54: list of Order objects
        expected_prices: Optional[Sequence[float]] = None,  # S54: paired with orders
    ) -> "ReconciliationReport":
        """Create reconciliation from two report dicts.

        Accepts output format from:
        - BaseBacktester._calculate_metrics()
        - PaperTrader.generate_report()
        """
        bt = cls._normalise(backtest_report, source="backtest")
        lv = cls._normalise(live_report, source="live")

        deltas = {}
        warnings = []

        # Compare numeric fields
        for fld in ("total_return", "sharpe_ratio", "sortino_ratio",
                     "max_drawdown", "win_rate", "total_fees"):
            bt_val = getattr(bt, fld)
            lv_val = getattr(lv, fld)
            delta = lv_val - bt_val
            deltas[f"delta_{fld}"] = delta

            # Warn on significant divergence
            if fld == "total_return" and abs(delta) > 0.05:
                warnings.append(
                    f"Return divergence > 5%: backtest={bt_val:.4f}, live={lv_val:.4f}"
                )
            if fld == "max_drawdown" and abs(delta) > 0.03:
                warnings.append(
                    f"Drawdown divergence > 3%: backtest={bt_val:.4f}, live={lv_val:.4f}"
                )
            if fld == "sharpe_ratio" and abs(delta) > 0.5:
                warnings.append(
                    f"Sharpe divergence > 0.5: backtest={bt_val:.4f}, live={lv_val:.4f}"
                )

        # Trade count comparison
        deltas["delta_num_trades"] = lv.num_trades - bt.num_trades
        if bt.num_trades > 0 and abs(deltas["delta_num_trades"]) / bt.num_trades > 0.2:
            warnings.append(
                f"Trade count divergence > 20%: backtest={bt.num_trades}, live={lv.num_trades}"
            )

        # Slippage
        deltas["live_avg_slippage"] = lv.avg_fill_slippage
        if lv.avg_fill_slippage > 0.002:  # > 0.2%
            warnings.append(
                f"High live slippage: {lv.avg_fill_slippage:.4%}"
            )

        # S54: order-level slippage decomposition
        by_order: List[OrderDivergence] = []
        if orders is not None:
            prices = list(expected_prices) if expected_prices is not None else []
            for i, order in enumerate(orders):
                fill_price = float(getattr(order, "avg_fill_price", 0.0))
                expected = prices[i] if i < len(prices) else fill_price
                qty = float(getattr(order, "filled_amount", getattr(order, "amount", 0.0)))
                if expected <= 0 or qty <= 0:
                    continue
                slip_frac = abs(fill_price - expected) / expected
                # positive = unfavourable (overpaid on buy, underpaid on sell)
                side = getattr(order, "side", "")
                if side == "buy":
                    slip_cost = (fill_price - expected) * qty
                else:
                    slip_cost = (expected - fill_price) * qty
                by_order.append(OrderDivergence(
                    order_id=str(getattr(order, "order_id", i)),
                    expected_price=expected,
                    fill_price=fill_price,
                    quantity=qty,
                    slippage=slip_frac,
                    slippage_cost=slip_cost,
                    side=side,
                ))
            if by_order:
                avg_order_slip = sum(d.slippage for d in by_order) / len(by_order)
                deltas["order_avg_slippage"] = avg_order_slip
                if avg_order_slip > 0.002:
                    warnings.append(
                        f"Order-level avg slippage > 0.2%: {avg_order_slip:.4%}"
                    )

        return cls(backtest=bt, live=lv, deltas=deltas, warnings=warnings, by_order=by_order)

    @staticmethod
    def _normalise(report: Dict[str, Any], source: str) -> NormalisedMetrics:
        """Map various report formats to NormalisedMetrics."""
        return NormalisedMetrics(
            total_return=float(report.get("total_return", 0.0)),
            sharpe_ratio=float(report.get("sharpe_ratio", 0.0)),
            sortino_ratio=float(report.get("sortino_ratio", 0.0)),
            max_drawdown=float(report.get("max_drawdown", 0.0)),
            num_trades=int(report.get("num_trades", report.get("total_trades", 0))),
            win_rate=float(report.get("win_rate", 0.0)),
            total_fees=float(report.get("total_fees", 0.0)),
            avg_fill_slippage=float(report.get("avg_fill_slippage", 0.0)),
            final_portfolio_value=float(report.get("final_portfolio_value", 0.0)),
            source=source,
            period_start=str(report.get("period_start", "")),
            period_end=str(report.get("period_end", "")),
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "RECONCILIATION REPORT: Backtest vs Live",
            "=" * 60,
            "",
            f"{'Metric':<25} {'Backtest':>12} {'Live':>12} {'Delta':>12}",
            "-" * 60,
        ]
        for fld in ("total_return", "sharpe_ratio", "sortino_ratio",
                     "max_drawdown", "win_rate", "total_fees"):
            bt_val = getattr(self.backtest, fld)
            lv_val = getattr(self.live, fld)
            delta = self.deltas.get(f"delta_{fld}", 0.0)
            fmt = ".4f" if fld != "total_fees" else ".2f"
            lines.append(
                f"{fld:<25} {bt_val:>12{fmt}} {lv_val:>12{fmt}} {delta:>+12{fmt}}"
            )

        lines.append(f"{'num_trades':<25} {self.backtest.num_trades:>12} "
                      f"{self.live.num_trades:>12} {self.deltas.get('delta_num_trades', 0):>+12}")
        lines.append(f"{'avg_fill_slippage':<25} {'n/a':>12} "
                      f"{self.live.avg_fill_slippage:>12.4%} {'':>12}")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Export as JSON dict (optionally write to file)."""
        data = {
            "backtest": asdict(self.backtest),
            "live": asdict(self.live),
            "deltas": self.deltas,
            "warnings": self.warnings,
            "by_order": [asdict(d) for d in self.by_order],  # S54
        }
        if path:
            Path(path).write_text(json.dumps(data, indent=2))
            logger.info("Reconciliation report saved to %s", path)
        return data
