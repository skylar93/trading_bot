"""
A6 Cost Decomposition — 4-axis P&L breakdown (Week 90).

Decomposes each fill's P&L into four additive components:

    total_pnl = signal_pnl + slippage_pnl + fee_pnl + funding_pnl

Where:
    signal_pnl    — strategy alpha: (mid_at_submit - entry_price) * qty  [sells only]
    slippage_pnl  — execution friction vs mid: (fill_price - mid_at_submit) * qty * side_sign
    fee_pnl       — -fee_paid
    funding_pnl   — -funding_accrued  (perps; 0 for spot)

Algebraic identity check:
    signal_pnl + slippage_pnl = (fill_price - entry_price) * qty  [for sells]
    → total_pnl = (fill_price - entry_price) * qty - fee - funding  [net realized PnL]

For buys (position-opening):
    signal_pnl = 0  (no realized PnL yet)
    slippage_pnl = -(fill_price - mid_at_submit) * qty  (cost if fill > mid)

Usage:
    from deployment.analysis.cost_decomposition import FillRecord, decompose, CostDecomposer

    fill = FillRecord(
        fill_id="f001",
        timestamp=datetime.utcnow(),
        side="sell",
        fill_price=110.0,
        qty=1.0,
        entry_price=100.0,
        mid_at_submit=109.5,
    )
    d = decompose(fill, fee_paid=0.11)
    # d.total_pnl == d.signal_pnl + d.slippage_pnl + d.fee_pnl + d.funding_pnl
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """Input descriptor for a single fill (buy or sell)."""
    fill_id: str
    timestamp: datetime
    side: str           # "buy" | "sell"
    fill_price: float
    qty: float
    entry_price: float  # 0.0 for buys (unrealized); used only for sells
    mid_at_submit: float  # mid-market price at decision time; use fill_price if unknown


@dataclass
class FillDecomposition:
    """4-axis P&L decomposition for a single fill."""
    fill_id: str
    timestamp: datetime
    signal_pnl: float       # strategy alpha (sells only; 0 for buys)
    slippage_pnl: float     # execution friction vs mid
    fee_pnl: float          # -fee_paid  (always ≤ 0)
    funding_pnl: float      # -funding_accrued (always ≤ 0 for longs)
    total_pnl: float        # algebraic sum of the four axes

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class DailyCostSummary:
    """Aggregate 4-axis breakdown for a single calendar day."""
    date: date
    num_fills: int
    num_sells: int
    total_signal_pnl: float
    total_slippage_pnl: float
    total_fee_pnl: float
    total_funding_pnl: float
    total_pnl: float
    avg_slippage_per_fill: float   # average |slippage_pnl| per fill


@dataclass
class CumulativeCostSummary:
    """Running totals across all tracked fills."""
    num_fills: int
    num_sells: int
    total_signal_pnl: float
    total_slippage_pnl: float
    total_fee_pnl: float
    total_funding_pnl: float
    total_pnl: float
    avg_slippage_per_fill: float


# ---------------------------------------------------------------------------
# Core decomposition function
# ---------------------------------------------------------------------------

def decompose(
    fill: FillRecord,
    fee_paid: float,
    funding_accrued: float = 0.0,
) -> FillDecomposition:
    """Decompose a single fill into 4 P&L axes.

    Parameters
    ----------
    fill :
        FillRecord with fill_price, mid_at_submit, entry_price, side.
    fee_paid :
        Total fee paid on this fill (positive number; stored as negative fee_pnl).
    funding_accrued :
        Funding payment accrued on this fill's notional (positive = cost).
        Pass 0.0 for spot (default).

    Returns
    -------
    FillDecomposition
        Decomposes P&L into signal / slippage / fee / funding axes.
        total_pnl is guaranteed to equal the algebraic sum.
    """
    if fill.side == "sell":
        # Strategy alpha: what the signal would have earned at mid
        signal_pnl = (fill.mid_at_submit - fill.entry_price) * fill.qty
        # Execution quality: improvement or degradation vs mid
        slippage_pnl = (fill.fill_price - fill.mid_at_submit) * fill.qty
    else:
        # Buy: no realized signal PnL yet; slippage is the cost above mid
        signal_pnl = 0.0
        slippage_pnl = -(fill.fill_price - fill.mid_at_submit) * fill.qty

    fee_pnl = -float(fee_paid)
    funding_pnl = -float(funding_accrued)
    total_pnl = signal_pnl + slippage_pnl + fee_pnl + funding_pnl

    return FillDecomposition(
        fill_id=fill.fill_id,
        timestamp=fill.timestamp,
        signal_pnl=signal_pnl,
        slippage_pnl=slippage_pnl,
        fee_pnl=fee_pnl,
        funding_pnl=funding_pnl,
        total_pnl=total_pnl,
    )


# ---------------------------------------------------------------------------
# CostDecomposer — accumulates fills and produces summaries
# ---------------------------------------------------------------------------

class CostDecomposer:
    """Accumulates FillDecomposition objects and produces daily/cumulative summaries.

    Thread-safety: not thread-safe; wrap in a lock if calling from multiple threads.
    """

    def __init__(self) -> None:
        self._fills: List[FillDecomposition] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, decomp: FillDecomposition) -> None:
        """Append a pre-computed FillDecomposition."""
        self._fills.append(decomp)

    def add_fill(
        self,
        fill: FillRecord,
        fee_paid: float,
        funding_accrued: float = 0.0,
    ) -> FillDecomposition:
        """Decompose and accumulate a fill in one step. Returns the FillDecomposition."""
        d = decompose(fill, fee_paid=fee_paid, funding_accrued=funding_accrued)
        self._fills.append(d)
        return d

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def daily_summary(self, target_date: date) -> Optional[DailyCostSummary]:
        """Aggregate fills for a specific calendar day (UTC date)."""
        day_fills = [
            f for f in self._fills
            if f.timestamp.date() == target_date
        ]
        if not day_fills:
            return None
        return self._summarise_day(target_date, day_fills)

    def all_daily_summaries(self) -> List[DailyCostSummary]:
        """Return one DailyCostSummary per calendar day that has fills."""
        by_date: Dict[date, List[FillDecomposition]] = defaultdict(list)
        for f in self._fills:
            by_date[f.timestamp.date()].append(f)
        return [
            self._summarise_day(d, fills)
            for d, fills in sorted(by_date.items())
        ]

    def cumulative_summary(self) -> CumulativeCostSummary:
        """Running totals across all tracked fills."""
        fills = self._fills
        n = len(fills)
        n_sells = sum(1 for f in fills if f.slippage_pnl != 0 or f.signal_pnl != 0)
        total_signal = sum(f.signal_pnl for f in fills)
        total_slip = sum(f.slippage_pnl for f in fills)
        total_fee = sum(f.fee_pnl for f in fills)
        total_fund = sum(f.funding_pnl for f in fills)
        total = sum(f.total_pnl for f in fills)
        avg_slip = abs(total_slip) / n if n > 0 else 0.0
        return CumulativeCostSummary(
            num_fills=n,
            num_sells=n_sells,
            total_signal_pnl=total_signal,
            total_slippage_pnl=total_slip,
            total_fee_pnl=total_fee,
            total_funding_pnl=total_fund,
            total_pnl=total,
            avg_slippage_per_fill=avg_slip,
        )

    def fills(self) -> List[FillDecomposition]:
        """Read-only view of accumulated fills."""
        return list(self._fills)

    # ------------------------------------------------------------------
    # Factory: load from audit.jsonl
    # ------------------------------------------------------------------

    @classmethod
    def from_audit_log(
        cls,
        path: Path,
        enable_funding: bool = False,
        mid_price_key: str = "mid_price",
    ) -> "CostDecomposer":
        """Build a CostDecomposer by replaying fill records from an audit log.

        Expected audit.jsonl record format (type=="fill"):
            {
              "ts": "2026-04-28T00:01:00Z",
              "type": "fill",
              "payload": {
                "fill_id": "f001",          # optional; derived from ts if absent
                "side": "buy"|"sell",
                "price": 65000.0,           # fill_price
                "mid_price": 64990.0,       # optional; defaults to price
                "quantity": 0.001,
                "fee": 6.5,
                "pnl": 0.0,                 # gross market move (sells only)
                "entry_price": 60000.0,     # optional; inferred from pnl if absent
                "funding": 0.0              # optional; only if enable_funding=True
              }
            }
        """
        decomposer = cls()
        path = Path(path)
        if not path.exists():
            logger.warning("audit log not found: %s", path)
            return decomposer

        _buy_prices: Dict[str, float] = {}  # symbol → last buy fill_price (for entry_price inference)
        n_skipped = 0

        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("audit log line %d: JSON parse error, skipping", lineno)
                    n_skipped += 1
                    continue

                if record.get("type") != "fill":
                    continue

                payload = record.get("payload", {})
                try:
                    ts_str = record.get("ts", "")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    side = payload["side"]
                    fill_price = float(payload["price"])
                    qty = float(payload["quantity"])
                    fee = float(payload.get("fee", 0.0))
                    mid = float(payload.get(mid_price_key, fill_price))
                    fill_id = payload.get("fill_id", f"fill_{lineno}")
                    funding = float(payload.get("funding", 0.0)) if enable_funding else 0.0

                    # Infer entry_price for sells
                    if side == "sell":
                        gross_pnl = float(payload.get("pnl", 0.0))
                        entry_price_raw = float(payload.get("entry_price", 0.0))
                        if entry_price_raw > 0:
                            entry_price = entry_price_raw
                        elif abs(qty) > 1e-10 and gross_pnl != 0.0:
                            # pnl = (fill_price - entry_price) * qty → entry = fill - pnl/qty
                            entry_price = fill_price - gross_pnl / qty
                        else:
                            entry_price = _buy_prices.get("_last", fill_price)
                    else:
                        entry_price = 0.0
                        _buy_prices["_last"] = fill_price

                    fill = FillRecord(
                        fill_id=fill_id,
                        timestamp=ts,
                        side=side,
                        fill_price=fill_price,
                        qty=qty,
                        entry_price=entry_price,
                        mid_at_submit=mid,
                    )
                    decomposer.add_fill(fill, fee_paid=fee, funding_accrued=funding)

                except (KeyError, ValueError) as exc:
                    logger.debug("audit log line %d: parse error %s, skipping", lineno, exc)
                    n_skipped += 1

        logger.info(
            "CostDecomposer loaded %d fills from %s (%d skipped)",
            len(decomposer._fills), path, n_skipped,
        )
        return decomposer

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_day(
        target_date: date,
        fills: Iterable[FillDecomposition],
    ) -> DailyCostSummary:
        fills_list = list(fills)
        n = len(fills_list)
        n_sells = sum(1 for f in fills_list if f.signal_pnl != 0.0 or (f.slippage_pnl != 0 and f.signal_pnl == 0))
        total_signal = sum(f.signal_pnl for f in fills_list)
        total_slip = sum(f.slippage_pnl for f in fills_list)
        total_fee = sum(f.fee_pnl for f in fills_list)
        total_fund = sum(f.funding_pnl for f in fills_list)
        total = sum(f.total_pnl for f in fills_list)
        avg_slip = abs(total_slip) / n if n > 0 else 0.0
        return DailyCostSummary(
            date=target_date,
            num_fills=n,
            num_sells=n_sells,
            total_signal_pnl=total_signal,
            total_slippage_pnl=total_slip,
            total_fee_pnl=total_fee,
            total_funding_pnl=total_fund,
            total_pnl=total,
            avg_slippage_per_fill=avg_slip,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_decomposition_identity(
    decomp: FillDecomposition,
    realized_pnl: float,
    tol: float = 0.01,
) -> bool:
    """Return True if 4-axis sum matches realized_pnl within tol dollars.

    Used for regression testing (A6.5).
    """
    computed = decomp.signal_pnl + decomp.slippage_pnl + decomp.fee_pnl + decomp.funding_pnl
    return abs(computed - realized_pnl) < tol and abs(decomp.total_pnl - realized_pnl) < tol
