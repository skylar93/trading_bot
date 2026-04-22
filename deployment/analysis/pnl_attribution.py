"""
P&L Attribution — Week 66 (S51).

Decomposes each trade's realised P&L into four additive components:

    net_pnl = market_move - slippage_cost - fees

Where:
    market_move   — gross price-change PnL: (exit_price - entry_price) * qty
    slippage_cost — execution quality cost (|fill_price - expected_price| / expected_price * qty * price)
    fees          — total transaction costs paid
    net_pnl       — model's bottom-line contribution

The decomposition satisfies:
    market_move = net_pnl + slippage_cost + fees

Usage:
    from deployment.analysis.pnl_attribution import PnLAttributor
    from deployment.paper_trader import Trade

    attributor = PnLAttributor()
    attributions = attributor.attribute(trades, slippage_records=trader._slippage_records)
    summary = attributor.summarise(attributions)
    print(summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency; only used when slippage_model is passed.
_SlippageModel = None


def _get_slippage_model_cls():
    global _SlippageModel
    if _SlippageModel is None:
        from deployment.execution.slippage_model import SlippageModel
        _SlippageModel = SlippageModel
    return _SlippageModel


@dataclass
class TradeAttribution:
    """Per-trade P&L decomposition."""
    trade_index: int
    side: str             # "sell" (only closing trades have realised PnL)
    entry_price: float
    exit_price: float
    quantity: float
    market_move: float    # (exit_price - entry_price) * quantity  [gross PnL]
    slippage_cost: float  # execution quality cost  (≥ 0)
    fees: float           # transaction costs paid  (≥ 0)
    net_pnl: float        # market_move - slippage_cost - fees

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionSummary:
    """Aggregate P&L attribution across all trades."""
    num_closing_trades: int
    total_market_move: float
    total_slippage_cost: float
    total_fees: float
    total_net_pnl: float
    slippage_pct_of_gross: float   # total_slippage_cost / |total_market_move|, 0 if gross==0
    fees_pct_of_gross: float       # total_fees / |total_market_move|, 0 if gross==0

    # Per-trade averages
    avg_market_move: float
    avg_slippage_cost: float
    avg_fees: float
    avg_net_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PnLAttributor:
    """
    Attributes realised P&L to its cost components.

    Parameters
    ----------
    fee_bps : float
        Expected fee in basis-points.  Used only when fee is not directly
        available on the trade object.  Default = 10 bps (0.1 %).
    slippage_model : SlippageModel, optional
        When provided, expected slippage is predicted per-trade from the model
        (vol, size, spread_bps features).  Residual between observed and
        predicted slippage is folded into market_move.
    """

    def __init__(self, fee_bps: float = 10.0, slippage_model=None) -> None:
        self._fee_bps = fee_bps
        self._slippage_model = slippage_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        trades: Sequence,
        slippage_records: Optional[Sequence[float]] = None,
        entry_price: Optional[float] = None,
        slippage_features: Optional[Sequence[Dict[str, float]]] = None,
    ) -> List[TradeAttribution]:
        """
        Compute per-trade attribution for a list of Trade objects.

        Parameters
        ----------
        trades :
            List of ``Trade`` dataclass instances from ``deployment.paper_trader``.
        slippage_records :
            Optional list of observed slippage fractions
            (``|fill_price - expected_price| / expected_price``).  Paired
            with closing trades in order; excess records are ignored.
        entry_price :
            Override for the entry price (used when the Trade object doesn't
            contain it directly — legacy path).  Usually None; the attributor
            infers entry price from ``trade.pnl``.
        slippage_features :
            Optional list of feature dicts (vol, size, spread_bps) for each
            closing trade.  When provided and a slippage_model is attached,
            expected_slippage is predicted from the model; the residual
            (observed − expected) is folded into market_move so that
            slippage_cost reflects only the *model-predicted* component.

        Returns
        -------
        List[TradeAttribution]
            One entry per *closing* (sell) trade with non-zero quantity.
        """
        slip_iter = iter(slippage_records or [])
        feat_iter = iter(slippage_features or [])
        results: List[TradeAttribution] = []
        _last_buy_price: Optional[float] = None

        for idx, trade in enumerate(trades):
            if trade.side == "buy":
                _last_buy_price = trade.price
                continue

            # ----- closing (sell) trade -----
            qty = float(trade.quantity)
            if qty < 1e-10:
                continue

            exit_p = float(trade.price)
            fee = float(trade.fee)

            # Recover entry_price:
            #   apply_sell returns pnl = (exit_price - entry_price) * qty
            #   So entry_price = exit_price - pnl / qty
            raw_pnl = float(trade.pnl)
            if abs(qty) > 1e-10 and raw_pnl != 0.0:
                inferred_entry = exit_p - raw_pnl / qty
            elif _last_buy_price is not None:
                inferred_entry = _last_buy_price
            elif entry_price is not None:
                inferred_entry = entry_price
            else:
                inferred_entry = exit_p  # no info → no market_move

            # Observed slippage fraction from records
            slip_frac = next(slip_iter, 0.0)
            observed_slip_cost = slip_frac * qty * exit_p

            # Model-predicted slippage (R8): if model + features available,
            # use predicted bps as slippage_cost; residual goes to market_move.
            feat = next(feat_iter, None)
            if self._slippage_model is not None and feat is not None:
                predicted_bps = self._slippage_model.predict(feat)
                expected_slip_cost = (predicted_bps / 10_000.0) * qty * exit_p
                # residual = observed − expected folds back into gross P&L
                residual = observed_slip_cost - expected_slip_cost
                slippage_cost = expected_slip_cost
            else:
                slippage_cost = observed_slip_cost
                residual = 0.0

            market_move = (exit_p - inferred_entry) * qty + residual
            net_pnl = market_move - slippage_cost - fee

            results.append(TradeAttribution(
                trade_index=idx,
                side="sell",
                entry_price=inferred_entry,
                exit_price=exit_p,
                quantity=qty,
                market_move=market_move,
                slippage_cost=slippage_cost,
                fees=fee,
                net_pnl=net_pnl,
            ))

        return results

    def summarise(self, attributions: List[TradeAttribution]) -> AttributionSummary:
        """Aggregate attribution list into a summary."""
        if not attributions:
            return AttributionSummary(
                num_closing_trades=0,
                total_market_move=0.0,
                total_slippage_cost=0.0,
                total_fees=0.0,
                total_net_pnl=0.0,
                slippage_pct_of_gross=0.0,
                fees_pct_of_gross=0.0,
                avg_market_move=0.0,
                avg_slippage_cost=0.0,
                avg_fees=0.0,
                avg_net_pnl=0.0,
            )

        n = len(attributions)
        total_mm = sum(a.market_move for a in attributions)
        total_slip = sum(a.slippage_cost for a in attributions)
        total_fees = sum(a.fees for a in attributions)
        total_net = sum(a.net_pnl for a in attributions)

        abs_gross = abs(total_mm)
        slip_pct = total_slip / abs_gross if abs_gross > 1e-10 else 0.0
        fees_pct = total_fees / abs_gross if abs_gross > 1e-10 else 0.0

        return AttributionSummary(
            num_closing_trades=n,
            total_market_move=total_mm,
            total_slippage_cost=total_slip,
            total_fees=total_fees,
            total_net_pnl=total_net,
            slippage_pct_of_gross=slip_pct,
            fees_pct_of_gross=fees_pct,
            avg_market_move=total_mm / n,
            avg_slippage_cost=total_slip / n,
            avg_fees=total_fees / n,
            avg_net_pnl=total_net / n,
        )

    def to_exporter_fields(self, summary: AttributionSummary) -> Dict[str, float]:
        """Return a dict suitable for passing to ``MetricsExporter.update(**fields)``."""
        return {
            "pnl_market_move": summary.total_market_move,
            "pnl_slippage_cost": summary.total_slippage_cost,
            "pnl_fees": summary.total_fees,
            "pnl_net": summary.total_net_pnl,
        }
