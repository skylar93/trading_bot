"""
Exchange Snapshot — Week 73 (F7)

Point-in-time read of exchange state: positions, open orders, balance.
All methods are best-effort: they return empty/zero on failure so callers
can decide severity (F8 halt policy lives in PaperTrader, not here).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExchangeSnapshot:
    """
    Wraps CCXT REST calls to produce a point-in-time exchange snapshot.

    Parameters
    ----------
    exchange :
        Initialised CCXT exchange object with REST credentials.
    symbol : str
        Default trading pair, e.g. ``"BTC/USDT"``.
    """

    def __init__(self, exchange, symbol: str = "BTC/USDT") -> None:
        self._exchange = exchange
        self.symbol = symbol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open positions for *symbol*.

        Returns a list of normalised position dicts::

            {"symbol": str, "qty": float, "entry_price": float,
             "side": str, "unrealised_pnl": float}

        Returns ``[]`` on API error or missing ``fetch_positions`` support.
        """
        sym = symbol or self.symbol
        try:
            raw = self._exchange.fetch_positions([sym])
            return [self._normalise_position(p) for p in raw]
        except Exception as exc:
            logger.warning("ExchangeSnapshot.get_positions failed: %s", exc)
            return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all open orders for *symbol*.

        Returns a list of normalised order dicts::

            {"order_id": str, "symbol": str, "side": str, "amount": float,
             "filled": float, "remaining": float, "price": float,
             "type": str, "status": str}

        Returns ``[]`` on API error.
        """
        sym = symbol or self.symbol
        try:
            raw = self._exchange.fetch_open_orders(sym)
            return [self._normalise_order(o) for o in raw]
        except Exception as exc:
            logger.warning("ExchangeSnapshot.get_open_orders failed: %s", exc)
            return []

    def get_balance(self) -> Dict[str, Any]:
        """Fetch account balance.

        Returns::

            {"free": {asset: float}, "used": {asset: float},
             "total": {asset: float}}

        Returns empty sub-dicts on API error.
        """
        try:
            raw = self._exchange.fetch_balance()
            return {
                "free":  {k: float(v) for k, v in (raw.get("free") or {}).items() if v},
                "used":  {k: float(v) for k, v in (raw.get("used") or {}).items() if v},
                "total": {k: float(v) for k, v in (raw.get("total") or {}).items() if v},
            }
        except Exception as exc:
            logger.warning("ExchangeSnapshot.get_balance failed: %s", exc)
            return {"free": {}, "used": {}, "total": {}}

    def snapshot(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Convenience: positions + open orders + balance in one call.

        Returns::

            {"symbol": str, "positions": [...], "open_orders": [...],
             "balance": {...}}
        """
        sym = symbol or self.symbol
        return {
            "symbol":      sym,
            "positions":   self.get_positions(sym),
            "open_orders": self.get_open_orders(sym),
            "balance":     self.get_balance(),
        }

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_position(raw: Dict[str, Any]) -> Dict[str, Any]:
        qty = raw.get("contracts") or raw.get("amount") or raw.get("size") or 0
        entry = raw.get("entryPrice") or raw.get("avgPrice") or raw.get("average") or 0
        return {
            "symbol":        raw.get("symbol", ""),
            "qty":           float(qty),
            "entry_price":   float(entry),
            "side":          (raw.get("side") or "long").lower(),
            "unrealised_pnl": float(raw.get("unrealizedPnl") or 0),
        }

    @staticmethod
    def _normalise_order(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "order_id":  str(raw.get("id", "")),
            "symbol":    raw.get("symbol", ""),
            "side":      (raw.get("side") or "").lower(),
            "amount":    float(raw.get("amount") or 0),
            "filled":    float(raw.get("filled") or 0),
            "remaining": float(raw.get("remaining") or 0),
            "price":     float(raw.get("price") or 0),
            "type":      raw.get("type", "market"),
            "status":    raw.get("status", "open"),
        }
