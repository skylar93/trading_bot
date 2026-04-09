"""
StateStore: SQLite-backed crash-recovery store for PaperTrader.

Phase 6 Week 56 (S1). Single-process WAL SQLite that holds the latest snapshot
of positions / orders / account state plus the full TradingState blob needed to
resume an in-flight episode after a crash.

Design notes
------------
* SQLite with ``PRAGMA journal_mode=WAL`` for crash safety + concurrent reads.
* `save_snapshot(state)` is called from PaperTrader on every step. It writes
  one row per symbol position, replaces the open orders table for that
  snapshot, upserts a single ``account_state`` row (id=1), and stores the full
  TradingState dict as JSON in ``account_state.full_state_json`` so recovery
  is exact (avoids floating-point loss when reconstructing portfolio_history).
* `load_latest()` returns the dict produced by ``TradingState.to_dict()`` or
  None if no snapshot has been written yet.
* Thread-safe: a single ``threading.RLock`` guards all DB access. SQLite
  connections are created with ``check_same_thread=False`` so the same
  connection can be reused across threads.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
    qty         REAL NOT NULL,
    avg_price   REAL NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS account_state (
    id               INTEGER PRIMARY KEY,
    cash             REAL NOT NULL,
    equity           REAL NOT NULL,
    updated_at       TEXT NOT NULL,
    full_state_json  TEXT NOT NULL
);
"""


class StateStore:
    """SQLite snapshot store. See module docstring for semantics."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Snapshot API
    # ------------------------------------------------------------------

    def save_snapshot(self, state: Dict[str, Any]) -> None:
        """
        Persist a TradingState snapshot.

        Parameters
        ----------
        state : dict
            The output of ``TradingState.to_dict()``. Must contain at least
            ``cash``, ``equity``, ``symbol``, ``position``, ``entry_price``.
        """
        self._reject_nonfinite(state)
        full_json = json.dumps(state, default=str)
        now = datetime.utcnow().isoformat()
        symbol = state.get("symbol", "DEFAULT")
        position = float(state.get("position", 0.0))
        entry_price = float(state.get("entry_price", 0.0))
        cash = float(state.get("cash", 0.0))
        equity = float(state.get("equity", cash))
        orders = state.get("orders", []) or []

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute(
                    "INSERT INTO positions(symbol, qty, avg_price, updated_at) "
                    "VALUES(?,?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET "
                    "qty=excluded.qty, avg_price=excluded.avg_price, "
                    "updated_at=excluded.updated_at",
                    (symbol, position, entry_price, now),
                )
                cur.execute("DELETE FROM orders")
                for od in orders:
                    cur.execute(
                        "INSERT INTO orders(order_id, symbol, side, qty, price, status, created_at) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (
                            str(od.get("order_id", "")),
                            str(od.get("symbol", symbol)),
                            str(od.get("side", "")),
                            float(od.get("qty", 0.0)),
                            float(od.get("price", 0.0)),
                            str(od.get("status", "")),
                            str(od.get("created_at", now)),
                        ),
                    )
                cur.execute(
                    "INSERT INTO account_state(id, cash, equity, updated_at, full_state_json) "
                    "VALUES(1,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "cash=excluded.cash, equity=excluded.equity, "
                    "updated_at=excluded.updated_at, full_state_json=excluded.full_state_json",
                    (cash, equity, now, full_json),
                )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent snapshot dict, or None if empty."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT full_state_json FROM account_state WHERE id=1"
            )
            row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def clear(self) -> None:
        """Wipe all snapshot data (testing / fresh-start)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute("DELETE FROM positions")
                cur.execute("DELETE FROM orders")
                cur.execute("DELETE FROM account_state")
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_nonfinite(state: Dict[str, Any]) -> None:
        """Refuse to persist NaN/inf values; better to crash than to silently
        save corrupted state we can't reason about on restore."""
        for k, v in state.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"Refusing to persist non-finite value for '{k}': {v!r}")

    # context manager sugar
    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
