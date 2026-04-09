"""
Immutable Audit Logger — append-only jsonl with SHA-256 hash chain.

Each record has the form:
    {"ts": <ISO>, "type": <str>, "payload": <dict>, "hash": <sha256hex>}

where hash = sha256(prev_hash_hex + json(payload)).

The chain starts with a genesis hash of "0" * 64.

Thread-safe: a single lock serialises all writes.  fsync is optional but
recommended in production (fsync=True).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_GENESIS_HASH = "0" * 64


def _sha256(prev_hash: str, payload: Dict[str, Any]) -> str:
    """Compute sha256(prev_hash + canonical_json(payload))."""
    raw = prev_hash + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class AuditLogger:
    """
    Append-only jsonl audit logger with hash-chain integrity.

    Parameters
    ----------
    log_path : str
        Path to the .jsonl file.  Parent directories must exist.
    fsync : bool
        If True, call os.fsync after every write (slower but crash-safe).
    """

    def __init__(self, log_path: str, fsync: bool = False) -> None:
        self._log_path = log_path
        self._fsync = fsync
        self._lock = threading.Lock()
        self._prev_hash: str = _GENESIS_HASH

        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

        # Replay existing records to restore chain state.
        if os.path.exists(log_path):
            self._prev_hash = self._replay_chain(log_path)

        self._fh = open(log_path, "a", encoding="utf-8")  # append mode
        logger.info("AuditLogger initialised | path=%s prev_hash=%.8s…", log_path, self._prev_hash)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_order(self, order: Any) -> None:
        """Record an order submission event."""
        payload = self._order_to_dict(order)
        self._write("order", payload)

    def log_fill(self, fill: Any) -> None:
        """Record an order fill event.

        fill may be an Order-like object or a plain dict.
        """
        if isinstance(fill, dict):
            payload = dict(fill)
        else:
            payload = self._order_to_dict(fill)
        self._write("fill", payload)

    def log_risk_event(self, event: Dict[str, Any]) -> None:
        """Record a risk management event (kill-switch, drawdown breach, etc.)."""
        self._write("risk_event", dict(event))

    def log_model_decision(self, action: Any, obs_hash: str) -> None:
        """Record a model decision.

        Parameters
        ----------
        action : scalar or list
            The action chosen by the model.
        obs_hash : str
            sha256 hex digest of the observation (use sha256(obs.tobytes())).
        """
        payload = {
            "action": action if isinstance(action, (int, float, str)) else list(action),
            "obs_hash": obs_hash,
        }
        self._write("model_decision", payload)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record_type: str, payload: Dict[str, Any]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            h = _sha256(self._prev_hash, payload)
            record = {"ts": ts, "type": record_type, "payload": payload, "hash": h}
            line = json.dumps(record, separators=(",", ":")) + "\n"
            self._fh.write(line)
            if self._fsync:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            self._prev_hash = h

    @staticmethod
    def _replay_chain(log_path: str) -> str:
        """Read existing log file and return the last hash (to resume chain)."""
        prev = _GENESIS_HASH
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    prev = record.get("hash", prev)
        except Exception as e:
            logger.warning("AuditLogger: failed to replay chain from %s: %s", log_path, e)
        return prev

    @staticmethod
    def _order_to_dict(order: Any) -> Dict[str, Any]:
        """Convert an Order object (or dict) to a plain dict."""
        if isinstance(order, dict):
            d = dict(order)
        elif hasattr(order, "__dict__"):
            d = {k: v for k, v in order.__dict__.items() if not k.startswith("_")}
        else:
            d = {"repr": str(order)}
        # Normalise datetime fields to ISO strings for JSON serialisation.
        for k, v in list(d.items()):
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d
