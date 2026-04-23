"""
Real-time schema drift guard — Week 83 (R13 / G6).

Validates each incoming OHLCV tick/bar from ccxt_live.py against the
expected schema at runtime.  Drift is reported immediately and, per the
``on_schema_drift`` config, either halts the trader or fires a warning alert.

Expected schema (from pandera_schema.py OHLCV_SCHEMA):
    Keys   : $open, $high, $low, $close, $volume (plus any extra allowed)
    Dtypes : float-compatible
    Ranges : all values > 0 (NaN / inf / negative ⟹ drift)

Drift triggers:
    1. Unexpected key set  — required key missing in incoming record
    2. Wrong dtype         — value not coercible to float
    3. Value range         — NaN, inf, or ≤ 0 for price/volume fields

Usage::

    from data.quality.stream_validator import StreamValidator, SchemaDrift

    validator = StreamValidator(on_schema_drift="halt", alerter=alerter)
    for tick in live_feed:
        try:
            validator.validate(tick)
        except SchemaDrift as e:
            # already alerted; halt or continue per policy
            break
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Required OHLCV keys that must be present in every tick record.
_REQUIRED_KEYS: Set[str] = {"$open", "$high", "$low", "$close", "$volume"}

# All required keys must be positive finite floats.
_POSITIVE_KEYS: Set[str] = _REQUIRED_KEYS


class SchemaDrift(Exception):
    """Raised when a tick fails schema validation and policy is 'halt'."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class StreamValidator:
    """Validate each incoming tick against the expected OHLCV schema.

    Parameters
    ----------
    on_schema_drift:
        ``"halt"``  — raise :class:`SchemaDrift` after alerting (default, safe).
        ``"warn"``  — alert only; do not raise, allow caller to continue.
    alerter:
        Optional :class:`deployment.monitoring.alerter.TradingAlerter` instance.
        When provided, ``alerter.schema_drift_detected()`` is called on every drift.
    """

    def __init__(
        self,
        on_schema_drift: str = "halt",
        alerter: Optional[Any] = None,
    ) -> None:
        if on_schema_drift not in ("halt", "warn"):
            raise ValueError(
                f"on_schema_drift must be 'halt' or 'warn', got {on_schema_drift!r}"
            )
        self.on_schema_drift = on_schema_drift
        self.alerter = alerter
        self._drift_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, record: Dict[str, Any]) -> None:
        """Validate a single tick/bar record.

        Parameters
        ----------
        record:
            Dict representing one OHLCV bar.  May contain extra keys.

        Raises
        ------
        SchemaDrift
            When drift is detected and ``on_schema_drift == "halt"``.
        """
        detail = self._check(record)
        if detail is None:
            return

        self._drift_count += 1
        logger.warning("Schema drift detected (count=%d): %s", self._drift_count, detail)

        if self.alerter is not None:
            self.alerter.schema_drift_detected(detail, on_drift=self.on_schema_drift)

        if self.on_schema_drift == "halt":
            raise SchemaDrift(detail)

    @property
    def drift_count(self) -> int:
        """Total number of drifting records observed since instantiation."""
        return self._drift_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check(self, record: Dict[str, Any]) -> Optional[str]:
        """Return a drift description string, or None if the record is clean."""
        # 1. Required key presence
        missing = _REQUIRED_KEYS - set(record.keys())
        if missing:
            return f"missing required keys: {sorted(missing)}"

        # 2. Dtype + value range for each required key
        for key in _POSITIVE_KEYS:
            raw = record[key]

            # Dtype: must be coercible to float
            try:
                val = float(raw)
            except (TypeError, ValueError):
                return f"key={key!r} value={raw!r} is not float-coercible (dtype={type(raw).__name__})"

            # NaN check
            if math.isnan(val):
                return f"key={key!r} value is NaN"

            # Inf check
            if math.isinf(val):
                return f"key={key!r} value is ±inf"

            # Positive range
            if val <= 0:
                return f"key={key!r} value={val} is not > 0 (expected positive price/volume)"

        return None
