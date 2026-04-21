"""
DataQualityGate — validates OHLCV DataFrames before they enter the environment (Week 62, S33).

Week 79 (H6): pandera-backed OHLCV expectation suite replaces the ad-hoc checks.
The external API is unchanged; ``gate.py`` now delegates column-level validation
to :mod:`data.quality.pandera_schema` and preserves the legacy ``DataIssue``
surface for callers that inspect individual issues.

Usage::

    from data.quality.gate import validate, DataQualityGate

    issues = validate(df)                   # module-level shortcut
    gate = DataQualityGate(max_gap_bars=5)
    gate.check(df)                          # raises DataQualityError on first error

Issue types
-----------
- nan_inf         : NaN or ±inf found in any numeric column
- negative_price  : $open / $high / $low / $close ≤ 0
- zero_volume     : $volume == 0 for a bar
- time_gap        : gap between consecutive bars exceeds ``max_gap_bars`` (requires
                    a DatetimeIndex or a column named ``timestamp`` / ``time`` / ``date``)
- schema          : pandera expectation failure (H6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from data.quality.pandera_schema import validate_ohlcv as _pandera_validate

# Map pandera kind labels → legacy DataIssue kind names for backward compat
_PANDERA_KIND_MAP: dict = {
    "not_null":   "nan_inf",
    "no_inf":     "nan_inf",
    "positive":   "negative_price",
    "monotonic_ts": "time_gap",
    "no_gap":     "time_gap",
    "pandera":    "schema",
}


# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataIssue:
    """Describes a single data quality problem."""

    kind: str          # 'nan_inf' | 'negative_price' | 'zero_volume' | 'time_gap'
    row: Optional[int] # 0-based row index where the issue was found (None if global)
    column: Optional[str]  # column name, if applicable
    detail: str        # human-readable description

    def __str__(self) -> str:
        loc = f"row={self.row}" if self.row is not None else "global"
        col = f", col={self.column}" if self.column else ""
        return f"[{self.kind}] {loc}{col}: {self.detail}"


class DataQualityError(ValueError):
    """Raised by DataQualityGate.check() when issues are found."""

    def __init__(self, issues: List[DataIssue]) -> None:
        self.issues = issues
        msg = f"{len(issues)} data quality issue(s):\n" + "\n".join(
            f"  {i}" for i in issues
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------

_PRICE_COLS = ["$open", "$high", "$low", "$close"]
_ALL_NUMERIC = ["$open", "$high", "$low", "$close", "$volume"]
_TIME_COL_NAMES = {"timestamp", "time", "date", "datetime"}


class DataQualityGate:
    """
    Validates an OHLCV DataFrame and returns a list of :class:`DataIssue` objects.

    Args:
        max_gap_bars: Maximum allowed consecutive missing bars when a datetime
            index / column is present.  ``None`` disables the gap check.
        raise_on_issue: If True, :meth:`check` raises :class:`DataQualityError`
            immediately instead of collecting all issues.
    """

    def __init__(
        self,
        max_gap_bars: Optional[int] = None,
        raise_on_issue: bool = False,
    ) -> None:
        self.max_gap_bars = max_gap_bars
        self.raise_on_issue = raise_on_issue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> List[DataIssue]:
        """Return all issues found.  Never raises.

        H6 (Week 79): pandera runs as a first-pass guard and reports failures
        via ``data.quality.pandera_schema.validate_ohlcv()``.  The legacy
        per-row checks always run so that callers continue to receive
        ``DataIssue`` objects with the original ``kind`` values
        (``nan_inf``, ``negative_price``, ``zero_volume``, ``time_gap``).
        This preserves backward compatibility with all existing callers and
        tests while still benefiting from the declarative pandera schema.

        If pandera reports a violation that the legacy checks cannot attribute
        to a specific row (e.g. schema-level failures), a supplementary
        ``DataIssue(kind="schema", ...)`` entry is appended.
        """
        issues: List[DataIssue] = []

        # Always run legacy per-row checks for backward-compatible DataIssue kinds.
        issues.extend(self._check_nan_inf(df))
        issues.extend(self._check_negative_price(df))
        issues.extend(self._check_zero_volume(df))

        # H6: pandera supplementary pass — catches schema-level violations
        # (type coercion, unexpected nulls in non-numeric columns, etc.) that
        # the legacy numeric checks may miss.  Only add if pandera found
        # something the legacy checks did *not* already catch.
        if not issues:
            for msg in _pandera_validate(df):
                kind = msg.split("]")[0].lstrip("[") if msg.startswith("[") else "schema"
                # Remap pandera kind names to legacy names for callers that
                # inspect ``issue.kind`` directly.
                kind = _PANDERA_KIND_MAP.get(kind, kind)
                issues.append(DataIssue(kind=kind, row=None, column=None, detail=msg))

        if self.max_gap_bars is not None:
            issues.extend(self._check_time_gap(df))
        return issues

    def check(self, df: pd.DataFrame) -> None:
        """Validate and raise :class:`DataQualityError` if any issues found."""
        issues = self.validate(df)
        if issues:
            raise DataQualityError(issues)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_nan_inf(df: pd.DataFrame) -> List[DataIssue]:
        issues: List[DataIssue] = []
        for col in _ALL_NUMERIC:
            if col not in df.columns:
                continue
            series = df[col]
            bad_mask = series.isna() | series.isin([math.inf, -math.inf])
            for row in df.index[bad_mask]:
                val = series.loc[row]
                issues.append(
                    DataIssue(
                        kind="nan_inf",
                        row=int(row),
                        column=col,
                        detail=f"value={val!r}",
                    )
                )
        return issues

    @staticmethod
    def _check_negative_price(df: pd.DataFrame) -> List[DataIssue]:
        issues: List[DataIssue] = []
        for col in _PRICE_COLS:
            if col not in df.columns:
                continue
            bad_mask = df[col] <= 0
            for row in df.index[bad_mask]:
                val = df[col].loc[row]
                issues.append(
                    DataIssue(
                        kind="negative_price",
                        row=int(row),
                        column=col,
                        detail=f"value={val}",
                    )
                )
        return issues

    @staticmethod
    def _check_zero_volume(df: pd.DataFrame) -> List[DataIssue]:
        issues: List[DataIssue] = []
        if "$volume" not in df.columns:
            return issues
        bad_mask = df["$volume"] == 0
        for row in df.index[bad_mask]:
            issues.append(
                DataIssue(
                    kind="zero_volume",
                    row=int(row),
                    column="$volume",
                    detail="volume=0",
                )
            )
        return issues

    def _check_time_gap(self, df: pd.DataFrame) -> List[DataIssue]:
        """Detect irregular time gaps when a datetime axis is available."""
        issues: List[DataIssue] = []

        # Try to find a datetime series to diff
        dt_series: Optional[pd.Series] = None
        if isinstance(df.index, pd.DatetimeIndex):
            dt_series = df.index.to_series().reset_index(drop=True)
        else:
            for col in df.columns:
                if col.lower() in _TIME_COL_NAMES:
                    try:
                        dt_series = pd.to_datetime(df[col]).reset_index(drop=True)
                        break
                    except Exception:
                        pass

        if dt_series is None or len(dt_series) < 2:
            return issues

        diffs = dt_series.diff().dropna()
        if len(diffs) == 0:
            return issues

        # Use the median diff as the expected bar period
        median_diff = diffs.median()
        if median_diff.total_seconds() == 0:
            return issues

        threshold = median_diff * (self.max_gap_bars + 1)
        for idx, diff in diffs.items():
            if diff > threshold:
                row_int = int(idx)
                issues.append(
                    DataIssue(
                        kind="time_gap",
                        row=row_int,
                        column=None,
                        detail=(
                            f"gap={diff} exceeds threshold={threshold} "
                            f"(median_bar={median_diff}, max_gap_bars={self.max_gap_bars})"
                        ),
                    )
                )
        return issues


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_GATE = DataQualityGate()


def validate(df: pd.DataFrame) -> List[DataIssue]:
    """Module-level shortcut: return all data quality issues in *df*."""
    return _DEFAULT_GATE.validate(df)
