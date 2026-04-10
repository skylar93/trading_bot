"""
CSVDataSource — lazy-loading file-backed DataSource (Week 62, S32).
"""

from __future__ import annotations

import pandas as pd

from data.sources.base import DataSource


class CSVDataSource(DataSource):
    """
    Loads OHLCV data from a CSV file on first access (lazy).

    Column auto-renaming:
    - plain names (open/high/low/close/volume) → $open/$high/$low/$close/$volume
    - dollar-prefixed names are kept as-is

    Args:
        path: Path to the CSV file.
        **read_csv_kwargs: Additional kwargs forwarded to ``pd.read_csv``
            (e.g. ``parse_dates=["timestamp"]``).
    """

    RENAME_MAP = {
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
    }
    REQUIRED_COLS = ["$open", "$high", "$low", "$close", "$volume"]

    def __init__(self, path: str, **read_csv_kwargs) -> None:
        self._path = path
        self._read_csv_kwargs = read_csv_kwargs
        self._df: pd.DataFrame | None = None  # loaded on first access

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._df is not None:
            return
        df = pd.read_csv(self._path, **self._read_csv_kwargs)
        # Normalise column names
        df = df.rename(columns={k: v for k, v in self.RENAME_MAP.items() if k in df.columns})
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSVDataSource: missing required columns {missing} in {self._path}"
            )
        self._df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def get_window(self, start: int, end: int) -> pd.DataFrame:
        self._load()
        return self._df.iloc[start:end]  # type: ignore[index]

    def latest(self) -> pd.Series:
        self._load()
        return self._df.iloc[-1]  # type: ignore[index]

    def __len__(self) -> int:
        self._load()
        return len(self._df)  # type: ignore[arg-type]

    def is_live(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Optional direct access (read-only contract)
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """Expose the underlying DataFrame (loaded on first access)."""
        self._load()
        return self._df  # type: ignore[return-value]
