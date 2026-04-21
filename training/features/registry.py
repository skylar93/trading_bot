"""
Feature Registry — Week 79 (H9).

Lightweight registry that tracks each feature's:
  - version     : integer, incremented when the computation code changes
  - code_hash   : SHA-256 of the source function/module text
  - input_schema: expected input column names (and optional dtypes)
  - output_cols : column names produced
  - description : human-readable summary

When a feature's code_hash drifts from the registered value, ``drift_report()``
flags it so that promotion candidates know which features changed.

Usage::

    from training.features.registry import FeatureRegistry

    reg = FeatureRegistry()
    reg.register(
        name="rsi_14",
        compute_fn=compute_rsi,
        input_cols=["$close"],
        output_cols=["rsi_14"],
        description="14-period RSI",
    )

    drifted = reg.drift_report()   # → {"rsi_14": {"old_hash": ..., "new_hash": ...}}
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path.home() / ".trading_bot" / "feature_registry.json"


class FeatureRegistry:
    """
    Lightweight feature registry backed by a local JSON file.

    Parameters
    ----------
    registry_path : Path or str, optional
        Where to persist the registry index.
    """

    def __init__(self, registry_path: Optional[str | Path] = None) -> None:
        self._path = Path(registry_path or _DEFAULT_REGISTRY_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write({})
        logger.info("FeatureRegistry at %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        compute_fn: Optional[Callable] = None,
        *,
        input_cols: Optional[List[str]] = None,
        output_cols: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        code_source: Optional[str] = None,
    ) -> int:
        """
        Register or update a feature.

        If the feature already exists and ``compute_fn`` / ``code_source`` has
        changed, the version is bumped.

        Returns
        -------
        int
            Current version number.
        """
        code_hash = self._hash_fn(compute_fn, code_source)

        with self._lock:
            index = self._read()
            existing = index.get(name)

            if existing is None:
                version = 1
                logger.info("Registering new feature: %s (v%d)", name, version)
            elif existing.get("code_hash") != code_hash:
                version = existing.get("version", 1) + 1
                logger.info(
                    "Feature %s code changed — bumping to v%d", name, version
                )
            else:
                version = existing.get("version", 1)
                logger.debug("Feature %s unchanged (v%d)", name, version)

            index[name] = {
                "name": name,
                "version": version,
                "code_hash": code_hash,
                "input_cols": list(input_cols or []),
                "output_cols": list(output_cols or []),
                "description": description,
                "tags": dict(tags or {}),
                "registered_at": (
                    existing.get("registered_at")
                    if existing and existing.get("code_hash") == code_hash
                    else datetime.now(timezone.utc).isoformat()
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write(index)

        return version

    def get(self, name: str) -> Dict[str, Any]:
        """Return the registry entry for *name*.

        Raises
        ------
        KeyError
            Feature not registered.
        """
        index = self._read()
        if name not in index:
            raise KeyError(f"Feature {name!r} not in registry")
        return dict(index[name])

    def list_features(self) -> List[Dict[str, Any]]:
        """Return all registered features sorted by name."""
        return sorted(self._read().values(), key=lambda e: e["name"])

    def drift_report(
        self,
        features: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Detect features whose code has drifted since registration.

        Parameters
        ----------
        features : dict[name → callable], optional
            Map of feature name to current compute function.  When provided,
            compares live hashes against registered hashes.

        Returns
        -------
        dict
            Keys are feature names with drift.
            Values are ``{"old_hash": ..., "new_hash": ..., "version": ...}``.
        """
        index = self._read()
        drifted: Dict[str, Dict[str, str]] = {}

        if not features:
            return drifted

        for name, fn in features.items():
            entry = index.get(name)
            if entry is None:
                drifted[name] = {
                    "old_hash": "(unregistered)",
                    "new_hash": self._hash_fn(fn),
                    "version": "0",
                    "status": "unregistered",
                }
                continue
            new_hash = self._hash_fn(fn)
            if new_hash != entry.get("code_hash"):
                drifted[name] = {
                    "old_hash": entry.get("code_hash", ""),
                    "new_hash": new_hash,
                    "version": str(entry.get("version", 0)),
                    "status": "code_changed",
                }

        if drifted:
            logger.warning(
                "Feature drift detected in %d feature(s): %s",
                len(drifted), list(drifted.keys()),
            )
        return drifted

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Check that *df* contains the expected output columns for each feature.

        Returns
        -------
        list[str]
            Error messages.  Empty means all expected columns are present.
        """
        index = self._read()
        names = feature_names or list(index.keys())
        errors: List[str] = []
        for name in names:
            entry = index.get(name)
            if entry is None:
                errors.append(f"[feature_registry] {name!r}: not registered")
                continue
            for col in entry.get("output_cols", []):
                if col not in df.columns:
                    errors.append(
                        f"[feature_registry] {name!r} v{entry['version']}: "
                        f"expected column {col!r} missing from DataFrame"
                    )
        return errors

    def __len__(self) -> int:
        return len(self._read())

    def __repr__(self) -> str:
        return f"FeatureRegistry(path={self._path!r}, n_features={len(self)})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_fn(
        fn: Optional[Callable],
        code_source: Optional[str] = None,
    ) -> str:
        """SHA-256 of the function source code (or explicit source string)."""
        if code_source is not None:
            source = code_source
        elif fn is not None:
            try:
                source = inspect.getsource(fn)
            except (OSError, TypeError):
                source = str(fn)
        else:
            source = ""
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _write(self, index: Dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(self._path)
