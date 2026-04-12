"""Lightweight local model registry (S59).

Stores model version metadata in a single JSON file — no MLflow or database
required.  Suitable for solo-dev workflows where the training machine is the
same as the deployment machine.

Each registered entry records:
    version   — auto-incremented integer (``v1``, ``v2``, …)
    name      — human-readable label (e.g. "ppo_v3_sharpe")
    path      — filesystem path to the saved model artefact
    metrics   — dict of scalar evaluation metrics (sharpe, sortino, …)
    config    — dict of hyperparameters / training config snapshot
    created_at — ISO-8601 UTC timestamp
    tags      — optional string→string metadata

Usage
-----
    registry = ModelRegistry("~/.trading_bot/model_registry.json")
    vid = registry.register(
        name="ppo_v3",
        path="/models/ppo_v3.zip",
        metrics={"sharpe": 1.42, "max_drawdown": 0.08},
        config={"lr": 3e-4, "n_steps": 2048},
    )
    entry = registry.get(vid)
    latest = registry.latest()
    registry.delete(vid)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """File-backed local model registry.

    Parameters
    ----------
    registry_path : str or Path
        Path to the JSON file used for storage.  Created on first write if it
        does not exist.  Parent directory must already exist.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self._path = Path(registry_path).expanduser().resolve()
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        path: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a new model version.

        Parameters
        ----------
        name : str
            Human-readable model name.
        path : str
            Filesystem path to the saved model artefact.
        metrics : dict, optional
            Evaluation metrics (sharpe, max_drawdown, etc.).
        config : dict, optional
            Training hyperparameters / config snapshot.
        tags : dict, optional
            Arbitrary string metadata.

        Returns
        -------
        str
            Version id (e.g. ``"v1"``).
        """
        with self._lock:
            next_n = len(self._data.get("versions", [])) + 1
            version_id = f"v{next_n}"
            entry: Dict[str, Any] = {
                "version": version_id,
                "name": name,
                "path": str(path),
                "metrics": dict(metrics or {}),
                "config": dict(config or {}),
                "tags": dict(tags or {}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._data.setdefault("versions", []).append(entry)
            self._save()
        logger.info(
            "ModelRegistry: registered %s (%s) at %s", version_id, name, path
        )
        return version_id

    def get(self, version_id: str) -> Dict[str, Any]:
        """Return entry for a specific version id.

        Raises
        ------
        KeyError
            If ``version_id`` is not in the registry.
        """
        with self._lock:
            for entry in self._data.get("versions", []):
                if entry["version"] == version_id:
                    return dict(entry)
        raise KeyError(f"Version {version_id!r} not found in registry")

    def latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently registered entry, or None if empty."""
        with self._lock:
            versions = self._data.get("versions", [])
            if not versions:
                return None
            return dict(versions[-1])

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all registered entries in registration order (oldest first)."""
        with self._lock:
            return [dict(e) for e in self._data.get("versions", [])]

    def delete(self, version_id: str) -> None:
        """Remove a version from the registry (metadata only — artefact not deleted).

        Raises
        ------
        KeyError
            If ``version_id`` does not exist.
        """
        with self._lock:
            before = len(self._data.get("versions", []))
            self._data["versions"] = [
                e for e in self._data.get("versions", [])
                if e["version"] != version_id
            ]
            if len(self._data["versions"]) == before:
                raise KeyError(f"Version {version_id!r} not found in registry")
            self._save()
        logger.info("ModelRegistry: deleted %s", version_id)

    def update_metrics(self, version_id: str, metrics: Dict[str, float]) -> None:
        """Merge new metrics into an existing version entry.

        Useful for adding post-deployment evaluation results.
        """
        with self._lock:
            for entry in self._data.get("versions", []):
                if entry["version"] == version_id:
                    entry["metrics"].update(metrics)
                    self._save()
                    return
        raise KeyError(f"Version {version_id!r} not found in registry")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                logger.info(
                    "ModelRegistry: loaded %d version(s) from %s",
                    len(data.get("versions", [])),
                    self._path,
                )
                return data
            except Exception as exc:
                logger.error(
                    "ModelRegistry: failed to load %s (%s); starting empty.",
                    self._path,
                    exc,
                )
        return {"versions": []}

    def _save(self) -> None:
        """Atomically write registry to disk (temp-file + rename)."""
        tmp = self._path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
        except Exception as exc:
            logger.error("ModelRegistry: save failed (%s)", exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        with self._lock:
            return len(self._data.get("versions", []))

    def __repr__(self) -> str:
        return f"ModelRegistry(path={self._path!r}, n_versions={len(self)})"
