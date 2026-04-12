"""
Model Registry — lightweight, file-based (no MLflow required).

Week 67 (S59): Store model versions, metrics, and config on disk so that
Week 68 (S62) rollback is possible.

Layout (under ``registry_dir``):
    <registry_dir>/
        registry.json       ← index of all versions
        models/
            v<N>/
                model.*     ← checkpoint file(s) copied here
                meta.json   ← version, metrics, config, timestamp, path

Usage::

    reg = ModelRegistry()
    ver = reg.register(
        model_path="/path/to/model.zip",
        metrics={"sharpe": 1.2, "max_dd": 0.08},
        config={"algo": "PPO", "learning_rate": 3e-4},
        tag="post-week66",
    )
    reg.set_active(ver)
    active = reg.get_active()    # returns meta dict
    reg.rollback(target_version) # copies version to active_model slot

Thread-safe for concurrent reads; writes are serialised with a file lock.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_DIR = os.path.join(
    os.path.expanduser("~"), ".trading_bot", "model_registry"
)


class ModelRegistry:
    """
    Local file-based model registry.

    Parameters
    ----------
    registry_dir : str
        Root directory for the registry.  Created on first use.
    """

    def __init__(self, registry_dir: str = _DEFAULT_REGISTRY_DIR) -> None:
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "models").mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._index_path = self._dir / "registry.json"

        if not self._index_path.exists():
            self._write_index({"versions": {}, "active": None})
        logger.info("ModelRegistry at %s", self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model_path: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        copy_files: bool = True,
    ) -> int:
        """Register a new model version.

        Parameters
        ----------
        model_path :
            Path to the model checkpoint file (or directory).
        metrics :
            Performance metrics at time of registration.
        config :
            Training / architecture config.
        tag :
            Optional free-text label.
        copy_files :
            If True (default), copy the checkpoint into the registry.
            Set to False if you only want to record the external path.

        Returns
        -------
        int
            The assigned version number.
        """
        with self._lock:
            index = self._read_index()
            existing = index.get("versions", {})
            version = max((int(v) for v in existing), default=0) + 1

            version_dir = self._dir / "models" / f"v{version}"
            version_dir.mkdir(parents=True, exist_ok=True)

            stored_path = str(model_path)
            if copy_files:
                src = Path(model_path)
                if src.is_dir():
                    dst = version_dir / src.name
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    stored_path = str(dst)
                elif src.is_file():
                    dst = version_dir / src.name
                    shutil.copy2(src, dst)
                    stored_path = str(dst)
                else:
                    logger.warning("model_path does not exist on disk: %s (path recorded only)", model_path)
                    stored_path = str(model_path)

            meta: Dict[str, Any] = {
                "version": version,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "model_path": stored_path,
                "original_path": str(model_path),
                "metrics": metrics or {},
                "config": config or {},
                "tag": tag or "",
            }
            meta_path = version_dir / "meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            index.setdefault("versions", {})[str(version)] = {
                "meta_path": str(meta_path),
                "registered_at": meta["registered_at"],
                "tag": tag or "",
            }
            self._write_index(index)

        logger.info("Registered model version %d from %s", version, model_path)
        return version

    def set_active(self, version: int) -> None:
        """Mark a registered version as the active model."""
        with self._lock:
            index = self._read_index()
            if str(version) not in index.get("versions", {}):
                raise KeyError(f"Version {version} not found in registry")
            index["active"] = version
            self._write_index(index)
        logger.info("Active model set to version %d", version)

    def get_active(self) -> Optional[Dict[str, Any]]:
        """Return metadata dict for the active version, or None."""
        index = self._read_index()
        active = index.get("active")
        if active is None:
            return None
        return self.get_version(int(active))

    def get_version(self, version: int) -> Dict[str, Any]:
        """Return metadata dict for a specific version.

        Raises
        ------
        KeyError
            If the version is not registered.
        """
        index = self._read_index()
        entry = index.get("versions", {}).get(str(version))
        if entry is None:
            raise KeyError(f"Version {version} not in registry")
        meta_path = Path(entry["meta_path"])
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json missing for version {version}: {meta_path}")
        return json.loads(meta_path.read_text())

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return summary of all registered versions (sorted ascending)."""
        index = self._read_index()
        result = []
        active = index.get("active")
        for v_str, entry in sorted(index.get("versions", {}).items(), key=lambda x: int(x[0])):
            result.append({
                "version": int(v_str),
                "registered_at": entry.get("registered_at", ""),
                "tag": entry.get("tag", ""),
                "is_active": int(v_str) == active,
            })
        return result

    def rollback(self, target_version: int, active_model_path: Optional[str] = None) -> str:
        """Switch the active model to ``target_version``.

        If ``active_model_path`` is given, the checkpoint file stored for
        ``target_version`` is also copied there so that a restarted
        PaperTrader picks it up automatically.

        Returns
        -------
        str
            Path to the rolled-back model checkpoint.
        """
        meta = self.get_version(target_version)
        stored = Path(meta["model_path"])

        if active_model_path is not None:
            dst = Path(active_model_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if stored.is_dir():
                shutil.copytree(stored, dst, dirs_exist_ok=True)
            elif stored.is_file():
                shutil.copy2(stored, dst)
            else:
                logger.warning("Stored path for version %d not found: %s", target_version, stored)

        self.set_active(target_version)
        logger.info("Rolled back to version %d (stored at %s)", target_version, stored)
        return str(stored)

    def delete_version(self, version: int) -> None:
        """Remove a version from the registry (does NOT delete files by default)."""
        with self._lock:
            index = self._read_index()
            versions = index.get("versions", {})
            if str(version) not in versions:
                raise KeyError(f"Version {version} not found")
            if index.get("active") == version:
                raise ValueError(
                    f"Cannot delete active version {version}. "
                    "Set a different active version first."
                )
            del versions[str(version)]
            index["versions"] = versions
            self._write_index(index)
        logger.info("Deleted version %d from registry index", version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_index(self) -> Dict[str, Any]:
        try:
            return json.loads(self._index_path.read_text())
        except Exception as e:
            logger.warning("Failed to read registry index: %s", e)
            return {"versions": {}, "active": None}

    def _write_index(self, index: Dict[str, Any]) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(self._index_path)
