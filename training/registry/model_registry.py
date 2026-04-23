"""
Model Registry — lightweight, file-based (no MLflow required).

Unified implementation satisfying both Week 67 (S59) and Week 68 (S62) APIs,
extended in Week 75 (G1) with a promotion state machine.

Week 67 API (simple, file-based):
    registry = ModelRegistry("path/to/registry.json")
    vid = registry.register(name="ppo_v1", path="/models/ppo.zip")
    # vid == "v1" and vid == 1 are BOTH True (VersionID type)
    entry = registry.get(vid)
    registry.update_metrics(vid, {"sharpe": 1.2})
    registry.delete(vid)
    len(registry)

Week 68 API (directory-based, with rollback):
    registry = ModelRegistry(registry_dir="/path/to/dir")
    ver = registry.register(model_path="/models/ppo.zip", metrics={...})
    registry.set_active(ver)
    registry.rollback(ver, active_model_path="/active/model.zip")
    registry.get_version(ver)
    registry.get_active()

Week 75 API (G1 — promotion state machine):
    registry.promote(ver, to_stage="staging", actor="alice", reason="backtest passed")
    registry.get_stage(ver)          # → "staging"
    registry.list_by_stage("canary") # → [VersionID, ...]
    VALID_TRANSITIONS = {
        "candidate": ["staging"],
        "staging":   ["canary", "retired"],
        "canary":    ["prod", "staging", "retired"],
        "prod":      ["retired"],
        "retired":   [],
    }

Constructor auto-detects mode:
    - Positional arg ending in ``.json`` → file mode (Week 67 compatible)
    - ``registry_dir=`` kwarg or positional directory path → dir mode (Week 68)

``VersionID`` is a subclass of ``int`` that additionally compares equal to
``"v{n}"`` strings so both ``assert vid == 1`` and ``assert vid == "v1"`` pass.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# H7: optional MLflow Model Registry sync
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# H7 — MLflow stage mapping
# ---------------------------------------------------------------------------

# Our stage vocabulary → MLflow built-in stages
_STAGE_TO_MLFLOW: Dict[str, str] = {
    "candidate": "None",
    "staging":   "Staging",
    "canary":    "Staging",   # MLflow has no canary; use Staging with tag
    "prod":      "Production",
    "retired":   "Archived",
}


class MLflowRegistryBridge:
    """
    Synchronises :class:`ModelRegistry` stage transitions to the MLflow
    Model Registry so MLflow UI shows live promotion state.

    Week 79 (H7): MLflow becomes the authoritative record for model artifacts.
    ``ModelRegistry`` JSON index remains as a fast local cache.

    Parameters
    ----------
    model_name : str
        Registered-model name in the MLflow Model Registry.
    tracking_uri : str, optional
        MLflow tracking URI.  Defaults to ``./mlruns``.
    """

    def __init__(self, model_name: str, tracking_uri: Optional[str] = None) -> None:
        if not HAS_MLFLOW:
            raise ImportError("mlflow is required for MLflowRegistryBridge")
        self.model_name = model_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()
        self._ensure_registered_model()

    def _ensure_registered_model(self) -> None:
        try:
            self._client.get_registered_model(self.model_name)
        except mlflow.exceptions.MlflowException:
            self._client.create_registered_model(
                self.model_name,
                description="Trading bot model — managed by ModelRegistry (H7)",
            )
            logger.info("Created MLflow registered model: %s", self.model_name)

    def log_version(
        self,
        version_int: int,
        run_id: Optional[str],
        artifact_path: str = "model",
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Create a new model version in the MLflow registry.

        Returns the MLflow version string or None if MLflow is unavailable.
        """
        if run_id is None:
            logger.warning("No run_id provided; skipping MLflow model version creation")
            return None
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            mv = self._client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id,
                tags={"registry_version": str(version_int), **(tags or {})},
            )
            if metrics:
                for k, v in metrics.items():
                    self._client.set_model_version_tag(
                        self.model_name, mv.version, f"metric.{k}", str(v)
                    )
            logger.info(
                "MLflow model version %s created for registry v%d",
                mv.version, version_int,
            )
            return mv.version
        except Exception as exc:
            logger.warning("MLflow log_version failed: %s", exc)
            return None

    def sync_stage(self, version_int: int, our_stage: str, mlflow_version: str) -> None:
        """Push a stage change to MLflow."""
        mlflow_stage = _STAGE_TO_MLFLOW.get(our_stage, "None")
        try:
            self._client.transition_model_version_stage(
                name=self.model_name,
                version=mlflow_version,
                stage=mlflow_stage,
                archive_existing_versions=(our_stage == "prod"),
            )
            if our_stage == "canary":
                self._client.set_model_version_tag(
                    self.model_name, mlflow_version, "stage_detail", "canary"
                )
            logger.info(
                "MLflow stage synced: v%d → %s (mlflow=%s)",
                version_int, our_stage, mlflow_stage,
            )
        except Exception as exc:
            logger.warning("MLflow sync_stage failed: %s", exc)

_DEFAULT_REGISTRY_DIR = os.path.join(
    os.path.expanduser("~"), ".trading_bot", "model_registry"
)

# ---------------------------------------------------------------------------
# G1 — Promotion state machine constants
# ---------------------------------------------------------------------------

VALID_STAGES = ("candidate", "staging", "canary", "prod", "retired")

VALID_TRANSITIONS: Dict[str, List[str]] = {
    "candidate": ["staging"],
    "staging":   ["canary", "retired"],
    "canary":    ["prod", "staging", "retired"],
    "prod":      ["retired"],
    "retired":   [],
}

PROMOTION_CRITERIA: Dict[Tuple[str, str], str] = {
    ("candidate", "staging"): "offline backtest: Sharpe ≥ 0.5, max_drawdown ≤ 30%",
    ("staging",   "canary"):  "walkforward eval pass + human approval",
    ("canary",    "prod"):    "7d canary traffic split, ruin prob CI < 1%, human approval",
    ("canary",    "staging"): "canary underperformance detected — demote",
    ("staging",   "retired"): "model deprecated",
    ("canary",    "retired"): "model deprecated",
    ("prod",      "retired"): "model replaced by newer prod version",
}


# ---------------------------------------------------------------------------
# VersionID — int subclass compatible with "v{n}" string comparison
# ---------------------------------------------------------------------------

class VersionID(int):
    """Integer version number that also compares equal to the ``"v{n}"`` string.

    Examples
    --------
    >>> v = VersionID(1)
    >>> v == 1      # True  (int comparison)
    >>> v == "v1"   # True  (string comparison)
    >>> str(v)      # "v1"
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if other.startswith("v"):
                try:
                    return int(self) == int(other[1:])
                except ValueError:
                    pass
            else:
                try:
                    return int(self) == int(other)
                except ValueError:
                    pass
            return False
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __str__(self) -> str:
        return f"v{int(self)}"

    def __repr__(self) -> str:
        return f"v{int(self)}"


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Local file-based model registry.

    Parameters
    ----------
    registry_path_or_dir : str or Path, optional
        Either a ``.json`` file path (Week 67 / simple mode) **or** a directory
        path (Week 68 / full mode with copy-files & rollback support).
    registry_dir : str or Path, optional
        Explicit directory path (Week 68 API).  Takes precedence over the
        positional argument.
    """

    def __init__(
        self,
        registry_path_or_dir: str | Path | None = None,
        *,
        registry_dir: str | Path | None = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ) -> None:
        if registry_dir is not None:
            self._mode = "dir"
            self._dir = Path(registry_dir)
            self._index_path = self._dir / "registry.json"
        elif registry_path_or_dir is not None:
            p = Path(registry_path_or_dir)
            if p.suffix == ".json":
                self._mode = "file"
                self._dir = p.parent
                self._index_path = p
            else:
                self._mode = "dir"
                self._dir = p
                self._index_path = self._dir / "registry.json"
        else:
            self._mode = "dir"
            self._dir = Path(_DEFAULT_REGISTRY_DIR)
            self._index_path = self._dir / "registry.json"

        self._dir.mkdir(parents=True, exist_ok=True)
        if self._mode == "dir":
            (self._dir / "models").mkdir(exist_ok=True)

        self._lock = threading.Lock()

        if not self._index_path.exists():
            self._write_index({"versions": {}, "active": None})

        # H7: optional MLflow bridge
        self._mlflow_bridge: Optional[MLflowRegistryBridge] = None
        if mlflow_model_name and HAS_MLFLOW:
            try:
                self._mlflow_bridge = MLflowRegistryBridge(
                    model_name=mlflow_model_name,
                    tracking_uri=mlflow_tracking_uri,
                )
                logger.info("MLflow bridge enabled: model=%s", mlflow_model_name)
            except Exception as exc:
                logger.warning("MLflow bridge init failed (non-fatal): %s", exc)

        logger.info("ModelRegistry at %s (mode=%s)", self._index_path, self._mode)

        # G1: ensure promotion index section exists
        with self._lock:
            index = self._read_index()
            if "stages" not in index:
                index["stages"] = {}
                self._write_index(index)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        model_path: str | None = None,
        *,
        path: str | None = None,
        name: str = "",
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        copy_files: bool = True,
        mlflow_run_id: Optional[str] = None,
        mlflow_artifact_path: str = "model",
    ) -> VersionID:
        """Register a new model version.

        Supports two calling conventions:
          - Week 67: ``register(name="ppo_v1", path="/models/ppo.zip")``
          - Week 68: ``register(model_path="/models/ppo.zip", metrics={...})``

        Parameters
        ----------
        model_path :
            Path to the checkpoint file/directory (positional or keyword).
        path :
            Alias for ``model_path`` (Week 67 compatibility).
        name :
            Human-readable label.
        metrics :
            Evaluation metrics dict.
        config :
            Hyperparameter snapshot.
        tag :
            Short free-text label (legacy; prefer ``name``).
        tags :
            Arbitrary string→string metadata.
        copy_files :
            Copy checkpoint into registry (dir mode only; ignored in file mode).

        Returns
        -------
        VersionID
            Version number.  Compares equal to both ``1`` and ``"v1"``.
        """
        resolved_path = path or model_path or ""

        with self._lock:
            index = self._read_index()
            existing = index.get("versions", {})
            version = max((int(v) for v in existing), default=0) + 1

            stored_path = str(resolved_path)

            if self._mode == "dir" and copy_files and resolved_path:
                version_dir = self._dir / "models" / f"v{version}"
                version_dir.mkdir(parents=True, exist_ok=True)

                src = Path(resolved_path)
                if src.is_dir():
                    dst = version_dir / src.name
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    stored_path = str(dst)
                elif src.is_file():
                    dst = version_dir / src.name
                    shutil.copy2(src, dst)
                    stored_path = str(dst)
                else:
                    logger.warning(
                        "model_path not on disk: %s (path recorded only)", resolved_path
                    )

            meta: Dict[str, Any] = {
                "version": version,
                "name": name or tag or "",
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "model_path": stored_path,
                "original_path": str(resolved_path),
                "metrics": dict(metrics or {}),
                "config": dict(config or {}),
                "tag": tag or name or "",
                "tags": dict(tags or {}),
                # Week 67 compat: also store as "path" key
                "path": stored_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            if self._mode == "dir":
                version_dir = self._dir / "models" / f"v{version}"
                version_dir.mkdir(parents=True, exist_ok=True)
                meta_path = version_dir / "meta.json"
                meta_path.write_text(json.dumps(meta, indent=2))
                index.setdefault("versions", {})[str(version)] = {
                    "meta_path": str(meta_path),
                    "registered_at": meta["registered_at"],
                    "tag": meta["tag"],
                    "name": meta["name"],
                }
            else:
                # file mode: store meta inline
                index.setdefault("versions", {})[str(version)] = meta

            self._write_index(index)

        # G1: auto-initialise stage to "candidate"
        with self._lock:
            index = self._read_index()
            index.setdefault("stages", {})[str(version)] = {
                "stage": "candidate",
                "history": [
                    {
                        "from_stage": None,
                        "to_stage": "candidate",
                        "actor": "system",
                        "reason": "initial registration",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }
            self._write_index(index)

        # H7: log to MLflow Model Registry if bridge is active
        if self._mlflow_bridge is not None and mlflow_run_id:
            mlflow_ver = self._mlflow_bridge.log_version(
                version_int=version,
                run_id=mlflow_run_id,
                artifact_path=mlflow_artifact_path,
                metrics=dict(metrics or {}),
                tags=dict(tags or {}),
            )
            if mlflow_ver:
                with self._lock:
                    idx = self._read_index()
                    idx["versions"][str(version)]["mlflow_version"] = mlflow_ver
                    self._write_index(idx)

        logger.info(
            "Registered model version %d (%s) from %s",
            version, meta["name"], resolved_path,
        )
        return VersionID(version)

    def get_version(self, version: int | str) -> Dict[str, Any]:
        """Return metadata dict for a specific version (accepts int or ``"v{n}"``).

        Raises
        ------
        KeyError
            If the version is not registered.
        """
        version_int = self._parse_version(version)
        index = self._read_index()
        entry = index.get("versions", {}).get(str(version_int))
        if entry is None:
            raise KeyError(f"Version {version} not in registry")

        if self._mode == "dir":
            meta_path = Path(entry["meta_path"])
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"meta.json missing for version {version_int}: {meta_path}"
                )
            meta = json.loads(meta_path.read_text())
        else:
            meta = dict(entry)

        meta["version"] = VersionID(version_int)
        return meta

    # Alias (Week 67 API)
    def get(self, version_id: int | str) -> Dict[str, Any]:
        """Alias for ``get_version``."""
        return self.get_version(version_id)

    def latest(self) -> Optional[Dict[str, Any]]:
        """Return metadata for the most recently registered version, or None."""
        index = self._read_index()
        versions = index.get("versions", {})
        if not versions:
            return None
        latest_ver = max(int(v) for v in versions)
        return self.get_version(latest_ver)

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return summary of all registered versions (sorted ascending)."""
        index = self._read_index()
        active = index.get("active")
        result = []
        for v_str, entry in sorted(
            index.get("versions", {}).items(), key=lambda x: int(x[0])
        ):
            result.append(
                {
                    "version": VersionID(int(v_str)),
                    "registered_at": entry.get("registered_at", ""),
                    "tag": entry.get("tag", ""),
                    "name": entry.get("name", ""),
                    "is_active": int(v_str) == active,
                }
            )
        return result

    def delete_version(self, version: int | str) -> None:
        """Remove a version from the registry index (files NOT deleted).

        Raises
        ------
        ValueError
            If this is the currently active version.
        KeyError
            If the version is not registered.
        """
        version_int = self._parse_version(version)
        with self._lock:
            index = self._read_index()
            versions = index.get("versions", {})
            if str(version_int) not in versions:
                raise KeyError(f"Version {version} not found")
            if index.get("active") == version_int:
                raise ValueError(
                    f"Cannot delete active version {version_int}. "
                    "Set a different active version first."
                )
            del versions[str(version_int)]
            index["versions"] = versions
            self._write_index(index)
        logger.info("Deleted version %d from registry index", version_int)

    # Alias (Week 67 API)
    def delete(self, version_id: int | str) -> None:
        """Alias for ``delete_version``."""
        self.delete_version(version_id)

    def update_metrics(
        self, version_id: int | str, metrics_update: Dict[str, float]
    ) -> None:
        """Merge ``metrics_update`` into the stored metrics for a version.

        Raises
        ------
        KeyError
            If the version is not registered.
        """
        version_int = self._parse_version(version_id)
        with self._lock:
            index = self._read_index()
            entry = index.get("versions", {}).get(str(version_int))
            if entry is None:
                raise KeyError(f"Version {version_id} not in registry")

            if self._mode == "dir":
                meta_path = Path(entry["meta_path"])
                meta = json.loads(meta_path.read_text())
                meta.setdefault("metrics", {}).update(metrics_update)
                meta_path.write_text(json.dumps(meta, indent=2))
            else:
                entry.setdefault("metrics", {}).update(metrics_update)
                self._write_index(index)

    # ------------------------------------------------------------------
    # Active version / rollback (Week 68 API)
    # ------------------------------------------------------------------

    def set_active(self, version: int | str) -> None:
        """Mark a registered version as the active model."""
        version_int = self._parse_version(version)
        with self._lock:
            index = self._read_index()
            if str(version_int) not in index.get("versions", {}):
                raise KeyError(f"Version {version} not found in registry")
            index["active"] = version_int
            self._write_index(index)
        logger.info("Active model set to version %d", version_int)

    def get_active(self) -> Optional[Dict[str, Any]]:
        """Return metadata dict for the active version, or None."""
        index = self._read_index()
        active = index.get("active")
        if active is None:
            return None
        return self.get_version(int(active))

    # ------------------------------------------------------------------
    # G1 — Promotion state machine
    # ------------------------------------------------------------------

    def promote(
        self,
        version: int | str,
        to_stage: str,
        *,
        actor: str = "unknown",
        reason: str = "",
        force: bool = False,
    ) -> None:
        """Transition ``version`` to ``to_stage``.

        Parameters
        ----------
        version : int or str
            Version to promote (accepts ``1`` or ``"v1"``).
        to_stage : str
            Target stage.  Must be a valid transition from the current stage
            unless ``force=True``.
        actor : str
            Human or system identifier performing the promotion.
        reason : str
            Free-text justification (stored in history).
        force : bool
            Skip transition-validity check.  Use for testing only.

        Raises
        ------
        KeyError
            Version not in registry.
        ValueError
            ``to_stage`` is not a valid stage or the transition is not allowed.
        """
        if to_stage not in VALID_STAGES:
            raise ValueError(
                f"Unknown stage {to_stage!r}. Valid stages: {VALID_STAGES}"
            )

        version_int = self._parse_version(version)
        with self._lock:
            index = self._read_index()
            if str(version_int) not in index.get("versions", {}):
                raise KeyError(f"Version {version} not in registry")

            stages = index.setdefault("stages", {})
            stage_entry = stages.setdefault(
                str(version_int),
                {
                    "stage": "candidate",
                    "history": [],
                    # G4 (Week 83): auto-demotion criteria stored for audit trail.
                    # PaperTrader sets traffic_pct=0; this field records the policy.
                    "auto_demote_criteria": {
                        "sigma_below_prod": 1.0,
                        "consecutive_hours": 6,
                    },
                },
            )
            current_stage = stage_entry["stage"]

            if not force:
                allowed = VALID_TRANSITIONS.get(current_stage, [])
                if to_stage not in allowed:
                    raise ValueError(
                        f"Transition {current_stage!r} → {to_stage!r} is not allowed. "
                        f"Allowed from {current_stage!r}: {allowed}"
                    )

            event: Dict[str, Any] = {
                "from_stage": current_stage,
                "to_stage": to_stage,
                "actor": actor,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            stage_entry["stage"] = to_stage
            stage_entry["history"].append(event)
            stages[str(version_int)] = stage_entry
            index["stages"] = stages
            self._write_index(index)

        logger.info(
            "Model v%d promoted: %s → %s  (actor=%s reason=%r)",
            version_int, current_stage, to_stage, actor, reason,
        )

        # H7: sync to MLflow if bridge is active
        if self._mlflow_bridge is not None:
            index = self._read_index()
            mlflow_ver = (
                index.get("versions", {})
                .get(str(version_int), {})
                .get("mlflow_version")
            )
            if mlflow_ver:
                self._mlflow_bridge.sync_stage(version_int, to_stage, mlflow_ver)

    def get_stage(self, version: int | str) -> str:
        """Return the current promotion stage of ``version``.

        Returns ``"candidate"`` for versions registered before G1 was deployed.

        Raises
        ------
        KeyError
            Version not in registry.
        """
        version_int = self._parse_version(version)
        index = self._read_index()
        if str(version_int) not in index.get("versions", {}):
            raise KeyError(f"Version {version} not in registry")
        stages = index.get("stages", {})
        return stages.get(str(version_int), {}).get("stage", "candidate")

    def get_promotion_history(self, version: int | str) -> List[Dict[str, Any]]:
        """Return the full promotion event history for ``version``."""
        version_int = self._parse_version(version)
        index = self._read_index()
        if str(version_int) not in index.get("versions", {}):
            raise KeyError(f"Version {version} not in registry")
        stages = index.get("stages", {})
        return list(stages.get(str(version_int), {}).get("history", []))

    def list_by_stage(self, stage: str) -> List[VersionID]:
        """Return all version IDs currently at ``stage``, sorted ascending."""
        if stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage {stage!r}")
        index = self._read_index()
        stages = index.get("stages", {})
        result = []
        for v_str in sorted(index.get("versions", {}).keys(), key=int):
            s = stages.get(v_str, {}).get("stage", "candidate")
            if s == stage:
                result.append(VersionID(int(v_str)))
        return result

    def check_promotion_conditions(
        self,
        version: int | str,
        to_stage: str,
    ) -> Tuple[bool, str]:
        """Check whether ``version`` can be promoted to ``to_stage``.

        Returns
        -------
        (ok, message) : (bool, str)
            ``ok=True`` when the transition is structurally valid.
            The caller must still verify empirical criteria (backtest metrics,
            canary performance) before calling ``promote()``.
        """
        try:
            version_int = self._parse_version(version)
        except (KeyError, ValueError) as exc:
            return False, str(exc)

        try:
            current = self.get_stage(version_int)
        except KeyError as exc:
            return False, str(exc)

        allowed = VALID_TRANSITIONS.get(current, [])
        if to_stage not in allowed:
            return False, (
                f"Transition {current!r} → {to_stage!r} not allowed. "
                f"Allowed: {allowed}"
            )

        criteria = PROMOTION_CRITERIA.get((current, to_stage), "")
        msg = f"OK: {current!r} → {to_stage!r}"
        if criteria:
            msg += f". Required criteria: {criteria}"
        return True, msg

    def rollback(
        self,
        target_version: int | str,
        active_model_path: Optional[str] = None,
    ) -> str:
        """Switch the active model to ``target_version``.

        Parameters
        ----------
        target_version :
            Version to roll back to.
        active_model_path :
            If given, copy the checkpoint to this path so a restarted
            PaperTrader picks it up automatically.

        Returns
        -------
        str
            Path to the rolled-back model checkpoint.
        """
        version_int = self._parse_version(target_version)
        meta = self.get_version(version_int)
        stored = Path(meta["model_path"])

        if active_model_path is not None:
            dst = Path(active_model_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if stored.is_dir():
                shutil.copytree(stored, dst, dirs_exist_ok=True)
            elif stored.is_file():
                shutil.copy2(stored, dst)
            else:
                logger.warning(
                    "Stored path for version %d not found: %s", version_int, stored
                )

        self.set_active(version_int)
        logger.info(
            "Rolled back to version %d (stored at %s)", version_int, stored
        )
        return str(stored)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        index = self._read_index()
        return len(index.get("versions", {}))

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(path={self._index_path!r}, "
            f"n_versions={len(self)})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_version(version: int | str) -> int:
        """Normalise a version to an integer (handles ``"v1"`` strings)."""
        if isinstance(version, str):
            s = version.lstrip("v")
            try:
                return int(s)
            except ValueError:
                raise KeyError(f"Invalid version id: {version!r}")
        return int(version)

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
