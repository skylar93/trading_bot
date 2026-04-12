"""Week 67/68 (S59, S62): ModelRegistry unit tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from training.registry.model_registry import ModelRegistry


@pytest.fixture
def reg(tmp_path):
    return ModelRegistry(registry_dir=str(tmp_path / "registry"))


@pytest.fixture
def dummy_model(tmp_path):
    """Create a dummy .zip checkpoint file."""
    p = tmp_path / "model.zip"
    p.write_bytes(b"FAKE_MODEL_WEIGHTS")
    return str(p)


class TestModelRegistryBasic:
    def test_register_returns_version_1_first(self, reg, dummy_model):
        ver = reg.register(model_path=dummy_model, metrics={"sharpe": 1.0})
        assert ver == 1

    def test_register_increments_version(self, reg, dummy_model):
        v1 = reg.register(model_path=dummy_model)
        v2 = reg.register(model_path=dummy_model)
        assert v2 == v1 + 1

    def test_get_version_returns_meta(self, reg, dummy_model):
        ver = reg.register(
            model_path=dummy_model,
            metrics={"sharpe": 1.5, "max_dd": 0.07},
            config={"algo": "PPO"},
            tag="test-run",
        )
        meta = reg.get_version(ver)
        assert meta["version"] == ver
        assert meta["metrics"]["sharpe"] == 1.5
        assert meta["config"]["algo"] == "PPO"
        assert meta["tag"] == "test-run"

    def test_get_version_unknown_raises(self, reg):
        with pytest.raises(KeyError):
            reg.get_version(999)

    def test_list_versions_empty(self, reg):
        assert reg.list_versions() == []

    def test_list_versions_sorted(self, reg, dummy_model):
        reg.register(dummy_model)
        reg.register(dummy_model)
        reg.register(dummy_model)
        versions = reg.list_versions()
        nums = [v["version"] for v in versions]
        assert nums == sorted(nums)

    def test_copy_files_stores_checkpoint(self, reg, dummy_model):
        ver = reg.register(dummy_model, copy_files=True)
        meta = reg.get_version(ver)
        assert Path(meta["model_path"]).exists()

    def test_no_copy_records_path_only(self, reg, dummy_model):
        ver = reg.register(dummy_model, copy_files=False)
        meta = reg.get_version(ver)
        assert meta["model_path"] == dummy_model


class TestModelRegistryActive:
    def test_active_none_initially(self, reg):
        assert reg.get_active() is None

    def test_set_active_updates_pointer(self, reg, dummy_model):
        ver = reg.register(dummy_model)
        reg.set_active(ver)
        active = reg.get_active()
        assert active is not None
        assert active["version"] == ver

    def test_set_active_unknown_raises(self, reg):
        with pytest.raises(KeyError):
            reg.set_active(999)

    def test_is_active_flag_in_list(self, reg, dummy_model):
        v1 = reg.register(dummy_model)
        v2 = reg.register(dummy_model)
        reg.set_active(v2)
        lst = {v["version"]: v for v in reg.list_versions()}
        assert not lst[v1]["is_active"]
        assert lst[v2]["is_active"]


class TestModelRegistryRollback:
    def test_rollback_sets_active(self, reg, dummy_model):
        v1 = reg.register(dummy_model)
        v2 = reg.register(dummy_model)
        reg.set_active(v2)
        reg.rollback(v1)
        assert reg.get_active()["version"] == v1

    def test_rollback_copies_to_target_path(self, reg, dummy_model, tmp_path):
        ver = reg.register(dummy_model)
        target = str(tmp_path / "active" / "model.zip")
        reg.rollback(ver, active_model_path=target)
        assert Path(target).exists()

    def test_rollback_unknown_raises(self, reg):
        with pytest.raises(KeyError):
            reg.rollback(999)


class TestModelRegistryDelete:
    def test_delete_removes_from_index(self, reg, dummy_model):
        v1 = reg.register(dummy_model)
        v2 = reg.register(dummy_model)
        reg.delete_version(v1)
        with pytest.raises(KeyError):
            reg.get_version(v1)
        assert reg.get_version(v2) is not None

    def test_delete_active_raises(self, reg, dummy_model):
        ver = reg.register(dummy_model)
        reg.set_active(ver)
        with pytest.raises(ValueError):
            reg.delete_version(ver)


class TestRollbackScript:
    """Integration test: rollback_model.py CLI."""

    def test_list_flag(self, reg, dummy_model):
        reg.register(dummy_model, tag="v1-tag")
        result = subprocess.run(
            [sys.executable, "scripts/rollback_model.py", "--list",
             "--registry-dir", str(reg._dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "v1-tag" in result.stdout

    def test_rollback_via_cli(self, reg, dummy_model):
        ver = reg.register(dummy_model)
        result = subprocess.run(
            [sys.executable, "scripts/rollback_model.py", str(ver),
             "--registry-dir", str(reg._dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert f"version {ver}" in result.stdout

    def test_rollback_unknown_version_exits_nonzero(self, reg):
        result = subprocess.run(
            [sys.executable, "scripts/rollback_model.py", "999",
             "--registry-dir", str(reg._dir)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
