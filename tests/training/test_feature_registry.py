"""Tests for training.features.registry — Week 79 (H9)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg(tmp_path):
    return FeatureRegistry(registry_path=tmp_path / "feature_registry.json")


def _fn_v1(df):
    return df["$close"].rolling(14).mean()


def _fn_v2(df):
    return df["$close"].rolling(14).mean() + 1  # different source


# ---------------------------------------------------------------------------
# Register / Get
# ---------------------------------------------------------------------------

class TestRegisterGet:
    def test_register_new_feature(self, reg):
        version = reg.register(
            "sma_14",
            compute_fn=_fn_v1,
            input_cols=["$close"],
            output_cols=["sma_14"],
            description="14-bar SMA",
        )
        assert version == 1

    def test_get_returns_metadata(self, reg):
        reg.register("sma_14", compute_fn=_fn_v1, input_cols=["$close"], output_cols=["sma_14"])
        entry = reg.get("sma_14")
        assert entry["name"] == "sma_14"
        assert entry["version"] == 1
        assert entry["input_cols"] == ["$close"]
        assert entry["output_cols"] == ["sma_14"]
        assert len(entry["code_hash"]) == 16

    def test_get_unknown_raises_key_error(self, reg):
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_len(self, reg):
        assert len(reg) == 0
        reg.register("a", compute_fn=_fn_v1, output_cols=["a"])
        reg.register("b", compute_fn=_fn_v2, output_cols=["b"])
        assert len(reg) == 2


# ---------------------------------------------------------------------------
# Version bumping on code change
# ---------------------------------------------------------------------------

class TestVersionBumping:
    def test_same_fn_no_version_bump(self, reg):
        reg.register("sma_14", compute_fn=_fn_v1, output_cols=["sma_14"])
        v2 = reg.register("sma_14", compute_fn=_fn_v1, output_cols=["sma_14"])
        assert v2 == 1

    def test_changed_fn_bumps_version(self, reg):
        reg.register("sma_14", compute_fn=_fn_v1, output_cols=["sma_14"])
        v2 = reg.register("sma_14", compute_fn=_fn_v2, output_cols=["sma_14"])
        assert v2 == 2

    def test_explicit_code_source(self, reg):
        reg.register("feat", code_source="source_v1", output_cols=["feat"])
        v2 = reg.register("feat", code_source="source_v2", output_cols=["feat"])
        assert v2 == 2


# ---------------------------------------------------------------------------
# list_features
# ---------------------------------------------------------------------------

class TestListFeatures:
    def test_sorted_by_name(self, reg):
        reg.register("zzz", compute_fn=_fn_v1, output_cols=["zzz"])
        reg.register("aaa", compute_fn=_fn_v2, output_cols=["aaa"])
        names = [f["name"] for f in reg.list_features()]
        assert names == sorted(names)

    def test_empty_list(self, reg):
        assert reg.list_features() == []


# ---------------------------------------------------------------------------
# drift_report
# ---------------------------------------------------------------------------

class TestDriftReport:
    def test_no_drift_when_fns_match(self, reg):
        reg.register("sma_14", compute_fn=_fn_v1, output_cols=["sma_14"])
        report = reg.drift_report({"sma_14": _fn_v1})
        assert report == {}

    def test_drift_detected_on_code_change(self, reg):
        reg.register("sma_14", compute_fn=_fn_v1, output_cols=["sma_14"])
        report = reg.drift_report({"sma_14": _fn_v2})
        assert "sma_14" in report
        assert report["sma_14"]["status"] == "code_changed"

    def test_unregistered_feature_flagged(self, reg):
        report = reg.drift_report({"new_feat": _fn_v1})
        assert "new_feat" in report
        assert report["new_feat"]["status"] == "unregistered"

    def test_empty_features_dict_no_drift(self, reg):
        reg.register("sma_14", compute_fn=_fn_v1, output_cols=["sma_14"])
        report = reg.drift_report({})
        assert report == {}


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_all_cols_present(self, reg):
        reg.register("sma_14", output_cols=["sma_14"])
        df = pd.DataFrame({"$close": [1.0], "sma_14": [1.0]})
        errors = reg.validate_dataframe(df)
        assert errors == []

    def test_missing_col_reported(self, reg):
        reg.register("sma_14", output_cols=["sma_14"])
        df = pd.DataFrame({"$close": [1.0]})
        errors = reg.validate_dataframe(df)
        assert any("sma_14" in e for e in errors)

    def test_unregistered_feature_name(self, reg):
        df = pd.DataFrame({"$close": [1.0]})
        errors = reg.validate_dataframe(df, feature_names=["nonexistent"])
        assert any("not registered" in e for e in errors)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_registry_persists_across_instances(self, tmp_path):
        path = tmp_path / "reg.json"
        reg1 = FeatureRegistry(registry_path=path)
        reg1.register("feat_a", compute_fn=_fn_v1, output_cols=["feat_a"])

        reg2 = FeatureRegistry(registry_path=path)
        assert len(reg2) == 1
        entry = reg2.get("feat_a")
        assert entry["version"] == 1
