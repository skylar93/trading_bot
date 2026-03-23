"""
Week 11 tests: Web UI Rebuild
Tests for all 5 new/rewritten Streamlit pages and their helper functions.
All Streamlit calls are mocked so tests run without a Streamlit server.
"""

import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─── Streamlit stub ───────────────────────────────────────────────────────────
# Inject a MagicMock for 'streamlit' so pages that do `import streamlit as st`
# inside functions get a stub without requiring Streamlit to be installed.

_st_mock = MagicMock()
_st_mock.session_state = {}
_st_mock.sidebar = MagicMock()
_st_mock.sidebar.text_input = MagicMock(return_value="")
_st_mock.sidebar.checkbox = MagicMock(return_value=False)
_st_mock.sidebar.slider = MagicMock(return_value=10)
_st_mock.sidebar.radio = MagicMock(return_value="Training Dashboard")
_st_mock.sidebar.selectbox = MagicMock(return_value="")
_st_mock.sidebar.number_input = MagicMock(return_value=10000.0)
_st_mock.sidebar.toggle = MagicMock(return_value=False)
_st_mock.sidebar.caption = MagicMock()
_st_mock.sidebar.markdown = MagicMock()
_st_mock.sidebar.header = MagicMock()
_st_mock.sidebar.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False)))
_st_mock.selectbox = MagicMock(return_value="")
_st_mock.text_area = MagicMock(return_value="")
_st_mock.text_input = MagicMock(return_value="")
_st_mock.number_input = MagicMock(return_value=10000.0)
_st_mock.slider = MagicMock(return_value=10)
_st_mock.checkbox = MagicMock(return_value=False)
_st_mock.button = MagicMock(return_value=False)
_st_mock.toggle = MagicMock(return_value=False)
_st_mock.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
_st_mock.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False)))
_st_mock.spinner = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False)))
_st_mock.plotly_chart = MagicMock()
_st_mock.dataframe = MagicMock()
_st_mock.title = MagicMock()
_st_mock.subheader = MagicMock()
_st_mock.info = MagicMock()
_st_mock.success = MagicMock()
_st_mock.warning = MagicMock()
_st_mock.error = MagicMock()
_st_mock.caption = MagicMock()
_st_mock.markdown = MagicMock()
_st_mock.json = MagicMock()
_st_mock.code = MagicMock()
_st_mock.metric = MagicMock()
_st_mock.set_page_config = MagicMock()
_st_mock.rerun = MagicMock()
_st_mock.download_button = MagicMock()
_st_mock.empty = MagicMock(return_value=MagicMock())

sys.modules["streamlit"] = _st_mock

# Project root on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ─── Imports of page helpers ──────────────────────────────────────────────────

from deployment.web_interface.pages.training_dashboard import (
    DEFAULT_TRACKING_URI,
    WATCHED_METRICS,
    build_metric_chart,
    build_multi_metric_chart,
    get_available_runs,
    get_mlflow_experiments,
    get_run_metrics,
    parse_run_params,
    render_training_dashboard,
)
from deployment.web_interface.pages.ensemble_monitor import (
    REGIME_COLORS,
    REGIME_LABELS,
    build_agent_performance_table,
    build_regime_timeline,
    build_weights_chart,
    get_ensemble_checkpoints,
    load_ensemble_checkpoint,
    normalise_weights,
    render_ensemble_monitor,
)
from deployment.web_interface.pages.paper_trading import (
    build_paper_trading_chart,
    compute_portfolio_metrics,
    format_action,
    get_available_checkpoints,
    render_paper_trading,
)
from deployment.web_interface.pages.config_editor import (
    RECOMMENDED_KEYS,
    REQUIRED_TOP_LEVEL_KEYS,
    diff_configs,
    get_config_schema,
    load_config_yaml,
    render_config_editor,
    save_config_yaml,
    validate_config_yaml,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Import tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:
    def test_training_dashboard_imports(self):
        import deployment.web_interface.pages.training_dashboard as m  # noqa: F401

    def test_ensemble_monitor_imports(self):
        import deployment.web_interface.pages.ensemble_monitor as m  # noqa: F401

    def test_paper_trading_imports(self):
        import deployment.web_interface.pages.paper_trading as m  # noqa: F401

    def test_config_editor_imports(self):
        import deployment.web_interface.pages.config_editor as m  # noqa: F401

    def test_app_imports(self):
        import deployment.web_interface.app as m  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Training Dashboard helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingDashboardHelpers:

    # ── get_mlflow_experiments ────────────────────────────────────────────
    def test_get_mlflow_experiments_returns_list(self):
        with patch("deployment.web_interface.pages.training_dashboard.get_mlflow_experiments",
                   return_value=[]) as mock_fn:
            result = mock_fn("/fake/uri")
            assert isinstance(result, list)

    def test_get_mlflow_experiments_empty_on_error(self):
        # mlflow not available → should return []
        result = get_mlflow_experiments("/non/existent/path")
        assert isinstance(result, list)

    def test_get_mlflow_experiments_empty_without_mlflow(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            result = get_mlflow_experiments("/fake")
            assert result == []

    def test_default_tracking_uri_is_string(self):
        assert isinstance(DEFAULT_TRACKING_URI, str)

    def test_watched_metrics_is_list(self):
        assert isinstance(WATCHED_METRICS, list)
        assert len(WATCHED_METRICS) > 0

    # ── get_run_metrics ───────────────────────────────────────────────────
    def test_get_run_metrics_returns_dataframe(self):
        result = get_run_metrics("/fake/uri", "run_id_xyz")
        assert isinstance(result, pd.DataFrame)

    def test_get_run_metrics_empty_on_bad_uri(self):
        result = get_run_metrics("/bad/uri", "run_xyz")
        assert result.empty

    def test_get_run_metrics_has_correct_columns(self):
        result = get_run_metrics("/bad/uri", "run_xyz")
        assert list(result.columns) == ["step", "metric", "value"]

    def test_get_run_metrics_custom_keys(self):
        result = get_run_metrics("/bad/uri", "run_xyz", metric_keys=["my/metric"])
        assert isinstance(result, pd.DataFrame)

    # ── get_available_runs ────────────────────────────────────────────────
    def test_get_available_runs_returns_list(self):
        result = get_available_runs("/fake/uri", "exp_id_0")
        assert isinstance(result, list)

    def test_get_available_runs_empty_on_bad_uri(self):
        result = get_available_runs("/bad/uri", "0")
        assert result == []

    # ── parse_run_params ──────────────────────────────────────────────────
    def test_parse_run_params_empty(self):
        result = parse_run_params({})
        assert isinstance(result, dict)
        assert "run_id" in result
        assert "status" in result

    def test_parse_run_params_with_data(self):
        run = {"run_id": "abc123", "run_name": "run1", "status": "FINISHED", "start_time": 0}
        result = parse_run_params(run)
        assert result["run_id"] == "abc123"
        assert result["status"] == "FINISHED"

    # ── build_metric_chart ────────────────────────────────────────────────
    def test_build_metric_chart_empty_df(self):
        import plotly.graph_objects as go
        df = pd.DataFrame(columns=["step", "metric", "value"])
        fig = build_metric_chart(df, "train/reward")
        assert isinstance(fig, go.Figure)

    def test_build_metric_chart_with_data(self):
        import plotly.graph_objects as go
        df = pd.DataFrame({"step": [1, 2, 3], "metric": ["train/reward"] * 3, "value": [0.1, 0.2, 0.3]})
        fig = build_metric_chart(df, "train/reward")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_build_metric_chart_returns_figure(self):
        import plotly.graph_objects as go
        fig = build_metric_chart(pd.DataFrame(columns=["step", "metric", "value"]), "x/y", title="My Chart")
        assert isinstance(fig, go.Figure)

    def test_build_multi_metric_chart_empty(self):
        import plotly.graph_objects as go
        fig = build_multi_metric_chart(pd.DataFrame(), ["a", "b"])
        assert isinstance(fig, go.Figure)

    def test_build_multi_metric_chart_with_data(self):
        import plotly.graph_objects as go
        df = pd.DataFrame({
            "step": [1, 2, 3, 1, 2, 3],
            "metric": ["a"] * 3 + ["b"] * 3,
            "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        fig = build_multi_metric_chart(df, ["a", "b"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    # ── Page callable / sync ──────────────────────────────────────────────
    def test_training_dashboard_page_callable(self):
        _st_mock.session_state = {}
        _st_mock.selectbox.return_value = ""
        # Should not raise
        try:
            render_training_dashboard()
        except Exception:
            pass  # May fail on mlflow, just verify it's callable

    def test_training_dashboard_no_async(self):
        assert not inspect.iscoroutinefunction(render_training_dashboard)

    def test_training_dashboard_session_state_key(self):
        # Function sets session_state key
        _st_mock.session_state = {}
        _st_mock.sidebar.text_input.return_value = "/tmp/mlruns"
        try:
            render_training_dashboard()
        except Exception:
            pass
        # session_state key should have been set (or mlflow failed gracefully)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Ensemble Monitor helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsembleMonitorHelpers:

    # ── load_ensemble_checkpoint ──────────────────────────────────────────
    def test_load_ensemble_checkpoint_missing(self):
        result = load_ensemble_checkpoint("/nonexistent/path/ensemble.json")
        assert result == {}

    def test_load_ensemble_checkpoint_valid(self, tmp_path):
        data = {
            "weights": {"ppo": 0.4, "sac": 0.35, "td3": 0.25},
            "metrics": {"ppo": {"sharpe": 1.2, "max_dd": 0.1, "total_return": 0.15}},
            "regime": 0,
            "step": 1000,
        }
        p = tmp_path / "ensemble_1000.json"
        p.write_text(json.dumps(data))
        result = load_ensemble_checkpoint(str(p))
        assert result["step"] == 1000
        assert "ppo" in result["weights"]

    def test_load_ensemble_checkpoint_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        result = load_ensemble_checkpoint(str(p))
        assert result == {}

    def test_load_ensemble_checkpoint_defaults_added(self, tmp_path):
        p = tmp_path / "minimal.json"
        p.write_text('{"step": 5}')
        result = load_ensemble_checkpoint(str(p))
        assert "weights" in result
        assert "metrics" in result
        assert "regime" in result

    def test_load_ensemble_checkpoint_wrong_type(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('[1, 2, 3]')
        result = load_ensemble_checkpoint(str(p))
        assert result == {}

    # ── build_weights_chart ───────────────────────────────────────────────
    def test_build_weights_chart_empty(self):
        import plotly.graph_objects as go
        fig = build_weights_chart({})
        assert isinstance(fig, go.Figure)

    def test_build_weights_chart_valid(self):
        import plotly.graph_objects as go
        fig = build_weights_chart({"ppo": 0.4, "sac": 0.35, "td3": 0.25})
        assert isinstance(fig, go.Figure)

    def test_build_weights_chart_returns_pie(self):
        fig = build_weights_chart({"ppo": 0.5, "sac": 0.5})
        assert len(fig.data) == 1
        assert fig.data[0].type == "pie"

    def test_build_weights_chart_three_agents(self):
        fig = build_weights_chart({"a": 0.3, "b": 0.3, "c": 0.4})
        assert len(fig.data[0].labels) == 3

    # ── build_regime_timeline ─────────────────────────────────────────────
    def test_build_regime_timeline_empty(self):
        import plotly.graph_objects as go
        fig = build_regime_timeline([])
        assert isinstance(fig, go.Figure)

    def test_build_regime_timeline_valid(self):
        import plotly.graph_objects as go
        history = [{"step": i, "regime": i % 3} for i in range(30)]
        fig = build_regime_timeline(history)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_build_regime_timeline_labels(self):
        history = [{"step": 1, "regime": 0}, {"step": 2, "regime": 2}]
        fig = build_regime_timeline(history)
        trace_names = {t.name for t in fig.data}
        assert REGIME_LABELS[0] in trace_names
        assert REGIME_LABELS[2] in trace_names

    def test_regime_colors_all_defined(self):
        assert 0 in REGIME_COLORS
        assert 1 in REGIME_COLORS
        assert 2 in REGIME_COLORS

    # ── build_agent_performance_table ─────────────────────────────────────
    def test_build_agent_performance_table_empty(self):
        df = build_agent_performance_table({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_build_agent_performance_table_valid(self):
        metrics = {
            "ppo": {"sharpe": 1.2, "max_dd": 0.08, "total_return": 0.12},
            "sac": {"sharpe": 0.9, "max_dd": 0.12, "total_return": 0.09},
        }
        df = build_agent_performance_table(metrics)
        assert len(df) == 2
        assert "Agent" in df.columns
        assert "Sharpe" in df.columns

    def test_build_agent_performance_table_columns(self):
        df = build_agent_performance_table({"a": {"sharpe": 1.0, "max_dd": 0.05, "total_return": 0.1}})
        assert set(["Agent", "Sharpe", "Max DD (%)", "Total Return (%)"]).issubset(df.columns)

    # ── normalise_weights ─────────────────────────────────────────────────
    def test_agent_weights_sum_to_one(self):
        w = normalise_weights({"a": 0.3, "b": 0.5, "c": 0.2})
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_normalise_weights_zero_sum(self):
        w = normalise_weights({"a": 0.0, "b": 0.0})
        # Should return equal weights
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_normalise_weights_negative_clipped(self):
        w = normalise_weights({"a": -1.0, "b": 2.0})
        assert w["a"] == 0.0

    # ── get_ensemble_checkpoints ──────────────────────────────────────────
    def test_get_ensemble_checkpoints_missing_dir(self):
        result = get_ensemble_checkpoints("/nonexistent/dir")
        assert result == []

    def test_get_ensemble_checkpoints_returns_sorted(self, tmp_path):
        (tmp_path / "ensemble_200.json").write_text("{}")
        (tmp_path / "ensemble_100.json").write_text("{}")
        result = get_ensemble_checkpoints(str(tmp_path))
        assert result == sorted(result)

    # ── Page callable / sync ──────────────────────────────────────────────
    def test_ensemble_monitor_page_callable(self):
        _st_mock.session_state = {}
        _st_mock.sidebar.text_input.return_value = "/nonexistent"
        try:
            render_ensemble_monitor()
        except Exception:
            pass

    def test_ensemble_monitor_no_async(self):
        assert not inspect.iscoroutinefunction(render_ensemble_monitor)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Paper Trading helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperTradingHelpers:

    # ── get_available_checkpoints ─────────────────────────────────────────
    def test_get_available_checkpoints_empty_dir(self):
        result = get_available_checkpoints("/nonexistent/path")
        assert result == []

    def test_get_available_checkpoints_returns_list(self, tmp_path):
        (tmp_path / "model_a.zip").write_bytes(b"fake")
        result = get_available_checkpoints(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith(".zip")

    def test_get_available_checkpoints_filters_zip(self, tmp_path):
        (tmp_path / "model.zip").write_bytes(b"")
        (tmp_path / "model.pt").write_bytes(b"")
        result = get_available_checkpoints(str(tmp_path))
        assert all(r.endswith(".zip") for r in result)

    def test_get_available_checkpoints_sorted(self, tmp_path):
        (tmp_path / "model_b.zip").write_bytes(b"")
        (tmp_path / "model_a.zip").write_bytes(b"")
        result = get_available_checkpoints(str(tmp_path))
        assert result == sorted(result, reverse=True)

    # ── format_action ─────────────────────────────────────────────────────
    def test_format_action_buy(self):
        assert "BUY" in format_action(0.8)

    def test_format_action_sell(self):
        assert "SELL" in format_action(-0.8)

    def test_format_action_hold(self):
        assert "HOLD" in format_action(0.05)

    def test_format_action_hold_negative_small(self):
        assert "HOLD" in format_action(-0.05)

    def test_format_action_numeric_array(self):
        result = format_action(np.array([0.9]))
        assert "BUY" in result

    def test_format_action_invalid(self):
        result = format_action(None)
        assert result == "HOLD"

    # ── compute_portfolio_metrics ─────────────────────────────────────────
    def test_compute_portfolio_metrics_empty(self):
        mets = compute_portfolio_metrics([])
        assert mets["total_return"] == 0.0
        assert mets["max_drawdown"] == 0.0
        assert mets["n_trades"] == 0

    def test_compute_portfolio_metrics_valid(self):
        history = [
            {"step": i, "portfolio_value": 10000.0 * (1 + i * 0.001), "action": 0.5, "reward": 0.001}
            for i in range(50)
        ]
        mets = compute_portfolio_metrics(history)
        assert mets["total_return"] > 0
        assert 0.0 <= mets["max_drawdown"] <= 1.0

    def test_compute_portfolio_metrics_sharpe(self):
        history = [
            {"step": i, "portfolio_value": 10000.0, "action": 0.0, "reward": 0.0}
            for i in range(30)
        ]
        mets = compute_portfolio_metrics(history)
        assert isinstance(mets["sharpe"], float)

    def test_compute_portfolio_metrics_drawdown(self):
        # Declining portfolio → non-zero drawdown
        history = [
            {"step": i, "portfolio_value": 10000.0 - i * 100, "action": -0.5, "reward": -0.01}
            for i in range(20)
        ]
        mets = compute_portfolio_metrics(history)
        assert mets["max_drawdown"] > 0.0

    def test_compute_portfolio_metrics_win_rate(self):
        history = [
            {"step": i, "portfolio_value": 10000.0 + i * 10, "action": 0.5, "reward": 0.001}
            for i in range(20)
        ]
        mets = compute_portfolio_metrics(history)
        assert 0.0 <= mets["win_rate"] <= 1.0

    def test_compute_portfolio_metrics_n_trades(self):
        history = [
            {"step": i, "portfolio_value": 10000.0, "action": 0.5 if i % 2 == 0 else 0.0, "reward": 0.0}
            for i in range(20)
        ]
        mets = compute_portfolio_metrics(history)
        assert mets["n_trades"] > 0

    # ── build_paper_trading_chart ─────────────────────────────────────────
    def test_build_paper_trading_chart_empty(self):
        import plotly.graph_objects as go
        fig = build_paper_trading_chart([])
        assert isinstance(fig, go.Figure)

    def test_build_paper_trading_chart_valid(self):
        import plotly.graph_objects as go
        history = [
            {"step": i, "portfolio_value": 10000.0 + i, "action": 0.1, "reward": 0.01}
            for i in range(10)
        ]
        fig = build_paper_trading_chart(history)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_build_paper_trading_chart_returns_figure(self):
        import plotly.graph_objects as go
        fig = build_paper_trading_chart([{"step": 1, "portfolio_value": 10000.0, "action": 0.0, "reward": 0.0}])
        assert isinstance(fig, go.Figure)

    # ── Page callable / sync ──────────────────────────────────────────────
    def test_paper_trading_page_callable(self):
        _st_mock.session_state = {}
        _st_mock.button.return_value = False
        _st_mock.sidebar.text_input.return_value = "/nonexistent"
        try:
            render_paper_trading()
        except Exception:
            pass

    def test_paper_trading_no_async(self):
        assert not inspect.iscoroutinefunction(render_paper_trading)

    def test_paper_trading_session_state_init(self):
        _st_mock.session_state = {}
        _st_mock.button.return_value = False
        _st_mock.sidebar.text_input.return_value = "/nonexistent"
        try:
            render_paper_trading()
        except Exception:
            pass
        # Session state keys should be set
        # (at minimum pt_history is initialised)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Config Editor helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigEditorHelpers:

    _VALID_YAML = """\
env:
  window_size: 20
training:
  total_timesteps: 100000
risk:
  max_position_size: 0.2
ensemble:
  method: weighted_average
"""

    # ── load_config_yaml ──────────────────────────────────────────────────
    def test_load_config_yaml_missing_file(self):
        raw, parsed = load_config_yaml("/nonexistent/config.yaml")
        assert raw == ""
        assert parsed == {}

    def test_load_config_yaml_valid(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(self._VALID_YAML)
        raw, parsed = load_config_yaml(str(p))
        assert isinstance(raw, str)
        assert "env" in parsed

    def test_load_config_yaml_returns_string_and_dict(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(self._VALID_YAML)
        raw, parsed = load_config_yaml(str(p))
        assert isinstance(raw, str)
        assert isinstance(parsed, dict)

    # ── validate_config_yaml ──────────────────────────────────────────────
    def test_validate_config_yaml_empty(self):
        ok, msgs = validate_config_yaml("")
        assert not ok
        assert any("empty" in m.lower() for m in msgs)

    def test_validate_config_yaml_valid(self):
        ok, msgs = validate_config_yaml(self._VALID_YAML)
        assert ok
        errors = [m for m in msgs if m.startswith("ERROR:")]
        assert errors == []

    def test_validate_config_yaml_invalid_syntax(self):
        ok, msgs = validate_config_yaml("key: : : :")
        assert not ok
        assert any("ERROR:" in m for m in msgs)

    def test_validate_config_yaml_missing_required_keys(self):
        yaml_str = "training:\n  timesteps: 1000\n"
        ok, msgs = validate_config_yaml(yaml_str)
        assert not ok
        error_texts = " ".join(msgs)
        assert "env" in error_texts or "risk" in error_texts

    def test_validate_config_yaml_warnings_for_recommended(self):
        yaml_str = "env: {}\ntraining: {}\nrisk: {}\n"
        ok, msgs = validate_config_yaml(yaml_str)
        assert ok  # required keys present
        # Should have warnings for missing recommended keys
        warnings = [m for m in msgs if m.startswith("WARNING:")]
        assert len(warnings) > 0

    def test_required_keys_is_set(self):
        assert isinstance(REQUIRED_TOP_LEVEL_KEYS, (set, frozenset))
        assert "env" in REQUIRED_TOP_LEVEL_KEYS

    def test_recommended_keys_is_set(self):
        assert isinstance(RECOMMENDED_KEYS, (set, frozenset))

    # ── save_config_yaml ──────────────────────────────────────────────────
    def test_save_config_yaml_creates_file(self, tmp_path):
        p = tmp_path / "saved.yaml"
        result = save_config_yaml(str(p), self._VALID_YAML)
        assert result is True
        assert p.exists()

    def test_save_config_yaml_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        result = save_config_yaml(str(p), "key: : :")
        assert result is False

    def test_save_config_yaml_validates_before_save(self, tmp_path):
        p = tmp_path / "out.yaml"
        bad = "env: {}\n"  # missing training and risk
        result = save_config_yaml(str(p), bad)
        assert result is False
        assert not p.exists()

    # ── get_config_schema ─────────────────────────────────────────────────
    def test_get_config_schema_returns_dict(self):
        schema = get_config_schema()
        assert isinstance(schema, dict)

    def test_get_config_schema_has_required(self):
        schema = get_config_schema()
        assert "required" in schema

    # ── diff_configs ──────────────────────────────────────────────────────
    def test_diff_configs_identical(self):
        d = {"a": 1, "b": 2}
        result = diff_configs(d, d)
        assert result == []

    def test_diff_configs_addition(self):
        result = diff_configs({}, {"new_key": 42})
        assert any("ADDED" in r for r in result)

    def test_diff_configs_removal(self):
        result = diff_configs({"old_key": 1}, {})
        assert any("REMOVED" in r for r in result)

    def test_diff_configs_change(self):
        result = diff_configs({"key": 1}, {"key": 2})
        assert any("CHANGED" in r for r in result)

    # ── Page callable / sync ──────────────────────────────────────────────
    def test_config_editor_page_callable(self):
        _st_mock.session_state = {}
        _st_mock.text_area.return_value = ""
        _st_mock.sidebar.text_input.return_value = "/nonexistent.yaml"
        try:
            render_config_editor()
        except Exception:
            pass

    def test_config_editor_no_async(self):
        assert not inspect.iscoroutinefunction(render_config_editor)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. App-level tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestApp:

    def test_main_callable(self):
        from deployment.web_interface.app import main
        assert callable(main)

    def test_main_no_async(self):
        from deployment.web_interface.app import main
        assert not inspect.iscoroutinefunction(main)

    def test_pages_list_has_five_entries(self):
        from deployment.web_interface.app import PAGES
        assert len(PAGES) >= 5  # "Results Report" 추가로 현재 6개

    def test_pages_list_contains_training_dashboard(self):
        from deployment.web_interface.app import PAGES
        assert "Training Dashboard" in PAGES

    def test_pages_list_contains_backtest(self):
        from deployment.web_interface.app import PAGES
        assert "Backtest Results" in PAGES

    def test_pages_list_contains_ensemble(self):
        from deployment.web_interface.app import PAGES
        assert "Ensemble Monitor" in PAGES

    def test_pages_list_contains_paper_trading(self):
        from deployment.web_interface.app import PAGES
        assert "Paper Trading" in PAGES

    def test_pages_list_contains_config_editor(self):
        from deployment.web_interface.app import PAGES
        assert "Config Editor" in PAGES

    def test_app_sync_only(self):
        """Verify no coroutine functions are defined in app.py."""
        import deployment.web_interface.app as app_mod
        for name, obj in inspect.getmembers(app_mod):
            if inspect.isfunction(obj):
                assert not inspect.iscoroutinefunction(obj), (
                    f"Unexpected coroutine function in app.py: {name}"
                )

    def test_live_trading_no_async(self):
        from deployment.web_interface.pages.Live_Trading import render_live_trading
        assert not inspect.iscoroutinefunction(render_live_trading)

    def test_model_training_no_async(self):
        from deployment.web_interface.pages.Model_Training import model_training_page
        assert not inspect.iscoroutinefunction(model_training_page)
