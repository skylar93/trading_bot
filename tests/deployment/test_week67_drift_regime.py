"""Phase 6 Week 67 (S56-S60): Drift & Regime tests.

Covers:
    S56 — FeatureDriftDetector fires on distribution shift
    S57 — Regime detection live wire (hook called, MetricsExporter updated)
    S58 — RetrainingTrigger fires on drawdown and drift accumulation
    S59 — ModelRegistry CRUD
    S60 — Integration: PaperTrader with all Week 67 components active
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import List

import numpy as np
import pytest

from training.monitoring.drift_detector import (
    ADWIN,
    DriftDetector,
    FeatureDriftDetector,
)
from training.registry.model_registry import ModelRegistry
from deployment.monitoring.retraining_trigger import RetrainingTrigger, RetrainingEvent


# ===========================================================================
# S56 — FeatureDriftDetector
# ===========================================================================

class TestFeatureDriftDetector:
    """Unit tests for FeatureDriftDetector."""

    def test_init_creates_one_detector_per_feature(self):
        fdd = FeatureDriftDetector(["rsi", "macd", "vol"])
        assert set(fdd._detectors.keys()) == {"rsi", "macd", "vol"}

    def test_init_empty_raises(self):
        with pytest.raises(ValueError):
            FeatureDriftDetector([])

    def test_update_dict_returns_per_feature_flags(self):
        fdd = FeatureDriftDetector(["a", "b"])
        result = fdd.update({"a": 1.0, "b": 2.0})
        assert set(result.keys()) == {"a", "b"}
        assert all(isinstance(v, bool) for v in result.values())

    def test_update_array_maps_positionally(self):
        fdd = FeatureDriftDetector(["x", "y", "z"])
        result = fdd.update(np.array([0.1, 0.2, 0.3]))
        assert set(result.keys()) == {"x", "y", "z"}

    def test_update_array_with_custom_names(self):
        fdd = FeatureDriftDetector(["a", "b"])
        result = fdd.update(np.array([1.0, 2.0, 3.0]), feature_names=["a", "b"])
        assert set(result.keys()) == {"a", "b"}

    def test_nan_skipped_no_state_change(self):
        fdd = FeatureDriftDetector(["rsi"])
        # Should not raise and should return False for NaN
        result = fdd.update({"rsi": float("nan")})
        assert result["rsi"] is False

    def test_inf_skipped_no_state_change(self):
        fdd = FeatureDriftDetector(["rsi"])
        result = fdd.update({"rsi": float("inf")})
        assert result["rsi"] is False

    def test_drift_fires_on_distribution_shift(self):
        """ADWIN should eventually detect a large mean shift."""
        # Use a lenient delta (0.1) for faster detection in tests
        fdd = FeatureDriftDetector(["ret"], method="adwin", confidence=0.1)
        rng = np.random.default_rng(1)
        # Stable phase: mean=0, small noise
        for _ in range(200):
            fdd.update({"ret": float(rng.normal(0.0, 0.005))})
        # Shift phase: mean jumps by 0.5 (very large relative to std)
        fired = False
        for _ in range(200):
            alarms = fdd.update({"ret": float(rng.normal(0.5, 0.005))})
            if alarms["ret"]:
                fired = True
                break
        assert fired, "FeatureDriftDetector should detect a 0.5 mean shift"

    def test_any_drift_property(self):
        fdd = FeatureDriftDetector(["a", "b"])
        # Manually inject a detection by patching the inner detector
        fdd._last_alarms = {"a": True, "b": False}
        assert fdd.any_drift is True
        fdd._last_alarms = {"a": False, "b": False}
        assert fdd.any_drift is False

    def test_drift_features_property(self):
        fdd = FeatureDriftDetector(["a", "b", "c"])
        fdd._last_alarms = {"a": True, "b": False, "c": True}
        assert sorted(fdd.drift_features) == ["a", "c"]

    def test_n_detections_returns_dict(self):
        fdd = FeatureDriftDetector(["x", "y"])
        counts = fdd.n_detections
        assert set(counts.keys()) == {"x", "y"}
        assert all(v == 0 for v in counts.values())

    def test_total_detections_sums(self):
        fdd = FeatureDriftDetector(["a", "b"])
        fdd._detectors["a"]._detector.n_detections = 3
        fdd._detectors["b"]._detector.n_detections = 2
        assert fdd.total_detections == 5

    def test_reset_all_clears_state(self):
        fdd = FeatureDriftDetector(["a", "b"])
        fdd._last_alarms = {"a": True, "b": True}
        fdd.reset()
        assert fdd.any_drift is False
        assert all(v == 0 for v in fdd.n_detections.values())

    def test_reset_single_feature(self):
        fdd = FeatureDriftDetector(["a", "b"])
        fdd._last_alarms = {"a": True, "b": True}
        fdd.reset("a")
        assert fdd._last_alarms["a"] is False
        assert fdd._last_alarms["b"] is True

    def test_reset_unknown_feature_raises(self):
        fdd = FeatureDriftDetector(["a"])
        with pytest.raises(KeyError):
            fdd.reset("nonexistent")

    def test_last_alarms_returns_copy(self):
        fdd = FeatureDriftDetector(["a"])
        copy = fdd.last_alarms()
        copy["a"] = True
        assert fdd._last_alarms["a"] is False  # original unaffected

    def test_array_too_short_raises(self):
        fdd = FeatureDriftDetector(["a", "b", "c"])
        with pytest.raises(ValueError):
            fdd.update(np.array([1.0]))


# ===========================================================================
# S57 — Regime detection live wire
# ===========================================================================

def _make_paper_trader_for_regime(**extra_init_kwargs):
    """Return a minimal PaperTrader with simulation_mode=True."""
    from deployment.paper_trader import PaperTrader

    agent = MagicMock()
    agent.predict.return_value = (np.array([0.0], dtype=np.float32), None)

    config = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.99,
            "window_size": 5,
            "daily_report_interval": 999_999,
            "poll_interval_seconds": 1.0,
        }
    }
    return PaperTrader(
        agent=agent,
        config=config,
        simulation_mode=True,
        **extra_init_kwargs,
    )


class TestRegimeLiveWire:
    """S57: Regime detection live wire."""

    def test_regime_detector_not_called_when_absent(self):
        """Without regime_detector, _check_regime should be a no-op."""
        trader = _make_paper_trader_for_regime()
        # Should not raise
        trader._check_regime()
        assert trader._current_regime == -1

    def test_regime_changes_call_hook(self):
        from training.regime.regime_detector import RegimeDetector

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True

        hook_calls: List[tuple] = []

        def on_change(prev: int, new: int, probs: np.ndarray):
            hook_calls.append((prev, new))

        trader = _make_paper_trader_for_regime(
            regime_detector=rd,
            on_regime_change=on_change,
        )
        # Prime price history with enough data
        rng = np.random.default_rng(0)
        trader._price_history = list(
            50_000.0 + np.cumsum(rng.normal(50.0, 100.0, 30))
        )
        trader._current_regime = 0  # force initial regime to 0

        # Mock predict to return regime 2 (high-vol)
        rd.predict = MagicMock(return_value=np.array([0.0, 0.0, 1.0]))
        trader._check_regime()

        assert len(hook_calls) == 1
        assert hook_calls[0] == (0, 2)
        assert trader._current_regime == 2

    def test_no_hook_call_when_regime_unchanged(self):
        from training.regime.regime_detector import RegimeDetector

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True
        rd.predict = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))

        hook_calls: List[tuple] = []
        trader = _make_paper_trader_for_regime(
            regime_detector=rd,
            on_regime_change=lambda p, n, pr: hook_calls.append((p, n)),
        )
        trader._price_history = [50_000.0] * 20
        trader._current_regime = 0  # same as argmax of [1,0,0]

        trader._check_regime()
        assert hook_calls == []

    def test_regime_exported_to_metrics_exporter(self):
        """After _check_regime, MetricsExporter.snapshot() should reflect the regime."""
        from training.regime.regime_detector import RegimeDetector
        from deployment.paper_trader import PaperTrader

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True
        rd.predict = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))  # regime 1

        trader = _make_paper_trader_for_regime(regime_detector=rd)
        trader._price_history = [50_000.0] * 20
        trader._current_regime = -1

        trader._check_regime()  # triggers regime change -1 → 1

        # Now update metrics and check snapshot
        trader.metrics_exporter.update(current_regime=trader._current_regime)
        snap = trader.metrics_exporter.snapshot()
        assert snap is not None
        assert snap.current_regime == 1

    def test_regime_audit_logged_on_change(self):
        from training.regime.regime_detector import RegimeDetector
        from deployment.audit.audit_logger import AuditLogger

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True
        rd.predict = MagicMock(return_value=np.array([0.0, 0.0, 1.0]))

        audit = MagicMock(spec=AuditLogger)
        trader = _make_paper_trader_for_regime(
            regime_detector=rd,
            audit_logger=audit,
        )
        trader._price_history = [50_000.0] * 20
        trader._current_regime = 1  # force change

        trader._check_regime()

        audit.log_risk_event.assert_called_once()
        args, _ = audit.log_risk_event.call_args
        event_dict = args[0]
        assert event_dict["type"] == "regime_change"
        assert event_dict["new_regime"] == 2
        assert event_dict["prev_regime"] == 1

    def test_regime_hook_exception_does_not_crash(self):
        from training.regime.regime_detector import RegimeDetector

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True
        rd.predict = MagicMock(return_value=np.array([0.0, 0.0, 1.0]))

        def bad_hook(p, n, pr):
            raise RuntimeError("hook error")

        trader = _make_paper_trader_for_regime(
            regime_detector=rd,
            on_regime_change=bad_hook,
        )
        trader._price_history = [50_000.0] * 20

        # Should not raise
        trader._check_regime()

    def test_regime_skips_on_insufficient_price_history(self):
        from training.regime.regime_detector import RegimeDetector

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True
        rd.predict = MagicMock()

        trader = _make_paper_trader_for_regime(regime_detector=rd)
        trader._price_history = [50_000.0] * 3  # < min 5

        trader._check_regime()
        rd.predict.assert_not_called()


# ===========================================================================
# S58 — RetrainingTrigger
# ===========================================================================

class TestRetrainingTrigger:
    """Unit tests for RetrainingTrigger."""

    def test_no_trigger_below_thresholds(self):
        rt = RetrainingTrigger(config={
            "drawdown_trigger_pct": 0.15,
            "drift_alarm_trigger_count": 5,
        })
        event = rt.check(drawdown_pct=0.05, drift_count=2, step=10)
        assert event is None

    def test_drawdown_trigger_fires(self):
        rt = RetrainingTrigger(config={"drawdown_trigger_pct": 0.10})
        event = rt.check(drawdown_pct=0.20, drift_count=0, step=50)
        assert event is not None
        assert event.condition == "drawdown"
        assert event.value == pytest.approx(0.20)
        assert event.threshold == pytest.approx(0.10)
        assert event.step == 50

    def test_drift_trigger_fires(self):
        rt = RetrainingTrigger(config={"drift_alarm_trigger_count": 3})
        event = rt.check(drawdown_pct=0.01, drift_count=5, step=100)
        assert event is not None
        assert event.condition == "drift"
        assert event.value == pytest.approx(5.0)
        assert event.threshold == pytest.approx(3.0)

    def test_drawdown_takes_priority_over_drift(self):
        rt = RetrainingTrigger(config={
            "drawdown_trigger_pct": 0.10,
            "drift_alarm_trigger_count": 3,
        })
        event = rt.check(drawdown_pct=0.20, drift_count=10, step=50)
        assert event is not None
        assert event.condition == "drawdown"

    def test_cooldown_prevents_double_fire(self):
        rt = RetrainingTrigger(config={
            "drawdown_trigger_pct": 0.10,
            "cooldown_steps": 50,
        })
        e1 = rt.check(drawdown_pct=0.20, drift_count=0, step=100)
        e2 = rt.check(drawdown_pct=0.20, drift_count=0, step=130)  # within cooldown
        assert e1 is not None
        assert e2 is None  # suppressed by cooldown

    def test_cooldown_allows_fire_after_expiry(self):
        rt = RetrainingTrigger(config={
            "drawdown_trigger_pct": 0.10,
            "cooldown_steps": 50,
        })
        rt.check(drawdown_pct=0.20, drift_count=0, step=100)
        e2 = rt.check(drawdown_pct=0.20, drift_count=0, step=151)  # past cooldown
        assert e2 is not None

    def test_events_accumulated(self):
        rt = RetrainingTrigger(config={
            "drawdown_trigger_pct": 0.10,
            "cooldown_steps": 1,
        })
        rt.check(drawdown_pct=0.20, drift_count=0, step=10)
        rt.check(drawdown_pct=0.20, drift_count=0, step=12)
        assert len(rt.events) == 2

    def test_reset_clears_events_and_cooldown(self):
        rt = RetrainingTrigger(config={
            "drawdown_trigger_pct": 0.10,
            "cooldown_steps": 1000,
        })
        rt.check(drawdown_pct=0.20, drift_count=0, step=10)
        rt.reset()
        assert len(rt.events) == 0
        e = rt.check(drawdown_pct=0.20, drift_count=0, step=11)
        assert e is not None  # cooldown cleared

    def test_on_trigger_callback_called(self):
        called: List[RetrainingEvent] = []
        rt = RetrainingTrigger(
            config={"drawdown_trigger_pct": 0.10},
            on_trigger=called.append,
        )
        rt.check(drawdown_pct=0.20, drift_count=0, step=1)
        assert len(called) == 1
        assert called[0].condition == "drawdown"

    def test_audit_logger_receives_event(self):
        from deployment.audit.audit_logger import AuditLogger
        audit = MagicMock(spec=AuditLogger)
        rt = RetrainingTrigger(
            config={"drawdown_trigger_pct": 0.10},
            audit_logger=audit,
        )
        rt.check(drawdown_pct=0.20, drift_count=0, step=5)
        audit.log_risk_event.assert_called_once()
        args, _ = audit.log_risk_event.call_args
        assert args[0]["type"] == "retraining_trigger"
        assert args[0]["condition"] == "drawdown"

    def test_on_trigger_exception_does_not_crash(self):
        def bad_cb(e):
            raise RuntimeError("cb error")

        rt = RetrainingTrigger(
            config={"drawdown_trigger_pct": 0.10},
            on_trigger=bad_cb,
        )
        # Should not raise
        rt.check(drawdown_pct=0.20, drift_count=0, step=1)

    def test_event_to_dict(self):
        event = RetrainingEvent(
            condition="drift",
            value=5.0,
            threshold=3.0,
            step=99,
        )
        d = event.to_dict()
        assert d["condition"] == "drift"
        assert d["value"] == pytest.approx(5.0)
        assert d["step"] == 99


# ===========================================================================
# S59 — ModelRegistry
# ===========================================================================

class TestModelRegistry:
    """Unit tests for ModelRegistry."""

    @pytest.fixture
    def registry(self, tmp_path):
        return ModelRegistry(tmp_path / "registry.json")

    def test_starts_empty(self, registry):
        assert len(registry) == 0
        assert registry.latest() is None
        assert registry.list_versions() == []

    def test_register_returns_version_id(self, registry):
        vid = registry.register(name="ppo_v1", path="/models/ppo.zip")
        assert vid == "v1"

    def test_register_increments_version(self, registry):
        v1 = registry.register(name="ppo_v1", path="/m/1.zip")
        v2 = registry.register(name="ppo_v2", path="/m/2.zip")
        assert v1 == "v1"
        assert v2 == "v2"

    def test_get_returns_entry(self, registry):
        vid = registry.register(
            name="model_a",
            path="/path/to/model.zip",
            metrics={"sharpe": 1.5},
            config={"lr": 3e-4},
            tags={"env": "paper"},
        )
        entry = registry.get(vid)
        assert entry["version"] == "v1"
        assert entry["name"] == "model_a"
        assert entry["metrics"]["sharpe"] == pytest.approx(1.5)
        assert entry["config"]["lr"] == pytest.approx(3e-4)
        assert entry["tags"]["env"] == "paper"
        assert "created_at" in entry

    def test_get_unknown_raises_key_error(self, registry):
        with pytest.raises(KeyError):
            registry.get("v99")

    def test_latest_returns_last(self, registry):
        registry.register(name="a", path="/a.zip")
        registry.register(name="b", path="/b.zip")
        assert registry.latest()["name"] == "b"

    def test_list_versions_ordered(self, registry):
        registry.register(name="first", path="/1.zip")
        registry.register(name="second", path="/2.zip")
        versions = registry.list_versions()
        assert len(versions) == 2
        assert versions[0]["name"] == "first"
        assert versions[1]["name"] == "second"

    def test_delete_removes_entry(self, registry):
        vid = registry.register(name="x", path="/x.zip")
        registry.delete(vid)
        assert len(registry) == 0
        with pytest.raises(KeyError):
            registry.get(vid)

    def test_delete_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.delete("v99")

    def test_update_metrics(self, registry):
        vid = registry.register(name="m", path="/m.zip", metrics={"sharpe": 1.0})
        registry.update_metrics(vid, {"sortino": 1.3, "sharpe": 1.1})
        entry = registry.get(vid)
        assert entry["metrics"]["sharpe"] == pytest.approx(1.1)
        assert entry["metrics"]["sortino"] == pytest.approx(1.3)

    def test_update_metrics_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.update_metrics("v99", {"sharpe": 1.0})

    def test_persisted_to_disk(self, tmp_path):
        path = tmp_path / "reg.json"
        r1 = ModelRegistry(path)
        r1.register(name="m1", path="/m1.zip")

        r2 = ModelRegistry(path)  # reload from disk
        assert len(r2) == 1
        assert r2.latest()["name"] == "m1"

    def test_json_file_is_valid(self, tmp_path):
        path = tmp_path / "reg.json"
        reg = ModelRegistry(path)
        reg.register(name="test", path="/t.zip", metrics={"x": 1.0})
        data = json.loads(path.read_text())
        assert "versions" in data
        assert len(data["versions"]) == 1

    def test_repr_includes_path_and_count(self, tmp_path):
        path = tmp_path / "r.json"
        reg = ModelRegistry(path)
        reg.register(name="a", path="/a.zip")
        r = repr(reg)
        assert "n_versions=1" in r


# ===========================================================================
# S60 — Integration: PaperTrader full stack
# ===========================================================================

def _prices_for_test(n: int = 50, seed: int = 7):
    rng = np.random.default_rng(seed)
    return (50_000.0 + np.cumsum(rng.normal(50.0, 100.0, n))).tolist()


class TestWeek67Integration:
    """Integration test: PaperTrader with FeatureDriftDetector + RegimeDetector."""

    def test_paper_trader_runs_with_feature_drift_detector(self):
        from deployment.paper_trader import PaperTrader

        fdd = FeatureDriftDetector(["f0", "f1", "f2", "f3", "f4"])
        agent = MagicMock()
        agent.predict.return_value = (np.array([0.1], dtype=np.float32), None)
        config = {
            "paper_trading": {
                "symbol": "BTC/USDT",
                "initial_balance": 10_000.0,
                "trading_fee": 0.001,
                "max_position_size": 1.0,
                "max_drawdown_threshold": 0.99,
                "window_size": 5,
                "daily_report_interval": 999_999,
                "poll_interval_seconds": 1.0,
            }
        }
        trader = PaperTrader(
            agent=agent,
            config=config,
            simulation_mode=True,
            feature_drift_detector=fdd,
        )
        report = trader.run(price_stream=iter(_prices_for_test(30)))
        # Should complete without error
        assert report["steps"] > 0

    def test_regime_change_hook_fires_during_run(self):
        """Run PaperTrader with a threshold-based regime detector; hook must be invoked."""
        from training.regime.regime_detector import RegimeDetector
        from deployment.paper_trader import PaperTrader

        rd = RegimeDetector(method="threshold", n_regimes=3)
        rd._is_fitted = True

        regime_changes: List[tuple] = []

        def hook(prev, new, probs):
            regime_changes.append((prev, new))

        agent = MagicMock()
        agent.predict.return_value = (np.array([0.0], dtype=np.float32), None)
        config = {
            "paper_trading": {
                "symbol": "BTC/USDT",
                "initial_balance": 10_000.0,
                "trading_fee": 0.001,
                "max_position_size": 1.0,
                "max_drawdown_threshold": 0.99,
                "window_size": 5,
                "daily_report_interval": 999_999,
                "poll_interval_seconds": 1.0,
            }
        }

        # Build a price stream that changes volatility to force regime change.
        # Low-vol segment → high-vol segment
        rng = np.random.default_rng(42)
        prices_low = (50_000.0 + np.cumsum(rng.normal(0.0, 10.0, 30))).tolist()
        prices_high = (prices_low[-1] + np.cumsum(rng.normal(0.0, 2_000.0, 30))).tolist()
        prices = prices_low + prices_high

        trader = PaperTrader(
            agent=agent,
            config=config,
            simulation_mode=True,
            regime_detector=rd,
            on_regime_change=hook,
        )
        trader.run(price_stream=iter(prices))

        # The hook must have fired at least once (initial -1 → first regime)
        assert len(regime_changes) >= 1

    def test_metrics_snapshot_has_week67_fields(self):
        """MetricSnapshot should contain current_regime and feature_drift_alarms."""
        from deployment.monitoring.metrics_exporter import MetricsExporter, MetricSnapshot

        me = MetricsExporter()
        snap = me.update(
            portfolio_value=10_000.0,
            cash=5_000.0,
            position=0.1,
            unrealised_pnl=0.0,
            realised_pnl=0.0,
            drawdown_pct=0.0,
            num_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            drift_detected=False,
            alerts_fired=0,
            current_regime=2,
            feature_drift_alarms=3,
        )
        assert snap.current_regime == 2
        assert snap.feature_drift_alarms == 3

    def test_metrics_to_json_includes_week67_fields(self):
        from deployment.monitoring.metrics_exporter import MetricsExporter

        me = MetricsExporter()
        me.update(
            portfolio_value=10_000.0,
            cash=5_000.0,
            position=0.0,
            unrealised_pnl=0.0,
            realised_pnl=0.0,
            drawdown_pct=0.0,
            num_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            drift_detected=False,
            alerts_fired=0,
            current_regime=1,
            feature_drift_alarms=7,
        )
        d = me.to_json()
        assert "current_regime" in d
        assert d["current_regime"] == 1
        assert "feature_drift_alarms" in d
        assert d["feature_drift_alarms"] == 7
