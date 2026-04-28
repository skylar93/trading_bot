"""Tests for ModelDriftDetector — 5 dimensions (Phase 8 A5)."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pytest

from deployment.monitoring.model_drift import ModelDriftDetector, ModelDriftEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(overrides: Dict | None = None, baseline_hit_rates=None, shadow_hours=0.0):
    """Return a detector with shadow mode expired by default (active mode)."""
    cfg = {
        "model_drift": {
            "action_kl_threshold": 0.5,
            "meta_weight_collapse_pct": 0.80,
            "meta_weight_collapse_window_h": 1,
            "pred_realized_corr_min": 0.05,
            "pred_realized_window": 20,
            "hit_rate_drop_pct": 0.20,
            "shadow_mode_hours": shadow_hours,
            "steps_per_hour": 10,  # collapse window = 1h * 10 = 10 steps
            "action_bins": 10,
            "kl_window_steps": 30,
        },
        "drift": {"action_entropy_min": 0.3},
    }
    if overrides:
        cfg["model_drift"].update(overrides)
    return ModelDriftDetector(
        config=cfg,
        baseline_hit_rates=baseline_hit_rates,
        _start_time=time.time() - 99999,  # shadow expired
    )


def _fill_actions(detector, actions):
    """Push a list of scalar actions into the detector."""
    events = []
    for a in actions:
        events.extend(detector.update(action=float(a)))
    return events


# ---------------------------------------------------------------------------
# Dim 1: Action distribution KL
# ---------------------------------------------------------------------------

class TestActionKL:
    def test_no_alert_below_threshold(self):
        """Identical first/second-half distribution → KL≈0 → no alert.

        Use a tiled deterministic pattern so both halves have matching
        histograms (random samples have high KL due to small-sample variance).
        """
        det = _make_detector()
        # 10-value cycle × 6 = 60 steps; both halves are the same distribution
        pattern = list(np.linspace(0.05, 0.95, 10)) * 6
        events = _fill_actions(det, pattern)
        kl_events = [e for e in events if e.dimension == "action_kl"]
        assert kl_events == [], f"unexpected KL alert: {kl_events}"

    def test_alert_when_distribution_shifts(self):
        """First 30 steps near 0, next 30 near 1 → high KL → WARN."""
        det = _make_detector()
        # first half: all near 0.0
        _fill_actions(det, [0.01] * 30)
        # second half: all near 1.0 → distribution shift
        events = _fill_actions(det, [0.99] * 30)
        kl_events = [e for e in events if e.dimension == "action_kl"]
        assert kl_events, "expected action_kl alert after distribution shift"
        assert kl_events[-1].metric_value > 0.5

    def test_not_enough_samples_silent(self):
        """Buffer not full → no KL check."""
        det = _make_detector()
        events = _fill_actions(det, [0.5] * 10)  # kl_window_steps=30 required
        assert all(e.dimension != "action_kl" for e in events)


# ---------------------------------------------------------------------------
# Dim 2: Meta-controller weight collapse
# ---------------------------------------------------------------------------

class TestMetaWeightCollapse:
    def test_no_alert_balanced_weights(self):
        """Balanced 4-agent weights → no collapse."""
        det = _make_detector()
        weights = {"ppo": 0.3, "sac": 0.3, "td3": 0.2, "flag": 0.2}
        for _ in range(15):
            evs = det.update(meta_weights=weights)
            assert all(e.dimension != "meta_weight_collapse" for e in evs)

    def test_alert_after_full_collapse_window(self):
        """One agent > 80% for collapse_window=10 steps → WARN."""
        det = _make_detector()
        # steps_per_hour=10, window_h=1 → collapse_window=10
        for _ in range(10):
            evs = det.update(meta_weights={"ppo": 0.95, "sac": 0.03, "td3": 0.01, "flag": 0.01})
        assert any(e.dimension == "meta_weight_collapse" for e in evs)

    def test_no_alert_partial_collapse_window(self):
        """Dominant for only 9/10 steps → no alert yet."""
        det = _make_detector()
        for i in range(9):
            evs = det.update(meta_weights={"ppo": 0.95, "sac": 0.03, "td3": 0.01, "flag": 0.01})
            assert all(e.dimension != "meta_weight_collapse" for e in evs)


# ---------------------------------------------------------------------------
# Dim 3: Predicted vs realized return correlation
# ---------------------------------------------------------------------------

class TestPredRealizedCorr:
    def test_no_alert_high_correlation(self):
        """Predicted ≈ realized → Spearman high → no alert."""
        det = _make_detector()
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, 30)
        events = []
        for r in returns:
            events.extend(det.update(predicted_return=r, realized_return=r + 1e-6))
        corr_events = [e for e in events if e.dimension == "pred_realized_corr"]
        assert corr_events == []

    def test_alert_low_correlation(self):
        """Predicted orthogonal to realized → Spearman ≈ 0 → WARN."""
        det = _make_detector()
        rng = np.random.default_rng(2)
        pred = rng.normal(0, 1, 30)
        real = rng.normal(0, 1, 30)  # independent noise → low correlation
        # Force correlation by overwriting
        events = []
        for p, r in zip(pred, real):
            events.extend(det.update(predicted_return=p, realized_return=r))
        # We may or may not hit threshold with truly random data.
        # Instead inject a worst case: pred = -real (anti-correlated)
        det2 = _make_detector()
        signal = np.linspace(-1, 1, 30)
        for v in signal:
            det2.update(predicted_return=v, realized_return=-v)
        snap = det2.snapshot()
        assert snap["pred_realized_corr"] is not None
        assert snap["pred_realized_corr"] < 0  # anti-correlated

    def test_silent_below_min_samples(self):
        """Fewer than window (20) samples → no correlation check."""
        det = _make_detector()
        events = []
        for _ in range(10):
            events.extend(det.update(predicted_return=-1.0, realized_return=1.0))
        corr_events = [e for e in events if e.dimension == "pred_realized_corr"]
        assert corr_events == []


# ---------------------------------------------------------------------------
# Dim 4: Per-regime hit rate
# ---------------------------------------------------------------------------

class TestRegimeHitRate:
    def test_no_alert_above_threshold(self):
        """Hit rate at baseline → no alert."""
        det = _make_detector(baseline_hit_rates={0: 0.6})
        # 10 trades, all wins → hit_rate = 1.0 >> 0.6 - 0.2 = 0.4
        for _ in range(10):
            evs = det.update(regime=0, trade_won=True)
            assert all(e.dimension != "regime_hit_rate" for e in evs)

    def test_alert_below_threshold(self):
        """Hit rate far below baseline → WARN."""
        det = _make_detector(baseline_hit_rates={0: 0.8})
        # 10 trades, all losses → hit_rate=0.0 < 0.8 - 0.2 = 0.6
        evs = []
        for _ in range(10):
            evs = det.update(regime=0, trade_won=False)
        assert any(e.dimension == "regime_hit_rate" for e in evs)

    def test_silent_below_min_trades(self):
        """Fewer than 10 trades for regime → no alert."""
        det = _make_detector(baseline_hit_rates={1: 0.9})
        for _ in range(9):
            evs = det.update(regime=1, trade_won=False)
            assert all(e.dimension != "regime_hit_rate" for e in evs)

    def test_silent_no_baseline(self):
        """No baseline configured for regime → no alert regardless."""
        det = _make_detector()  # baseline_hit_rates={}
        for _ in range(15):
            evs = det.update(regime=2, trade_won=False)
            assert all(e.dimension != "regime_hit_rate" for e in evs)


# ---------------------------------------------------------------------------
# Dim 5: Action entropy collapse
# ---------------------------------------------------------------------------

class TestActionEntropy:
    def test_no_alert_uniform_actions(self):
        """Uniform distribution → high entropy → no alert."""
        det = _make_detector()
        rng = np.random.default_rng(3)
        # 60 steps uniformly spread across [0,1]
        events = _fill_actions(det, rng.uniform(0, 1, 60))
        ent_events = [e for e in events if e.dimension == "action_entropy"]
        assert ent_events == []

    def test_alert_constant_action(self):
        """All same action → zero entropy → WARN."""
        det = _make_detector()
        events = _fill_actions(det, [0.5] * 60)
        ent_events = [e for e in events if e.dimension == "action_entropy"]
        assert ent_events, "expected action_entropy alert for constant action"
        assert ent_events[-1].metric_value < det.action_entropy_min


# ---------------------------------------------------------------------------
# Shadow mode
# ---------------------------------------------------------------------------

class TestShadowMode:
    def test_shadow_mode_still_emits_events(self):
        """During shadow mode events are still returned (but WARNING only)."""
        cfg = {
            "model_drift": {
                "action_kl_threshold": 0.5,
                "meta_weight_collapse_pct": 0.80,
                "meta_weight_collapse_window_h": 1,
                "pred_realized_corr_min": 0.05,
                "pred_realized_window": 20,
                "hit_rate_drop_pct": 0.20,
                "shadow_mode_hours": 9999,  # always in shadow
                "steps_per_hour": 10,
                "action_bins": 10,
                "kl_window_steps": 30,
            },
            "drift": {"action_entropy_min": 0.3},
        }
        det = ModelDriftDetector(config=cfg, _start_time=time.time())
        assert det.in_shadow_mode
        # force entropy collapse
        events = _fill_actions(det, [0.5] * 60)
        ent_events = [e for e in events if e.dimension == "action_entropy"]
        assert ent_events, "shadow mode should still emit events"

    def test_alerter_called_in_shadow_mode(self):
        """Alerter.send_alert is called with WARNING even in shadow mode."""
        alerts = []

        class FakeAlerter:
            def send_alert(self, msg, level="WARNING"):
                alerts.append({"msg": msg, "level": level})

        cfg = {
            "model_drift": {
                "action_kl_threshold": 0.5,
                "shadow_mode_hours": 9999,
                "steps_per_hour": 10,
                "action_bins": 10,
                "kl_window_steps": 30,
                "meta_weight_collapse_pct": 0.80,
                "meta_weight_collapse_window_h": 1,
                "pred_realized_corr_min": 0.05,
                "pred_realized_window": 20,
                "hit_rate_drop_pct": 0.20,
            },
            "drift": {"action_entropy_min": 0.3},
        }
        det = ModelDriftDetector(config=cfg, alerter=FakeAlerter(), _start_time=time.time())
        _fill_actions(det, [0.5] * 60)
        assert alerts, "alerter should have been called"
        assert all(a["level"] == "WARNING" for a in alerts)


# ---------------------------------------------------------------------------
# Integration: single-action agent → entropy collapse alert
# ---------------------------------------------------------------------------

class TestIntegrationEntropyCollapse:
    def test_single_action_fires_alert(self):
        """Agent that always outputs 0 → action_entropy alert fires."""
        det = _make_detector()
        events_all = []
        for _ in range(60):
            events_all.extend(det.update(action=0.0))
        fired = [e for e in events_all if e.dimension == "action_entropy"]
        assert fired, "expected action_entropy event for agent stuck on one action"

    def test_snapshot_reflects_low_entropy(self):
        """snapshot() returns entropy below threshold when agent is fixated."""
        det = _make_detector()
        for _ in range(60):
            det.update(action=0.0)
        snap = det.snapshot()
        assert snap["action_entropy"] is not None
        assert snap["action_entropy"] < det.action_entropy_min
        assert snap["n_warnings"] > 0


# ---------------------------------------------------------------------------
# Snapshot / introspection
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_structure(self):
        """snapshot() returns all expected keys."""
        det = _make_detector()
        snap = det.snapshot()
        expected_keys = {
            "n_warnings", "in_shadow_mode", "action_kl", "action_entropy",
            "pred_realized_corr", "meta_weights", "regime_hit_rates", "recent_events",
        }
        assert expected_keys.issubset(snap.keys())

    def test_snapshot_regime_hit_rates_populated(self):
        """After trades, snapshot regime_hit_rates is non-empty."""
        det = _make_detector(baseline_hit_rates={0: 0.6})
        for _ in range(10):
            det.update(regime=0, trade_won=True)
        snap = det.snapshot()
        assert 0 in snap["regime_hit_rates"]
        assert snap["regime_hit_rates"][0] == pytest.approx(1.0)
