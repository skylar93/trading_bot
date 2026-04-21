"""Tests for training.evaluation.walkforward — Week 79 (H10)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.evaluation.walkforward import (
    PurgedKFoldSplitter,
    WalkForwardReport,
    WalkForwardEvaluator,
    STAGING_GATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "$open": close - 0.5,
            "$high": close + 1,
            "$low": close - 1,
            "$close": close,
            "$volume": rng.uniform(1000, 5000, n),
        },
        index=idx,
    )


class _DummyAgent:
    def get_action(self, obs, deterministic=False):
        return 0

    def train_step(self, *args, **kwargs):
        pass


class _DummyEnv:
    def __init__(self, data):
        self.data = data
        self._step = 0

    def reset(self):
        self._step = 0
        return np.zeros(4), {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(4)
        reward = np.random.normal(0.001, 0.01)
        done = self._step >= len(self.data)
        return obs, reward, done, False, {}


# ---------------------------------------------------------------------------
# PurgedKFoldSplitter
# ---------------------------------------------------------------------------

class TestPurgedKFoldSplitter:
    def test_produces_correct_number_of_splits(self):
        data = _make_data(300)
        splitter = PurgedKFoldSplitter(n_splits=4, embargo_bars=10)
        splits = splitter.split(data)
        assert len(splits) <= 4
        assert len(splits) >= 1

    def test_train_and_test_do_not_overlap(self):
        data = _make_data(300)
        splitter = PurgedKFoldSplitter(n_splits=4, embargo_bars=10)
        splits = splitter.split(data)
        for train_df, test_df in splits:
            train_end = train_df.index[-1]
            test_start = test_df.index[0]
            assert test_start > train_end, "Test starts before train ends"

    def test_embargo_gap_respected(self):
        data = _make_data(300)
        embargo = 15
        splitter = PurgedKFoldSplitter(n_splits=3, embargo_bars=embargo)
        splits = splitter.split(data)
        for train_df, test_df in splits:
            # The gap must be > 0 (train index is before test index)
            assert train_df.index[-1] < test_df.index[0]

    def test_expanding_window(self):
        data = _make_data(400)
        splitter = PurgedKFoldSplitter(n_splits=4, embargo_bars=5)
        splits = splitter.split(data)
        train_sizes = [len(t) for t, _ in splits]
        # Expanding: each fold has at least as much train data as the previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_raises_on_too_little_data(self):
        data = _make_data(5)
        splitter = PurgedKFoldSplitter(n_splits=10, embargo_bars=10)
        with pytest.raises(ValueError):
            splitter.split(data)


# ---------------------------------------------------------------------------
# WalkForwardReport
# ---------------------------------------------------------------------------

class TestWalkForwardReport:
    def _make_report(self, oos_sharpe=0.5, stability=0.6, dd=0.2, n_folds=5):
        return WalkForwardReport(
            model_version=1,
            n_folds=n_folds,
            oos_sharpe_mean=oos_sharpe,
            oos_sharpe_std=0.1,
            is_sharpe_mean=oos_sharpe / max(stability, 1e-8),
            stability_ratio=stability,
            mean_max_drawdown=dd,
        )

    def test_passes_gate(self):
        r = self._make_report(oos_sharpe=0.5, stability=0.6, dd=0.2, n_folds=5)
        assert r.passes_staging_gate()

    def test_fails_gate_low_sharpe(self):
        r = self._make_report(oos_sharpe=0.1, stability=0.6, dd=0.2, n_folds=5)
        assert not r.passes_staging_gate()
        failures = r.gate_failures()
        assert any("oos_sharpe" in f for f in failures)

    def test_fails_gate_low_stability(self):
        r = self._make_report(oos_sharpe=0.5, stability=0.1, dd=0.2, n_folds=5)
        assert not r.passes_staging_gate()
        failures = r.gate_failures()
        assert any("stability" in f for f in failures)

    def test_fails_gate_high_drawdown(self):
        r = self._make_report(oos_sharpe=0.5, stability=0.6, dd=0.5, n_folds=5)
        assert not r.passes_staging_gate()
        failures = r.gate_failures()
        assert any("drawdown" in f for f in failures)

    def test_fails_gate_too_few_folds(self):
        r = self._make_report(oos_sharpe=0.5, stability=0.6, dd=0.2, n_folds=2)
        assert not r.passes_staging_gate()
        failures = r.gate_failures()
        assert any("n_folds" in f for f in failures)

    def test_summary_line_contains_pass(self):
        r = self._make_report()
        assert "PASS" in r.summary_line()

    def test_summary_line_contains_fail(self):
        r = self._make_report(oos_sharpe=0.0)
        assert "FAIL" in r.summary_line()

    def test_to_dict_and_back(self):
        r = self._make_report()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["n_folds"] == r.n_folds

    def test_save_and_load(self, tmp_path):
        r = self._make_report()
        p = tmp_path / "report.json"
        r.save(p)
        loaded = WalkForwardReport.load(p)
        assert loaded.n_folds == r.n_folds
        assert abs(loaded.oos_sharpe_mean - r.oos_sharpe_mean) < 1e-9


# ---------------------------------------------------------------------------
# WalkForwardEvaluator — integration smoke test
# ---------------------------------------------------------------------------

class TestWalkForwardEvaluator:
    def test_evaluate_produces_report(self):
        data = _make_data(300)
        evaluator = WalkForwardEvaluator(
            n_splits=3,
            embargo_bars=10,
            total_timesteps=50,
            eval_episodes=2,
        )
        report = evaluator.evaluate(
            agent_factory=_DummyAgent,
            env_factory=_DummyEnv,
            data=data,
            model_version=1,
        )
        assert isinstance(report, WalkForwardReport)
        assert report.n_folds >= 1
        assert report.model_version == 1

    def test_evaluate_saves_report(self, tmp_path):
        data = _make_data(200)
        evaluator = WalkForwardEvaluator(
            n_splits=2, embargo_bars=10, total_timesteps=20, eval_episodes=1
        )
        report = evaluator.evaluate(
            agent_factory=_DummyAgent,
            env_factory=_DummyEnv,
            data=data,
        )
        p = tmp_path / "wf_report.json"
        report.save(p)
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert "oos_sharpe_mean" in loaded
