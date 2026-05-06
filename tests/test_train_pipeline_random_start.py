"""Regression tests: random_start_eval and eval_episodes flow through run_walk_forward.

These tests guard against the config keys being silently dropped or defaulted
incorrectly before they reach WalkForwardValidator.validate().

Uses mock to avoid running actual training (fast, no GPU needed).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from training.train_pipeline import run_walk_forward
from training.validation.walk_forward import FoldResult, WalkForwardResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    dates = pd.date_range("2025-01-01", periods=300, freq="h")
    rng = np.random.default_rng(0)
    base = rng.normal(100, 1, 300).cumsum()
    return pd.DataFrame(
        {
            "$open": base,
            "$high": base + 0.5,
            "$low": base - 0.5,
            "$close": base,
            "$volume": np.abs(rng.normal(1000, 100, 300)),
        },
        index=dates,
    )


def _stub_result(n_folds: int = 2) -> WalkForwardResult:
    folds = [
        FoldResult(
            fold_idx=i,
            train_size=200,
            test_size=50,
            is_sharpe=0.5,
            oos_sharpe=0.3,
            oos_total_return=0.01,
            oos_total_return_random=0.008,
            oos_trade_count_mean=3.0,
            oos_trade_count_random_mean=4.0,
        )
        for i in range(n_folds)
    ]
    return WalkForwardResult(folds=folds)


def _base_config(**wf_overrides) -> dict:
    cfg = {
        "env": {
            "type": "single_asset_rl",
            "window_size": 10,
            "trading_fee": 0.0001,
        },
        "agent": {"type": "sb3_cvar_ppo"},
        "training": {"total_timesteps": 500},
        "walk_forward": {
            "enabled": True,
            "n_splits": 2,
            "eval_episodes": 5,
            "random_start_eval": False,
            **wf_overrides,
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRandomStartEvalPassthrough:
    """random_start_eval in walk_forward config must reach validator.validate()."""

    def test_random_start_eval_true_forwarded(self, small_df):
        cfg = _base_config(random_start_eval=True, eval_episodes=10)
        with patch(
            "training.train_pipeline.WalkForwardValidator.validate",
            return_value=_stub_result(),
        ) as mock_validate:
            run_walk_forward(cfg, small_df)
        _, kwargs = mock_validate.call_args
        assert kwargs.get("random_start_eval") is True, (
            "random_start_eval=True in config was not forwarded to validator.validate()"
        )

    def test_random_start_eval_false_forwarded(self, small_df):
        cfg = _base_config(random_start_eval=False, eval_episodes=5)
        with patch(
            "training.train_pipeline.WalkForwardValidator.validate",
            return_value=_stub_result(),
        ) as mock_validate:
            run_walk_forward(cfg, small_df)
        _, kwargs = mock_validate.call_args
        assert kwargs.get("random_start_eval") is False

    def test_random_start_eval_defaults_to_false(self, small_df):
        """If key absent from config, defaults to False (no accidental activation)."""
        cfg = _base_config()
        del cfg["walk_forward"]["random_start_eval"]
        with patch(
            "training.train_pipeline.WalkForwardValidator.validate",
            return_value=_stub_result(),
        ) as mock_validate:
            run_walk_forward(cfg, small_df)
        _, kwargs = mock_validate.call_args
        assert kwargs.get("random_start_eval") is False


class TestEvalEpisodesPassthrough:
    """eval_episodes in walk_forward config must reach validator.validate()."""

    def test_eval_episodes_forwarded(self, small_df):
        cfg = _base_config(eval_episodes=20)
        with patch(
            "training.train_pipeline.WalkForwardValidator.validate",
            return_value=_stub_result(),
        ) as mock_validate:
            run_walk_forward(cfg, small_df)
        _, kwargs = mock_validate.call_args
        assert kwargs.get("eval_episodes") == 20, (
            f"eval_episodes=20 not forwarded; got {kwargs.get('eval_episodes')}"
        )

    def test_eval_episodes_default_is_5(self, small_df):
        """Default eval_episodes is 5 per train_pipeline.py line 122."""
        cfg = _base_config()
        del cfg["walk_forward"]["eval_episodes"]
        with patch(
            "training.train_pipeline.WalkForwardValidator.validate",
            return_value=_stub_result(),
        ) as mock_validate:
            run_walk_forward(cfg, small_df)
        _, kwargs = mock_validate.call_args
        assert kwargs.get("eval_episodes") == 5


class TestResultPassthrough:
    """run_walk_forward must return the WalkForwardResult from validator unchanged."""

    def test_result_returned_verbatim(self, small_df):
        expected = _stub_result(n_folds=2)
        cfg = _base_config()
        with patch(
            "training.train_pipeline.WalkForwardValidator.validate",
            return_value=expected,
        ):
            result = run_walk_forward(cfg, small_df)
        assert result is expected


class TestPhase8BetaConfigIntegration:
    """Verify the Phase 8-Beta ablation configs pass random_start_eval=True."""

    @pytest.mark.parametrize("config_path", [
        "config/phase8_beta/B1_inactivity.yaml",
        "config/phase8_beta/B2_sharpe_weight.yaml",
        "config/phase8_beta/B3_sharpe_clip.yaml",
    ])
    def test_beta_configs_enable_random_start(self, config_path, small_df):
        import yaml
        from config.loader import load_raw, _deep_merge
        try:
            cfg = load_raw("config/base.yaml")
        except FileNotFoundError:
            pytest.skip("config/base.yaml not found — run from project root")
        try:
            with open(config_path, encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, override)
        except FileNotFoundError:
            pytest.skip(f"{config_path} not found — Phase 8-Beta configs not yet created")

        wf_cfg = cfg.get("walk_forward", {})
        assert wf_cfg.get("random_start_eval") is True, (
            f"{config_path} must set walk_forward.random_start_eval: true"
        )
        assert wf_cfg.get("eval_episodes", 0) == 20, (
            f"{config_path} must set walk_forward.eval_episodes: 20"
        )
