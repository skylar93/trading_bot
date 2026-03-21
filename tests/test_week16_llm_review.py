"""
Week 16 Tests: LLMReviewPanel (training/review/llm_review_panel.py)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.review.llm_review_panel import (
    AgentBehaviorSummary,
    LLMReviewPanel,
    ReviewResult,
    _DRY_RUN_RESPONSE,
    _MAX_WEIGHT_CHANGE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_summary(**overrides):
    rng = np.random.default_rng(0)
    prices = (50_000.0 + np.cumsum(rng.normal(100, 200, 100))).tolist()
    actions = rng.uniform(-1, 1, 100).tolist()
    defaults = dict(
        actions=actions,
        portfolio_values=prices,
        trades=[
            {"side": "buy",  "price": 50_200.0, "quantity": 0.1, "pnl": 0.0},
            {"side": "sell", "price": 51_000.0, "quantity": 0.1, "pnl": 80.0},
            {"side": "buy",  "price": 50_500.0, "quantity": 0.1, "pnl": 0.0},
            {"side": "sell", "price": 49_800.0, "quantity": 0.1, "pnl": -70.0},
        ],
        current_reward_weights={
            "pnl": 0.35,
            "sharpe": 0.25,
            "downside_risk": 0.15,
            "drawdown": 0.15,
            "transaction_cost": 0.10,
        },
        symbol="BTC/USDT",
        period_label="test-fold-1",
    )
    defaults.update(overrides)
    return AgentBehaviorSummary(**defaults)


def _make_panel(dry_run=True) -> LLMReviewPanel:
    return LLMReviewPanel(dry_run=dry_run)


# ---------------------------------------------------------------------------
# Test 1: AgentBehaviorSummary creation
# ---------------------------------------------------------------------------

def test_agent_behavior_summary_creation():
    s = _make_summary()
    assert s.symbol == "BTC/USDT"
    assert len(s.actions) == 100
    assert len(s.portfolio_values) == 100
    assert isinstance(s.action_distribution, dict)


# ---------------------------------------------------------------------------
# Test 2: Derived statistics computed in __post_init__
# ---------------------------------------------------------------------------

def test_derived_statistics_computed():
    s = _make_summary()
    assert isinstance(s.total_return, float)
    assert isinstance(s.sharpe_ratio, float)
    assert 0.0 <= s.max_drawdown <= 1.0
    assert 0.0 <= s.win_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 3: Action distribution sums to 1
# ---------------------------------------------------------------------------

def test_action_distribution_sums_to_one():
    s = _make_summary()
    total = (
        s.action_distribution["buy_pct"]
        + s.action_distribution["sell_pct"]
        + s.action_distribution["hold_pct"]
    )
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: LLMReviewPanel initialization
# ---------------------------------------------------------------------------

def test_review_panel_initialization():
    panel = LLMReviewPanel(dry_run=True)
    assert panel.dry_run is True
    assert panel.max_weight_change == _MAX_WEIGHT_CHANGE


# ---------------------------------------------------------------------------
# Test 5: Dry-run returns ReviewResult
# ---------------------------------------------------------------------------

def test_dry_run_returns_review():
    panel = _make_panel(dry_run=True)
    result = panel.review(_make_summary())
    assert isinstance(result, ReviewResult)


# ---------------------------------------------------------------------------
# Test 6: ReviewResult has all five perspectives populated
# ---------------------------------------------------------------------------

def test_review_result_has_five_perspectives():
    result = _make_panel().review(_make_summary())
    for field in ("strategy", "risk", "execution", "data", "improvement"):
        assert getattr(result, field), f"Field '{field}' should not be empty"


# ---------------------------------------------------------------------------
# Test 7–11: Each perspective individually non-empty
# ---------------------------------------------------------------------------

def test_strategy_review_populated():
    result = _make_panel().review(_make_summary())
    assert len(result.strategy) > 10


def test_risk_review_populated():
    result = _make_panel().review(_make_summary())
    assert len(result.risk) > 10


def test_execution_review_populated():
    result = _make_panel().review(_make_summary())
    assert len(result.execution) > 10


def test_data_review_populated():
    result = _make_panel().review(_make_summary())
    assert len(result.data) > 10


def test_improvement_review_populated():
    result = _make_panel().review(_make_summary())
    assert len(result.improvement) > 10


# ---------------------------------------------------------------------------
# Test 12: Confidence score in [0, 1]
# ---------------------------------------------------------------------------

def test_confidence_score_range():
    result = _make_panel().review(_make_summary())
    assert 0.0 <= result.confidence_score <= 1.0


# ---------------------------------------------------------------------------
# Test 13: Dry-run suggested weights are within max_weight_change
# ---------------------------------------------------------------------------

def test_dry_run_suggested_weights_valid():
    panel = _make_panel()
    result = panel.review(_make_summary())
    for key, delta in result.suggested_weight_changes.items():
        assert abs(delta) <= panel.max_weight_change + 1e-9, (
            f"Weight change for {key} ({delta}) exceeds limit"
        )


# ---------------------------------------------------------------------------
# Test 14: adjust_reward_weights applies bounded changes
# ---------------------------------------------------------------------------

def test_weight_adjustment_bounded():
    panel = _make_panel()
    current = {"pnl": 0.35, "sharpe": 0.25, "downside_risk": 0.15,
               "drawdown": 0.15, "transaction_cost": 0.10}
    # Inject large suggested change
    result = ReviewResult(suggested_weight_changes={"drawdown": 0.99})
    new_weights = panel.adjust_reward_weights(result, current)
    # Change should be clamped
    change = new_weights["drawdown"] * sum(current.values()) - current["drawdown"]
    assert change <= panel.max_weight_change + 1e-6


# ---------------------------------------------------------------------------
# Test 15: adjust_reward_weights renormalises to sum 1
# ---------------------------------------------------------------------------

def test_weight_adjustment_renormalises():
    panel = _make_panel()
    current = {"pnl": 0.35, "sharpe": 0.25, "downside_risk": 0.15,
               "drawdown": 0.15, "transaction_cost": 0.10}
    result = ReviewResult(suggested_weight_changes={"drawdown": 0.03, "transaction_cost": -0.02})
    new_weights = panel.adjust_reward_weights(result, current)
    assert abs(sum(new_weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 16: adjust_reward_weights – positive delta increases weight
# ---------------------------------------------------------------------------

def test_weight_adjustment_direction():
    panel = _make_panel()
    current = {"pnl": 0.5, "sharpe": 0.5}
    result = ReviewResult(suggested_weight_changes={"pnl": 0.04})
    new_weights = panel.adjust_reward_weights(result, current)
    # pnl raw value before normalisation increases
    # After normalisation pnl should be > 0.5
    assert new_weights["pnl"] > 0.5


# ---------------------------------------------------------------------------
# Test 17: adjust_reward_weights – unchanged when no suggestions
# ---------------------------------------------------------------------------

def test_weight_adjustment_no_change_when_empty():
    panel = _make_panel()
    current = {"pnl": 0.35, "sharpe": 0.25, "downside_risk": 0.15,
               "drawdown": 0.15, "transaction_cost": 0.10}
    result = ReviewResult(suggested_weight_changes={})
    new_weights = panel.adjust_reward_weights(result, current)
    # Weights should be identical (just renormalised, which is identity when they sum to 1)
    for k in current:
        assert abs(new_weights[k] - current[k]) < 1e-6


# ---------------------------------------------------------------------------
# Test 18: log_to_mlflow calls log_artifact
# ---------------------------------------------------------------------------

def test_mlflow_logging_creates_artifact():
    panel = _make_panel()
    result = panel.review(_make_summary())
    mock_mlflow = MagicMock()
    panel.log_to_mlflow(result, mock_mlflow)
    mock_mlflow.log_artifact.assert_called_once()
    mock_mlflow.log_metric.assert_called()


# ---------------------------------------------------------------------------
# Test 19: Prompt construction contains key fields
# ---------------------------------------------------------------------------

def test_prompt_construction():
    panel = _make_panel()
    summary = _make_summary()
    prompt = panel._build_prompt(summary)
    assert "BTC/USDT" in prompt
    assert "Sharpe" in prompt
    assert "strategy" in prompt
    assert "suggested_weight_changes" in prompt


# ---------------------------------------------------------------------------
# Test 20: _parse_response handles JSON correctly
# ---------------------------------------------------------------------------

def test_parse_response_extracts_weights():
    panel = _make_panel()
    raw = json.dumps({
        "strategy": "momentum following",
        "risk": "tail risk present",
        "execution": "good timing",
        "data": "price important",
        "improvement": "increase drawdown weight",
        "suggested_weight_changes": {"drawdown": 0.03, "transaction_cost": -0.01},
        "confidence_score": 0.8,
    })
    result = panel._parse_response(raw)
    assert result.strategy == "momentum following"
    assert result.suggested_weight_changes["drawdown"] == pytest.approx(0.03)
    assert result.confidence_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test 21: _parse_response clamps large weight changes
# ---------------------------------------------------------------------------

def test_parse_response_clamps_large_changes():
    panel = _make_panel()
    raw = json.dumps({
        "strategy": "x", "risk": "x", "execution": "x",
        "data": "x", "improvement": "x",
        "suggested_weight_changes": {"drawdown": 0.99},
        "confidence_score": 0.5,
    })
    result = panel._parse_response(raw)
    assert result.suggested_weight_changes["drawdown"] <= _MAX_WEIGHT_CHANGE


# ---------------------------------------------------------------------------
# Test 22: _parse_response handles malformed JSON gracefully
# ---------------------------------------------------------------------------

def test_parse_response_handles_malformed_json():
    panel = _make_panel()
    result = panel._parse_response("this is not json {{{")
    assert isinstance(result, ReviewResult)
    # Falls back: raw text goes into strategy field
    assert len(result.strategy) > 0


# ---------------------------------------------------------------------------
# Test 23: Review panel with minimal summary (single data point)
# ---------------------------------------------------------------------------

def test_review_panel_with_minimal_summary():
    summary = AgentBehaviorSummary(
        actions=[0.1],
        portfolio_values=[10_000.0],
        trades=[],
        current_reward_weights={"pnl": 1.0},
        symbol="ETH/USDT",
        period_label="minimal",
    )
    panel = _make_panel()
    result = panel.review(summary)
    assert isinstance(result, ReviewResult)
    assert result.confidence_score >= 0.0


# ---------------------------------------------------------------------------
# Test 24: Unknown weight key in suggestions is skipped gracefully
# ---------------------------------------------------------------------------

def test_unknown_weight_key_skipped():
    panel = _make_panel()
    current = {"pnl": 0.6, "sharpe": 0.4}
    result = ReviewResult(
        suggested_weight_changes={"pnl": 0.03, "nonexistent_key": 0.05}
    )
    new_weights = panel.adjust_reward_weights(result, current)
    assert "nonexistent_key" not in new_weights
    assert abs(sum(new_weights.values()) - 1.0) < 1e-6
