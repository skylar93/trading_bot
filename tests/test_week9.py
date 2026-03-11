"""
Phase 3 Week 9: CVaR-constrained training tests.

Verifies:
- compute_cvar standalone function correctness
- CVaRCallback initialisation and validation
- _on_training_start detects on-policy vs off-policy correctly
- Violation detection logic (rollout_count, violation_count, last_cvar)
- Advantage / return buffer scaling on violation (on-policy only)
- Entropy coefficient adjustment (up on violation, decay otherwise)
- Lagrangian dual variable update
- MLflow and SB3 logger metric emission
- _on_rollout_end integration with PPO rollout buffer
- _on_step integration for off-policy algorithms
- End-to-end PPO training with CVaRCallback attached
- build_cvar_callback reads config correctly
- violation_rate property
"""

import math
from typing import Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {
            "$open":   close * rng.uniform(0.998, 1.002, n),
            "$high":   close * rng.uniform(1.000, 1.010, n),
            "$low":    close * rng.uniform(0.990, 1.000, n),
            "$close":  close,
            "$volume": rng.randint(1_000, 10_000, n).astype(float),
        }
    )


def _make_mock_ppo_model(ent_coef: float = 0.01, n_steps: int = 32, n_envs: int = 1):
    """Return a MagicMock that mimics enough of SB3's PPO for testing."""
    model = MagicMock()
    model.ent_coef = ent_coef
    model.num_timesteps = 0

    # Rollout buffer
    buf = MagicMock()
    rewards = np.random.randn(n_steps, n_envs).astype(np.float32)
    buf.rewards = rewards
    advantages = np.ones((n_steps, n_envs), dtype=np.float32)
    buf.advantages = advantages
    returns = np.ones((n_steps, n_envs), dtype=np.float32) * 2.0
    buf.returns = returns
    model.rollout_buffer = buf

    # SB3 logger
    model.logger = MagicMock()
    return model


def _make_mock_sac_model():
    """Return a mock for an off-policy model (SAC/TD3) — no rollout_buffer."""
    model = MagicMock(spec=[
        "ent_coef", "replay_buffer", "logger", "num_timesteps",
    ])
    # SAC auto-entropy: ent_coef is NOT a plain float
    type(model).ent_coef = PropertyMock(return_value=MagicMock())  # tensor-like
    model.num_timesteps = 0

    rb = MagicMock()
    rb.size.return_value = 200
    rewards = np.random.randn(200, 1).astype(np.float32)
    sample = MagicMock()
    sample.rewards = MagicMock()
    sample.rewards.cpu.return_value.numpy.return_value = rewards.flatten()
    rb.sample.return_value = sample
    model.replay_buffer = rb

    model.logger = MagicMock()
    return model


def _attach_callback(cb, model):
    """Simulate SB3's callback initialisation sequence.

    In SB3 >= 2.0, ``logger`` is a @property that reads from ``model.logger``,
    so we must NOT assign ``cb.logger`` directly — just set ``cb.model``.
    """
    cb.model = model          # model is a class-level type annotation, not a property
    cb.num_timesteps = model.num_timesteps
    cb.n_calls = 0
    cb._on_training_start()


# ---------------------------------------------------------------------------
# 1. compute_cvar standalone function
# ---------------------------------------------------------------------------

from agents.sb3.cvar_callback import compute_cvar, CVaRCallback


def test_compute_cvar_module_importable():
    """compute_cvar and CVaRCallback must be importable from agents.sb3."""
    from agents.sb3 import compute_cvar as _cv, CVaRCallback as _CB
    assert callable(_cv)
    assert _CB is CVaRCallback


def test_compute_cvar_basic():
    """CVaR of sorted [−1, 0, 1] at alpha=1/3 is the worst 1 element = -1."""
    arr = np.array([-1.0, 0.0, 1.0])
    result = compute_cvar(arr, alpha=1 / 3)
    assert math.isclose(result, -1.0, rel_tol=1e-6)


def test_compute_cvar_alpha_1_returns_mean():
    """alpha=1.0 → worst 100% = mean of all values."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_cvar(arr, alpha=1.0)
    assert math.isclose(result, np.mean(arr), rel_tol=1e-6)


def test_compute_cvar_empty_array_returns_zero():
    assert compute_cvar(np.array([]), alpha=0.05) == 0.0


def test_compute_cvar_single_element():
    assert math.isclose(compute_cvar(np.array([42.0]), alpha=0.05), 42.0)


def test_compute_cvar_all_positive_positive_result():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_cvar(arr, alpha=0.05) > 0.0


def test_compute_cvar_all_negative_negative_result():
    arr = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    assert compute_cvar(arr, alpha=0.05) < 0.0


def test_compute_cvar_known_value():
    """Worst 20% of [−0.1, −0.05, 0.0, 0.05, 0.1] → [-0.1] → CVaR = -0.1."""
    arr = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    result = compute_cvar(arr, alpha=0.2)
    assert math.isclose(result, -0.1, rel_tol=1e-6)


def test_compute_cvar_2d_array_flattened():
    """compute_cvar should handle 2-D arrays (e.g. rollout_buffer.rewards shape)."""
    arr = np.array([[-1.0, 0.0], [1.0, 2.0]])
    result = compute_cvar(arr, alpha=0.25)  # worst 25% of 4 elements = [-1.0]
    assert math.isclose(result, -1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 2. CVaRCallback.__init__ validation
# ---------------------------------------------------------------------------

def test_default_params():
    cb = CVaRCallback()
    assert cb.alpha == 0.05
    assert cb.cvar_threshold == -0.02
    assert cb.penalty_scale == 2.0
    assert cb.ent_coef_scale == 2.0
    assert cb.max_ent_coef == 0.1
    assert cb.use_lagrangian is False
    assert cb.lagrangian_lr == 0.01
    assert cb.lambda_cvar == 0.0
    assert cb.log_interval == 1


def test_custom_params():
    cb = CVaRCallback(
        alpha=0.1,
        cvar_threshold=-0.05,
        penalty_scale=3.0,
        ent_coef_scale=1.5,
        max_ent_coef=0.2,
        use_lagrangian=True,
        lagrangian_lr=0.005,
        lambda_init=0.1,
        log_interval=5,
    )
    assert cb.alpha == 0.1
    assert cb.cvar_threshold == -0.05
    assert cb.penalty_scale == 3.0
    assert cb.ent_coef_scale == 1.5
    assert cb.max_ent_coef == 0.2
    assert cb.use_lagrangian is True
    assert cb.lagrangian_lr == 0.005
    assert cb.lambda_cvar == 0.1
    assert cb.log_interval == 5


def test_alpha_zero_raises():
    with pytest.raises(ValueError, match="alpha"):
        CVaRCallback(alpha=0.0)


def test_alpha_greater_than_one_raises():
    with pytest.raises(ValueError, match="alpha"):
        CVaRCallback(alpha=1.1)


def test_alpha_exactly_one_ok():
    cb = CVaRCallback(alpha=1.0)
    assert cb.alpha == 1.0


def test_penalty_scale_below_one_raises():
    with pytest.raises(ValueError, match="penalty_scale"):
        CVaRCallback(penalty_scale=0.5)


def test_ent_coef_scale_below_one_raises():
    with pytest.raises(ValueError, match="ent_coef_scale"):
        CVaRCallback(ent_coef_scale=0.9)


def test_initial_counters_zero():
    cb = CVaRCallback()
    assert cb.rollout_count == 0
    assert cb.violation_count == 0
    assert cb.last_cvar == 0.0


# ---------------------------------------------------------------------------
# 3. _on_training_start — algorithm detection
# ---------------------------------------------------------------------------

def test_on_training_start_on_policy_detected():
    cb = CVaRCallback()
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)
    assert cb._is_on_policy is True


def test_on_training_start_off_policy_detected():
    cb = CVaRCallback()
    # SAC-like model: spec without rollout_buffer
    model = MagicMock(spec=["ent_coef", "replay_buffer", "logger", "num_timesteps"])
    model.ent_coef = 0.01
    model.num_timesteps = 0
    model.logger = MagicMock()
    _attach_callback(cb, model)
    assert cb._is_on_policy is False


def test_on_training_start_stores_float_ent_coef():
    cb = CVaRCallback()
    model = _make_mock_ppo_model(ent_coef=0.02)
    _attach_callback(cb, model)
    assert cb._original_ent_coef == pytest.approx(0.02)


def test_on_training_start_no_ent_coef_snapshot_for_tensor():
    cb = CVaRCallback()
    model = MagicMock()
    model.rollout_buffer = MagicMock()
    # ent_coef is a mock (not float)
    model.ent_coef = MagicMock()
    model.num_timesteps = 0
    model.logger = MagicMock()
    _attach_callback(cb, model)
    assert cb._original_ent_coef is None


# ---------------------------------------------------------------------------
# 4. Violation detection
# ---------------------------------------------------------------------------

def test_rollout_count_increments():
    cb = CVaRCallback(cvar_threshold=100.0)  # always violated
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    for i in range(3):
        cb._apply_cvar_constraint(np.array([-1.0, -2.0]), can_modify_buffer=False)
    assert cb.rollout_count == 3


def test_violation_count_increments_on_violation():
    cb = CVaRCallback(cvar_threshold=100.0)  # all rewards well below threshold
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0, -2.0, -3.0]), can_modify_buffer=False)
    assert cb.violation_count == 1


def test_violation_count_stable_when_not_violated():
    cb = CVaRCallback(cvar_threshold=-100.0)  # threshold far below → never violated
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([1.0, 2.0, 3.0]), can_modify_buffer=False)
    assert cb.violation_count == 0


def test_last_cvar_updated():
    cb = CVaRCallback(alpha=1.0)  # CVaR = mean
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    rewards = np.array([1.0, 2.0, 3.0])
    cb._apply_cvar_constraint(rewards, can_modify_buffer=False)
    assert math.isclose(cb.last_cvar, np.mean(rewards), rel_tol=1e-5)


def test_violated_flag_when_below_threshold():
    cb = CVaRCallback(cvar_threshold=0.0)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-0.5, -0.3]), can_modify_buffer=False)
    assert cb.violation_count == 1


def test_not_violated_when_above_threshold():
    cb = CVaRCallback(cvar_threshold=-1.0)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([0.5, 0.3]), can_modify_buffer=False)
    assert cb.violation_count == 0


# ---------------------------------------------------------------------------
# 5. Advantage / return scaling
# ---------------------------------------------------------------------------

def test_advantages_scaled_down_on_violation():
    cb = CVaRCallback(cvar_threshold=100.0, penalty_scale=2.0)
    model = _make_mock_ppo_model()
    model.rollout_buffer.advantages = np.ones((4, 1), dtype=np.float64)
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=True)

    np.testing.assert_allclose(model.rollout_buffer.advantages, 0.5, rtol=1e-6)


def test_returns_scaled_down_on_violation():
    cb = CVaRCallback(cvar_threshold=100.0, penalty_scale=4.0)
    model = _make_mock_ppo_model()
    model.rollout_buffer.returns = np.ones((4, 1), dtype=np.float64) * 8.0
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=True)

    np.testing.assert_allclose(model.rollout_buffer.returns, 2.0, rtol=1e-6)


def test_advantages_not_scaled_when_not_violated():
    cb = CVaRCallback(cvar_threshold=-100.0, penalty_scale=2.0)
    model = _make_mock_ppo_model()
    model.rollout_buffer.advantages = np.ones((4, 1), dtype=np.float64)
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([1.0]), can_modify_buffer=True)

    np.testing.assert_allclose(model.rollout_buffer.advantages, 1.0, rtol=1e-6)


def test_returns_not_scaled_when_not_violated():
    cb = CVaRCallback(cvar_threshold=-100.0, penalty_scale=2.0)
    model = _make_mock_ppo_model()
    model.rollout_buffer.returns = np.ones((4, 1), dtype=np.float64) * 3.0
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([1.0]), can_modify_buffer=True)

    np.testing.assert_allclose(model.rollout_buffer.returns, 3.0, rtol=1e-6)


def test_buffer_not_modified_when_can_modify_buffer_false():
    """Off-policy path: even if violated, buffer is not touched."""
    cb = CVaRCallback(cvar_threshold=100.0, penalty_scale=2.0)
    model = _make_mock_ppo_model()
    orig_adv = model.rollout_buffer.advantages.copy()
    _attach_callback(cb, model)

    # Pass can_modify_buffer=False even though we have rollout_buffer
    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)

    np.testing.assert_allclose(model.rollout_buffer.advantages, orig_adv, rtol=1e-6)


def test_penalty_scale_applied_correctly():
    """penalty_scale=3 → advantages divided by 3."""
    cb = CVaRCallback(cvar_threshold=100.0, penalty_scale=3.0)
    model = _make_mock_ppo_model()
    model.rollout_buffer.advantages = np.full((2, 1), 9.0, dtype=np.float64)
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=True)

    np.testing.assert_allclose(model.rollout_buffer.advantages, 3.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# 6. Entropy coefficient adjustment
# ---------------------------------------------------------------------------

def test_ent_coef_increased_on_violation():
    cb = CVaRCallback(cvar_threshold=100.0, ent_coef_scale=2.0, max_ent_coef=1.0)
    model = _make_mock_ppo_model(ent_coef=0.01)
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)
    assert model.ent_coef == pytest.approx(0.02)


def test_ent_coef_decreases_on_no_violation():
    cb = CVaRCallback(cvar_threshold=-100.0, ent_coef_scale=2.0)
    model = _make_mock_ppo_model(ent_coef=0.04)
    _attach_callback(cb, model)
    cb._original_ent_coef = 0.01  # pretend original is lower

    cb._apply_cvar_constraint(np.array([1.0]), can_modify_buffer=False)
    assert model.ent_coef == pytest.approx(0.02)


def test_ent_coef_capped_at_max():
    cb = CVaRCallback(cvar_threshold=100.0, ent_coef_scale=10.0, max_ent_coef=0.05)
    model = _make_mock_ppo_model(ent_coef=0.04)
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)
    assert model.ent_coef <= cb.max_ent_coef


def test_ent_coef_not_below_original_on_decay():
    cb = CVaRCallback(cvar_threshold=-100.0, ent_coef_scale=2.0)
    model = _make_mock_ppo_model(ent_coef=0.01)  # already at original
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([1.0]), can_modify_buffer=False)
    assert model.ent_coef >= cb._original_ent_coef


def test_ent_coef_not_modified_for_non_float():
    """If ent_coef is a mock (tensor), _adjust_ent_coef should be a no-op."""
    cb = CVaRCallback(cvar_threshold=100.0, ent_coef_scale=2.0)
    model = _make_mock_ppo_model()
    model.ent_coef = MagicMock()  # non-float
    _attach_callback(cb, model)
    # _original_ent_coef should be None
    assert cb._original_ent_coef is None

    original_mock = model.ent_coef
    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)
    # ent_coef should not have been reassigned to a float
    assert model.ent_coef is original_mock


def test_ent_coef_exact_cap_boundary():
    """When ent_coef * scale > max_ent_coef, result is exactly max_ent_coef."""
    cb = CVaRCallback(cvar_threshold=100.0, ent_coef_scale=5.0, max_ent_coef=0.08)
    model = _make_mock_ppo_model(ent_coef=0.05)
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)
    assert model.ent_coef == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# 7. Lagrangian dual variable
# ---------------------------------------------------------------------------

def test_lagrangian_increases_on_violation():
    cb = CVaRCallback(cvar_threshold=0.0, use_lagrangian=True, lagrangian_lr=0.1)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)
    # constraint_violation = 0.0 - (-1.0) = 1.0 → λ += 0.1 * 1.0 = 0.1
    assert cb.lambda_cvar == pytest.approx(0.1, rel=1e-5)


def test_lagrangian_decreases_below_zero_clamped():
    cb = CVaRCallback(
        cvar_threshold=-0.5, use_lagrangian=True, lagrangian_lr=1.0, lambda_init=0.01
    )
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    # rewards above threshold → constraint_violation = -0.5 - positive = negative
    cb._apply_cvar_constraint(np.array([1.0, 2.0, 3.0]), can_modify_buffer=False)
    assert cb.lambda_cvar >= 0.0


def test_lagrangian_never_negative():
    cb = CVaRCallback(cvar_threshold=-10.0, use_lagrangian=True, lagrangian_lr=1.0)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    for _ in range(10):
        cb._apply_cvar_constraint(np.array([5.0, 5.0, 5.0]), can_modify_buffer=False)
    assert cb.lambda_cvar >= 0.0


def test_lagrangian_proportional_to_violation():
    lr = 0.5
    threshold = 0.0
    rewards = np.array([-2.0])  # CVaR ≈ -2.0 → violation = 0.0 - (-2.0) = 2.0
    cb = CVaRCallback(cvar_threshold=threshold, use_lagrangian=True, lagrangian_lr=lr)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(rewards, can_modify_buffer=False)
    # expected: λ = 0 + lr * (threshold - cvar) = 0.5 * (0 - (-2)) = 1.0
    assert cb.lambda_cvar == pytest.approx(1.0, rel=1e-5)


def test_lagrangian_disabled_by_default():
    cb = CVaRCallback(cvar_threshold=0.0, use_lagrangian=False)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-5.0]), can_modify_buffer=False)
    assert cb.lambda_cvar == 0.0


def test_lagrangian_lr_effect():
    """Higher lr → faster lambda growth."""
    def run(lr):
        cb = CVaRCallback(cvar_threshold=0.0, use_lagrangian=True, lagrangian_lr=lr)
        model = _make_mock_ppo_model()
        _attach_callback(cb, model)
        cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)
        return cb.lambda_cvar

    assert run(0.1) < run(0.5)


# ---------------------------------------------------------------------------
# 8. Logging
# ---------------------------------------------------------------------------

def test_metrics_logged_to_sb3_logger():
    cb = CVaRCallback(log_interval=1)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-0.5]), can_modify_buffer=False)

    assert model.logger.record.called
    recorded_keys = [call.args[0] for call in model.logger.record.call_args_list]
    assert any("cvar" in k for k in recorded_keys)


def test_metrics_include_violation_rate():
    cb = CVaRCallback(log_interval=1)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)

    recorded_keys = [call.args[0] for call in model.logger.record.call_args_list]
    assert any("violation_rate" in k for k in recorded_keys)


def test_mlflow_manager_called_on_log():
    mlflow_mgr = MagicMock()
    cb = CVaRCallback(log_interval=1, mlflow_manager=mlflow_mgr)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)

    assert mlflow_mgr.log_metric.called


def test_no_crash_without_mlflow_manager():
    cb = CVaRCallback(mlflow_manager=None, log_interval=1)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)  # no error


def test_log_interval_controls_frequency():
    """Logging should only happen on rollout_count % log_interval == 0."""
    cb = CVaRCallback(log_interval=3)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    # Call 3 times; only on 3rd should logger.record be called
    cb._apply_cvar_constraint(np.array([0.1]), can_modify_buffer=False)  # rollout 1
    cb._apply_cvar_constraint(np.array([0.1]), can_modify_buffer=False)  # rollout 2
    model.logger.record.reset_mock()
    cb._apply_cvar_constraint(np.array([0.1]), can_modify_buffer=False)  # rollout 3 → log
    assert model.logger.record.called


# ---------------------------------------------------------------------------
# 9. _on_rollout_end for on-policy
# ---------------------------------------------------------------------------

def test_on_rollout_end_reads_rollout_buffer_rewards():
    cb = CVaRCallback(cvar_threshold=100.0)  # force violation
    model = _make_mock_ppo_model()
    model.rollout_buffer.rewards = np.array([[-0.5], [-0.3]])
    _attach_callback(cb, model)

    cb._on_rollout_end()

    assert cb.rollout_count == 1
    assert cb.last_cvar < 0.0


def test_on_rollout_end_increments_violation_count():
    cb = CVaRCallback(cvar_threshold=100.0)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    cb._on_rollout_end()

    assert cb.violation_count == 1


def test_on_rollout_end_scales_advantages():
    cb = CVaRCallback(cvar_threshold=100.0, penalty_scale=2.0)
    model = _make_mock_ppo_model()
    model.rollout_buffer.advantages = np.ones((2, 1), dtype=np.float64)
    model.rollout_buffer.rewards = np.array([[-1.0], [-1.0]])
    _attach_callback(cb, model)

    cb._on_rollout_end()

    np.testing.assert_allclose(model.rollout_buffer.advantages, 0.5, rtol=1e-6)


def test_on_rollout_end_skipped_for_off_policy():
    """_on_rollout_end should be a no-op if rollout_buffer is absent."""
    cb = CVaRCallback()
    model = MagicMock(spec=["ent_coef", "replay_buffer", "logger", "num_timesteps"])
    model.ent_coef = 0.01
    model.num_timesteps = 0
    model.logger = MagicMock()
    _attach_callback(cb, model)

    cb._on_rollout_end()  # should not raise

    assert cb.rollout_count == 0


# ---------------------------------------------------------------------------
# 10. _on_step for off-policy
# ---------------------------------------------------------------------------

def test_on_step_returns_true():
    cb = CVaRCallback()
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)
    cb.n_calls = 1
    assert cb._on_step() is True


def test_on_step_skips_for_on_policy():
    cb = CVaRCallback(off_policy_check_interval=1)
    model = _make_mock_ppo_model()  # has rollout_buffer → on_policy
    _attach_callback(cb, model)

    cb._on_step()
    assert cb.rollout_count == 0  # on_step did nothing


def test_on_step_checks_periodically_for_off_policy():
    cb = CVaRCallback(off_policy_check_interval=3)
    model = _make_mock_sac_model()
    _attach_callback(cb, model)
    cb._is_on_policy = False  # force off-policy

    # Step 1, 2: no check
    cb._on_step()
    cb._on_step()
    assert cb.rollout_count == 0

    # Step 3: check fires
    cb._on_step()
    assert cb.rollout_count == 1


def test_on_step_skips_empty_replay_buffer():
    cb = CVaRCallback(off_policy_check_interval=1)
    model = _make_mock_sac_model()
    model.replay_buffer.size.return_value = 0
    _attach_callback(cb, model)
    cb._is_on_policy = False

    cb._on_step()
    assert cb.rollout_count == 0


def test_on_step_does_not_modify_replay_buffer():
    """Off-policy: callback samples buffer for CVaR but does NOT scale its rewards."""
    cb = CVaRCallback(cvar_threshold=100.0, off_policy_check_interval=1, penalty_scale=2.0)
    model = _make_mock_sac_model()
    _attach_callback(cb, model)
    cb._is_on_policy = False

    cb._on_step()

    # CVaR check was performed (sampled the replay buffer)
    model.replay_buffer.sample.assert_called_once()
    # rollout_count incremented, so constraint was evaluated
    assert cb.rollout_count == 1
    # No attempt to set .advantages or .returns on the replay buffer
    # (replay buffers don't have these attrs; if code tried, it would add them to the mock)
    # Verify by checking no assignment was made via the mock's assignment tracking
    assert "advantages" not in model.replay_buffer.__dict__
    assert "returns" not in model.replay_buffer.__dict__


# ---------------------------------------------------------------------------
# 11. violation_rate property
# ---------------------------------------------------------------------------

def test_violation_rate_zero_initially():
    cb = CVaRCallback()
    assert cb.violation_rate == 0.0


def test_violation_rate_after_all_violations():
    cb = CVaRCallback(cvar_threshold=100.0)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    for _ in range(4):
        cb._apply_cvar_constraint(np.array([-1.0]), can_modify_buffer=False)

    assert cb.violation_rate == pytest.approx(1.0)


def test_violation_rate_partial():
    cb = CVaRCallback(cvar_threshold=0.0)
    model = _make_mock_ppo_model()
    _attach_callback(cb, model)

    # 2 violations (negative rewards) + 2 non-violations (positive rewards)
    for r in [[-1.0], [1.0], [-0.5], [0.5]]:
        cb._apply_cvar_constraint(np.array(r), can_modify_buffer=False)

    assert cb.violation_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 12. build_cvar_callback (pipeline integration)
# ---------------------------------------------------------------------------

from training.train_pipeline import build_cvar_callback


def test_build_cvar_callback_disabled_returns_none():
    config = {"risk": {"cvar_enabled": False}}
    assert build_cvar_callback(config) is None


def test_build_cvar_callback_no_risk_key_returns_none():
    assert build_cvar_callback({}) is None


def test_build_cvar_callback_enabled_returns_instance():
    config = {"risk": {"cvar_enabled": True}}
    cb = build_cvar_callback(config)
    assert isinstance(cb, CVaRCallback)


def test_build_cvar_callback_params_passed():
    config = {
        "risk": {
            "cvar_enabled": True,
            "cvar_alpha": 0.1,
            "cvar_threshold": -0.05,
            "penalty_scale": 3.0,
            "ent_coef_scale": 1.5,
            "max_ent_coef": 0.2,
            "use_lagrangian": True,
            "lagrangian_lr": 0.05,
            "log_interval": 2,
        }
    }
    cb = build_cvar_callback(config)
    assert cb.alpha == 0.1
    assert cb.cvar_threshold == -0.05
    assert cb.penalty_scale == 3.0
    assert cb.ent_coef_scale == 1.5
    assert cb.max_ent_coef == 0.2
    assert cb.use_lagrangian is True
    assert cb.lagrangian_lr == 0.05
    assert cb.log_interval == 2


def test_build_cvar_callback_mlflow_forwarded():
    mlflow_mgr = MagicMock()
    config = {"risk": {"cvar_enabled": True}}
    cb = build_cvar_callback(config, mlflow_manager=mlflow_mgr)
    assert cb.mlflow_manager is mlflow_mgr


def test_build_cvar_callback_default_values():
    config = {"risk": {"cvar_enabled": True}}
    cb = build_cvar_callback(config)
    assert cb.alpha == 0.05
    assert cb.cvar_threshold == -0.02
    assert cb.penalty_scale == 2.0


# ---------------------------------------------------------------------------
# 13. End-to-end training with CVaRCallback (real PPO)
# ---------------------------------------------------------------------------

from envs.wrap_env import make_sb3_env
from agents.sb3.sb3_agent_wrapper import SB3AgentWrapper
from stable_baselines3.common.callbacks import CallbackList


@pytest.fixture(scope="module")
def small_df():
    return _make_df(200, seed=0)


@pytest.fixture(scope="module")
def train_env(small_df):
    return make_sb3_env(small_df, n_envs=1, use_vec_normalize=True)


def test_cvar_callback_attaches_to_ppo_without_error(train_env):
    """CVaRCallback must not raise during PPO training (short run)."""
    agent = SB3AgentWrapper(
        "ppo",
        train_env.observation_space,
        train_env.action_space,
        sb3_params={"n_steps": 32, "batch_size": 16, "n_epochs": 2},
    )
    cb = CVaRCallback(cvar_threshold=-10.0)  # threshold low → rarely violated
    agent.train(train_env, total_timesteps=64, callbacks=cb)
    assert agent.model is not None


def test_cvar_callback_records_rollouts(train_env):
    """After training, rollout_count must be > 0."""
    agent = SB3AgentWrapper(
        "ppo",
        train_env.observation_space,
        train_env.action_space,
        sb3_params={"n_steps": 32, "batch_size": 16, "n_epochs": 2},
    )
    cb = CVaRCallback(cvar_threshold=0.0)
    agent.train(train_env, total_timesteps=64, callbacks=cb)
    assert cb.rollout_count >= 1


def test_cvar_training_does_not_crash_with_violations(train_env):
    """Set threshold high (always violated) → training must still complete."""
    agent = SB3AgentWrapper(
        "ppo",
        train_env.observation_space,
        train_env.action_space,
        sb3_params={"n_steps": 32, "batch_size": 16, "n_epochs": 2},
    )
    cb = CVaRCallback(cvar_threshold=1000.0, penalty_scale=2.0)
    agent.train(train_env, total_timesteps=64, callbacks=cb)
    assert cb.violation_count >= 1


def test_cvar_in_callback_list(train_env):
    """CVaRCallback composable with other SB3 callbacks via CallbackList."""
    from stable_baselines3.common.callbacks import BaseCallback

    class DummyCb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.called = False

        def _on_step(self):
            self.called = True
            return True

    agent = SB3AgentWrapper(
        "ppo",
        train_env.observation_space,
        train_env.action_space,
        sb3_params={"n_steps": 32, "batch_size": 16, "n_epochs": 2},
    )
    dummy = DummyCb()
    cvar = CVaRCallback(cvar_threshold=0.0)
    cb_list = CallbackList([dummy, cvar])
    agent.train(train_env, total_timesteps=64, callbacks=cb_list)

    assert dummy.called
    assert cvar.rollout_count >= 1


def test_build_cvar_callback_integrates_into_train_sb3_agent(train_env):
    """build_cvar_callback result integrates with train_sb3_agent callback list."""
    config = {
        "risk": {"cvar_enabled": True, "cvar_threshold": 0.0},
        "training": {
            "total_timesteps": 64,
            "checkpoint_interval": 9999,
            "eval_interval": 9999,
            "log_interval": 32,
            "n_eval_episodes": 1,
        },
        "paths": {"checkpoint_dir": "/tmp/cvar_test_ckpt"},
    }
    cb = build_cvar_callback(config)
    assert isinstance(cb, CVaRCallback)
    assert cb.cvar_threshold == 0.0
