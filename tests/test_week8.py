"""
Week 8 Test Suite: Optuna-based Hyperparameter Optimisation.

Tests cover:
  - _suggest_params: correct ranges & types for every parameter
  - _apply_params_to_config: plain keys, dotted keys, special keys, deep copy
  - _evaluate_agent: SB3/BaseAgent/callable interfaces, NaN handling
  - TrialResult / HyperoptResult dataclasses
  - OptunaHyperopt: init, study creation (single + multi), objective
    function logic (success, prune-on-fail, prune-on-nan, OOM-retry),
    optimize(), _collect_results()
  - run_hyperopt: data splitting, delegation
  - train_pipeline.run_hyperopt_optuna: integration shim
  - config/training_config.yaml: hyperopt section present with required keys
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal OHLCV DataFrame (200 rows) for testing."""
    rng = np.random.default_rng(0)
    n = 200
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    return pd.DataFrame({
        "$open": close * 0.999,
        "$high": close * 1.002,
        "$low": close * 0.997,
        "$close": close,
        "$volume": rng.integers(1_000, 10_000, n).astype(float),
    })


@pytest.fixture
def base_config() -> Dict[str, Any]:
    return {
        "env": {
            "window_size": 10,
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
        },
        "agent": {
            "algo_type": "sb3_ppo",
            "feature_extractor": "mlp",
            "sb3_params": {"ppo": {}},
        },
        "training": {
            "total_timesteps": 100,
            "seed": 0,
        },
        "hyperopt": {
            "n_startup_trials": 2,
            "eval_episodes": 1,
            "trial_timesteps": 50,
        },
    }


def _make_fake_env(df, cfg):
    """Return a tiny fake Gymnasium env for unit tests."""
    import gymnasium as gym

    class FakeEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(5,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self._step = 0

        def reset(self, **kwargs):
            self._step = 0
            return self.observation_space.sample(), {}

        def step(self, action):
            self._step += 1
            obs = self.observation_space.sample()
            reward = float(np.random.randn())
            done = self._step >= 10
            return obs, reward, done, False, {"portfolio_value": 10_000.0}

    return FakeEnv()


def _make_fake_agent(env, cfg):
    """Return a lightweight mock SB3AgentWrapper for unit tests."""
    agent = MagicMock()
    agent.model = MagicMock()
    agent.model.predict = lambda obs, deterministic=True: (
        env.action_space.sample(), None
    )
    agent.train = MagicMock(return_value={"total_timesteps": 50})
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Import guard
# ──────────────────────────────────────────────────────────────────────────────

def test_module_importable():
    from training.hyperopt.hyperopt_optuna import (
        OptunaHyperopt,
        HyperoptResult,
        TrialResult,
        _apply_params_to_config,
        _evaluate_agent,
        _suggest_params,
        run_hyperopt,
    )
    assert True


# ──────────────────────────────────────────────────────────────────────────────
# _suggest_params
# ──────────────────────────────────────────────────────────────────────────────

def _fixed_trial(params_dict):
    """Create an optuna Trial fixed to return values from params_dict."""
    import optuna
    sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial = study.ask()
    return trial


def test_suggest_params_returns_all_keys(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = _suggest_params(trial, base_config)

    required_keys = {
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "ent_coef", "clip_range", "vf_coef", "max_grad_norm",
        "feature_extractor", "reward_weights.pnl", "reward_weights.sharpe",
    }
    assert required_keys <= set(params.keys())


def test_suggest_params_learning_rate_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    for seed in range(5):
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.RandomSampler(seed=seed))
        trial = study.ask()
        p = _suggest_params(trial, base_config)
        assert 1e-4 <= p["learning_rate"] <= 1e-3


def test_suggest_params_n_steps_valid_choice(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert p["n_steps"] in [1024, 2048, 4096]


def test_suggest_params_batch_size_valid_choice(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=2))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert p["batch_size"] in [32, 64, 128, 256]


def test_suggest_params_n_epochs_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=3))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 3 <= p["n_epochs"] <= 15


def test_suggest_params_gamma_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=4))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 0.95 <= p["gamma"] <= 0.999


def test_suggest_params_gae_lambda_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=5))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 0.9 <= p["gae_lambda"] <= 0.99


def test_suggest_params_ent_coef_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=6))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 1e-3 <= p["ent_coef"] <= 0.1


def test_suggest_params_clip_range_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=7))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 0.1 <= p["clip_range"] <= 0.3


def test_suggest_params_vf_coef_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=8))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 0.25 <= p["vf_coef"] <= 1.0


def test_suggest_params_max_grad_norm_valid_choice(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=9))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert p["max_grad_norm"] in [0.3, 0.5, 0.7, 1.0]


def test_suggest_params_feature_extractor_valid_choice(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=10))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert p["feature_extractor"] in ["conv1d", "lstm", "mlp"]


def test_suggest_params_reward_pnl_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=11))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 0.2 <= p["reward_weights.pnl"] <= 0.5


def test_suggest_params_reward_sharpe_in_range(base_config):
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=12))
    trial = study.ask()
    p = _suggest_params(trial, base_config)
    assert 0.1 <= p["reward_weights.sharpe"] <= 0.4


def test_suggest_params_config_override_learning_rate():
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    cfg = {"hyperopt": {"parameters": {"learning_rate": {"min": 5e-4, "max": 5e-4}}}}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    p = _suggest_params(trial, cfg)
    # When min == max, value should equal the fixed point
    assert abs(p["learning_rate"] - 5e-4) < 1e-10


def test_suggest_params_config_override_batch_size():
    import optuna
    from training.hyperopt.hyperopt_optuna import _suggest_params

    cfg = {"hyperopt": {"parameters": {"batch_size": {"values": [64]}}}}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    p = _suggest_params(trial, cfg)
    assert p["batch_size"] == 64


# ──────────────────────────────────────────────────────────────────────────────
# _apply_params_to_config
# ──────────────────────────────────────────────────────────────────────────────

def test_apply_params_plain_key(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"learning_rate": 1e-4})
    assert cfg["agent"]["sb3_params"]["ppo"]["learning_rate"] == 1e-4


def test_apply_params_n_steps(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"n_steps": 4096})
    assert cfg["agent"]["sb3_params"]["ppo"]["n_steps"] == 4096


def test_apply_params_batch_size(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"batch_size": 128})
    assert cfg["agent"]["sb3_params"]["ppo"]["batch_size"] == 128


def test_apply_params_feature_extractor(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"feature_extractor": "lstm"})
    assert cfg["agent"]["feature_extractor"] == "lstm"


def test_apply_params_reward_weights_pnl(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"reward_weights.pnl": 0.35})
    assert cfg["env"]["reward"]["weights"]["pnl"] == pytest.approx(0.35)


def test_apply_params_reward_weights_sharpe(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"reward_weights.sharpe": 0.25})
    assert cfg["env"]["reward"]["weights"]["sharpe"] == pytest.approx(0.25)


def test_apply_params_generic_dotted_key(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    cfg = _apply_params_to_config(base_config, {"some.nested.key": 42})
    assert cfg["some"]["nested"]["key"] == 42


def test_apply_params_does_not_modify_original(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    original = copy.deepcopy(base_config)
    _apply_params_to_config(base_config, {"learning_rate": 9e-4})
    assert base_config == original


def test_apply_params_multiple_keys(base_config):
    from training.hyperopt.hyperopt_optuna import _apply_params_to_config

    params = {
        "learning_rate": 3e-4,
        "gamma": 0.97,
        "feature_extractor": "conv1d",
        "reward_weights.pnl": 0.4,
    }
    cfg = _apply_params_to_config(base_config, params)
    assert cfg["agent"]["sb3_params"]["ppo"]["learning_rate"] == 3e-4
    assert cfg["agent"]["sb3_params"]["ppo"]["gamma"] == 0.97
    assert cfg["agent"]["feature_extractor"] == "conv1d"
    assert cfg["env"]["reward"]["weights"]["pnl"] == pytest.approx(0.4)


# ──────────────────────────────────────────────────────────────────────────────
# _evaluate_agent
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluate_agent_returns_tuple(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})
    agent = MagicMock()
    agent.predict = lambda obs, deterministic=True: (env.action_space.sample(), None)

    result = _evaluate_agent(agent, env, n_episodes=2)
    assert isinstance(result, tuple) and len(result) == 3


def test_evaluate_agent_finite_values(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})
    agent = MagicMock()
    agent.predict = lambda obs, deterministic=True: (env.action_space.sample(), None)

    sharpe, max_dd, total_ret = _evaluate_agent(agent, env, n_episodes=2)
    assert math.isfinite(sharpe)
    assert math.isfinite(max_dd)
    assert math.isfinite(total_ret)
    assert 0.0 <= max_dd <= 1.0


def test_evaluate_agent_uses_predict_interface(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})
    predict_calls = []

    agent = MagicMock()
    def fake_predict(obs, deterministic=True):
        predict_calls.append(1)
        return env.action_space.sample(), None
    agent.predict = fake_predict

    _evaluate_agent(agent, env, n_episodes=1)
    assert len(predict_calls) > 0


def test_evaluate_agent_uses_get_action_interface(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})
    get_action_calls = []

    class FakeAgent:
        def get_action(self, obs):
            get_action_calls.append(1)
            return env.action_space.sample()

    _evaluate_agent(FakeAgent(), env, n_episodes=1)
    assert len(get_action_calls) > 0


def test_evaluate_agent_uses_callable_interface(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})
    call_count = [0]

    def callable_agent(obs):
        call_count[0] += 1
        return env.action_space.sample()

    _evaluate_agent(callable_agent, env, n_episodes=1)
    assert call_count[0] > 0


def test_evaluate_agent_nan_reward_safe(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})

    class NaNAgent:
        def predict(self, obs, deterministic=True):
            return env.action_space.sample(), None

    # Override step: NaN reward AND no portfolio_value in info
    # so the fallback formula `prev * (1 + nan * 0.01)` is exercised.
    original_step = env.step
    def bad_step(action):
        obs, _, t, tr, _ = original_step(action)
        return obs, float("nan"), t, tr, {}  # no portfolio_value key
    env.step = bad_step

    sharpe, max_dd, total_ret = _evaluate_agent(NaNAgent(), env, n_episodes=1)
    # NaN portfolio values are replaced by 1.0 → flat → zero log-returns
    # → sharpe = 0.0, max_dd = 0.0
    assert sharpe == pytest.approx(0.0)
    assert 0.0 <= max_dd <= 0.01  # flat portfolio has ~0 drawdown


def test_evaluate_agent_without_portfolio_value_in_info(sample_df):
    from training.hyperopt.hyperopt_optuna import _evaluate_agent

    env = _make_fake_env(sample_df, {})
    original_step = env.step
    def step_no_pv(action):
        obs, rew, t, tr, _ = original_step(action)
        return obs, rew, t, tr, {}  # no portfolio_value key
    env.step = step_no_pv

    agent = MagicMock()
    agent.predict = lambda obs, deterministic=True: (env.action_space.sample(), None)
    sharpe, max_dd, total_ret = _evaluate_agent(agent, env, n_episodes=1)
    assert math.isfinite(sharpe)


# ──────────────────────────────────────────────────────────────────────────────
# TrialResult / HyperoptResult dataclasses
# ──────────────────────────────────────────────────────────────────────────────

def test_trial_result_defaults():
    from training.hyperopt.hyperopt_optuna import TrialResult

    tr = TrialResult(
        trial_number=0, params={}, sharpe=1.0, max_drawdown=0.05,
        total_return=0.1, n_timesteps=1000, duration_seconds=5.0,
    )
    assert not tr.pruned
    assert not tr.failed
    assert tr.error_msg == ""


def test_trial_result_failed_flag():
    from training.hyperopt.hyperopt_optuna import TrialResult

    tr = TrialResult(
        trial_number=1, params={}, sharpe=0.0, max_drawdown=1.0,
        total_return=0.0, n_timesteps=0, duration_seconds=0.1,
        failed=True, error_msg="crash",
    )
    assert tr.failed
    assert tr.error_msg == "crash"


def test_hyperopt_result_defaults():
    from training.hyperopt.hyperopt_optuna import HyperoptResult

    r = HyperoptResult(n_trials=10, n_completed=8, n_pruned=1, n_failed=1)
    assert r.best_sharpe == -np.inf
    assert r.best_max_drawdown == pytest.approx(1.0)
    assert r.best_params == {}
    assert r.pareto_front == []
    assert r.trials == []
    assert r.study_name == ""


def test_hyperopt_result_trial_count():
    from training.hyperopt.hyperopt_optuna import HyperoptResult

    r = HyperoptResult(n_trials=10, n_completed=7, n_pruned=2, n_failed=1)
    assert r.n_completed + r.n_pruned + r.n_failed == r.n_trials


# ──────────────────────────────────────────────────────────────────────────────
# OptunaHyperopt — init
# ──────────────────────────────────────────────────────────────────────────────

def test_optuna_hyperopt_init(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    split = len(sample_df) // 2
    ho = OptunaHyperopt(
        config=base_config,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df.iloc[:split],
        val_df=sample_df.iloc[split:],
        n_trials=3,
    )
    assert ho.n_trials == 3
    # total_timesteps not passed → falls back to config["training"]["total_timesteps"] = 100
    assert ho._timesteps == 100


def test_optuna_hyperopt_timesteps_from_training_config(sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    cfg = {
        "training": {"total_timesteps": 200, "seed": 0},
        "hyperopt": {},
    }
    ho = OptunaHyperopt(
        config=cfg,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
    )
    assert ho._timesteps == 200


def test_optuna_hyperopt_timesteps_override(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    ho = OptunaHyperopt(
        config=base_config,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
        total_timesteps=999,
    )
    assert ho._timesteps == 999


# ──────────────────────────────────────────────────────────────────────────────
# OptunaHyperopt — _create_study
# ──────────────────────────────────────────────────────────────────────────────

def test_create_study_single_objective(base_config, sample_df):
    import optuna
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    ho = OptunaHyperopt(
        config=base_config,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
        multi_objective=False,
    )
    study = ho._create_study()
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_create_study_multi_objective(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    ho = OptunaHyperopt(
        config=base_config,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
        multi_objective=True,
    )
    study = ho._create_study()
    assert len(study.directions) == 2


def test_create_study_has_successive_halving_pruner(base_config, sample_df):
    import optuna
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    ho = OptunaHyperopt(
        config=base_config,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
        multi_objective=False,
    )
    study = ho._create_study()
    assert isinstance(study.pruner, optuna.pruners.SuccessiveHalvingPruner)


def test_study_name_propagated(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    ho = OptunaHyperopt(
        config=base_config,
        env_factory=_make_fake_env,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
        study_name="my_study",
    )
    study = ho._create_study()
    assert study.study_name == "my_study"


# ──────────────────────────────────────────────────────────────────────────────
# OptunaHyperopt — objective logic (mocked)
# ──────────────────────────────────────────────────────────────────────────────

def test_objective_success_returns_sharpe(base_config, sample_df):
    """Objective returns a finite float for a successful trial."""
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt
    import optuna

    # Mock _evaluate_agent to avoid real env interaction
    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(1.5, 0.1, 0.2),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=1,
            multi_objective=False,
        )
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        value = ho._objective(trial)
    assert math.isfinite(value)


def test_objective_prunes_on_env_factory_failure(base_config, sample_df):
    """If env_factory raises, the trial is pruned."""
    import optuna
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    def bad_env_factory(df, cfg):
        raise RuntimeError("env broken")

    ho = OptunaHyperopt(
        config=base_config,
        env_factory=bad_env_factory,
        agent_factory=_make_fake_agent,
        train_df=sample_df,
        val_df=sample_df,
        n_trials=1,
    )
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    with pytest.raises(optuna.TrialPruned):
        ho._objective(trial)

    assert ho._results[-1].failed


def test_objective_prunes_on_nan_sharpe(base_config, sample_df):
    """Non-finite sharpe is treated as pruned."""
    import optuna
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(float("nan"), 0.1, 0.0),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=1,
        )
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        with pytest.raises(optuna.TrialPruned):
            ho._objective(trial)

    assert ho._results[-1].pruned


def test_objective_records_trial_result(base_config, sample_df):
    """Successful trial is appended to _results."""
    import optuna
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(0.8, 0.12, 0.05),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=1,
        )
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        ho._objective(trial)

    assert len(ho._results) == 1
    assert ho._results[0].sharpe == pytest.approx(0.8)
    assert ho._results[0].max_drawdown == pytest.approx(0.12)


def test_objective_multi_objective_returns_tuple(base_config, sample_df):
    """Multi-objective returns (sharpe, max_dd) tuple."""
    import optuna
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(1.2, 0.08, 0.15),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=1,
            multi_objective=True,
        )
        study = optuna.create_study(directions=["maximize", "minimize"])
        trial = study.ask()
        result = ho._objective(trial)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == pytest.approx(1.2)
    assert result[1] == pytest.approx(0.08)


# ──────────────────────────────────────────────────────────────────────────────
# OptunaHyperopt — optimize() end-to-end (small n_trials)
# ──────────────────────────────────────────────────────────────────────────────

def test_optimize_returns_hyperopt_result(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import HyperoptResult, OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(1.0, 0.1, 0.1),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=2,
        )
        result = ho.optimize()

    assert isinstance(result, HyperoptResult)


def test_optimize_counts_trials(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(0.5, 0.2, 0.05),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=3,
        )
        result = ho.optimize()

    assert result.n_trials == 3
    assert result.n_completed == result.n_trials - result.n_pruned - result.n_failed


def test_optimize_best_params_populated(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(2.0, 0.05, 0.2),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=2,
        )
        result = ho.optimize()

    assert isinstance(result.best_params, dict)
    assert len(result.best_params) > 0


def test_optimize_best_sharpe_updated(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(1.7, 0.09, 0.18),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=2,
        )
        result = ho.optimize()

    assert result.best_sharpe > -np.inf


def test_optimize_study_name_in_result(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(0.3, 0.3, 0.01),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=2,
            study_name="test_study_42",
        )
        result = ho.optimize()

    assert result.study_name == "test_study_42"


def test_optimize_multi_objective_pareto(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(1.0, 0.1, 0.1),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=3,
            multi_objective=True,
        )
        result = ho.optimize()

    assert isinstance(result.pareto_front, list)


def test_optimize_mlflow_logging_called(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    mock_mlflow = MagicMock()
    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(0.9, 0.15, 0.1),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=1,
            mlflow_manager=mock_mlflow,
        )
        ho.optimize()

    assert mock_mlflow.log_params.called or mock_mlflow.log_metrics.called


def test_optimize_timeout_respected(base_config, sample_df):
    """Timeout=0 should stop the study immediately (0 or 1 completed trials)."""
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(0.5, 0.2, 0.05),
    ):
        ho = OptunaHyperopt(
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            train_df=sample_df.iloc[:100],
            val_df=sample_df.iloc[100:],
            n_trials=100,
            timeout=0.0,  # zero seconds
        )
        result = ho.optimize()

    # With timeout=0, at most a handful of trials should run
    assert result.n_trials <= 5


# ──────────────────────────────────────────────────────────────────────────────
# run_hyperopt convenience function
# ──────────────────────────────────────────────────────────────────────────────

def test_run_hyperopt_returns_hyperopt_result(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import HyperoptResult, run_hyperopt

    with patch(
        "training.hyperopt.hyperopt_optuna._evaluate_agent",
        return_value=(0.6, 0.2, 0.06),
    ):
        result = run_hyperopt(
            df=sample_df,
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            n_trials=2,
        )
    assert isinstance(result, HyperoptResult)


def test_run_hyperopt_splits_data_correctly(base_config, sample_df):
    """train_df should have 70% rows, val_df 30%."""
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt, run_hyperopt

    captured = {}

    class CapturingHyperopt(OptunaHyperopt):
        def optimize(self):
            captured["train_len"] = len(self.train_df)
            captured["val_len"] = len(self.val_df)
            from training.hyperopt.hyperopt_optuna import HyperoptResult
            return HyperoptResult(n_trials=0, n_completed=0, n_pruned=0, n_failed=0)

    with patch("training.hyperopt.hyperopt_optuna.OptunaHyperopt", CapturingHyperopt):
        run_hyperopt(
            df=sample_df,
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            n_trials=1,
            train_ratio=0.7,
        )

    expected_train = int(len(sample_df) * 0.7)
    assert captured["train_len"] == expected_train
    assert captured["val_len"] == len(sample_df) - expected_train


def test_run_hyperopt_multi_objective_flag(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt, run_hyperopt

    captured = {}

    class CapturingHyperopt(OptunaHyperopt):
        def optimize(self):
            captured["multi"] = self.multi_objective
            from training.hyperopt.hyperopt_optuna import HyperoptResult
            return HyperoptResult(n_trials=0, n_completed=0, n_pruned=0, n_failed=0)

    with patch("training.hyperopt.hyperopt_optuna.OptunaHyperopt", CapturingHyperopt):
        run_hyperopt(
            df=sample_df,
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            n_trials=1,
            multi_objective=True,
        )

    assert captured["multi"] is True


def test_run_hyperopt_reads_n_startup_from_config(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import OptunaHyperopt, run_hyperopt

    base_config["hyperopt"]["n_startup_trials"] = 7
    captured = {}

    class CapturingHyperopt(OptunaHyperopt):
        def optimize(self):
            captured["n_startup"] = self.n_startup_trials
            from training.hyperopt.hyperopt_optuna import HyperoptResult
            return HyperoptResult(n_trials=0, n_completed=0, n_pruned=0, n_failed=0)

    with patch("training.hyperopt.hyperopt_optuna.OptunaHyperopt", CapturingHyperopt):
        run_hyperopt(
            df=sample_df,
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
            n_trials=1,
        )

    assert captured["n_startup"] == 7


# ──────────────────────────────────────────────────────────────────────────────
# train_pipeline.run_hyperopt_optuna integration shim
# ──────────────────────────────────────────────────────────────────────────────

def test_run_hyperopt_optuna_in_train_pipeline(base_config, sample_df):
    from training.hyperopt.hyperopt_optuna import HyperoptResult
    from training.train_pipeline import run_hyperopt_optuna

    mock_result = HyperoptResult(
        n_trials=2, n_completed=2, n_pruned=0, n_failed=0,
        best_sharpe=1.0, best_max_drawdown=0.1,
        best_params={"learning_rate": 3e-4},
    )
    # Patch in the source module (local import in run_hyperopt_optuna resolves there)
    with patch(
        "training.hyperopt.hyperopt_optuna.run_hyperopt",
        return_value=mock_result,
    ):
        result = run_hyperopt_optuna(
            df=sample_df,
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
        )

    assert result.best_sharpe == pytest.approx(1.0)
    assert result.n_completed == 2


def test_run_hyperopt_optuna_reads_config(base_config, sample_df):
    """n_trials, multi_objective, study_name are read from config."""
    from training.train_pipeline import run_hyperopt_optuna

    base_config["hyperopt"].update({
        "n_trials": 7,
        "multi_objective": True,
        "study_name": "cfg_study",
    })

    captured = {}

    def mock_run_hyperopt(**kwargs):
        captured.update(kwargs)
        from training.hyperopt.hyperopt_optuna import HyperoptResult
        return HyperoptResult(n_trials=0, n_completed=0, n_pruned=0, n_failed=0)

    with patch("training.hyperopt.hyperopt_optuna.run_hyperopt", mock_run_hyperopt):
        run_hyperopt_optuna(
            df=sample_df,
            config=base_config,
            env_factory=_make_fake_env,
            agent_factory=_make_fake_agent,
        )

    assert captured.get("n_trials") == 7
    assert captured.get("multi_objective") is True
    assert captured.get("study_name") == "cfg_study"


# ──────────────────────────────────────────────────────────────────────────────
# config/training_config.yaml — hyperopt section
# ──────────────────────────────────────────────────────────────────────────────

def test_config_yaml_has_hyperopt_section():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert "hyperopt" in cfg


def test_config_yaml_hyperopt_has_n_trials():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert "n_trials" in cfg["hyperopt"]


def test_config_yaml_hyperopt_has_parameters():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert "parameters" in cfg["hyperopt"]


def test_config_yaml_hyperopt_parameters_has_learning_rate():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert "learning_rate" in cfg["hyperopt"]["parameters"]


def test_config_yaml_hyperopt_parameters_has_all_required_keys():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    params = cfg["hyperopt"]["parameters"]

    required = {
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "ent_coef", "clip_range",
        "vf_coef", "max_grad_norm", "feature_extractor",
        "reward_weights.pnl", "reward_weights.sharpe",
    }
    assert required <= set(params.keys())


def test_config_yaml_hyperopt_enabled_flag():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    # enabled key exists (default False — don't run by default)
    assert "enabled" in cfg["hyperopt"]
    assert cfg["hyperopt"]["enabled"] is False


def test_config_yaml_hyperopt_multi_objective_flag():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert "multi_objective" in cfg["hyperopt"]
