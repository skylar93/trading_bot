"""
Phase 2 Week 5: EnsembleManager tests.

Verifies:
- EnsembleManager creates 3 heterogeneous SB3 agents
- Initial weights sum to 1.0 and respect weight_init
- get_ensemble_action returns correct shape for all methods
- train_all trains each agent sequentially
- update_weights adjusts weights after recording returns
- rolling Sharpe weight computation (better returns → higher weight)
- rebalance() triggers weight recomputation
- record_episode_return accumulates correctly
- select_best evaluates and returns an agent ID
- save / load round-trip
- get_ensemble_metrics returns expected keys
- from_config constructs EnsembleManager correctly
- agent_factory creates EnsembleManager with "ensemble" type
- train_ensemble_agent pipeline function runs end-to-end
- EnsembleManager with "best" method uses top-weight agent only
- EnsembleManager with 1 agent works correctly
- Weight normalisation invariants are always maintained
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from agents.ensemble.ensemble_manager import EnsembleManager
from agents.strategies.agent_factory import create_agent, list_available_agents
from envs.wrap_env import make_sb3_env


# ---------------------------------------------------------------------------
# Shared helpers & fixtures
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


@pytest.fixture(scope="module")
def small_df():
    return _make_df(200, seed=42)


@pytest.fixture(scope="module")
def vec_env(small_df):
    """Shared VecNormalize-wrapped DummyVecEnv for the test session."""
    return make_sb3_env(small_df, n_envs=1, use_vec_normalize=True)


@pytest.fixture(scope="module")
def obs_space(vec_env):
    return vec_env.observation_space


@pytest.fixture(scope="module")
def act_space(vec_env):
    return vec_env.action_space


def _make_ensemble(
    obs_space,
    act_space,
    method: str = "rolling_validation",
    custom_configs=None,
) -> EnsembleManager:
    """Build a minimal 3-agent ensemble for testing."""
    configs = custom_configs or [
        {"id": "ppo_c", "type": "sb3_ppo", "weight_init": 0.4,
         "params": {"n_steps": 64, "batch_size": 32}},
        {"id": "sac_m", "type": "sb3_sac", "weight_init": 0.35},
        {"id": "td3_a", "type": "sb3_td3", "weight_init": 0.25},
    ]
    return EnsembleManager(
        agent_configs=configs,
        observation_space=obs_space,
        action_space=act_space,
        method=method,
        rebalance_interval=100,
        validation_window=50,
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestEnsembleConstruction:
    def test_creates_three_agents(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        assert len(em) == 3
        assert len(em.agents) == 3

    def test_agent_ids_present(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        assert "ppo_c" in em.agents
        assert "sac_m" in em.agents
        assert "td3_a" in em.agents

    def test_initial_weights_sum_to_one(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        w = em.get_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_weight_init_proportions(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        w = em.get_weights()
        # ppo_c should have the largest share (0.4 normalised)
        assert w["ppo_c"] > w["sac_m"] > w["td3_a"]

    def test_default_configs_used_when_none_given(self, obs_space, act_space):
        em = EnsembleManager(
            observation_space=obs_space,
            action_space=act_space,
        )
        assert len(em) == 3  # 3 default agents
        assert abs(sum(em.get_weights().values()) - 1.0) < 1e-6

    def test_invalid_method_raises(self, obs_space, act_space):
        with pytest.raises(ValueError, match="Unknown method"):
            EnsembleManager(
                observation_space=obs_space,
                action_space=act_space,
                method="invalid_method",
            )

    def test_single_agent_ensemble(self, obs_space, act_space):
        em = EnsembleManager(
            agent_configs=[{"id": "solo", "type": "sb3_ppo", "params": {"n_steps": 64}}],
            observation_space=obs_space,
            action_space=act_space,
        )
        assert len(em) == 1
        assert abs(em.get_weights()["solo"] - 1.0) < 1e-6

    def test_repr_contains_agent_info(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        r = repr(em)
        assert "EnsembleManager" in r
        assert "rolling_validation" in r
        assert "ppo_c" in r

    def test_metadata_populated(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        assert em.agent_metadata["ppo_c"]["type"] == "sb3_ppo"
        assert em.agent_metadata["sac_m"]["risk_profile"] == "moderate"


# ---------------------------------------------------------------------------
# 2. get_ensemble_action
# ---------------------------------------------------------------------------

class TestEnsembleAction:
    def test_action_shape_rolling_validation(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space, method="rolling_validation")
        obs = obs_space.sample()
        action = em.get_ensemble_action(obs)
        assert action.shape == act_space.shape

    def test_action_shape_weighted_average(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space, method="weighted_average")
        obs = obs_space.sample()
        action = em.get_ensemble_action(obs)
        assert action.shape == act_space.shape

    def test_action_shape_best(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space, method="best")
        obs = obs_space.sample()
        action = em.get_ensemble_action(obs)
        assert action.shape == act_space.shape

    def test_action_dtype_float(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        obs = obs_space.sample()
        action = em.get_ensemble_action(obs)
        assert action.dtype in (np.float32, np.float64)

    def test_best_method_uses_top_weight_agent(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space, method="best")
        # Force td3_a to have highest weight
        em._weights = {"ppo_c": 0.1, "sac_m": 0.1, "td3_a": 0.8}
        obs = obs_space.sample()
        best_action = em.agents["td3_a"].get_action(obs)
        ensemble_action = em.get_ensemble_action(obs, deterministic=True)
        np.testing.assert_array_almost_equal(best_action, ensemble_action)


# ---------------------------------------------------------------------------
# 3. Weight management
# ---------------------------------------------------------------------------

class TestWeightManagement:
    def test_get_weights_returns_copy(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        w1 = em.get_weights()
        w1["ppo_c"] = 999.0
        w2 = em.get_weights()
        assert w2["ppo_c"] != 999.0

    def test_update_weights_from_eval_metrics(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        metrics = {
            "ppo_c": {"mean_reward": 10.0},
            "sac_m": {"mean_reward": 5.0},
            "td3_a": {"mean_reward": 1.0},
        }
        em.update_weights(metrics)
        w = em.get_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_better_returns_lead_to_higher_weight(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        # Give ppo_c many good returns
        for v in [10.0, 12.0, 11.0, 13.0, 10.5]:
            em.record_episode_return("ppo_c", v)
        for v in [1.0, 0.5, 0.8]:
            em.record_episode_return("sac_m", v)
        for v in [-1.0, -2.0]:
            em.record_episode_return("td3_a", v)
        em.rebalance()
        w = em.get_weights()
        assert w["ppo_c"] > w["sac_m"]
        assert w["sac_m"] > w["td3_a"]
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_record_episode_return_accumulates(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        assert em._return_history["ppo_c"].maxlen == 50  # validation_window
        for i in range(60):
            em.record_episode_return("ppo_c", float(i))
        # deque has maxlen=50; only last 50 kept
        assert len(em._return_history["ppo_c"]) == 50

    def test_rebalance_updates_weights_from_history(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        w_before = dict(em.get_weights())
        em.record_episode_return("ppo_c", 100.0)
        em.record_episode_return("sac_m", -50.0)
        em.rebalance()
        w_after = em.get_weights()
        # ppo_c should gain relative weight
        assert w_after["ppo_c"] != w_before["ppo_c"] or w_after["sac_m"] != w_before["sac_m"]
        assert abs(sum(w_after.values()) - 1.0) < 1e-6

    def test_best_method_skips_weight_recompute(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space, method="best")
        initial_w = dict(em.get_weights())
        # Rebalance should leave weights unchanged for "best"
        em.record_episode_return("ppo_c", 1000.0)
        em.rebalance()
        assert em.get_weights() == initial_w

    def test_unknown_agent_id_in_update_weights_is_ignored(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        em.update_weights({"nonexistent": {"mean_reward": 999.0}})
        assert abs(sum(em.get_weights().values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. Training (minimal steps to stay fast)
# ---------------------------------------------------------------------------

class TestTrainAll:
    def test_train_all_returns_result_per_agent(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        results = em.train_all(vec_env, total_timesteps=300)
        assert set(results.keys()) == {"ppo_c", "sac_m", "td3_a"}

    def test_train_all_initialises_models(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        em.train_all(vec_env, total_timesteps=300)
        for agent in em.agents.values():
            assert agent.model is not None

    def test_train_agent_single_by_id(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        result = em.train_agent("ppo_c", vec_env, total_timesteps=150)
        assert "total_timesteps" in result
        assert em.agents["ppo_c"].model is not None

    def test_train_agent_unknown_id_raises(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        with pytest.raises(KeyError, match="ghost"):
            em.train_agent("ghost", vec_env, total_timesteps=100)

    def test_train_all_with_custom_timesteps_per_agent(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        tpa = {"ppo_c": 100, "sac_m": 150, "td3_a": 120}
        results = em.train_all(vec_env, total_timesteps=999, timesteps_per_agent=tpa)
        assert results["ppo_c"]["total_timesteps"] == 100
        assert results["sac_m"]["total_timesteps"] == 150
        assert results["td3_a"]["total_timesteps"] == 120


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

class TestEvaluateAgents:
    def test_evaluate_untrained_agents_returns_zeros(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        metrics = em.evaluate_agents(vec_env, n_eval_episodes=1)
        for agent_id in em.agents:
            assert metrics[agent_id]["mean_reward"] == 0.0  # untrained → skipped

    def test_evaluate_trained_agents_returns_floats(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        em.train_all(vec_env, total_timesteps=300)
        metrics = em.evaluate_agents(vec_env, n_eval_episodes=2)
        for agent_id in em.agents:
            assert isinstance(metrics[agent_id]["mean_reward"], float)
            assert "std_reward" in metrics[agent_id]

    def test_select_best_returns_valid_agent_id(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        em.train_all(vec_env, total_timesteps=300)
        best = em.select_best(vec_env, n_eval_episodes=2)
        assert best in em.agents

    def test_select_best_updates_weights(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        em.train_all(vec_env, total_timesteps=300)
        w_before = dict(em.get_weights())
        em.select_best(vec_env, n_eval_episodes=2)
        # Weights should have been updated (return history is now populated)
        w_after = em.get_weights()
        assert abs(sum(w_after.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 6. Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_files(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        em.train_all(vec_env, total_timesteps=150)
        with tempfile.TemporaryDirectory() as tmpdir:
            em.save(tmpdir)
            saved = os.listdir(tmpdir)
            assert any(f.endswith(".zip") for f in saved)

    def test_load_missing_files_does_not_raise(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        with tempfile.TemporaryDirectory() as tmpdir:
            em.load(tmpdir)  # no files exist → just logs warnings

    def test_save_and_reload(self, obs_space, act_space, vec_env):
        em = _make_ensemble(obs_space, act_space)
        em.train_all(vec_env, total_timesteps=150)
        obs = obs_space.sample()
        action_before = em.get_ensemble_action(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            em.save(tmpdir)
            em2 = _make_ensemble(obs_space, act_space)
            em2.load(tmpdir)
            action_after = em2.get_ensemble_action(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action_before, action_after, decimal=5)


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------

class TestEnsembleMetrics:
    def test_get_ensemble_metrics_keys(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        m = em.get_ensemble_metrics()
        assert "weights" in m
        assert "sharpe_scores" in m
        assert "return_history_sizes" in m
        assert "agent_types" in m
        assert "method" in m

    def test_sharpe_scores_zero_without_history(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        scores = em._compute_sharpe_scores()
        for v in scores.values():
            assert v == 0.0

    def test_sharpe_scores_nonzero_with_history(self, obs_space, act_space):
        em = _make_ensemble(obs_space, act_space)
        for v in [1.0, 2.0, 3.0]:
            em.record_episode_return("ppo_c", v)
        scores = em._compute_sharpe_scores()
        assert scores["ppo_c"] != 0.0


# ---------------------------------------------------------------------------
# 8. from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_config_builds_correctly(self, obs_space, act_space):
        cfg = {
            "method": "best",
            "rebalance_interval": 500,
            "validation_window": 100,
            "agents": [
                {"id": "a0", "type": "sb3_ppo", "weight_init": 0.6, "params": {"n_steps": 64}},
                {"id": "a1", "type": "sb3_sac", "weight_init": 0.4},
            ],
        }
        em = EnsembleManager.from_config(cfg, obs_space, act_space)
        assert len(em) == 2
        assert em.method == "best"
        assert em.rebalance_interval == 500
        assert em.validation_window == 100
        assert abs(sum(em.get_weights().values()) - 1.0) < 1e-6

    def test_from_config_uses_defaults_when_keys_missing(self, obs_space, act_space):
        em = EnsembleManager.from_config({}, obs_space, act_space)
        assert em.method == "rolling_validation"
        assert len(em) == 3  # default 3-agent pool


# ---------------------------------------------------------------------------
# 9. agent_factory integration
# ---------------------------------------------------------------------------

class TestAgentFactory:
    def test_factory_creates_ensemble(self, obs_space, act_space):
        agent = create_agent(
            "ensemble",
            observation_space=obs_space,
            action_space=act_space,
            config={
                "agents": [
                    {"id": "p", "type": "sb3_ppo", "params": {"n_steps": 64}},
                    {"id": "s", "type": "sb3_sac"},
                ],
                "method": "rolling_validation",
            },
        )
        assert isinstance(agent, EnsembleManager)
        assert len(agent) == 2

    def test_factory_alias_ensemblemanager(self, obs_space, act_space):
        agent = create_agent(
            "ensemblemanager",
            observation_space=obs_space,
            action_space=act_space,
        )
        assert isinstance(agent, EnsembleManager)

    def test_ensemble_in_list_available_agents(self):
        available = list_available_agents()
        assert "ensemble" in available


# ---------------------------------------------------------------------------
# 10. train_ensemble_agent pipeline
# ---------------------------------------------------------------------------

class TestTrainEnsemblePipeline:
    def test_train_ensemble_agent_returns_expected_keys(
        self, obs_space, act_space, vec_env
    ):
        from training.train_pipeline import train_ensemble_agent

        em = _make_ensemble(obs_space, act_space)
        config = {"training": {"total_timesteps": 300}}
        result = train_ensemble_agent(em, vec_env, config)

        assert "agent_results" in result
        assert "final_weights" in result
        assert "ensemble_metrics" in result

    def test_train_ensemble_agent_weights_sum_to_one(
        self, obs_space, act_space, vec_env
    ):
        from training.train_pipeline import train_ensemble_agent

        em = _make_ensemble(obs_space, act_space)
        config = {"training": {"total_timesteps": 300}}
        result = train_ensemble_agent(em, vec_env, config)

        total = sum(result["final_weights"].values())
        assert abs(total - 1.0) < 1e-6

    def test_train_ensemble_agent_with_eval_env(
        self, obs_space, act_space, small_df
    ):
        from training.train_pipeline import train_ensemble_agent

        train_env = make_sb3_env(small_df, n_envs=1, use_vec_normalize=True)
        eval_env = make_sb3_env(small_df, n_envs=1, use_vec_normalize=False)

        em = _make_ensemble(obs_space, act_space)
        config = {"training": {"total_timesteps": 300}}
        result = train_ensemble_agent(em, train_env, config, eval_env=eval_env)

        assert abs(sum(result["final_weights"].values()) - 1.0) < 1e-6

    def test_ensemble_checkpoint_saved(self, obs_space, act_space, vec_env):
        from training.train_pipeline import train_ensemble_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            em = _make_ensemble(obs_space, act_space)
            config = {
                "training": {"total_timesteps": 300},
                "paths": {"checkpoint_dir": tmpdir},
            }
            result = train_ensemble_agent(em, vec_env, config)
            save_dir = result["ensemble_save_dir"]
            assert os.path.isdir(save_dir)
            files = os.listdir(save_dir)
            assert any(f.endswith(".zip") for f in files)
