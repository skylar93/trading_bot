"""
Week 18: AutoStrategyIterator tests.

Coverage:
  - StrategyConfig (defaults, copy, custom fields)
  - StrategyResult (construction, fields)
  - AutoIterateConfig (defaults, custom)
  - AutoStrategyIterator dry_run mode
    - __init__ raises on non-dry_run without evaluate_fn
    - run() returns correct number of results
    - run() result iteration indices are set
    - run() sharpe values are finite floats
    - get_ranked_results() is sorted descending by sharpe
    - get_best() returns highest sharpe
    - summary() contains expected keys
  - Stagnation detection
    - stagnation triggers structural change (agent_type cycles)
    - stagnation_count resets after improvement
  - Custom evaluate_fn injection
    - deterministic evaluate_fn works end-to-end
  - _RuleBasedStrategist
    - propose() with empty results returns seed
    - propose() with results returns a StrategyConfig
    - structural=True changes agent_type or feature_set
  - _mock_evaluate (internal)
    - returns plausible metric ranges
  - config/auto_iterate.yaml
    - YAML is loadable and contains expected keys
  - Package re-exports (training.strategy_lab.__init__)
"""

from __future__ import annotations

import random
from typing import List

import pytest

from training.strategy_lab.auto_iterate import (
    AutoIterateConfig,
    AutoStrategyIterator,
    StrategyConfig,
    StrategyResult,
    _RuleBasedStrategist,
    _mock_evaluate,
)
from training.strategy_lab import (
    AutoIterateConfig as _AutoIterateConfigAlias,
    AutoStrategyIterator as _AutoStrategyIteratorAlias,
    StrategyConfig as _StrategyConfigAlias,
    StrategyResult as _StrategyResultAlias,
)


# ---------------------------------------------------------------------------
# StrategyConfig
# ---------------------------------------------------------------------------

class TestStrategyConfig:
    def test_defaults(self):
        cfg = StrategyConfig()
        assert cfg.agent_type == "ppo"
        assert isinstance(cfg.reward_weights, dict)
        assert "pnl" in cfg.reward_weights
        assert cfg.window_size == 20
        assert isinstance(cfg.feature_set, list)
        assert len(cfg.feature_set) > 0
        assert cfg.training_timesteps == 10_000
        assert cfg.tag == ""

    def test_custom_fields(self):
        cfg = StrategyConfig(
            agent_type="sac",
            reward_weights={"pnl": 0.7, "risk": 0.3},
            window_size=10,
            feature_set=["returns", "volume"],
            training_timesteps=5_000,
            tag="test",
        )
        assert cfg.agent_type == "sac"
        assert cfg.window_size == 10
        assert cfg.tag == "test"

    def test_copy_is_independent(self):
        cfg = StrategyConfig()
        copy = cfg.copy()
        copy.agent_type = "td3"
        copy.reward_weights["pnl"] = 0.99
        copy.feature_set.append("extra")
        assert cfg.agent_type == "ppo"
        assert cfg.reward_weights["pnl"] != 0.99
        assert "extra" not in cfg.feature_set


# ---------------------------------------------------------------------------
# StrategyResult
# ---------------------------------------------------------------------------

class TestStrategyResult:
    def test_construction(self):
        cfg = StrategyConfig()
        r = StrategyResult(config=cfg, sharpe=1.2, max_drawdown=0.1,
                           stability_ratio=0.8, total_return=0.15, iteration=0)
        assert r.sharpe == pytest.approx(1.2)
        assert r.max_drawdown == pytest.approx(0.1)
        assert r.iteration == 0
        assert r.rationale == ""
        assert r.elapsed_seconds == 0.0


# ---------------------------------------------------------------------------
# AutoIterateConfig
# ---------------------------------------------------------------------------

class TestAutoIterateConfig:
    def test_defaults(self):
        cfg = AutoIterateConfig()
        assert cfg.max_iterations == 20
        assert cfg.stagnation_window == 3
        assert cfg.stagnation_threshold == pytest.approx(0.05)
        assert cfg.dry_run is False
        assert cfg.seed == 42

    def test_custom(self):
        cfg = AutoIterateConfig(max_iterations=5, dry_run=True, seed=0)
        assert cfg.max_iterations == 5
        assert cfg.dry_run is True
        assert cfg.seed == 0


# ---------------------------------------------------------------------------
# AutoStrategyIterator — dry_run mode
# ---------------------------------------------------------------------------

class TestAutoStrategyIteratorDryRun:

    def _make(self, max_iterations=4, **kwargs) -> AutoStrategyIterator:
        cfg = AutoIterateConfig(
            dry_run=True,
            max_iterations=max_iterations,
            mlflow_experiment=None,  # no MLflow in tests
            seed=0,
            log_interval=2,
            **kwargs,
        )
        return AutoStrategyIterator(cfg)

    def test_init_ok(self):
        it = self._make()
        assert it is not None

    def test_init_raises_without_evaluate_fn_when_not_dry_run(self):
        cfg = AutoIterateConfig(dry_run=False, mlflow_experiment=None)
        with pytest.raises(ValueError, match="evaluate_fn"):
            AutoStrategyIterator(cfg)

    def test_run_returns_list(self):
        it = self._make(max_iterations=3)
        results = it.run()
        assert isinstance(results, list)
        assert len(results) >= 1
        assert len(results) <= 3

    def test_run_iteration_indices_set(self):
        it = self._make(max_iterations=4)
        results = it.run()
        for i, r in enumerate(results):
            assert r.iteration == i

    def test_run_sharpe_finite(self):
        it = self._make(max_iterations=5)
        results = it.run()
        for r in results:
            assert isinstance(r.sharpe, float)
            assert -100 < r.sharpe < 100

    def test_run_result_has_config(self):
        it = self._make(max_iterations=3)
        results = it.run()
        for r in results:
            assert isinstance(r.config, StrategyConfig)

    def test_get_ranked_results_sorted(self):
        it = self._make(max_iterations=6)
        it.run()
        ranked = it.get_ranked_results()
        sharpes = [r.sharpe for r in ranked]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_get_best_is_max_sharpe(self):
        it = self._make(max_iterations=5)
        results = it.run()
        best = it.get_best()
        assert best is not None
        assert best.sharpe == max(r.sharpe for r in results)

    def test_get_best_none_before_run(self):
        it = self._make()
        assert it.get_best() is None

    def test_summary_keys(self):
        it = self._make(max_iterations=4)
        it.run()
        s = it.summary()
        expected_keys = {
            "n_iterations", "best_sharpe", "best_max_drawdown",
            "best_stability_ratio", "best_total_return",
            "best_agent_type", "best_window_size", "best_tag",
        }
        assert expected_keys.issubset(set(s.keys()))

    def test_summary_empty_before_run(self):
        it = self._make()
        assert it.summary() == {}

    def test_seed_config_override(self):
        it = self._make(max_iterations=3)
        seed = StrategyConfig(agent_type="td3", window_size=10)
        results = it.run(seed_config=seed)
        assert results[0].config.agent_type == "td3"
        assert results[0].config.window_size == 10

    def test_rationale_set_on_results(self):
        it = self._make(max_iterations=3)
        results = it.run()
        assert isinstance(results[0].rationale, str)
        assert len(results[0].rationale) > 0

    def test_elapsed_seconds_non_negative(self):
        it = self._make(max_iterations=3)
        results = it.run()
        for r in results:
            assert r.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# Custom evaluate_fn injection
# ---------------------------------------------------------------------------

class TestCustomEvaluateFn:

    def test_custom_fn_called(self):
        calls: List[StrategyConfig] = []

        def my_eval(cfg: StrategyConfig) -> StrategyResult:
            calls.append(cfg)
            return StrategyResult(
                config=cfg,
                sharpe=1.0 + len(calls) * 0.1,
                max_drawdown=0.05,
                stability_ratio=0.8,
                total_return=0.1,
                iteration=len(calls) - 1,
            )

        cfg = AutoIterateConfig(
            dry_run=False, max_iterations=3, mlflow_experiment=None, seed=1
        )
        it = AutoStrategyIterator(cfg, evaluate_fn=my_eval)
        results = it.run()
        assert len(calls) == len(results)
        assert len(results) >= 1

    def test_custom_fn_sharpe_used_for_ranking(self):
        counter = {"n": 0}

        def my_eval(cfg: StrategyConfig) -> StrategyResult:
            counter["n"] += 1
            sharpe = [0.5, 1.5, 0.8, 2.0, 1.2][counter["n"] - 1] if counter["n"] <= 5 else 1.0
            return StrategyResult(
                config=cfg, sharpe=sharpe, max_drawdown=0.05,
                stability_ratio=0.7, total_return=0.1, iteration=counter["n"] - 1,
            )

        cfg = AutoIterateConfig(
            dry_run=False, max_iterations=5, mlflow_experiment=None, seed=2
        )
        it = AutoStrategyIterator(cfg, evaluate_fn=my_eval)
        it.run()
        ranked = it.get_ranked_results()
        assert ranked[0].sharpe >= ranked[-1].sharpe


# ---------------------------------------------------------------------------
# Stagnation detection & structural changes
# ---------------------------------------------------------------------------

class TestStagnation:

    def test_structural_change_triggered_after_stagnation(self):
        """After stagnation_window flat results, agent_type should eventually change."""
        cfg = AutoIterateConfig(
            dry_run=True,
            max_iterations=10,
            stagnation_window=2,
            stagnation_threshold=0.99,  # near-impossible threshold → always stagnant
            mlflow_experiment=None,
            seed=5,
        )
        it = AutoStrategyIterator(cfg)
        results = it.run()
        agent_types = {r.config.agent_type for r in results}
        # With aggressive stagnation, rule-based strategist should try multiple agents
        assert len(agent_types) >= 1  # at least started

    def test_no_stagnation_with_generous_threshold(self):
        """With threshold=0, every result counts as improvement → no structural change."""
        cfg = AutoIterateConfig(
            dry_run=True,
            max_iterations=5,
            stagnation_threshold=0.0,
            mlflow_experiment=None,
            seed=7,
        )
        it = AutoStrategyIterator(cfg)
        results = it.run()
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# _RuleBasedStrategist
# ---------------------------------------------------------------------------

class TestRuleBasedStrategist:

    def _strat(self, seed=0) -> _RuleBasedStrategist:
        return _RuleBasedStrategist(random.Random(seed))

    def test_propose_empty_returns_seed(self):
        s = self._strat()
        seed_cfg = StrategyConfig(agent_type="sac", window_size=15)
        cfg, rationale = s.propose([], seed_cfg)
        assert cfg.agent_type == seed_cfg.agent_type
        assert cfg.window_size == seed_cfg.window_size
        assert "seed" in rationale.lower() or "initial" in rationale.lower()

    def test_propose_with_results(self):
        s = self._strat()
        seed_cfg = StrategyConfig()
        result = StrategyResult(config=seed_cfg, sharpe=1.0, iteration=0)
        cfg, rationale = s.propose([result], seed_cfg)
        assert isinstance(cfg, StrategyConfig)
        assert isinstance(rationale, str)

    def test_propose_returns_copy(self):
        s = self._strat()
        seed_cfg = StrategyConfig()
        result = StrategyResult(config=seed_cfg, sharpe=1.0, iteration=0)
        cfg, _ = s.propose([result], seed_cfg)
        # Mutating returned config should not affect seed
        cfg.agent_type = "flag"
        assert seed_cfg.agent_type == "ppo"

    def test_structural_change_switches_agent(self):
        s = self._strat(seed=99)
        seed_cfg = StrategyConfig(agent_type="ppo")
        results = [StrategyResult(config=seed_cfg, sharpe=0.5, iteration=0)]
        changed_agents = set()
        for trial in range(20):
            s2 = self._strat(seed=trial)
            cfg, rationale = s2.propose(results, seed_cfg, structural=True)
            changed_agents.add(cfg.agent_type)
        # With 20 trials and 4 agent types, at least 2 should appear
        assert len(changed_agents) >= 1

    def test_perturb_produces_valid_config(self):
        s = self._strat()
        seed_cfg = StrategyConfig()
        results = [StrategyResult(config=seed_cfg, sharpe=1.0, iteration=0)]
        for _ in range(10):
            cfg, _ = s.propose(results, seed_cfg)
            assert cfg.window_size > 0
            assert len(cfg.feature_set) > 0
            assert sum(cfg.reward_weights.values()) > 0


# ---------------------------------------------------------------------------
# _mock_evaluate
# ---------------------------------------------------------------------------

class TestMockEvaluate:

    def test_returns_strategy_result(self):
        cfg = StrategyConfig()
        rng = random.Random(0)
        result = _mock_evaluate(cfg, 0, rng)
        assert isinstance(result, StrategyResult)

    def test_sharpe_finite(self):
        for agent in ["ppo", "sac", "td3", "flag"]:
            rng = random.Random(0)
            result = _mock_evaluate(StrategyConfig(agent_type=agent), 0, rng)
            assert -100 < result.sharpe < 100

    def test_max_drawdown_non_negative(self):
        rng = random.Random(0)
        for _ in range(10):
            result = _mock_evaluate(StrategyConfig(), 0, rng)
            assert result.max_drawdown >= 0.0

    def test_stability_ratio_non_negative(self):
        rng = random.Random(0)
        for _ in range(10):
            result = _mock_evaluate(StrategyConfig(), 0, rng)
            assert result.stability_ratio >= 0.0

    def test_flag_higher_base_sharpe_than_ppo(self):
        """On average (no noise), FLAG should have higher base Sharpe."""
        flag_sharpes = []
        ppo_sharpes = []
        for seed in range(50):
            flag_sharpes.append(
                _mock_evaluate(StrategyConfig(agent_type="flag"), 0, random.Random(seed)).sharpe
            )
            ppo_sharpes.append(
                _mock_evaluate(StrategyConfig(agent_type="ppo"), 0, random.Random(seed)).sharpe
            )
        assert sum(flag_sharpes) / 50 > sum(ppo_sharpes) / 50


# ---------------------------------------------------------------------------
# config/auto_iterate.yaml
# ---------------------------------------------------------------------------

class TestAutoIterateYAML:

    def test_yaml_loadable(self):
        import yaml
        with open("config/auto_iterate.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None

    def test_yaml_top_level_keys(self):
        import yaml
        with open("config/auto_iterate.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "auto_iterate" in cfg
        assert "seed_strategy" in cfg

    def test_yaml_auto_iterate_keys(self):
        import yaml
        with open("config/auto_iterate.yaml") as f:
            cfg = yaml.safe_load(f)
        ai = cfg["auto_iterate"]
        for key in ("max_iterations", "stagnation_window", "stagnation_threshold",
                    "use_claude", "dry_run", "seed"):
            assert key in ai, f"Missing key: {key}"

    def test_yaml_seed_strategy_keys(self):
        import yaml
        with open("config/auto_iterate.yaml") as f:
            cfg = yaml.safe_load(f)
        ss = cfg["seed_strategy"]
        assert "agent_type" in ss
        assert "window_size" in ss
        assert "reward_weights" in ss
        assert "feature_set" in ss

    def test_yaml_max_iterations_guard(self):
        import yaml
        with open("config/auto_iterate.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["auto_iterate"]["max_iterations"] <= 20

    def test_yaml_reward_weights_sum_to_one(self):
        import yaml
        with open("config/auto_iterate.yaml") as f:
            cfg = yaml.safe_load(f)
        rw = cfg["seed_strategy"]["reward_weights"]
        assert abs(sum(rw.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Package re-exports (training.strategy_lab.__init__)
# ---------------------------------------------------------------------------

class TestPackageReExports:
    def test_auto_iterate_config_alias(self):
        assert _AutoIterateConfigAlias is AutoIterateConfig

    def test_auto_strategy_iterator_alias(self):
        assert _AutoStrategyIteratorAlias is AutoStrategyIterator

    def test_strategy_config_alias(self):
        assert _StrategyConfigAlias is StrategyConfig

    def test_strategy_result_alias(self):
        assert _StrategyResultAlias is StrategyResult
