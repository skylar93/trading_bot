"""
Week 18: Automated Strategy Iteration & Final Integration.

AutoStrategyIterator — Claude-powered or rule-based strategy discovery loop.

Loop::

    1. Generate strategy hypothesis (StrategyConfig + rationale string)
    2. Evaluate via injectable evaluate_fn (or built-in mock for dry_run)
    3. Rank by Sharpe → max_drawdown → stability_ratio
    4. Propose modifications (rule-based perturb or Anthropic Claude API)
    5. Repeat until convergence or max_iterations

Stagnation detection:
    - If last 3 iterations show <5% Sharpe improvement → structural change
    - Structural changes: agent_type, window_size, reward_weights, feature_set

Guard rails:
    - max_iterations default 20
    - max_training_minutes per iteration (enforced by evaluate_fn contract)
    - All results logged to MLflow (optional, gracefully guarded)

Optional Claude API integration:
    - Set use_claude=True and ANTHROPIC_API_KEY env var
    - Falls back to rule-based if unavailable / dry_run=True

Usage::

    # Dry-run (no external calls, deterministic for testing)
    cfg = AutoIterateConfig(dry_run=True, max_iterations=3)
    it = AutoStrategyIterator(cfg)
    results = it.run()
    ranked = it.get_ranked_results()
    best = ranked[0]

    # Production (real evaluate_fn)
    def my_eval(strategy_cfg: StrategyConfig) -> StrategyResult:
        ...
    it = AutoStrategyIterator(cfg, evaluate_fn=my_eval)
    results = it.run()
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic_lib  # noqa: F401
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import mlflow as _mlflow_lib
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Parameterisation of one strategy hypothesis.

    Attributes:
        agent_type: One of "ppo", "sac", "td3", "flag".
        reward_weights: Mapping of reward component name → weight (must sum ~1.0).
        window_size: Look-back window (bars) fed into the observation.
        feature_set: List of feature names included in the observation.
        training_timesteps: Env steps for this iteration's training run.
        tag: Human-readable label (auto-filled by the iterator).
    """

    agent_type: str = "ppo"
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {"pnl": 0.6, "risk": 0.3, "holding": 0.1}
    )
    window_size: int = 20
    feature_set: List[str] = field(
        default_factory=lambda: ["returns", "volume", "rsi", "macd"]
    )
    training_timesteps: int = 10_000
    tag: str = ""

    def copy(self) -> "StrategyConfig":
        return copy.deepcopy(self)


@dataclass
class StrategyResult:
    """Backtest / evaluation outcome for one strategy.

    Attributes:
        config: The StrategyConfig that produced this result.
        sharpe: Annualised Sharpe ratio.
        max_drawdown: Maximum drawdown as a decimal (0.10 = 10 %).
        stability_ratio: Sharpe consistency across walk-forward folds
                         (mean_sharpe / (1 + std_sharpe)).  ∈ [0, ∞).
        total_return: Total portfolio return as a decimal.
        iteration: Loop iteration index (0-based).
        rationale: Why this config was proposed.
        elapsed_seconds: Wall-clock time taken by evaluate_fn.
    """

    config: StrategyConfig
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    stability_ratio: float = 0.0
    total_return: float = 0.0
    iteration: int = 0
    rationale: str = ""
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Internal strategist helpers
# ---------------------------------------------------------------------------

_AGENT_TYPES = ["ppo", "sac", "td3", "flag"]
_WINDOW_SIZES = [5, 10, 15, 20, 30, 50]
_FEATURE_SETS: List[List[str]] = [
    ["returns", "volume"],
    ["returns", "volume", "rsi"],
    ["returns", "volume", "rsi", "macd"],
    ["returns", "volume", "rsi", "macd", "bollinger"],
    ["returns", "volume", "rsi", "macd", "bollinger", "atr", "sentiment"],
]
_REWARD_WEIGHT_PRESETS: List[Dict[str, float]] = [
    {"pnl": 0.8, "risk": 0.1, "holding": 0.1},
    {"pnl": 0.6, "risk": 0.3, "holding": 0.1},
    {"pnl": 0.5, "risk": 0.4, "holding": 0.1},
    {"pnl": 0.4, "risk": 0.5, "holding": 0.1},
    {"pnl": 0.6, "risk": 0.2, "holding": 0.2},
]


class _RuleBasedStrategist:
    """Perturb-and-explore strategy proposer.

    On the first call returns the seed config unchanged.
    Subsequently, mutates the best-seen config using small perturbations.
    On a structural-change request, switches agent_type, window_size, or
    feature_set.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng

    def propose(
        self,
        results: List[StrategyResult],
        seed: StrategyConfig,
        structural: bool = False,
    ) -> Tuple[StrategyConfig, str]:
        if not results:
            return seed.copy(), "initial seed config"

        best = max(results, key=lambda r: r.sharpe)
        cfg = best.config.copy()

        if structural:
            return self._structural_change(cfg, results)
        return self._perturb(cfg)

    def _perturb(self, cfg: StrategyConfig) -> Tuple[StrategyConfig, str]:
        """Small continuous / discrete perturbation."""
        choice = self._rng.randint(0, 2)
        if choice == 0:
            # Perturb reward weights
            preset = self._rng.choice(_REWARD_WEIGHT_PRESETS)
            cfg.reward_weights = dict(preset)
            rationale = f"reward_weights → {preset}"
        elif choice == 1:
            # Perturb window_size ±1 step
            idx = _WINDOW_SIZES.index(cfg.window_size) if cfg.window_size in _WINDOW_SIZES else 3
            idx = max(0, min(len(_WINDOW_SIZES) - 1, idx + self._rng.choice([-1, 1])))
            cfg.window_size = _WINDOW_SIZES[idx]
            rationale = f"window_size → {cfg.window_size}"
        else:
            # Perturb feature_set ±1 feature group
            idx = 0
            for i, fs in enumerate(_FEATURE_SETS):
                if set(fs) == set(cfg.feature_set):
                    idx = i
                    break
            idx = max(0, min(len(_FEATURE_SETS) - 1, idx + self._rng.choice([-1, 1])))
            cfg.feature_set = list(_FEATURE_SETS[idx])
            rationale = f"feature_set → {cfg.feature_set}"
        return cfg, rationale

    def _structural_change(
        self, cfg: StrategyConfig, results: List[StrategyResult]
    ) -> Tuple[StrategyConfig, str]:
        """Major jump: switch agent_type or combine several changes at once."""
        tried_agents = {r.config.agent_type for r in results}
        untried = [a for a in _AGENT_TYPES if a not in tried_agents]

        if untried:
            cfg.agent_type = self._rng.choice(untried)
            rationale = f"structural: switch agent_type → {cfg.agent_type} (untried)"
        else:
            # All agents tried — flip to opposite window + largest feature set
            cfg.agent_type = self._rng.choice(_AGENT_TYPES)
            cfg.window_size = self._rng.choice(_WINDOW_SIZES)
            cfg.feature_set = list(self._rng.choice(_FEATURE_SETS))
            cfg.reward_weights = dict(self._rng.choice(_REWARD_WEIGHT_PRESETS))
            rationale = (
                f"structural: full reset — agent={cfg.agent_type}, "
                f"window={cfg.window_size}, features={len(cfg.feature_set)}"
            )
        return cfg, rationale


class _ClaudeStrategist:
    """Optional Anthropic Claude API-backed strategy proposer.

    Sends a compact JSON summary of the last N results to Claude and asks for
    a concrete StrategyConfig modification.  Falls back to _RuleBasedStrategist
    on any error.
    """

    _SYSTEM = (
        "You are a quant trading strategy optimiser. "
        "Given a list of recent backtest results (JSON), propose ONE modification "
        "to improve the Sharpe ratio. Reply in JSON with keys: "
        "agent_type, reward_weights, window_size, feature_set, rationale. "
        "Keep reward_weights summing to 1.0. "
        "agent_type must be one of: ppo, sac, td3, flag. "
        "window_size must be one of: 5, 10, 15, 20, 30, 50."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._fallback = _RuleBasedStrategist(random.Random())

    def propose(
        self,
        results: List[StrategyResult],
        seed: StrategyConfig,
        structural: bool = False,
    ) -> Tuple[StrategyConfig, str]:
        import json

        if not results:
            return seed.copy(), "initial seed (Claude)"

        summary = [
            {
                "iteration": r.iteration,
                "agent_type": r.config.agent_type,
                "reward_weights": r.config.reward_weights,
                "window_size": r.config.window_size,
                "feature_set": r.config.feature_set,
                "sharpe": round(r.sharpe, 4),
                "max_drawdown": round(r.max_drawdown, 4),
                "stability_ratio": round(r.stability_ratio, 4),
            }
            for r in results[-5:]  # last 5 results only (token limit)
        ]
        prompt = json.dumps({"results": summary, "structural_change_requested": structural})

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from text (Claude may wrap in ```json)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            proposal = json.loads(text)

            best = max(results, key=lambda r: r.sharpe)
            cfg = best.config.copy()
            cfg.agent_type = proposal.get("agent_type", cfg.agent_type)
            rw = proposal.get("reward_weights", cfg.reward_weights)
            # Normalise weights
            total = sum(rw.values()) or 1.0
            cfg.reward_weights = {k: v / total for k, v in rw.items()}
            ws = proposal.get("window_size", cfg.window_size)
            cfg.window_size = ws if ws in _WINDOW_SIZES else cfg.window_size
            fs = proposal.get("feature_set", cfg.feature_set)
            cfg.feature_set = fs if isinstance(fs, list) and fs else cfg.feature_set
            rationale = f"Claude: {proposal.get('rationale', 'no rationale')}"
            return cfg, rationale

        except Exception as exc:  # noqa: BLE001
            logger.warning("Claude API call failed (%s); falling back to rule-based.", exc)
            return self._fallback.propose(results, seed, structural)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AutoIterateConfig:
    """Configuration for the AutoStrategyIterator.

    Attributes:
        max_iterations: Hard cap on loop iterations (default 20).
        max_training_minutes: Per-iteration training budget.  Passed to
                              evaluate_fn via StrategyConfig.training_timesteps
                              (iterator does not enforce wall-clock time itself).
        stagnation_window: Number of consecutive iterations with <stagnation_threshold
                           Sharpe improvement before a structural change is triggered.
        stagnation_threshold: Minimum fractional Sharpe improvement to reset stagnation
                              counter (0.05 = 5 %).
        use_claude: If True and ANTHROPIC_API_KEY is set, use Claude API strategist.
        claude_model: Anthropic model ID for the strategist.
        mlflow_experiment: MLflow experiment name (None = no MLflow logging).
        dry_run: If True, skip external calls; use deterministic mock evaluate_fn.
        seed: Random seed for reproducibility.
        log_interval: Log a summary every N iterations.
    """

    max_iterations: int = 20
    max_training_minutes: int = 30
    stagnation_window: int = 3
    stagnation_threshold: float = 0.05
    use_claude: bool = False
    claude_model: str = "claude-haiku-4-5-20251001"
    mlflow_experiment: Optional[str] = "auto_strategy_iteration"
    dry_run: bool = False
    seed: int = 42
    log_interval: int = 5


# ---------------------------------------------------------------------------
# Mock evaluate function (dry_run / testing)
# ---------------------------------------------------------------------------

def _mock_evaluate(cfg: StrategyConfig, iteration: int, rng: random.Random) -> StrategyResult:
    """Deterministic-ish mock that returns plausible metrics.

    Sharpe trends upward with training_timesteps but adds noise.
    """
    base_sharpe = {
        "ppo": 0.8, "sac": 1.0, "td3": 0.9, "flag": 1.1
    }.get(cfg.agent_type, 0.8)

    feature_bonus = len(cfg.feature_set) * 0.03
    window_bonus = -abs(cfg.window_size - 20) * 0.01
    reward_bonus = cfg.reward_weights.get("pnl", 0.6) * 0.2

    sharpe = base_sharpe + feature_bonus + window_bonus + reward_bonus
    sharpe += rng.gauss(0, 0.1)
    sharpe = max(sharpe, -2.0)

    max_dd = max(0.0, 0.25 - cfg.reward_weights.get("risk", 0.3) * 0.3 + rng.gauss(0, 0.02))
    stability = max(0.0, sharpe / (1 + abs(rng.gauss(0, 0.15))))
    total_return = sharpe * 0.1 + rng.gauss(0, 0.05)

    return StrategyResult(
        config=cfg,
        sharpe=sharpe,
        max_drawdown=max_dd,
        stability_ratio=stability,
        total_return=total_return,
        iteration=iteration,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AutoStrategyIterator:
    """Automated strategy discovery loop.

    Parameters
    ----------
    config:
        AutoIterateConfig with loop hyper-parameters.
    evaluate_fn:
        Callable ``(StrategyConfig) → StrategyResult``.  If None and
        ``config.dry_run=True`` a deterministic mock is used.  In production,
        supply a function that trains an SB3 agent and runs a backtest.
    seed_config:
        Initial StrategyConfig to start exploration from.  Falls back to
        default if None.
    """

    def __init__(
        self,
        config: AutoIterateConfig,
        evaluate_fn: Optional[Callable[[StrategyConfig], StrategyResult]] = None,
        seed_config: Optional[StrategyConfig] = None,
    ):
        self.config = config
        self._rng = random.Random(config.seed)
        self._results: List[StrategyResult] = []
        self._seed = seed_config or StrategyConfig()

        # Resolve evaluate function
        if evaluate_fn is not None:
            self._evaluate_fn = evaluate_fn
        elif config.dry_run:
            _rng_ref = self._rng
            self._evaluate_fn = lambda cfg, it=0: _mock_evaluate(cfg, it, _rng_ref)
        else:
            raise ValueError(
                "evaluate_fn must be provided when dry_run=False. "
                "Supply a callable (StrategyConfig) → StrategyResult."
            )

        # Resolve strategist
        if config.use_claude and _ANTHROPIC_AVAILABLE and not config.dry_run:
            self._strategist: Any = _ClaudeStrategist(model=config.claude_model)
        else:
            self._strategist = _RuleBasedStrategist(self._rng)

        # MLflow run handle (lazy init)
        self._mlflow_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, seed_config: Optional[StrategyConfig] = None) -> List[StrategyResult]:
        """Execute the discovery loop.

        Returns
        -------
        List[StrategyResult]
            All evaluated results in iteration order (not ranked).
        """
        if seed_config is not None:
            self._seed = seed_config

        self._results = []
        stagnation_count = 0
        current_cfg = self._seed.copy()
        current_cfg.tag = "seed"

        self._mlflow_start()

        for iteration in range(self.config.max_iterations):
            # --- Generate next config ---
            if iteration == 0:
                cfg = current_cfg
                rationale = "initial seed config"
            else:
                structural = stagnation_count >= self.config.stagnation_window
                cfg, rationale = self._strategist.propose(
                    self._results, self._seed, structural=structural
                )
                if structural:
                    logger.info(
                        "Iteration %d: stagnation detected → structural change: %s",
                        iteration, rationale,
                    )
                    stagnation_count = 0
                cfg.tag = f"iter_{iteration}"

            # --- Evaluate ---
            t0 = time.monotonic()
            if self.config.dry_run:
                result = _mock_evaluate(cfg, iteration, self._rng)
                result.rationale = rationale
            else:
                result = self._evaluate_fn(cfg)
                result.rationale = rationale
                result.iteration = iteration
            result.elapsed_seconds = time.monotonic() - t0

            self._results.append(result)
            self._mlflow_log(result)

            # --- Stagnation check ---
            if len(self._results) >= 2:
                prev_best = max(self._results[:-1], key=lambda r: r.sharpe).sharpe
                improvement = (result.sharpe - prev_best) / (abs(prev_best) + 1e-8)
                if improvement < self.config.stagnation_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

            # --- Logging ---
            if (iteration + 1) % self.config.log_interval == 0 or iteration == 0:
                best_so_far = max(self._results, key=lambda r: r.sharpe)
                logger.info(
                    "Iter %d/%d | sharpe=%.3f | max_dd=%.3f | stability=%.3f | best_so_far=%.3f | %s",
                    iteration + 1,
                    self.config.max_iterations,
                    result.sharpe,
                    result.max_drawdown,
                    result.stability_ratio,
                    best_so_far.sharpe,
                    rationale[:60],
                )

            # --- Convergence check ---
            if self._has_converged():
                logger.info("Convergence detected at iteration %d. Stopping early.", iteration + 1)
                break

        self._mlflow_end()
        return list(self._results)

    def get_ranked_results(self) -> List[StrategyResult]:
        """Return all results sorted by (sharpe DESC, -max_drawdown DESC)."""
        return sorted(
            self._results,
            key=lambda r: (r.sharpe, -r.max_drawdown),
            reverse=True,
        )

    def get_best(self) -> Optional[StrategyResult]:
        """Return the single best result by Sharpe."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.sharpe)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for MLflow / reporting."""
        if not self._results:
            return {}
        best = self.get_best()
        assert best is not None
        return {
            "n_iterations": len(self._results),
            "best_sharpe": best.sharpe,
            "best_max_drawdown": best.max_drawdown,
            "best_stability_ratio": best.stability_ratio,
            "best_total_return": best.total_return,
            "best_agent_type": best.config.agent_type,
            "best_window_size": best.config.window_size,
            "best_tag": best.config.tag,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_converged(self) -> bool:
        """Return True if the last stagnation_window results are all within threshold."""
        w = self.config.stagnation_window
        if len(self._results) < w + 1:
            return False
        recent = self._results[-w:]
        sharpes = [r.sharpe for r in recent]
        spread = max(sharpes) - min(sharpes)
        return spread < self.config.stagnation_threshold * abs(sum(sharpes) / len(sharpes) + 1e-8)

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _mlflow_start(self) -> None:
        if not _MLFLOW_AVAILABLE or self.config.mlflow_experiment is None:
            return
        try:
            import mlflow
            mlflow.set_experiment(self.config.mlflow_experiment)
            run = mlflow.start_run(run_name="auto_iterate")
            self._mlflow_run_id = run.info.run_id
            mlflow.log_params({
                "max_iterations": self.config.max_iterations,
                "stagnation_window": self.config.stagnation_window,
                "stagnation_threshold": self.config.stagnation_threshold,
                "seed": self.config.seed,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow start failed: %s", exc)

    def _mlflow_log(self, result: StrategyResult) -> None:
        if not _MLFLOW_AVAILABLE or self._mlflow_run_id is None:
            return
        try:
            import mlflow
            step = result.iteration
            mlflow.log_metrics(
                {
                    "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "stability_ratio": result.stability_ratio,
                    "total_return": result.total_return,
                    "elapsed_seconds": result.elapsed_seconds,
                },
                step=step,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow log failed at iteration %d: %s", result.iteration, exc)

    def _mlflow_end(self) -> None:
        if not _MLFLOW_AVAILABLE or self._mlflow_run_id is None:
            return
        try:
            import mlflow
            s = self.summary()
            mlflow.log_metrics({k: v for k, v in s.items() if isinstance(v, (int, float))})
            mlflow.end_run()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow end failed: %s", exc)
