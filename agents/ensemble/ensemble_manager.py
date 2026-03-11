"""
EnsembleManager: coordinates a heterogeneous ensemble of SB3 agents.

Supports three ensemble methods:
  - rolling_validation : rolling Sharpe → softmax weights (default)
  - weighted_average   : same as rolling_validation (alias)
  - best               : winner-take-all (highest-weight agent acts alone)

Typical usage:
    manager = EnsembleManager(agent_configs, obs_space, act_space)
    manager.train_all(env, total_timesteps=300_000)
    action = manager.get_ensemble_action(obs)
    manager.update_weights(eval_metrics)
    manager.save("checkpoints/ensemble")
"""

import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from agents.sb3.sb3_agent_wrapper import SB3AgentWrapper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.regime.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)

# Default heterogeneous agent pool (PPO / SAC / TD3)
_DEFAULT_AGENT_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "ppo_conservative",
        "type": "sb3_ppo",
        "risk_profile": "conservative",
        "weight_init": 0.4,
        "params": {
            "ent_coef": 0.005,
            "clip_range": 0.1,
            "learning_rate": 3e-4,
            "n_steps": 2048,
        },
    },
    {
        "id": "sac_moderate",
        "type": "sb3_sac",
        "risk_profile": "moderate",
        "weight_init": 0.35,
        "params": {
            "ent_coef": "auto",
            "learning_rate": 3e-4,
        },
    },
    {
        "id": "td3_aggressive",
        "type": "sb3_td3",
        "risk_profile": "aggressive",
        "weight_init": 0.25,
        "params": {
            "learning_rate": 1e-3,
            "policy_delay": 2,
        },
    },
]


class EnsembleManager:
    """
    Manages a heterogeneous ensemble of SB3 agents (PPO, SAC, TD3).

    Weights are computed from a rolling Sharpe ratio over recent episode
    returns.  Higher Sharpe → higher ensemble weight via softmax.

    Attributes:
        agents          : {agent_id: SB3AgentWrapper}
        agent_metadata  : {agent_id: {type, risk_profile}}
        method          : weighting method string
        rebalance_interval : steps between external rebalancing calls
    """

    def __init__(
        self,
        agent_configs: Optional[List[Dict[str, Any]]] = None,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        method: str = "rolling_validation",
        rebalance_interval: int = 1000,
        validation_window: int = 200,
        softmax_temperature: float = 1.0,
        feature_extractor: Optional[str] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto",
    ) -> None:
        """
        Args:
            agent_configs: Per-agent config list. Each dict may contain:
                - id           (str, optional)  : unique agent identifier
                - type         (str)             : algo, e.g. "sb3_ppo"
                - weight_init  (float, optional) : initial ensemble weight
                - params       (dict, optional)  : SB3 algo hyperparams
                - risk_profile (str, optional)   : metadata label
            observation_space: Shared Gymnasium observation space.
            action_space: Shared Gymnasium action space.
            method: "rolling_validation" | "weighted_average" | "best"
            rebalance_interval: Steps between weight updates (used externally).
            validation_window: Max episode returns kept per agent for Sharpe.
            softmax_temperature: Temperature for softmax weight computation.
            feature_extractor: "conv1d", "lstm", or None (shared across agents).
            feature_extractor_kwargs: Kwargs forwarded to the feature extractor.
            device: PyTorch device string ("auto", "cpu", "cuda").
        """
        if method not in ("rolling_validation", "weighted_average", "best"):
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: rolling_validation, weighted_average, best"
            )

        self.method = method
        self.rebalance_interval = rebalance_interval
        self.validation_window = validation_window
        self.softmax_temperature = softmax_temperature
        self._device = device

        configs = agent_configs if agent_configs is not None else _DEFAULT_AGENT_CONFIGS

        # ── Build per-agent state ──────────────────────────────────────
        self.agents: Dict[str, SB3AgentWrapper] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        self._weights: Dict[str, float] = {}
        self._return_history: Dict[str, deque] = {}

        n = len(configs)
        for i, cfg in enumerate(configs):
            agent_id = cfg.get("id", f"agent_{i}")
            weight_init = float(cfg.get("weight_init", 1.0 / n))
            algo_type = cfg.get("type", "sb3_ppo")
            params = cfg.get("params", {})
            risk_profile = cfg.get("risk_profile", "moderate")

            self.agents[agent_id] = SB3AgentWrapper(
                algo_type=algo_type,
                observation_space=observation_space,
                action_space=action_space,
                feature_extractor=feature_extractor,
                feature_extractor_kwargs=feature_extractor_kwargs or {},
                sb3_params=params,
                device=device,
            )
            self.agent_metadata[agent_id] = {
                "type": algo_type,
                "risk_profile": risk_profile,
            }
            self._weights[agent_id] = weight_init
            self._return_history[agent_id] = deque(maxlen=validation_window)

        self._normalise_weights()

        # ── Week 6: regime detector (optional) ────────────────────────────────
        self._regime_detector: Optional["RegimeDetector"] = None

        logger.info(
            "EnsembleManager created: %d agents (%s), method=%s",
            len(self.agents),
            list(self.agents.keys()),
            method,
        )

    # ──────────────────────────────────────────────────────────────────
    # Ensemble action
    # ──────────────────────────────────────────────────────────────────

    def get_ensemble_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Return a single ensemble action for the given observation.

        - "best"  : only the top-weight agent contributes.
        - others  : weighted average of all agents' actions.
        """
        if self.method == "best":
            best_id = max(self._weights, key=self._weights.__getitem__)
            return self.agents[best_id].get_action(observation, deterministic=deterministic)

        agent_ids = list(self.agents.keys())
        actions = np.stack(
            [self.agents[aid].get_action(observation, deterministic=deterministic) for aid in agent_ids]
        )  # shape (N, action_dim)
        weights = np.array([self._weights[aid] for aid in agent_ids])
        return np.sum(weights[:, None] * actions, axis=0)

    # ──────────────────────────────────────────────────────────────────
    # Weight management
    # ──────────────────────────────────────────────────────────────────

    def get_weights(self) -> Dict[str, float]:
        """Return a copy of the current ensemble weights."""
        return dict(self._weights)

    def update_weights(self, eval_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Update weights from per-agent evaluation metrics.

        Args:
            eval_metrics: {agent_id: {"mean_reward": float, ...}}
                          As returned by evaluate_agents().
        """
        for agent_id, metrics in eval_metrics.items():
            if agent_id in self._return_history:
                self._return_history[agent_id].append(
                    float(metrics.get("mean_reward", 0.0))
                )
        self._recompute_weights()

    def record_episode_return(self, agent_id: str, episode_return: float) -> None:
        """Manually record one episode return for a specific agent."""
        if agent_id in self._return_history:
            self._return_history[agent_id].append(float(episode_return))

    def rebalance(self) -> None:
        """Recompute weights from accumulated return history."""
        self._recompute_weights()

    # ── Week 6: Regime-aware weight updates ───────────────────────────────────

    def set_regime_detector(self, detector: "RegimeDetector") -> None:
        """Attach a pre-fitted RegimeDetector to the ensemble."""
        self._regime_detector = detector
        logger.info("RegimeDetector attached to EnsembleManager (method=%s).", detector.method)

    def update_weights_regime_aware(
        self,
        eval_metrics: Dict[str, Dict[str, float]],
        prices: Optional[np.ndarray] = None,
        regime: Optional[int] = None,
    ) -> None:
        """
        Update ensemble weights using both performance and market regime.

        The update proceeds in two stages:
          1. Performance stage: standard rolling-Sharpe rebalancing from
             eval_metrics (same as update_weights).
          2. Regime stage: multiplicative adjustment based on each agent's
             risk_profile and the current market regime.

        Args:
            eval_metrics: {agent_id: {"mean_reward": float, ...}}
            prices:       Recent price array fed to the attached regime_detector
                          to infer the current regime. Ignored if *regime* is
                          supplied directly.
            regime:       Regime label (0=low_vol, 1=medium_vol, 2=high_vol).
                          If None, inferred from *prices* via the attached
                          detector (if any).
        """
        # Stage 1 — performance update (populates return history)
        self.update_weights(eval_metrics)

        # Stage 2 — regime adjustment
        if regime is None and self._regime_detector is not None and prices is not None:
            try:
                regime = self._regime_detector.predict_regime(prices)
            except Exception as exc:
                logger.warning("Regime inference failed: %s. Skipping regime adjustment.", exc)
                return

        if regime is None:
            return  # no regime info available, skip

        if self._regime_detector is not None:
            multipliers = self._regime_detector.get_weight_multipliers(regime)
        else:
            # Fallback multipliers (same as RegimeDetector defaults)
            _defaults: Dict[int, Dict[str, float]] = {
                0: {"conservative": 1.5, "moderate": 1.0, "aggressive": 0.5},
                1: {"conservative": 1.0, "moderate": 1.0, "aggressive": 1.0},
                2: {"conservative": 0.8, "moderate": 1.5, "aggressive": 0.3},
            }
            multipliers = _defaults.get(regime, {k: 1.0 for k in ("conservative", "moderate", "aggressive")})

        for agent_id in self._weights:
            risk_profile = self.agent_metadata[agent_id].get("risk_profile", "moderate")
            mult = multipliers.get(risk_profile, 1.0)
            self._weights[agent_id] *= max(mult, 1e-6)

        self._normalise_weights()
        logger.debug(
            "Regime-aware weights (regime=%d): %s",
            regime,
            {k: f"{v:.3f}" for k, v in self._weights.items()},
        )

    def get_current_regime(self) -> Optional[int]:
        """Return the current regime from the attached detector (or None)."""
        if self._regime_detector is not None:
            return self._regime_detector.current_regime
        return None

    def get_regime_info(self) -> Dict[str, Any]:
        """Return regime metadata dict (empty if no detector attached)."""
        if self._regime_detector is None:
            return {}
        return {
            "regime": self._regime_detector.current_regime,
            "regime_name": self._regime_detector.current_regime_name,
            "probs": self._regime_detector.current_probs.tolist(),
        }

    # ──────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────

    def train_all(
        self,
        env,
        total_timesteps: int,
        callbacks_per_agent: Optional[Dict[str, Any]] = None,
        timesteps_per_agent: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Train each agent in the ensemble sequentially on the same env.

        Args:
            env: Gymnasium-compatible env or VecEnv (shared by all agents).
            total_timesteps: Budget split equally unless timesteps_per_agent given.
            callbacks_per_agent: {agent_id: SB3 callback(s)}.
            timesteps_per_agent: {agent_id: int} for unequal time allocation.

        Returns:
            {agent_id: train_result_dict}
        """
        callbacks_per_agent = callbacks_per_agent or {}
        n = len(self.agents)
        default_steps = total_timesteps // n

        results: Dict[str, Any] = {}
        for agent_id, agent in self.agents.items():
            steps = (
                timesteps_per_agent[agent_id]
                if timesteps_per_agent and agent_id in timesteps_per_agent
                else default_steps
            )
            cb = callbacks_per_agent.get(agent_id)
            logger.info(
                "Training %s (%s) for %s steps",
                agent_id,
                self.agent_metadata[agent_id]["type"],
                f"{steps:,}",
            )
            results[agent_id] = agent.train(env, total_timesteps=steps, callbacks=cb)

        return results

    def train_agent(
        self,
        agent_id: str,
        env,
        total_timesteps: int,
        callbacks: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train a single named agent."""
        if agent_id not in self.agents:
            raise KeyError(
                f"Agent '{agent_id}' not found. Available: {list(self.agents.keys())}"
            )
        return self.agents[agent_id].train(env, total_timesteps=total_timesteps, callbacks=callbacks)

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────

    def evaluate_agents(
        self,
        eval_env,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate every agent on eval_env.

        Returns:
            {agent_id: {"mean_reward": float, "std_reward": float}}
        """
        from stable_baselines3.common.evaluation import evaluate_policy

        metrics: Dict[str, Dict[str, float]] = {}
        for agent_id, agent in self.agents.items():
            if agent.model is None:
                logger.warning("Agent '%s' has no trained model — skipping.", agent_id)
                metrics[agent_id] = {"mean_reward": 0.0, "std_reward": 0.0}
                continue

            mean_r, std_r = evaluate_policy(
                agent.model,
                eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
            )
            metrics[agent_id] = {
                "mean_reward": float(mean_r),
                "std_reward": float(std_r),
            }
            logger.info(
                "  %s: mean_reward=%.4f ± %.4f", agent_id, mean_r, std_r
            )
        return metrics

    def select_best(
        self,
        validation_env,
        n_eval_episodes: int = 5,
    ) -> str:
        """
        Evaluate all agents, update weights, and return the best agent ID.
        """
        metrics = self.evaluate_agents(validation_env, n_eval_episodes=n_eval_episodes)
        self.update_weights(metrics)
        best_id = max(metrics, key=lambda k: metrics[k]["mean_reward"])
        logger.info(
            "Best agent: %s (mean_reward=%.4f)",
            best_id,
            metrics[best_id]["mean_reward"],
        )
        return best_id

    # ──────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save all agent models to *directory*."""
        os.makedirs(directory, exist_ok=True)
        for agent_id, agent in self.agents.items():
            path = os.path.join(directory, agent_id)
            agent.save(path)
        logger.info("Ensemble saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load agent models from *directory* (skips missing files).

        Note: bypasses SB3AgentWrapper.load() to avoid the classmethod name
        collision in that class; directly loads into agent.model.
        """
        for agent_id, agent in self.agents.items():
            path = os.path.join(directory, agent_id)
            if os.path.exists(path + ".zip"):
                agent.model = agent._algo_class.load(path, device=agent._device)
                logger.info("  Loaded %s from %s", agent_id, path)
            else:
                logger.warning("  No saved model found for %s at %s", agent_id, path)

    # ──────────────────────────────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────────────────────────────

    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of ensemble diagnostics."""
        metrics: Dict[str, Any] = {
            "weights": dict(self._weights),
            "sharpe_scores": self._compute_sharpe_scores(),
            "return_history_sizes": {k: len(v) for k, v in self._return_history.items()},
            "agent_types": {k: v["type"] for k, v in self.agent_metadata.items()},
            "method": self.method,
        }
        if self._regime_detector is not None:
            metrics["regime"] = self.get_regime_info()
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _recompute_weights(self) -> None:
        """Recompute weights from return history using current method."""
        if self.method == "best":
            return  # "best" uses existing weights as selector; no update needed

        scores = self._compute_sharpe_scores()
        ids = list(scores.keys())
        vals = np.array([scores[aid] for aid in ids], dtype=float)

        # Softmax with temperature (numerically stable)
        vals_t = vals / max(self.softmax_temperature, 1e-8)
        exp_v = np.exp(vals_t - vals_t.max())
        softmax_w = exp_v / exp_v.sum()

        for i, agent_id in enumerate(ids):
            self._weights[agent_id] = float(softmax_w[i])

        self._normalise_weights()
        logger.debug("Weights updated: %s", self._weights)

    def _compute_sharpe_scores(self) -> Dict[str, float]:
        """Rolling Sharpe ratio per agent (mean/std of recent returns)."""
        scores: Dict[str, float] = {}
        for agent_id, history in self._return_history.items():
            arr = np.array(history, dtype=float)
            if len(arr) < 2:
                scores[agent_id] = 0.0
            else:
                scores[agent_id] = float(np.mean(arr) / (np.std(arr, ddof=1) + 1e-8))
        return scores

    def _normalise_weights(self) -> None:
        """Ensure weights are non-negative and sum to 1."""
        total = sum(self._weights.values())
        if total <= 0:
            n = len(self._weights)
            for k in self._weights:
                self._weights[k] = 1.0 / n
        else:
            for k in self._weights:
                self._weights[k] /= total

    # ──────────────────────────────────────────────────────────────────
    # Class constructor from config dict
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ) -> "EnsembleManager":
        """
        Build an EnsembleManager from a config dict.

        Expected top-level keys (all optional):
            agents              : list[dict]  – per-agent configs
            method              : str
            rebalance_interval  : int
            validation_window   : int
            softmax_temperature : float
            feature_extractor   : str | None
            feature_extractor_kwargs : dict
            device              : str
        """
        return cls(
            agent_configs=config.get("agents"),
            observation_space=observation_space,
            action_space=action_space,
            method=config.get("method", "rolling_validation"),
            rebalance_interval=config.get("rebalance_interval", 1000),
            validation_window=config.get("validation_window", 200),
            softmax_temperature=config.get("softmax_temperature", 1.0),
            feature_extractor=config.get("feature_extractor"),
            feature_extractor_kwargs=config.get("feature_extractor_kwargs", {}),
            device=config.get("device", "auto"),
        )

    def __repr__(self) -> str:
        agents_str = ", ".join(
            f"{aid}({meta['type']})" for aid, meta in self.agent_metadata.items()
        )
        return f"EnsembleManager(method={self.method}, agents=[{agents_str}])"

    def __len__(self) -> int:
        return len(self.agents)
