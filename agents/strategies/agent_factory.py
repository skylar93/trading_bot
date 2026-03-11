"""
Agent Factory: creates trading agents by type.

Supported agent types:
  - sb3_ppo / sb3_sac / sb3_td3 / sb3_a2c  (SB3-based, recommended)
  - momentum / meanreversion                 (strategy-specialised PPO agents)
  - multi / multiagent                       (MultiAgentManager)
  - assetspecific                            (AssetSpecificAgentFactory)
"""

from typing import Any, Dict, List, Optional

import gymnasium as gym
import logging
import numpy as np
import torch

from agents.sb3.sb3_agent_wrapper import SB3AgentWrapper

# Multi / strategy agents (kept)
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
from agents.strategies.multi.multi_agent_manager import MultiAgentManager

# Advanced
from agents.strategies.advanced.asset_specific_agents import AssetSpecificAgentFactory

logger = logging.getLogger(__name__)

SB3_TYPES = {"sb3ppo", "sb3sac", "sb3td3", "sb3a2c", "ppo", "sac", "td3", "a2c"}
ENSEMBLE_TYPES = {"ensemble", "ensemblemanager", "ensemblesb3"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_agent(
    agent_type: str,
    strategy: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    observation_space: Optional[gym.spaces.Space] = None,
    action_space: Optional[gym.spaces.Space] = None,
):
    """
    Create a trading agent.

    Args:
        agent_type: Agent type string (see module docstring)
        strategy: Optional strategy hint for multi-strategy agents
        config: Configuration dictionary
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space

    Returns:
        Instantiated agent object
    """
    if config is None:
        config = {}

    # Normalise type string
    agent_type_norm = agent_type.lower().replace("_", "").replace("-", "")
    if strategy:
        strategy = strategy.lower().replace("_", "").replace("-", "")

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if observation_space is None:
        obs_dim = config.get("observation_size", 10)
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    if action_space is None:
        action_dim = config.get("action_dim", 1)
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    logger.info(f"Creating agent: type={agent_type}, strategy={strategy}")

    # ------------------------------------------------------------------
    # SB3-based agents (primary path)
    # ------------------------------------------------------------------
    if agent_type_norm in SB3_TYPES:
        # Normalise to sb3_xxx format expected by the wrapper
        algo = agent_type_norm.replace("sb3", "")  # "" for plain "ppo" etc.
        algo_key = algo if algo else agent_type_norm  # "ppo", "sac", ...

        return SB3AgentWrapper(
            algo_type=algo_key,
            observation_space=observation_space,
            action_space=action_space,
            feature_extractor=config.get("feature_extractor"),
            feature_extractor_kwargs=config.get("feature_extractor_kwargs", {}),
            sb3_params=config.get("sb3_params", {}),
            device=device,
            verbose=config.get("verbose", 0),
        )

    # ------------------------------------------------------------------
    # Momentum PPO
    # ------------------------------------------------------------------
    elif agent_type_norm in ("momentum", "momentumppo"):
        return MomentumPPOAgent(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **_filtered(config),
        )

    # ------------------------------------------------------------------
    # Mean-reversion PPO
    # ------------------------------------------------------------------
    elif agent_type_norm in ("meanreversion", "meanreversionppo"):
        return MeanReversionPPOAgent(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **_filtered(config),
        )

    # ------------------------------------------------------------------
    # MultiAgentManager
    # ------------------------------------------------------------------
    elif agent_type_norm in ("multi", "multiagent", "multiagentmanager"):
        return MultiAgentManager(
            agent_configs=config.get("agent_configs", []),
            device=device,
            ensemble_method=config.get("ensemble_method", "weighted"),
        )

    # ------------------------------------------------------------------
    # Ensemble of SB3 agents (PPO + SAC + TD3)
    # ------------------------------------------------------------------
    elif agent_type_norm in ENSEMBLE_TYPES:
        from agents.ensemble.ensemble_manager import EnsembleManager

        return EnsembleManager(
            agent_configs=config.get("agents"),
            observation_space=observation_space,
            action_space=action_space,
            method=config.get("method", "rolling_validation"),
            rebalance_interval=config.get("rebalance_interval", 1000),
            validation_window=config.get("validation_window", 200),
            softmax_temperature=config.get("softmax_temperature", 1.0),
            feature_extractor=config.get("feature_extractor"),
            feature_extractor_kwargs=config.get("feature_extractor_kwargs", {}),
            device=device,
        )

    # ------------------------------------------------------------------
    # Asset-specific agents
    # ------------------------------------------------------------------
    elif agent_type_norm == "assetspecific":
        return AssetSpecificAgentFactory.create_agent(
            asset_id=config.get("asset_id", "unknown"),
            asset_type=config.get("asset_type", "unknown"),
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

    # ------------------------------------------------------------------
    # Unknown type
    # ------------------------------------------------------------------
    else:
        available = list_available_agents()
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Available types: {list(available.keys())}"
        )


def list_available_agents() -> Dict[str, str]:
    """Return a mapping of agent type → description."""
    return {
        "sb3_ppo": "Stable-Baselines3 PPO (recommended)",
        "sb3_sac": "Stable-Baselines3 SAC (off-policy, continuous actions)",
        "sb3_td3": "Stable-Baselines3 TD3 (off-policy, continuous actions)",
        "sb3_a2c": "Stable-Baselines3 A2C",
        "momentum": "Momentum-based strategy PPO agent",
        "meanreversion": "Mean-reversion strategy PPO agent",
        "multi": "MultiAgentManager (ensemble of agents)",
        "ensemble": "EnsembleManager — heterogeneous SB3 ensemble (PPO + SAC + TD3)",
        "assetspecific": "Asset-class-specific agent",
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_SKIP_KEYS = {"type", "strategy", "observation_space", "action_space", "device",
              "feature_extractor", "feature_extractor_kwargs", "sb3_params",
              "verbose", "algo_type"}


def _filtered(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove factory-level keys before passing config to an agent constructor."""
    return {k: v for k, v in config.items() if k not in _SKIP_KEYS}
