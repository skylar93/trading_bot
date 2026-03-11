"""Lightweight mock factory used by legacy test files."""
import numpy as np
import logging
from typing import Dict, Any, Optional
import gymnasium as gym

from agents.strategies.base_agent import BaseAgent

logger = logging.getLogger(__name__)

dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
dummy_act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


class _MockAgent(BaseAgent):
    """Minimal agent that returns random actions – no external deps."""

    def __init__(self, observation_space=None, action_space=None, **kwargs):
        super().__init__(
            observation_space or dummy_obs_space,
            action_space or dummy_act_space,
        )

    def get_action(self, observation, deterministic: bool = False):
        return self.action_space.sample()

    def predict(self, observation):
        return self.action_space.sample()

    def train_step(self, experience, **kwargs):
        return {"loss": 0.0}

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


def create_test_agent(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
    observation_space=None,
    action_space=None,
) -> BaseAgent:
    config = config or {}
    obs = observation_space or dummy_obs_space
    act = action_space or dummy_act_space
    return _MockAgent(obs, act)


def create_test_multi_agent_manager(
    agent_configs: list,
    ensemble_method: str = "weighted",
    device: str = "cpu",
) -> Dict[str, BaseAgent]:
    agents: Dict[str, BaseAgent] = {}
    for cfg in agent_configs:
        agent_id = cfg.get("id", f"agent_{len(agents)}")
        agents[agent_id] = create_test_agent(
            agent_type=cfg.get("type", "mock"),
            config=cfg,
        )

    def act_method(observations, deterministic=False):
        return {
            aid: agent.get_action(observations.get(aid, dummy_obs_space.sample()), deterministic)
            for aid, agent in agents.items()
        }

    def train_step_method(experiences):
        return {aid: {"loss": 0.0} for aid in agents}

    agents["act"] = act_method  # type: ignore[assignment]
    agents["train_step"] = train_step_method  # type: ignore[assignment]
    agents["ensemble_method"] = ensemble_method  # type: ignore[assignment]
    return agents
