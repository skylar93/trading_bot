from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, observation_space=None, action_space=None):
        """Initialize base agent

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
        """
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Get action from agent

        Args:
            state: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def train(
        self, env, total_timesteps: int = 10000, batch_size: int = 64
    ) -> Dict[str, Any]:
        """Train agent

        Args:
            env: Training environment
            total_timesteps: Total number of timesteps to train for
            batch_size: Size of each training batch

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent state

        Args:
            path: Path to save state to
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load agent state

        Args:
            path: Path to load state from
        """
        pass
