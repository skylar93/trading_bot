from abc import ABC, abstractmethod
import numpy as np
import logging
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """
    Base class for all agents
    
    Features:
    - Common interface for all agent implementations
    - Standardized methods for actions, training, and model management
    - Supports both individual parameter and experience dictionary formats
    - Compatible with shared experience buffer
    
    Implementation Notes:
    - Subclasses should override _update rather than train_step for compatibility
    - Handles conversion between different parameter formats
    - Provides standardized logging setup
    
    Recent Changes:
    - Added support for experience dictionary in train_step
    - Enhanced compatibility with shared experience buffer
    - Added proper docstrings and type hints
    """

    def __init__(self, observation_space=None, action_space=None):
        """Initialize base agent

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.logger = logging.getLogger(self.__class__.__name__)

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
    
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """
        Train the agent on a single step of experience.
        
        This method supports multiple calling conventions:
        1. train_step(state, action, reward, next_state, done[, info])
        2. train_step(experience=experience_dict)
        
        The experience_dict format should have keys:
        - observation/state
        - action
        - reward
        - next_observation/next_state
        - done
        - info (optional)
        
        Args:
            *args: Positional arguments (state, action, reward, next_state, done)
            **kwargs: Keyword arguments, including possibly 'experience'
            
        Returns:
            Dictionary of training metrics
        """
        # Handle experience dictionary format
        experience = kwargs.get("experience")
        if experience is not None:
            # Extract individual components from the experience dict
            state = experience.get("observation", experience.get("state"))
            action = experience.get("action")
            reward = experience.get("reward", 0.0)
            next_state = experience.get("next_observation", experience.get("next_state"))
            done = experience.get("done", False)
            info = experience.get("info", {})
            
            # Call the actual implementation with extracted components
            return self._update(state, action, reward, next_state, done, info)
        
        # Handle individual parameter format (traditional)
        elif len(args) >= 5:
            # Extract required parameters
            state, action, reward, next_state, done = args[:5]
            # Get optional info dictionary if provided
            info = args[5] if len(args) > 5 else {}
            
            # Call the actual implementation
            return self._update(state, action, reward, next_state, done, info)
        
        # If neither format is provided correctly
        else:
            raise ValueError(
                "train_step requires either an experience dictionary or "
                "state, action, reward, next_state, done parameters"
            )
    
    def _update(self, state, action, reward, next_state, done, info=None) -> Dict[str, float]:
        """
        Actual implementation of the update logic. Subclasses should override this method.
        
        This default implementation returns an empty metrics dictionary.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            Dictionary of training metrics
        """
        return {}

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

    def get_action_with_hidden_state(self, state: np.ndarray, deterministic: bool = False):
        """
        Get action and hidden state from agent.
        
        For agents that don't have internal hidden states, we return a dummy hidden state.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, hidden_state)
        """
        action = self.get_action(state, deterministic)
        # Return dummy hidden state if not overridden by subclass
        return action, np.zeros(10)
