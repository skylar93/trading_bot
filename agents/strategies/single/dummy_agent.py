"""
Dummy agent for testing that makes small random trades
"""

import logging
from typing import Dict, Any
import numpy as np
from agents.base.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    """Dummy agent for testing that makes small random trades"""
    
    def __init__(self, observation_space=None, action_space=None, **kwargs):
        """Initialize dummy agent
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space)
        self.step_count = -1  # Start at -1 so first increment gives 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Log unused config keys
        unused_keys = [key for key in kwargs.keys() if key not in self.__init__.__code__.co_varnames]
        if unused_keys:
            self.logger.warning(f"Ignoring unused config keys in DummyAgent: {unused_keys}")
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Generate small random actions for testing
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action to take
        """
        self.step_count += 1
        
        # First action should be non-zero but small for consistency test
        if self.step_count == 0:
            return np.array([0.5])
            
        # Trade every 5 steps with small magnitude
        if self.step_count % 5 == 0:
            action = 0.5 if (self.step_count // 5) % 2 == 0 else -0.5
            self.logger.info(f"DummyAgent taking action: {action}")
            return np.array([action])
            
        return np.array([0.0])
    
    def predict(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Alias for get_action to maintain API compatibility with other agents
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action to take
        """
        return self.get_action(state, deterministic)
    
    def train(self, env, total_timesteps: int = 10000, batch_size: int = 64) -> Dict[str, Any]:
        """Train agent (dummy implementation)
        
        Args:
            env: Training environment
            total_timesteps: Total number of timesteps to train for
            batch_size: Size of each training batch
            
        Returns:
            Empty dictionary since this is a dummy agent
        """
        return {}
    
    def save(self, path: str):
        """Save agent state (dummy implementation)
        
        Args:
            path: Path to save state to
        """
        pass
    
    def load(self, path: str):
        """Load agent state (dummy implementation)
        
        Args:
            path: Path to load state from
        """
        pass 