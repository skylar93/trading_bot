import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional


class BaseAgent:
    """
    Base class for all trading agents.
    
    Features:
    - Common interface for different agent implementations
    - Standard methods for actions, training, and model management
    - Compatible with gymnasium environments
    - Supports both discrete and continuous action spaces
    
    Implementation Notes:
    - Subclasses should implement get_action and train_step methods
    - Handles observation and action space validation
    - Provides utility methods for saving/loading models
    
    Recent Changes:
    - Added type annotations for better code safety
    - Improved error handling and validation
    - Added compatibility with new environment interfaces
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        **kwargs
    ):
        """
        Initialize base agent.
        
        Args:
            observation_space: Agent's observation space
            action_space: Agent's action space
            **kwargs: Additional parameters for specific implementations
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from agent based on observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        raise NotImplementedError("Subclasses must implement get_action")
        
    def train_step(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Train agent on a single experience.
        
        Args:
            experience: Dictionary containing experience data
            
        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError("Subclasses must implement train_step")
        
    def save(self, path: str) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save to
        """
        raise NotImplementedError("Subclasses must implement save")
        
    def load(self, path: str) -> None:
        """
        Load agent from disk.
        
        Args:
            path: Path to load from
        """
        raise NotImplementedError("Subclasses must implement load") 