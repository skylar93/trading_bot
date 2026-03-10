import gymnasium as gym
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all agents.
    
    Features:
    - Common interface for different agent implementations
    - Abstract methods for action selection and training
    - Proper state management and persistence
    - Support for deterministic/evaluation modes
    - Standardized action shape handling
    
    Implementation Notes:
    - Provides basic agent functionality
    - All agent implementations should inherit from this class
    - Implements action space shape compatibility handling
    - Ensures consistent action format across different agents
    
    Recent Changes:
    - Added proper multi-dimensional action space handling
    - Improved action shape compatibility checking
    - Enhanced error handling for action generation
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
        
        # Store action shape for proper action formatting
        if isinstance(action_space, gym.spaces.Box):
            self.action_shape = action_space.shape
            self.action_dim = int(np.prod(action_space.shape))
        else:
            self.action_shape = (1,)
            self.action_dim = 1
            
        logger.debug(f"Initialized BaseAgent with action shape: {self.action_shape}")
        
    def get_action(self, observation: np.ndarray, deterministic: bool = False, eval_mode: bool = False) -> np.ndarray:
        """
        Get action from agent based on observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic action selection
            eval_mode: Whether the agent is in evaluation mode (equivalent to deterministic)
            
        Returns:
            Selected action with shape matching action_space
        """
        # This should be implemented by subclasses
        # When implementing, ensure the returned action has the correct shape
        raise NotImplementedError("Subclasses must implement get_action")
        
    def _ensure_action_shape(self, action: np.ndarray) -> np.ndarray:
        """
        Ensure the action has the correct shape according to action_space.
        
        Args:
            action: Action array from policy
            
        Returns:
            Action array with correct shape
        """
        # If action is already the right shape, return it
        if action.shape == self.action_shape:
            return action
            
        # If action is 1D but should be multi-dimensional
        if action.shape == (1,) and self.action_shape != (1,):
            logger.warning(
                f"Action shape mismatch: got {action.shape}, expected {self.action_shape}. "
                f"Adapting action shape."
            )
            
            # Create a new action array with the right shape
            # Fill with the single value or zeros
            adapted_action = np.zeros(self.action_shape, dtype=np.float32)
            
            # Copy the first value or distribute evenly
            if len(action) == 1:
                # For multi-asset environments, we'll use a fixed strategy:
                # - First asset: Use the actual action value
                # - Other assets: Use small negative values (light sell)
                adapted_action[0] = action[0]  # Main action for first asset
                for i in range(1, self.action_shape[0]):
                    adapted_action[i] = -0.1  # Light sell for other assets
            else:
                # If we have multiple values but wrong shape,
                # take as many as we can and reshape
                adapted_action.flat[:min(len(action), adapted_action.size)] = action.flat[:min(len(action), adapted_action.size)]
                
            return adapted_action
            
        # If action is multi-dimensional but should be 1D
        if len(action.shape) > 1 and self.action_shape == (1,):
            logger.warning(
                f"Action shape mismatch: got {action.shape}, expected {self.action_shape}. "
                f"Taking first element."
            )
            return np.array([action.flat[0]], dtype=np.float32)
            
        # Any other case, try to reshape or pad
        logger.warning(
            f"Action shape mismatch: got {action.shape}, expected {self.action_shape}. "
            f"Attempting to reshape."
        )
        
        try:
            # Try to reshape
            reshaped = action.reshape(self.action_shape)
            return reshaped
        except ValueError:
            # If reshape fails, create new array and copy as much as possible
            adapted_action = np.zeros(self.action_shape, dtype=np.float32)
            flat_size = min(action.size, adapted_action.size)
            adapted_action.flat[:flat_size] = action.flat[:flat_size]
            return adapted_action
        
    def train_step(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Train agent on a single experience.
        
        Args:
            experience: Experience dictionary
            
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