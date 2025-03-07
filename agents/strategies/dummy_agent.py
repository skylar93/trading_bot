import numpy as np
import torch
import logging
import gymnasium as gym
from typing import Dict, Any, Tuple, List, Optional, Union
import os

from agents.strategies.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DummyAgent(BaseAgent):
    """
    Dummy Agent for testing purposes
    
    Features:
    - Implements a minimal agent interface for testing
    - Returns random or fixed actions
    - Tracks calls for test verification
    - Handles various input formats
    - Supports flexible method signatures
    
    Implementation Notes:
    - Designed for use in test environments
    - Can mimic different agent types by setting agent_type
    - Customizable behavior via config parameters
    - Records history of calls for verification
    
    Recent Changes:
    - Enhanced method signature compatibility
    - Added support for various experience formats
    - Improved observation/action handling
    - Added predict method alias
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: str = "cpu",
        agent_type: str = "generic",
        fixed_action: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the DummyAgent.
        
        Args:
            observation_space: Observation space from gym
            action_space: Action space from gym
            device: Device to use (CPU/GPU)
            agent_type: Type of agent to mimic ("momentum", "mean_reversion", etc.)
            fixed_action: If provided, always return this action instead of random
            **kwargs: Additional configuration parameters
        """
        super().__init__(observation_space, action_space)
        self.device = device
        self.agent_type = agent_type
        self.fixed_action = fixed_action
        
        # Store all kwargs for test verification
        self.config = kwargs
        
        # Initialize call counters for testing
        self.get_action_calls = 0
        self.train_step_calls = 0
        self.save_calls = 0
        self.load_calls = 0
        
        # Track action history
        self.action_history = []
        
        # Initialize dummy network for compatibility with tests
        self.policy = None
        self.optimizer = None
        
        # Add attributes expected by tests for specific agent types
        if agent_type == "momentum" or agent_type == "momentumppo":
            self.momentum_window = kwargs.get("momentum_window", 20)
        
        if agent_type == "mean_reversion" or agent_type == "meanreversionppo":
            self.rsi_window = kwargs.get("rsi_window", 14)
            self.bollinger_window = kwargs.get("bollinger_window", 20)
            self.bollinger_std = kwargs.get("bollinger_std", 2.0)
            
        # For PPO compatibility
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_param = kwargs.get("clip_param", 0.2)
        self.value_coef = kwargs.get("value_coef", 0.5)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        
        # For multi-agent compatibility
        self.min_share_reward = kwargs.get("min_share_reward", 0.0)
        self.synergy_score = 0.5  # Default synergy score for tests
        
        # For testing
        self.training_step = 0
        
        logger.info(f"Initialized DummyAgent with type {agent_type}")
        
    def get_action(
        self, 
        observation: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Return a random or fixed action based on the agent configuration.
        
        Args:
            observation: State observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action array with shape matching action_space
        """
        self.get_action_calls += 1
        
        # Return fixed action if specified
        if self.fixed_action is not None:
            action = np.array([self.fixed_action], dtype=np.float32)
        else:
            # Generate action based on agent type
            if self.agent_type == "momentum" or self.agent_type == "momentumppo":
                # For momentum, return slightly positive bias
                action = np.array([0.1], dtype=np.float32)
                
                # Try to look at observation to determine trend
                if len(observation) > 0 and isinstance(observation[0], np.ndarray):
                    # If 2D observation, check last few values for trend
                    closes = observation[-5:, 3] if observation.shape[1] > 3 else np.zeros(5)
                    if len(closes) > 1 and closes[-1] > closes[0]:
                        # Uptrend - momentum agent should go long
                        action = np.array([0.7], dtype=np.float32)
                    else:
                        # Downtrend - momentum agent should go short
                        action = np.array([-0.7], dtype=np.float32)
                    
            elif self.agent_type == "mean_reversion" or self.agent_type == "meanreversionppo":
                # For mean reversion, look for overbought/oversold
                action = np.array([0.0], dtype=np.float32)
                
                # Try to look at observation to determine mean reversion opportunity
                if len(observation) > 0 and isinstance(observation[0], np.ndarray):
                    # If 2D observation, check last few values for trend
                    closes = observation[-5:, 3] if observation.shape[1] > 3 else np.zeros(5)
                    if len(closes) > 1:
                        # Contrarian approach
                        if closes[-1] > closes[0]:
                            # Price went up - mean reversion expects down
                            action = np.array([-0.7], dtype=np.float32)
                        else:
                            # Price went down - mean reversion expects up
                            action = np.array([0.7], dtype=np.float32)
            else:
                # Random action for generic agent
                action = self.action_space.sample()
                
                # Ensure action is in correct format
                if isinstance(action, np.ndarray) and action.shape != (1,):
                    action = np.array([action[0]], dtype=np.float32)
                else:
                    action = np.array([action], dtype=np.float32)
        
        # Clip action to ensure it's within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Record action for testing
        self.action_history.append(action)
        
        return action
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Alias for get_action with deterministic=True.
        
        Args:
            observation: The current state observation
            
        Returns:
            Deterministic action
        """
        return self.get_action(observation, deterministic=True)
    
    def train_step(
        self,
        state: np.ndarray = None,
        action: np.ndarray = None,
        reward: float = None,
        next_state: np.ndarray = None,
        done: bool = None,
        info: Dict[str, Any] = None,
        experience: Dict[str, Any] = None,
        batch_size: int = None,
        num_epochs: int = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Simulated training step that returns dummy metrics.
        
        Supports multiple method signatures for compatibility with different tests:
        - Individual parameters (state, action, reward, etc.)
        - Single experience dict
        - Additional kwargs for advanced PPO parameters
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
            info: Additional information
            experience: Alternative single dict parameter
            batch_size: PPO batch size (ignored)
            num_epochs: PPO training epochs (ignored)
            **kwargs: Additional parameters for compatibility
            
        Returns:
            Dictionary of training metrics
        """
        self.train_step_calls += 1
        self.training_step += 1
        
        # Return different metrics based on agent type
        base_metrics = {
            "policy_loss": 0.01 / (1 + self.training_step),
            "value_loss": 0.05 / (1 + self.training_step),
            "entropy": 0.5 / (1 + self.training_step),
            "exploration_rate": 0.1 / (1 + 0.1 * self.training_step)
        }
        
        # Add agent-specific metrics
        if self.agent_type == "momentum" or self.agent_type == "momentumppo":
            base_metrics["momentum_bias"] = 0.3
            
        elif self.agent_type == "mean_reversion" or self.agent_type == "meanreversionppo": 
            base_metrics["rsi_value"] = 50.0
            base_metrics["reversion_bias"] = 0.2
            
        return base_metrics
    
    def _calculate_momentum_features(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate momentum features for testing.
        
        Args:
            state: Price and volume data
            
        Returns:
            Momentum feature array
        """
        return np.array([0.5, 0.1, 0.3])
    
    def _calculate_volatility_features(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate volatility features for testing.
        
        Args:
            state: Price and volume data
            
        Returns:
            Volatility feature array
        """
        return np.array([0.2])
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """
        Calculate RSI for testing.
        
        Args:
            prices: Array of prices
            
        Returns:
            RSI value
        """
        return 50.0
    
    def _calculate_bollinger_bands(
        self, 
        prices: np.ndarray,
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands for testing.
        
        Args:
            prices: Array of prices
            period: Period for moving average
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (middle, upper, lower) bands
        """
        if len(prices) == 0:
            return 100.0, 110.0, 90.0
        
        price = prices[-1]
        return price, price * 1.05, price * 0.95
    
    def _calculate_reversion_features(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate mean reversion features for testing.
        
        Args:
            state: Price and volume data
            
        Returns:
            Mean reversion feature array
        """
        return np.array([0.2, 0.3, 0.4])
    
    def save(self, path: str) -> None:
        """
        Simulated model saving.
        
        Args:
            path: Path to save the model
        """
        self.save_calls += 1
        logger.debug(f"DummyAgent: Simulated saving to {path}")
        
    def load(self, path: str) -> None:
        """
        Simulated model loading.
        
        Args:
            path: Path to load the model from
        """
        self.load_calls += 1
        logger.debug(f"DummyAgent: Simulated loading from {path}")
        
    def reset(self) -> None:
        """Reset counters and history for fresh testing."""
        self.get_action_calls = 0
        self.train_step_calls = 0
        self.save_calls = 0
        self.load_calls = 0
        self.action_history = [] 