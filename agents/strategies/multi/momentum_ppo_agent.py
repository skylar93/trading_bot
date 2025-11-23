import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from agents.strategies.single.ppo_agent import PPOAgent
from gymnasium import spaces
import pandas as pd

logger = logging.getLogger(__name__)

class MomentumPPOAgent(PPOAgent):
    """
    Momentum strategy PPO agent that specializes in trend-following.
    Inherits from base PPO agent but adds momentum-specific features and logic.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        momentum_window: int = 20,
        volatility_window: int = 20,
        trend_window: int = 50,
        momentum_threshold: float = 0.02,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        c3: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: float = 0.015,
        device: str = None,
        **kwargs
    ):
        """
        Initialize Momentum PPO Agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            momentum_window: Window size for momentum calculation
            volatility_window: Window size for volatility calculation
            trend_window: Window size for trend calculation
            momentum_threshold: Threshold for momentum signals (default: 0.02)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            c3: KL divergence coefficient
            batch_size: Batch size for updates
            n_epochs: Number of epochs per update
            target_kl: Target KL divergence
            device: Device to use for computations
            **kwargs: Additional arguments
        """
        # Calculate total features (base features + momentum indicators)
        n_momentum_features = 3  # momentum, volatility, trend
        
        if isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) == 2:  # (window_size, features)
                total_features = observation_space.shape[0] * observation_space.shape[1] + n_momentum_features
                
                flat_obs_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_features,),
                    dtype=np.float32
                )
            else:
                raise ValueError("Observation space must be 2D (window_size, features)")
            
            self.original_obs_space = observation_space
            self.n_momentum_features = n_momentum_features
        else:
            raise ValueError("Observation space must be Box")
        
        # Initialize base PPO agent with flattened observation space
        super().__init__(
            observation_space=flat_obs_space,
            action_space=action_space,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            c1=c1,
            c2=c2,
            c3=c3,
            batch_size=batch_size,
            n_epochs=n_epochs,
            target_kl=target_kl,
            device=device,
            **kwargs
        )
        
        # Momentum specific parameters
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.momentum_threshold = momentum_threshold
        self.strategy = "momentum"
        
        # Log unused config keys
        unused_keys = [key for key in kwargs.keys() if key not in self.__init__.__code__.co_varnames]
        if unused_keys:
            self.logger.warning(f"Ignoring unused config keys in MomentumPPOAgent: {unused_keys}")
        
        logger.info(
            f"Initialized MomentumPPOAgent with window={self.momentum_window}, "
            f"volatility_window={self.volatility_window}, "
            f"trend_window={self.trend_window}, "
            f"momentum_threshold={self.momentum_threshold}"
        )
    
    def _calculate_momentum_features(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate momentum-specific features from the state.
        
        Args:
            state: Raw state observation
            
        Returns:
            Momentum features as numpy array
        """
        # Ensure state is 2D
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        elif len(state.shape) == 3:
            # If state is (batch, window, features), use the last window
            state = state[:, -self.momentum_window:, :]
            
        # Extract close prices (assuming OHLCV format)
        if state.shape[-1] >= 4:  # Ensure we have enough features
            close_prices = state[..., 3]  # Get close prices
            
            # Calculate momentum indicators with safety checks
            if len(close_prices.shape) == 2:  # Batch case
                # Use available window size if momentum_window is too large
                available_window = min(close_prices.shape[1], self.momentum_window)
                if available_window < 2:  # Need at least 2 points for momentum
                    return np.zeros((close_prices.shape[0], 3), dtype=np.float32)
                
                momentum = close_prices[:, -1] / close_prices[:, -available_window] - 1
                volatility = np.std(close_prices[:, -available_window:], axis=1)
                trend = np.array([
                    np.polyfit(range(available_window), prices[-available_window:], 1)[0]
                    for prices in close_prices
                ])
            else:  # Single case
                available_window = min(len(close_prices), self.momentum_window)
                if available_window < 2:  # Need at least 2 points for momentum
                    return np.zeros(3, dtype=np.float32)
                
                momentum = close_prices[-1] / close_prices[-available_window] - 1
                volatility = np.std(close_prices[-available_window:])
                trend = np.polyfit(
                    range(available_window),
                    close_prices[-available_window:],
                    1
                )[0]
            
            # Handle NaN/inf values
            momentum = np.nan_to_num(momentum, nan=0.0, posinf=1.0, neginf=-1.0)
            volatility = np.nan_to_num(volatility, nan=0.0, posinf=1.0, neginf=0.0)
            trend = np.nan_to_num(trend, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return np.column_stack([momentum, volatility, trend]) if len(state.shape) > 2 else np.array([momentum, volatility, trend])
        else:
            # If we don't have enough features, return zero features
            shape = (state.shape[0], 3) if len(state.shape) > 2 else (3,)
            return np.zeros(shape, dtype=np.float32)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from PPO policy network with momentum feature augmentation.

        This method uses actual RL policy (not rule-based):
        1. Calculate momentum-specific features (momentum, volatility, trend)
        2. Augment state with these features
        3. Pass augmented state to parent PPO policy network

        Args:
            state: Current state observation (window_size, features) or (batch, window, features)
            deterministic: Whether to use deterministic action

        Returns:
            Action as numpy array with shape (1,)
        """
        # Convert DataFrame to numpy if needed
        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()

        # Calculate momentum features (momentum, volatility, trend)
        momentum_features = self._calculate_momentum_features(state)

        # Augment state with momentum features
        augmented_state = self._augment_state_with_features(state, momentum_features)

        # Use parent PPO agent's get_action with augmented state
        # This calls the actual RL policy network, not rule-based logic
        action = super().get_action(augmented_state, deterministic)

        return action

    def _augment_state_with_features(self, state: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Augment state with momentum features for the policy network.

        Args:
            state: Original state (window_size, n_features) or (batch, window, features)
            features: Momentum features (3,) or (batch, 3)

        Returns:
            Augmented state as flattened array matching observation space
        """
        # Flatten state
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            # Ensure features have batch dimension
            if len(features.shape) == 1:
                features = np.tile(features, (batch_size, 1))
            augmented = np.concatenate([flat_state, features], axis=1)
        else:  # (window, features)
            flat_state = state.reshape(-1)
            augmented = np.concatenate([flat_state, features])

        return augmented
    
    def train_step(self, state: np.ndarray, action: np.ndarray, 
                  reward: float, next_state: np.ndarray, 
                  done: bool, agent_id: str = None) -> Dict[str, float]:
        """
        Train the agent on a single state transition with momentum considerations.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            agent_id: Optional agent identifier for multi-agent scenarios
            
        Returns:
            Dictionary of training metrics
        """
        # Calculate momentum features for both states
        state_momentum = self._calculate_momentum_features(state)
        next_state_momentum = self._calculate_momentum_features(next_state)
        
        # Augment states with momentum features
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            flat_next_state = next_state.reshape(batch_size, -1)
            augmented_state = np.concatenate([flat_state, state_momentum], axis=1)
            augmented_next_state = np.concatenate([flat_next_state, next_state_momentum], axis=1)
        else:  # (window, features)
            flat_state = state.reshape(1, -1)
            flat_next_state = next_state.reshape(1, -1)
            augmented_state = np.concatenate([flat_state, state_momentum.reshape(1, -1)], axis=1)
            augmented_next_state = np.concatenate([flat_next_state, next_state_momentum.reshape(1, -1)], axis=1)
        
        # Add momentum-based reward modification
        momentum_reward = 0.0
        if len(state_momentum.shape) > 1:
            momentum = state_momentum[:, 0]
            momentum_reward = np.where(
                (momentum > self.momentum_threshold) & (action > 0) |
                (momentum < -self.momentum_threshold) & (action < 0),
                0.1, 0.0
            )
        else:
            momentum = state_momentum[0]
            action_value = action[0] if isinstance(action, np.ndarray) else action
            
            if (momentum > self.momentum_threshold and action_value > 0) or \
               (momentum < -self.momentum_threshold and action_value < 0):
                momentum_reward = 0.1  # Reward for following momentum
            
        modified_reward = reward + momentum_reward
        
        # Train with modified states and reward
        metrics = super().train_step(
            augmented_state.reshape(-1), action, modified_reward, augmented_next_state.reshape(-1), done
        )
        
        # If training failed, return empty metrics
        if metrics is None:
            return {
                "momentum_reward": float(momentum_reward),
                "momentum_value": float(momentum),
                "momentum_volatility": float(state_momentum[1] if len(state_momentum.shape) == 1 else state_momentum[0, 1]),
                "momentum_trend": float(state_momentum[2] if len(state_momentum.shape) == 1 else state_momentum[0, 2])
            }
        
        # Add momentum-specific metrics
        if len(state_momentum.shape) > 1:
            metrics.update({
                "momentum_reward": float(np.mean(momentum_reward)),
                "momentum_value": float(np.mean(state_momentum[:, 0])),
                "momentum_volatility": float(np.mean(state_momentum[:, 1])),
                "momentum_trend": float(np.mean(state_momentum[:, 2]))
            })
        else:
            metrics.update({
                "momentum_reward": float(momentum_reward),
                "momentum_value": float(state_momentum[0]),
                "momentum_volatility": float(state_momentum[1]),
                "momentum_trend": float(state_momentum[2])
            })
        
        return metrics
    
    def learn_from_shared_experience(self, shared_buffer: list) -> Dict[str, float]:
        """
        Learn from shared experience buffer with momentum strategy focus.
        
        Args:
            shared_buffer: List of experiences from all agents
            
        Returns:
            Dictionary of training metrics
        """
        if not shared_buffer:
            return {
                "shared_policy_loss": 0.0,
                "shared_value_loss": 0.0,
                "shared_entropy": 0.0
            }
            
        # Filter for experiences that align with momentum strategy
        relevant_exp = []
        
        for exp in shared_buffer:
            state = exp["state"]
            action = exp["action"]
            reward = exp["reward"]
            next_state = exp["next_state"]
            done = exp.get("done", False)
            
            # Ensure state has correct shape (window_size, features)
            if len(state.shape) == 1:
                window_size = self.original_obs_space.shape[0]
                n_features = self.original_obs_space.shape[1]
                state = state[:-self.n_momentum_features].reshape(window_size, n_features)
                next_state = next_state[:-self.n_momentum_features].reshape(window_size, n_features)
            
            # Calculate momentum for the state
            momentum_features = self._calculate_momentum_features(state)
            momentum = momentum_features[0] if len(momentum_features.shape) == 1 else momentum_features[:, 0]
            
            # Include experience if it follows momentum strategy
            if isinstance(momentum, np.ndarray):
                if np.any((momentum > self.momentum_threshold) & (action > 0) & (reward > 0)) or \
                   np.any((momentum < -self.momentum_threshold) & (action < 0) & (reward > 0)):
                    # Prepare augmented states
                    flat_state = state.reshape(1, -1)  # Make it 2D
                    flat_next_state = next_state.reshape(1, -1)  # Make it 2D
                    
                    # Normalize states with clipping
                    normalized_state = np.clip(self._normalize_state(flat_state.reshape(-1)), -10, 10)
                    normalized_next_state = np.clip(self._normalize_state(flat_next_state.reshape(-1)), -10, 10)
                    
                    # Normalize momentum features
                    normalized_momentum = np.clip(momentum_features / (np.abs(momentum_features).max() + 1e-8), -1, 1)
                    
                    # Combine with momentum features
                    augmented_state = np.concatenate([
                        normalized_state.reshape(-1),  # Flatten to 1D
                        normalized_momentum.reshape(-1)  # Flatten to 1D
                    ])
                    
                    next_momentum = self._calculate_momentum_features(next_state)
                    normalized_next_momentum = np.clip(next_momentum / (np.abs(next_momentum).max() + 1e-8), -1, 1)
                    
                    augmented_next_state = np.concatenate([
                        normalized_next_state.reshape(-1),  # Flatten to 1D
                        normalized_next_momentum.reshape(-1)  # Flatten to 1D
                    ])
                    
                    relevant_exp.append({
                        "state": augmented_state,
                        "action": action,
                        "reward": reward,
                        "next_state": augmented_next_state,
                        "done": done
                    })
            else:
                if (momentum > self.momentum_threshold and action > 0 and reward > 0) or \
                   (momentum < -self.momentum_threshold and action < 0 and reward > 0):
                    # Prepare augmented states
                    flat_state = state.reshape(1, -1)  # Make it 2D
                    flat_next_state = next_state.reshape(1, -1)  # Make it 2D
                    
                    # Normalize states with clipping
                    normalized_state = np.clip(self._normalize_state(flat_state.reshape(-1)), -10, 10)
                    normalized_next_state = np.clip(self._normalize_state(flat_next_state.reshape(-1)), -10, 10)
                    
                    # Normalize momentum features
                    normalized_momentum = np.clip(momentum_features / (np.abs(momentum_features).max() + 1e-8), -1, 1)
                    
                    # Combine with momentum features
                    augmented_state = np.concatenate([
                        normalized_state.reshape(-1),  # Flatten to 1D
                        normalized_momentum.reshape(-1)  # Flatten to 1D
                    ])
                    
                    next_momentum = self._calculate_momentum_features(next_state)
                    normalized_next_momentum = np.clip(next_momentum / (np.abs(next_momentum).max() + 1e-8), -1, 1)
                    
                    augmented_next_state = np.concatenate([
                        normalized_next_state.reshape(-1),  # Flatten to 1D
                        normalized_next_momentum.reshape(-1)  # Flatten to 1D
                    ])
                    
                    relevant_exp.append({
                        "state": augmented_state,
                        "action": action,
                        "reward": reward,
                        "next_state": augmented_next_state,
                        "done": done
                    })
        
        # Learn from filtered experiences
        if not relevant_exp:
            return {
                "shared_policy_loss": 0.0,
                "shared_value_loss": 0.0,
                "shared_entropy": 0.0
            }
            
        return super().learn_from_shared_experience(relevant_exp)
