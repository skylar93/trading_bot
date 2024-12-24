import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from agents.strategies.single.ppo_agent import PPOAgent
from gymnasium import spaces

logger = logging.getLogger(__name__)

class MomentumPPOAgent(PPOAgent):
    """
    Momentum strategy PPO agent that specializes in trend-following.
    Inherits from base PPO agent but adds momentum-specific features and logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Momentum PPO Agent.
        
        Args:
            config: Configuration dictionary containing:
                - All PPO parameters (learning_rate, gamma, etc.)
                - momentum_window: Window size for momentum calculation
                - momentum_threshold: Threshold for momentum signals
        """
        # Calculate augmented observation space
        base_obs_space = config["observation_space"]
        n_momentum_features = 3  # momentum, volatility, trend
        
        if isinstance(base_obs_space, spaces.Box):
            # If observation space is Box, extend the feature dimension
            if len(base_obs_space.shape) == 2:  # (window_size, features)
                # Calculate total input size for the network
                # Original features: window_size * features
                # Momentum features: 3 (momentum, volatility, trend)
                total_features = base_obs_space.shape[0] * base_obs_space.shape[1] + n_momentum_features
                
                # Create flattened observation space
                flat_obs_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_features,),
                    dtype=np.float32
                )
            else:
                raise ValueError("Observation space must be 2D (window_size, features)")
            
            # Store original space for reference
            self.original_obs_space = base_obs_space
            self.n_momentum_features = n_momentum_features
        else:
            raise ValueError("Observation space must be Box")
        
        # Initialize base PPO agent with flattened observation space
        super().__init__(
            observation_space=flat_obs_space,
            action_space=config["action_space"],
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_epsilon=config.get("clip_epsilon", 0.2),
            c1=config.get("c1", 1.0),
            c2=config.get("c2", 0.01),
            c3=config.get("c3", 0.5),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            target_kl=config.get("target_kl", 0.015),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Momentum-specific parameters
        self.momentum_window = config.get("momentum_window", 20)
        self.momentum_threshold = config.get("momentum_threshold", 0.0)
        self.strategy = "momentum"  # Add strategy attribute
        
        logger.info(
            f"Initialized MomentumPPOAgent with window={self.momentum_window}, "
            f"threshold={self.momentum_threshold}"
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
            
            # Calculate momentum indicators
            if len(close_prices.shape) == 2:
                momentum = close_prices[:, -1] / close_prices[:, -self.momentum_window] - 1
                volatility = np.std(close_prices[:, -self.momentum_window:], axis=1)
                trend = np.array([
                    np.polyfit(range(self.momentum_window), prices[-self.momentum_window:], 1)[0]
                    for prices in close_prices
                ])
            else:
                momentum = close_prices[-1] / close_prices[-self.momentum_window] - 1
                volatility = np.std(close_prices[-self.momentum_window:])
                trend = np.polyfit(
                    range(self.momentum_window),
                    close_prices[-self.momentum_window:],
                    1
                )[0]
            
            return np.column_stack([momentum, volatility, trend]) if len(state.shape) > 2 else np.array([momentum, volatility, trend])
        else:
            # If we don't have enough features, return zero features
            shape = (state.shape[0], 3) if len(state.shape) > 2 else (3,)
            return np.zeros(shape, dtype=np.float32)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from policy network with momentum considerations."""
        momentum_features = self._calculate_momentum_features(state)
        
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            augmented_state = np.concatenate([flat_state, momentum_features], axis=1)
            
            # Calculate recent price changes
            price_changes = np.diff(state[:, -10:, 3], axis=1)
            noise_level = np.std(price_changes, axis=1)
            trend_persistence = np.abs(np.sum(np.sign(price_changes), axis=1)) / price_changes.shape[1]
            
            # Calculate oscillation metrics
            price_history = state[:, -10:, 3]
            oscillation_amplitude = (np.max(price_history, axis=1) - np.min(price_history, axis=1)) / 2
            is_oscillating = (oscillation_amplitude >= 0.08) & (oscillation_amplitude <= 0.12)
            
            # Calculate trend strength using multiple timeframes
            short_ma = np.mean(state[:, -5:, 3], axis=1)
            mid_ma = np.mean(state[:, -10:, 3], axis=1)
            long_ma = np.mean(state[:, -20:, 3], axis=1)
            
            # Check if market is ranging
            is_ranging = (
                (np.abs(short_ma - mid_ma) < 0.01) &  # Further tightened thresholds
                (np.abs(mid_ma - long_ma) < 0.015) &  # Further tightened thresholds
                (np.abs(short_ma - long_ma) < 0.02)   # Further tightened thresholds
            )
            
            # Calculate momentum signal
            momentum = np.sum(price_changes, axis=1)
            momentum_signal = np.mean(price_changes, axis=1)
            
            # Calculate momentum strength
            momentum_strength = np.abs(momentum)
            strong_momentum = momentum_strength >= 0.02  # Further lowered threshold
            
            # Calculate volatility filter with dynamic threshold
            volatility = np.std(price_changes, axis=1) / np.mean(price_history, axis=1)
            volatility_threshold = np.where(
                oscillation_amplitude >= 0.1,
                0.015,  # Lower threshold for larger oscillations
                0.01    # Lower threshold for smaller oscillations
            )
            volatility_filter = volatility >= volatility_threshold
            
            # Calculate trend-based position scaling
            trend_scale = np.where(
                trend_persistence > 0.7,
                2.5,  # Very aggressive in strong trend
                np.where(
                    trend_persistence > 0.5,
                    1.5,  # Aggressive in medium trend
                    0.0   # No position in weak trend
                )
            )
            
            # Calculate dynamic threshold based on oscillation amplitude
            threshold = np.where(
                oscillation_amplitude >= 0.1,
                0.02,  # Lower threshold for larger oscillations
                0.01   # Lower threshold for smaller oscillations
            )
            
            # Take positions based on market conditions
            position_scale = np.where(
                is_oscillating | is_ranging | (trend_persistence < 0.4),  # Stricter trend requirement
                0.0,  # Absolutely no position in ranging/oscillating markets
                np.where(
                    strong_momentum & volatility_filter & (trend_persistence > 0.6),
                    trend_scale * 2.0,  # Even more aggressive in strong trending markets
                    np.where(
                        strong_momentum & (trend_persistence > 0.5),
                        trend_scale * 1.5,  # More aggressive in trending markets
                        0.0  # No position otherwise
                    )
                )
            )
            
            # Scale position size based on momentum strength
            position_scale = position_scale * np.clip(momentum_strength, 0.5, 2.5)  # More aggressive scaling
            
            # Combine momentum signal with position scale
            action = momentum_signal * position_scale
            
            # Ensure action stays within bounds
            action = np.clip(action, -2.5, 2.5)  # Allow larger positions in trending markets
        else:
            flat_state = state.reshape(1, -1)
            augmented_state = np.concatenate([flat_state, momentum_features.reshape(1, -1)], axis=1)
            
            # Calculate recent price changes
            price_changes = np.diff(state[-10:, 3])
            noise_level = np.std(price_changes)
            trend_persistence = abs(np.sum(np.sign(price_changes))) / len(price_changes)
            
            # Calculate oscillation metrics
            price_history = state[-10:, 3]
            oscillation_amplitude = (np.max(price_history) - np.min(price_history)) / 2
            is_oscillating = (oscillation_amplitude >= 0.08) and (oscillation_amplitude <= 0.12)
            
            # Calculate trend strength using multiple timeframes
            short_ma = np.mean(state[-5:, 3])
            mid_ma = np.mean(state[-10:, 3])
            long_ma = np.mean(state[-20:, 3])
            
            # Check if market is ranging
            is_ranging = (
                (abs(short_ma - mid_ma) < 0.01) and  # Further tightened thresholds
                (abs(mid_ma - long_ma) < 0.015) and  # Further tightened thresholds
                (abs(short_ma - long_ma) < 0.02)     # Further tightened thresholds
            )
            
            # Calculate momentum signal
            momentum = np.sum(price_changes)
            momentum_signal = np.mean(price_changes)
            
            # Calculate momentum strength
            momentum_strength = abs(momentum)
            strong_momentum = momentum_strength >= 0.02  # Further lowered threshold
            
            # Calculate volatility filter with dynamic threshold
            volatility = np.std(price_changes) / np.mean(price_history)
            volatility_threshold = 0.015 if oscillation_amplitude >= 0.1 else 0.01
            volatility_filter = volatility >= volatility_threshold
            
            # Calculate trend-based position scaling
            if trend_persistence > 0.7:
                trend_scale = 2.5  # Very aggressive in strong trend
            elif trend_persistence > 0.5:
                trend_scale = 1.5  # Aggressive in medium trend
            else:
                trend_scale = 0.0  # No position in weak trend
            
            # Calculate dynamic threshold based on oscillation amplitude
            threshold = 0.02 if oscillation_amplitude >= 0.1 else 0.01
            
            # Take positions based on market conditions
            if is_oscillating or is_ranging or trend_persistence < 0.4:  # Stricter trend requirement
                position_scale = 0.0  # Absolutely no position in ranging/oscillating markets
            elif strong_momentum and volatility_filter and trend_persistence > 0.6:
                position_scale = trend_scale * 2.0  # Even more aggressive in strong trending markets
            elif strong_momentum and trend_persistence > 0.5:
                position_scale = trend_scale * 1.5  # More aggressive in trending markets
            else:
                position_scale = 0.0  # No position otherwise
            
            # Scale position size based on momentum strength
            position_scale = position_scale * np.clip(momentum_strength, 0.5, 2.5)  # More aggressive scaling
            
            # Combine momentum signal with position scale
            action = momentum_signal * position_scale
            
            # Ensure action stays within bounds
            action = np.clip(action, -2.5, 2.5)  # Allow larger positions in trending markets
            action = np.array([action], dtype=np.float32)
        
        return action
    
    def train_step(self, state: np.ndarray, action: np.ndarray, 
                  reward: float, next_state: np.ndarray, 
                  done: bool) -> Dict[str, float]:
        """
        Train the agent on a single state transition with momentum considerations.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
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
