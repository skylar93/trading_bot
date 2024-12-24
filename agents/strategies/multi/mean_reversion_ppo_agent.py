import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from agents.strategies.single.ppo_agent import PPOAgent
from gymnasium import spaces

logger = logging.getLogger(__name__)

class MeanReversionPPOAgent(PPOAgent):
    """
    Mean Reversion strategy PPO agent that specializes in trading when assets deviate from their mean.
    Inherits from base PPO agent but adds mean reversion specific features and logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mean Reversion PPO Agent.
        
        Args:
            config: Configuration dictionary containing:
                - All PPO parameters (learning_rate, gamma, etc.)
                - rsi_window: Window size for RSI calculation
                - bb_window: Window size for Bollinger Bands
                - bb_std: Number of standard deviations for Bollinger Bands
                - oversold_threshold: RSI threshold for oversold condition
                - overbought_threshold: RSI threshold for overbought condition
        """
        # Calculate augmented observation space
        base_obs_space = config["observation_space"]
        n_reversion_features = 3  # RSI, BB_upper_dist, BB_lower_dist
        
        if isinstance(base_obs_space, spaces.Box):
            if len(base_obs_space.shape) == 2:  # (window_size, features)
                total_features = base_obs_space.shape[0] * base_obs_space.shape[1] + n_reversion_features
                
                flat_obs_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_features,),
                    dtype=np.float32
                )
            else:
                raise ValueError("Observation space must be 2D (window_size, features)")
            
            self.original_obs_space = base_obs_space
            self.n_reversion_features = n_reversion_features
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
        
        # Mean reversion specific parameters
        self.rsi_window = config.get("rsi_window", 14)
        self.bb_window = config.get("bb_window", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.overbought_threshold = config.get("overbought_threshold", 70)
        self.strategy = "mean_reversion"  # Add strategy attribute
        
        logger.info(
            f"Initialized MeanReversionPPOAgent with RSI window={self.rsi_window}, "
            f"BB window={self.bb_window}, BB std={self.bb_std}"
        )
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """
        Calculate RSI using traditional approach with Wilder's smoothing.
        
        Args:
            prices: Array of price values
            
        Returns:
            RSI value between 0 and 100
        """
        if len(prices) < self.rsi_window + 1:
            return 50.0  # Return neutral RSI for insufficient data
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Get the window we'll use for calculation
        window_deltas = deltas[-self.rsi_window:]  # Use last rsi_window changes
        
        # Calculate gains and losses
        gains = np.maximum(window_deltas, 0)
        losses = -np.minimum(window_deltas, 0)
        
        # Calculate smoothed averages
        avg_gain = np.sum(gains) / self.rsi_window
        avg_loss = np.sum(losses) / self.rsi_window
        
        # Calculate final RSI
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(np.clip(rsi, 0.0, 100.0))
    
    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate Bollinger Bands for a price series."""
        if len(prices) < self.bb_window:
            return prices[-1], prices[-1]  # Return current price as both bands if insufficient data
            
        window_prices = prices[-self.bb_window:]  # Use the last window_size prices
        mean = np.mean(window_prices)
        std = np.std(window_prices)
        
        upper = mean + self.bb_std * std
        lower = mean - self.bb_std * std
        
        # Ensure bands don't cross
        upper = max(upper, mean)
        lower = min(lower, mean)
        
        return upper, lower
    
    def _calculate_reversion_features(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate mean reversion specific features from the state.
        
        Args:
            state: Raw state observation
            
        Returns:
            Mean reversion features as numpy array
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if state.shape[-1] >= 4:  # Ensure we have enough features
            if len(state.shape) == 3:  # (batch, window, features)
                close_prices = state[..., 3]  # Get close prices for all batches
            else:  # (window, features)
                close_prices = state[:, 3]  # Get close prices for single sample
            
            if len(close_prices.shape) == 2:  # Batch processing
                rsi_values = []
                bb_upper_values = []
                bb_lower_values = []
                
                for prices in close_prices:
                    rsi = self._calculate_rsi(prices)
                    bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
                    current_price = prices[-1]
                    
                    rsi_values.append(rsi)
                    bb_upper_values.append((bb_upper - current_price) / current_price)
                    bb_lower_values.append((current_price - bb_lower) / current_price)
                
                rsi = np.array(rsi_values)
                bb_upper_dist = np.array(bb_upper_values)
                bb_lower_dist = np.array(bb_lower_values)
            else:  # Single sample processing
                rsi = self._calculate_rsi(close_prices)
                bb_upper, bb_lower = self._calculate_bollinger_bands(close_prices)
                current_price = close_prices[-1]
                bb_upper_dist = (bb_upper - current_price) / current_price
                bb_lower_dist = (current_price - bb_lower) / current_price
            
            return np.column_stack([rsi, bb_upper_dist, bb_lower_dist]) if len(state.shape) > 2 else np.array([rsi, bb_upper_dist, bb_lower_dist])
        else:
            shape = (state.shape[0], 3) if len(state.shape) > 2 else (3,)
            return np.zeros(shape, dtype=np.float32)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from policy network with mean reversion considerations.
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action as numpy array
        """
        reversion_features = self._calculate_reversion_features(state)
        
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            augmented_state = np.concatenate([flat_state, reversion_features], axis=1)
        else:  # (window, features)
            flat_state = state.reshape(1, -1)
            augmented_state = np.concatenate([flat_state, reversion_features.reshape(1, -1)], axis=1)
        
        # Get base action from policy network
        base_action = super().get_action(augmented_state.reshape(-1), deterministic)
        
        # Apply mean reversion based action modification
        if len(reversion_features.shape) > 1:
            rsi = reversion_features[:, 0]
            bb_upper_dist = reversion_features[:, 1]
            bb_lower_dist = reversion_features[:, 2]
            
            # Calculate mean reversion signals with stronger bias
            oversold_signal = (rsi < self.oversold_threshold) & (bb_lower_dist < 0.01)  # Stricter BB condition
            overbought_signal = (rsi > self.overbought_threshold) & (bb_upper_dist < 0.01)  # Stricter BB condition
            
            # Calculate action bias based on signals with stronger mean reversion
            action_bias = np.zeros_like(base_action)
            action_bias[oversold_signal] = 1.0  # Strong buy bias
            action_bias[overbought_signal] = -1.0  # Strong sell bias
            
            # Calculate signal strength based on distance from thresholds
            oversold_strength = np.clip((self.oversold_threshold - rsi) / self.oversold_threshold, 0, 1)
            overbought_strength = np.clip((rsi - self.overbought_threshold) / (100 - self.overbought_threshold), 0, 1)
            
            # Combine signal strengths
            signal_strength = np.where(oversold_signal, oversold_strength,
                                     np.where(overbought_signal, overbought_strength, 0.1))
            
            # Blend base action with bias (more weight on bias when signal is strong)
            action = signal_strength * action_bias + (1 - signal_strength) * base_action
            
            # Add mean reversion scaling based on BB distances
            bb_signal = np.maximum(bb_upper_dist, bb_lower_dist)
            action *= (1.0 + bb_signal)  # Scale action by BB distance
            
            # Ensure action stays within bounds
            action = np.clip(action, -1.0, 1.0)
        else:
            rsi = reversion_features[0]
            bb_upper_dist = reversion_features[1]
            bb_lower_dist = reversion_features[2]
            
            # Calculate signal strength based on distance from thresholds
            if rsi < self.oversold_threshold and bb_lower_dist < 0.01:
                action_bias = 1.0  # Strong buy bias
                signal_strength = np.clip((self.oversold_threshold - rsi) / self.oversold_threshold, 0.5, 0.95)
            elif rsi > self.overbought_threshold and bb_upper_dist < 0.01:
                action_bias = -1.0  # Strong sell bias
                signal_strength = np.clip((rsi - self.overbought_threshold) / (100 - self.overbought_threshold), 0.5, 0.95)
            else:
                action_bias = 0.0
                signal_strength = 0.1
            
            # Blend base action with bias (more weight on bias when signal is strong)
            action = signal_strength * action_bias + (1 - signal_strength) * base_action[0]
            
            # Add mean reversion scaling based on BB distances
            bb_signal = max(bb_upper_dist, bb_lower_dist)
            action *= (1.0 + bb_signal)  # Scale action by BB distance
            
            # Ensure action stays within bounds and wrap in array
            action = np.array([np.clip(action, -1.0, 1.0)])
            
        return action
    
    def train_step(self, state: np.ndarray, action: np.ndarray, 
                  reward: float, next_state: np.ndarray, 
                  done: bool) -> Dict[str, float]:
        """
        Train the agent on a single state transition with mean reversion considerations.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        state_reversion = self._calculate_reversion_features(state)
        next_state_reversion = self._calculate_reversion_features(next_state)
        
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            flat_next_state = next_state.reshape(batch_size, -1)
            augmented_state = np.concatenate([flat_state, state_reversion], axis=1)
            augmented_next_state = np.concatenate([flat_next_state, next_state_reversion], axis=1)
        else:  # (window, features)
            flat_state = state.reshape(1, -1)
            flat_next_state = next_state.reshape(1, -1)
            augmented_state = np.concatenate([flat_state, state_reversion.reshape(1, -1)], axis=1)
            augmented_next_state = np.concatenate([flat_next_state, next_state_reversion.reshape(1, -1)], axis=1)
        
        # Add mean reversion based reward modification
        reversion_reward = 0.0
        if len(state_reversion.shape) > 1:
            rsi = state_reversion[:, 0]
            bb_upper_dist = state_reversion[:, 1]
            bb_lower_dist = state_reversion[:, 2]
            
            # Calculate price movement
            current_prices = state[..., 3]
            next_prices = next_state[..., 3]
            price_changes = (next_prices - current_prices) / current_prices
            
            # Calculate mean reversion signals
            oversold_signal = (rsi < self.oversold_threshold) & (bb_lower_dist < 0.02)
            overbought_signal = (rsi > self.overbought_threshold) & (bb_upper_dist < 0.02)
            
            # Calculate rewards for different scenarios
            buy_reward = np.where(
                oversold_signal & (action > 0.2) & (price_changes > 0),
                0.5 * np.abs(price_changes) + 0.2,  # Strong reward for profitable buy
                np.where(
                    oversold_signal & (action > 0),
                    0.1,  # Small reward for correct direction
                    0.0
                )
            )
            
            sell_reward = np.where(
                overbought_signal & (action < -0.2) & (price_changes < 0),
                0.5 * np.abs(price_changes) + 0.2,  # Strong reward for profitable sell
                np.where(
                    overbought_signal & (action < 0),
                    0.1,  # Small reward for correct direction
                    0.0
                )
            )
            
            # Combine rewards
            reversion_reward = buy_reward + sell_reward
        else:
            rsi = state_reversion[0]
            bb_upper_dist = state_reversion[1]
            bb_lower_dist = state_reversion[2]
            
            # Calculate price movement
            current_price = state[..., 3][-1]
            next_price = next_state[..., 3][-1]
            price_change = (next_price - current_price) / current_price
            
            # Calculate rewards for mean reversion trades
            if rsi < self.oversold_threshold:  # Oversold condition
                if action[0] > 0.2 and price_change > 0:  # Strong buy and price went up
                    reversion_reward = 0.5 * abs(price_change) + 0.2  # Strong reward for profitable buy
                elif action[0] > 0:  # Any buy action
                    reversion_reward = 0.1  # Small reward for correct direction
            elif rsi > self.overbought_threshold:  # Overbought condition
                if action[0] < -0.2 and price_change < 0:  # Strong sell and price went down
                    reversion_reward = 0.5 * abs(price_change) + 0.2  # Strong reward for profitable sell
                elif action[0] < 0:  # Any sell action
                    reversion_reward = 0.1  # Small reward for correct direction
            
        modified_reward = reward + reversion_reward
        
        metrics = super().train_step(
            augmented_state.reshape(-1), action, modified_reward, augmented_next_state.reshape(-1), done
        )
        
        if metrics is None:
            return {
                "reversion_reward": float(reversion_reward),
                "rsi_value": float(rsi if isinstance(rsi, (int, float)) else rsi[0]),
                "bb_upper_dist": float(bb_upper_dist if isinstance(bb_upper_dist, (int, float)) else bb_upper_dist[0]),
                "bb_lower_dist": float(bb_lower_dist if isinstance(bb_lower_dist, (int, float)) else bb_lower_dist[0])
            }
        
        if len(state_reversion.shape) > 1:
            metrics.update({
                "reversion_reward": float(np.mean(reversion_reward)),
                "rsi_value": float(np.mean(state_reversion[:, 0])),
                "bb_upper_dist": float(np.mean(state_reversion[:, 1])),
                "bb_lower_dist": float(np.mean(state_reversion[:, 2]))
            })
        else:
            metrics.update({
                "reversion_reward": float(reversion_reward),
                "rsi_value": float(state_reversion[0]),
                "bb_upper_dist": float(state_reversion[1]),
                "bb_lower_dist": float(state_reversion[2])
            })
        
        return metrics
    
    def learn_from_shared_experience(self, shared_buffer: list) -> Dict[str, float]:
        """
        Learn from shared experience buffer with mean reversion strategy focus.
        
        Args:
            shared_buffer: List of experience tuples from other agents
            
        Returns:
            Dictionary of training metrics
        """
        # Filter for experiences that align with mean reversion strategy
        filtered_buffer = []
        for exp in shared_buffer:
            state, action, reward, next_state, done = exp
            state_reversion = self._calculate_reversion_features(state)
            
            if len(state_reversion.shape) > 1:
                rsi = state_reversion[:, 0]
                bb_upper_dist = state_reversion[:, 1]
                bb_lower_dist = state_reversion[:, 2]
                
                # Only learn from experiences that match our strategy
                if ((rsi < self.oversold_threshold).any() and (bb_lower_dist < 0.01).any()) or \
                   ((rsi > self.overbought_threshold).any() and (bb_upper_dist < 0.01).any()):
                    filtered_buffer.append(exp)
            else:
                rsi = state_reversion[0]
                bb_upper_dist = state_reversion[1]
                bb_lower_dist = state_reversion[2]
                
                if (rsi < self.oversold_threshold and bb_lower_dist < 0.01) or \
                   (rsi > self.overbought_threshold and bb_upper_dist < 0.01):
                    filtered_buffer.append(exp)
        
        # Learn from filtered experiences
        if filtered_buffer:
            return super().learn_from_shared_experience(filtered_buffer)
        else:
            return {}
