import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from agents.strategies.single.ppo_agent import PPOAgent
from gymnasium import spaces
import pandas as pd

logger = logging.getLogger(__name__)

class MeanReversionPPOAgent(PPOAgent):
    """
    Mean Reversion strategy PPO agent that specializes in trading when assets deviate from their mean.
    Inherits from base PPO agent but adds mean reversion specific features and logic.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        rsi_window: int = 14,
        bb_window: int = 20,
        bb_std: float = 2.0,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
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
        Initialize Mean Reversion PPO Agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            rsi_window: Window size for RSI calculation
            bb_window: Window size for Bollinger Bands
            bb_std: Number of standard deviations for Bollinger Bands
            oversold_threshold: RSI threshold for oversold condition
            overbought_threshold: RSI threshold for overbought condition
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
        # Calculate total features (base features + mean reversion indicators)
        n_reversion_features = 3  # RSI, BB_upper_dist, BB_lower_dist
        
        if isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) == 2:  # (window_size, features)
                total_features = observation_space.shape[0] * observation_space.shape[1] + n_reversion_features
                
                flat_obs_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_features,),
                    dtype=np.float32
                )
            else:
                raise ValueError("Observation space must be 2D (window_size, features)")
            
            self.original_obs_space = observation_space
            self.n_reversion_features = n_reversion_features
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
        
        # Mean reversion specific parameters
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.strategy = "mean_reversion"  # Add strategy attribute
        self.EPS = 1e-8  # Add epsilon constant for safe division
        
        # Log unused config keys
        unused_keys = [key for key in kwargs.keys() if key not in self.__init__.__code__.co_varnames]
        if unused_keys:
            self.logger.warning(f"Ignoring unused config keys in MeanReversionPPOAgent: {unused_keys}")
        
        logger.info(
            f"Initialized MeanReversionPPOAgent with RSI window={self.rsi_window}, "
            f"BB window={self.bb_window}, BB std={self.bb_std}"
        )
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """
        Calculate RSI using traditional approach with Wilder's smoothing.
        Handles NaN values safely.
        
        Args:
            prices: Array of price values
            
        Returns:
            RSI value between 0 and 100
        """
        # Handle NaN values in prices
        prices = np.nan_to_num(prices, nan=np.nanmean(prices) if np.any(~np.isnan(prices)) else 0.0)
        
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
        
        # Ensure result is valid
        rsi = np.nan_to_num(rsi, nan=50.0)
        return float(np.clip(rsi, 0.0, 100.0))
    
    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Bollinger Bands for a price series.
        Handles NaN values safely.
        """
        # Handle NaN values in prices
        prices = np.nan_to_num(prices, nan=np.nanmean(prices) if np.any(~np.isnan(prices)) else 0.0)
        
        if len(prices) < self.bb_window:
            return prices[-1], prices[-1]  # Return current price as both bands if insufficient data
            
        window_prices = prices[-self.bb_window:]  # Use the last window_size prices
        mean = np.mean(window_prices)
        std = np.std(window_prices)
        
        upper = mean + self.bb_std * std
        lower = mean - self.bb_std * std
        
        # Ensure bands don't cross and handle any NaN results
        upper = np.nan_to_num(upper, nan=mean)
        lower = np.nan_to_num(lower, nan=mean)
        upper = max(upper, mean)
        lower = min(lower, mean)
        
        return upper, lower
    
    def _calculate_reversion_features(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate mean reversion specific features from the state.
        Handles NaN values safely.
        
        Recent Changes:
        - Improved handling of NaN/Inf values with better bounds checking
        - Added protection against division by zero
        - Added clipping for extreme values
        
        Args:
            state: Raw state observation
            
        Returns:
            Mean reversion features as numpy array
        """
        # Handle NaN values in state with more specific bounds
        state = np.nan_to_num(state, nan=0.0, posinf=1e10, neginf=-1e10)
        
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
                    # Ensure prices are valid
                    prices = np.nan_to_num(prices, nan=np.nanmean(prices) if np.any(~np.isnan(prices)) else 0.0)
                    
                    # Calculate RSI and Bollinger Bands
                    rsi = self._calculate_rsi(prices)
                    bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
                    current_price = prices[-1] if len(prices) > 0 else 0.0
                    
                    # Avoid division by zero
                    if current_price < self.EPS:
                        current_price = self.EPS
                    
                    # Calculate distances with protection
                    bb_upper_dist = (bb_upper - current_price) / current_price
                    bb_lower_dist = (current_price - bb_lower) / current_price
                    
                    # Clip to reasonable ranges
                    bb_upper_dist = np.clip(bb_upper_dist, -1.0, 1.0)
                    bb_lower_dist = np.clip(bb_lower_dist, -1.0, 1.0)
                    
                    rsi_values.append(rsi)
                    bb_upper_values.append(bb_upper_dist)
                    bb_lower_values.append(bb_lower_dist)
                
                rsi = np.array(rsi_values)
                bb_upper_dist = np.array(bb_upper_values)
                bb_lower_dist = np.array(bb_lower_values)
            else:  # Single sample processing
                # Ensure prices are valid
                prices = np.nan_to_num(close_prices, nan=np.nanmean(close_prices) if np.any(~np.isnan(close_prices)) else 0.0)
                
                # Calculate RSI and Bollinger Bands
                rsi = self._calculate_rsi(prices)
                bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
                current_price = prices[-1] if len(prices) > 0 else 0.0
                
                # Avoid division by zero
                if current_price < self.EPS:
                    current_price = self.EPS
                
                # Calculate distances with protection
                bb_upper_dist = (bb_upper - current_price) / current_price
                bb_lower_dist = (current_price - bb_lower) / current_price
                
                # Clip to reasonable ranges
                bb_upper_dist = np.clip(bb_upper_dist, -1.0, 1.0)
                bb_lower_dist = np.clip(bb_lower_dist, -1.0, 1.0)
            
            # Create feature array and handle any remaining NaN/Inf values
            features = np.column_stack([rsi, bb_upper_dist, bb_lower_dist]) if len(state.shape) > 2 else np.array([rsi, bb_upper_dist, bb_lower_dist])
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
        else:
            shape = (state.shape[0], 3) if len(state.shape) > 2 else (3,)
            return np.zeros(shape, dtype=np.float32)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from policy network with mean reversion strategy.
        
        Handles different input shapes:
        - 2D: (window_size, features)
        - 3D: (batch_size, window_size, features)
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action
            
        Returns:
            Action as numpy array with shape (1,)
        """
        # Convert DataFrame to numpy if needed
        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()
        
        # Calculate mean reversion features
        mean_rev_features = self._calculate_reversion_features(state)
        
        # Calculate trend strength
        if len(state.shape) == 3:  # (batch_size, window_size, features)
            close_prices = state[..., 3]  # Get close prices for all batches
            if close_prices.shape[1] < 10:
                trend_strength = np.zeros(close_prices.shape[0])
            else:
                denominator = close_prices[:, -10]
                safe_mask = (denominator != 0) & ~np.isnan(denominator) & ~np.isinf(denominator)
                trend_strength = np.zeros(close_prices.shape[0])
                trend_strength[safe_mask] = (close_prices[safe_mask, -1] - denominator[safe_mask]) / denominator[safe_mask]
                trend_strength = np.nan_to_num(trend_strength, nan=0.0, posinf=0.0, neginf=0.0)
        else:  # (window_size, features)
            close_prices = state[:, 3]  # Get close prices
            if len(close_prices) < 10:
                trend_strength = 0.0
            else:
                denominator = close_prices[-10]
                if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                    trend_strength = 0.0
                else:
                    trend_strength = (close_prices[-1] - denominator) / denominator
                    trend_strength = np.nan_to_num(trend_strength, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract RSI and Bollinger Bands features
        rsi = mean_rev_features[0] if len(mean_rev_features.shape) == 1 else mean_rev_features[:, 0]
        bb_upper_dist = mean_rev_features[1] if len(mean_rev_features.shape) == 1 else mean_rev_features[:, 1]
        bb_lower_dist = mean_rev_features[2] if len(mean_rev_features.shape) == 1 else mean_rev_features[:, 2]
        
        # Generate action based on trend and mean reversion signals
        if isinstance(trend_strength, np.ndarray):  # Batch case
            # Take opposite action to trend (mean reversion)
            action = np.where(
                trend_strength > 0.02,  # Strong uptrend
                -1.0,  # Strong sell signal (expect reversal)
                np.where(
                    trend_strength < -0.02,  # Strong downtrend
                    1.0,  # Strong buy signal (expect reversal)
                    0.0  # No clear trend
                )
            )
            
            # Further strengthen based on RSI
            action = np.where(
                rsi > self.overbought_threshold,
                -1.0,  # Strong sell when overbought
                np.where(
                    rsi < self.oversold_threshold,
                    1.0,  # Strong buy when oversold
                    action
                )
            )
            
        else:  # Single case
            # Take opposite action to trend (mean reversion)
            if trend_strength > 0.02:  # Strong uptrend
                action = -1.0  # Strong sell signal (expect reversal)
            elif trend_strength < -0.02:  # Strong downtrend
                action = 1.0  # Strong buy signal (expect reversal)
            else:
                action = 0.0  # No clear trend
            
            # Further strengthen based on RSI
            if rsi > self.overbought_threshold:
                action = -1.0  # Strong sell when overbought
            elif rsi < self.oversold_threshold:
                action = 1.0  # Strong buy when oversold
        
        # Handle any NaN/inf values and clip
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        
        # Ensure action is a numpy array with shape (1,)
        if isinstance(action, (float, np.float32, np.float64)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.shape != (1,):
            action = action.reshape(1)
        
        return action
    
    def train_step(self, state: np.ndarray, action: np.ndarray, 
                  reward: float, next_state: np.ndarray, 
                  done: bool, agent_id: str = None) -> Dict[str, float]:
        """
        Train the agent on a single state transition with mean reversion considerations.
        
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
            
            # Reward for correctly predicting reversals
            reversion_reward = np.where(
                (rsi > self.overbought_threshold) & (action < 0) |  # Selling when overbought
                (rsi < self.oversold_threshold) & (action > 0),     # Buying when oversold
                0.1, 0.0
            )
        else:
            rsi = state_reversion[0]
            bb_upper_dist = state_reversion[1]
            bb_lower_dist = state_reversion[2]
            action_value = action[0] if isinstance(action, np.ndarray) else action
            
            # Reward for correctly predicting reversals
            if (rsi > self.overbought_threshold and action_value < 0) or \
               (rsi < self.oversold_threshold and action_value > 0):
                reversion_reward = 0.1
        
        modified_reward = reward + reversion_reward
        
        # Train with modified states and reward
        metrics = super().train_step(
            augmented_state.reshape(-1), action, modified_reward, augmented_next_state.reshape(-1), done
        )
        
        # If training failed, return empty metrics
        if metrics is None:
            return {
                "reversion_reward": float(reversion_reward),
                "rsi_value": float(rsi),
                "bb_upper_dist": float(bb_upper_dist),
                "bb_lower_dist": float(bb_lower_dist)
            }
        
        # Add reversion-specific metrics
        if len(state_reversion.shape) > 1:
            metrics.update({
                "reversion_reward": float(np.mean(reversion_reward)),
                "rsi_value": float(np.mean(rsi)),
                "bb_upper_dist": float(np.mean(bb_upper_dist)),
                "bb_lower_dist": float(np.mean(bb_lower_dist))
            })
        else:
            metrics.update({
                "reversion_reward": float(reversion_reward),
                "rsi_value": float(rsi),
                "bb_upper_dist": float(bb_upper_dist),
                "bb_lower_dist": float(bb_lower_dist)
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
