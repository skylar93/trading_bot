import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from gymnasium import spaces
import logging
import torch
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiAgentTradingEnv(gym.Env):
    """Multi-agent cryptocurrency trading environment"""

    def __init__(
        self,
        data: pd.DataFrame,
        agent_configs: List[Dict],
        window_size: int = 60,
        trading_fee: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Multi-agent Trading Environment

        Args:
            data: DataFrame with OHLCV data
            agent_configs: List of agent configurations
            window_size: Size of observation window
            trading_fee: Trading fee as decimal
            device: Device to use for computations
        """
        super().__init__()

        self.data = data
        self.window_size = window_size
        self.trading_fee = trading_fee
        self.device = device

        # Initialize agents
        self.agents = [config["id"] for config in agent_configs]
        self.agent_configs = {config["id"]: config for config in agent_configs}

        # Set up observation and action spaces for each agent
        self.observation_spaces = {}
        self.action_spaces = {}

        for agent_id in self.agents:
            # Observation space: OHLCV data for window_size steps
            n_features = self._get_n_features(agent_id)
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size, n_features),  # Changed from flat_dim to (window_size, n_features)
                dtype=np.float32,
            )

            # Action space: continuous action between -1 (full sell) and 1 (full buy)
            self.action_spaces[agent_id] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )

        # Initialize shared experience buffer
        self.shared_buffer = []
        self.shared_buffer_size = 10000

        # Initialize agent-specific attributes
        self.reset()

        logger.info(
            f"Initialized MultiAgentTradingEnv with {len(self.agents)} agents"
        )

    def _get_n_features(self, agent_id: str) -> int:
        """Get number of features for an agent based on its strategy"""
        # Simplified: Use same number of features for all agents
        base_features = len(self.data.columns)  # OHLCV data
        return base_features  # Fixed number of features for all agents

    def _calculate_strategy_features(self, agent_id: str) -> np.ndarray:
        """Calculate strategy-specific features"""
        strategy = self.agent_configs[agent_id]["strategy"]

        if strategy == "momentum":
            return self._calculate_momentum_features(agent_id)
        elif strategy == "mean_reversion":
            return self._calculate_mean_reversion_features(agent_id)
        elif strategy == "market_making":
            return self._calculate_market_making_features(agent_id)
        return np.array([])

    def _calculate_momentum_features(self, agent_id: str) -> np.ndarray:
        """
        Calculate momentum strategy features.
        
        Recent Changes:
        - Added protection against NaN/Inf values
        - Added protection against division by zero
        - Added bounds checking for calculated features
        """
        config = self.agent_configs[agent_id]
        lookback = config.get("lookback", 20)

        # Ensure we have enough data for the lookback period
        if self.current_step < lookback:
            return np.zeros(3)  # Return zero features if not enough data

        # Get price window
        close_prices = self.data["$close"].values
        price_window = close_prices[self.current_step - lookback : self.current_step]
        
        # Handle potential NaN values in price window
        price_window = np.nan_to_num(price_window, nan=np.nanmean(price_window) if np.any(~np.isnan(price_window)) else 0.0)
        
        # Ensure we have valid prices
        if len(price_window) == 0 or np.all(price_window == 0):
            return np.zeros(3)

        # Calculate momentum with protection against division by zero
        if price_window[0] == 0:
            momentum = 0
        else:
            momentum = price_window[-1] / price_window[0] - 1

        # Calculate volatility
        volatility = np.std(price_window)

        # Calculate trend using matching x and y arrays
        x = np.arange(len(price_window))
        trend = np.polyfit(x, price_window, 1)[0] if len(price_window) > 1 else 0

        # Create feature array and sanitize
        features = np.array([momentum, volatility, trend])
        
        # Handle any NaN or Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values
        features[0] = np.clip(features[0], -10, 10)  # Clip momentum to reasonable range
        
        return features

    def _calculate_mean_reversion_features(self, agent_id: str) -> np.ndarray:
        """
        Calculate mean reversion strategy features.
        
        Recent Changes:
        - Added protection against NaN/Inf values
        - Added protection against division by zero
        - Added bounds checking for calculated features
        """
        config = self.agent_configs[agent_id]
        window = config.get("window", 50)

        # Ensure we have enough data
        if self.current_step < window:
            return np.zeros(4)  # Return zero features if not enough data

        # Get price window
        close_prices = self.data["$close"].values
        price_window = close_prices[self.current_step - window : self.current_step]
        
        # Handle potential NaN values in price window
        price_window = np.nan_to_num(price_window, nan=np.nanmean(price_window) if np.any(~np.isnan(price_window)) else 0.0)
        
        # Ensure we have valid prices
        if len(price_window) == 0 or np.all(price_window == 0):
            return np.zeros(4)

        # Calculate indicators
        mean = np.mean(price_window)
        std = np.std(price_window)
        current_price = price_window[-1]
        
        # Add protection against division by zero with epsilon
        eps = 1e-8
        
        # Calculate z-score with protection against zero std
        if std < eps:
            zscore = 0
        else:
            zscore = (current_price - mean) / std
            
        # Calculate mean distance with protection against zero price
        if abs(current_price) < eps:
            mean_dist = 0
        else:
            mean_dist = (current_price - mean) / current_price
        
        # Create feature array and sanitize
        features = np.array([mean, std, zscore, mean_dist])
        
        # Handle any NaN or Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values
        features[2] = np.clip(features[2], -10, 10)  # Clip zscore to reasonable range
        features[3] = np.clip(features[3], -1, 1)    # Clip mean_dist to [-1, 1]
        
        return features

    def _calculate_market_making_features(self, agent_id: str) -> np.ndarray:
        """
        Calculate market making strategy features.
        
        Recent Changes:
        - Added protection against NaN/Inf values
        - Added bounds checking for calculated features
        """
        config = self.agent_configs[agent_id]

        # Ensure we have enough data
        if self.current_step < 20:  # Need at least 20 steps for volatility
            return np.zeros(5)  # Return zero features if not enough data

        # Get current data
        high = self.data["$high"].values[self.current_step]
        low = self.data["$low"].values[self.current_step]
        close = self.data["$close"].values[self.current_step]
        volume = self.data["$volume"].values[self.current_step]
        
        # Handle potential NaN values
        high = np.nan_to_num(high, nan=close)
        low = np.nan_to_num(low, nan=close)
        close = np.nan_to_num(close, nan=0.0)
        volume = np.nan_to_num(volume, nan=0.0)
        
        # Calculate market making indicators
        spread = high - low
        
        # Calculate volatility with protection against NaN
        price_window = self.data["$close"].values[self.current_step - 20 : self.current_step]
        price_window = np.nan_to_num(price_window, nan=np.nanmean(price_window) if np.any(~np.isnan(price_window)) else 0.0)
        volatility = np.std(price_window)
        
        bid_strength = close - low
        ask_strength = high - close

        # Create feature array and sanitize
        features = np.array([spread, volume, volatility, bid_strength, ask_strength])
        
        # Handle any NaN or Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return features

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Get observation for an agent.
        
        Recent Changes:
        - Added protection against NaN/Inf values in observations
        - Added validation to ensure observation has correct shape
        """
        # Get base OHLCV data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Handle negative start index with padding
        if start_idx < 0:
            # Create a padded observation with zeros
            obs = np.zeros((self.window_size, len(self.data.columns)))
            # Fill in the available data
            available_data = self.data.iloc[:end_idx].values
            obs[-len(available_data):] = available_data
        else:
            # Normal case: slice the window
            obs = self.data.iloc[start_idx:end_idx].values
        
        # Verify we have exactly window_size rows
        if len(obs) != self.window_size:
            self.logger.warning(
                f"Observation shape mismatch: got {len(obs)} rows, expected {self.window_size}. Padding with zeros."
            )
            # Create correctly sized observation with zeros
            correct_obs = np.zeros((self.window_size, obs.shape[1] if len(obs) > 0 else len(self.data.columns)))
            # Fill in the available data
            if len(obs) > 0:
                correct_obs[-len(obs):] = obs
            obs = correct_obs
        
        # Handle NaN and Inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e10, neginf=-1e10)

        # Return 2D array with shape (window_size, n_features)
        return obs.astype(np.float32)

    def _add_to_shared_buffer(self, experience: Dict):
        """Add experience to shared buffer"""
        self.shared_buffer.append(experience)
        if len(self.shared_buffer) > self.shared_buffer_size:
            self.shared_buffer.pop(0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset environment

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observations, info)
        """
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balances = {
            agent_id: self.agent_configs[agent_id]["initial_balance"]
            for agent_id in self.agents
        }
        self.positions = {agent_id: 0.0 for agent_id in self.agents}
        self.trades = {agent_id: [] for agent_id in self.agents}

        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        info = {
            agent_id: {
                "balance": self.balances[agent_id],
                "position": self.positions[agent_id],
                "portfolio_value": self.balances[agent_id],
            }
            for agent_id in self.agents
        }

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """Execute one step in the environment"""
        # Get current price
        current_price = float(self.data.iloc[self.current_step]["$close"])

        # Initialize return values
        observations = {}
        rewards = {}
        dones = {}
        truncated = {}
        infos = {}

        # Process each agent's action
        for agent_id in self.agents:
            action = actions[agent_id][0]  # Extract scalar action
            config = self.agent_configs[agent_id]

            # Calculate transaction costs (with agent-specific multiplier)
            fee_multiplier = config.get("fee_multiplier", 1.0)
            trading_fee = self.trading_fee * fee_multiplier

            # Execute trade
            if abs(action) > 1e-5:  # Non-zero action
                if action > 0:  # Buy
                    max_shares = self.balances[agent_id] / (
                        current_price * (1 + trading_fee)
                    )
                    shares = max_shares * action
                    cost = shares * current_price * (1 + trading_fee)

                    if cost <= self.balances[agent_id]:
                        self.positions[agent_id] += shares
                        self.balances[agent_id] -= cost
                else:  # Sell
                    shares = self.positions[agent_id] * abs(action)
                    revenue = shares * current_price * (1 - trading_fee)

                    self.positions[agent_id] -= shares
                    self.balances[agent_id] += revenue

            # Calculate portfolio value
            portfolio_value = self.balances[agent_id] + (
                self.positions[agent_id] * current_price
            )

            # Calculate reward (strategy-specific)
            reward = self._calculate_reward(agent_id, portfolio_value)

            # Store experience in shared buffer
            experience = {
                "agent_id": agent_id,
                "state": self._get_observation(agent_id),
                "action": action,
                "reward": reward,
                "portfolio_value": portfolio_value,
            }
            self._add_to_shared_buffer(experience)

            # Update return values
            observations[agent_id] = self._get_observation(agent_id)
            rewards[agent_id] = reward
            dones[agent_id] = self.current_step >= len(self.data) - 1
            truncated[agent_id] = False
            infos[agent_id] = {
                "balance": self.balances[agent_id],
                "position": self.positions[agent_id],
                "portfolio_value": portfolio_value,
            }

        # Move to next step
        self.current_step += 1

        return observations, rewards, dones, truncated, infos

    def _calculate_reward(
        self, agent_id: str, portfolio_value: float
    ) -> float:
        """Calculate reward based on agent's strategy"""
        strategy = self.agent_configs[agent_id]["strategy"]

        if strategy == "momentum":
            # Reward based on trend following
            return (
                portfolio_value
                / self.agent_configs[agent_id]["initial_balance"]
                - 1
            ) * 100

        elif strategy == "mean_reversion":
            # Reward based on mean reversion opportunities
            window = self.agent_configs[agent_id].get("window", 50)
            mean = np.mean(
                self.data["$close"].values[
                    self.current_step - window : self.current_step
                ]
            )
            current = self.data["$close"].values[self.current_step]
            deviation = abs(current - mean) / mean
            return (
                (
                    portfolio_value
                    / self.agent_configs[agent_id]["initial_balance"]
                    - 1
                )
                * 100
                * (1 + deviation)
            )

        elif strategy == "market_making":
            # Reward based on spread capture
            spread = (
                self.data["$high"].values[self.current_step]
                - self.data["$low"].values[self.current_step]
            )
            volume = self.data["$volume"].values[self.current_step]
            return (
                (
                    portfolio_value
                    / self.agent_configs[agent_id]["initial_balance"]
                    - 1
                )
                * 100
                * (1 + spread * volume)
            )

        return (
            portfolio_value / self.agent_configs[agent_id]["initial_balance"]
            - 1
        ) * 100

    def render(self):
        """Render the environment"""
        pass  # Implement if visualization is needed

    def close(self):
        """Clean up resources"""
        pass

    @property
    def observation_space(self):
        """Combined observation space for all agents"""
        return self.observation_spaces

    @property
    def action_space(self):
        """Combined action space for all agents"""
        return self.action_spaces
