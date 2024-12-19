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
    
    def __init__(self, 
                 data: pd.DataFrame,
                 agent_configs: List[Dict],
                 window_size: int = 60,
                 trading_fee: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
        self.agents = [config['id'] for config in agent_configs]
        self.agent_configs = {config['id']: config for config in agent_configs}
        
        # Set up observation and action spaces for each agent
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for agent_id in self.agents:
            # Observation space: price data + technical indicators + position info
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size, self._get_n_features(agent_id)),
                dtype=np.float32
            )
            
            # Action space: continuous action between -1 (full sell) and 1 (full buy)
            self.action_spaces[agent_id] = spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32
            )
        
        # Initialize shared experience buffer
        self.shared_buffer = []
        self.shared_buffer_size = 10000
        
        # Initialize agent-specific attributes
        self.reset()
        
        logger.info(f"Initialized MultiAgentTradingEnv with {len(self.agents)} agents")
    
    def _get_n_features(self, agent_id: str) -> int:
        """Get number of features for an agent based on its strategy"""
        base_features = len(self.data.columns)  # OHLCV data
        
        strategy = self.agent_configs[agent_id]['strategy']
        if strategy == 'momentum':
            return base_features + 3  # Add momentum indicators
        elif strategy == 'mean_reversion':
            return base_features + 4  # Add mean reversion indicators
        elif strategy == 'market_making':
            return base_features + 5  # Add order book features
        return base_features
    
    def _calculate_strategy_features(self, agent_id: str) -> np.ndarray:
        """Calculate strategy-specific features"""
        strategy = self.agent_configs[agent_id]['strategy']
        
        if strategy == 'momentum':
            return self._calculate_momentum_features(agent_id)
        elif strategy == 'mean_reversion':
            return self._calculate_mean_reversion_features(agent_id)
        elif strategy == 'market_making':
            return self._calculate_market_making_features(agent_id)
        return np.array([])
    
    def _calculate_momentum_features(self, agent_id: str) -> np.ndarray:
        """Calculate momentum strategy features"""
        config = self.agent_configs[agent_id]
        lookback = config.get('lookback', 20)
        
        # Calculate momentum indicators
        close_prices = self.data['$close'].values
        momentum = close_prices[self.current_step] / close_prices[self.current_step - lookback] - 1
        volatility = np.std(close_prices[self.current_step-lookback:self.current_step])
        trend = np.polyfit(range(lookback), close_prices[self.current_step-lookback:self.current_step], 1)[0]
        
        return np.array([momentum, volatility, trend])
    
    def _calculate_mean_reversion_features(self, agent_id: str) -> np.ndarray:
        """Calculate mean reversion strategy features"""
        config = self.agent_configs[agent_id]
        window = config.get('window', 50)
        
        # Calculate mean reversion indicators
        close_prices = self.data['$close'].values[self.current_step-window:self.current_step]
        mean = np.mean(close_prices)
        std = np.std(close_prices)
        zscore = (close_prices[-1] - mean) / std
        mean_dist = (close_prices[-1] - mean) / close_prices[-1]
        
        return np.array([mean, std, zscore, mean_dist])
    
    def _calculate_market_making_features(self, agent_id: str) -> np.ndarray:
        """Calculate market making strategy features"""
        config = self.agent_configs[agent_id]
        
        # Calculate market making indicators
        spread = self.data['$high'].values[self.current_step] - self.data['$low'].values[self.current_step]
        volume = self.data['$volume'].values[self.current_step]
        volatility = np.std(self.data['$close'].values[self.current_step-20:self.current_step])
        bid_strength = (self.data['$close'].values[self.current_step] - self.data['$low'].values[self.current_step])
        ask_strength = (self.data['$high'].values[self.current_step] - self.data['$close'].values[self.current_step])
        
        return np.array([spread, volume, volatility, bid_strength, ask_strength])
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for an agent"""
        # Get base OHLCV data
        obs = self.data.iloc[self.current_step-self.window_size:self.current_step].values
        
        # Add strategy-specific features
        strategy_features = self._calculate_strategy_features(agent_id)
        if len(strategy_features) > 0:
            strategy_features = np.tile(strategy_features, (self.window_size, 1))
            obs = np.hstack([obs, strategy_features])
        
        return obs.astype(np.float32)
    
    def _add_to_shared_buffer(self, experience: Dict):
        """Add experience to shared buffer"""
        self.shared_buffer.append(experience)
        if len(self.shared_buffer) > self.shared_buffer_size:
            self.shared_buffer.pop(0)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
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
            agent_id: self.agent_configs[agent_id]['initial_balance']
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
                'balance': self.balances[agent_id],
                'position': self.positions[agent_id],
                'portfolio_value': self.balances[agent_id]
            }
            for agent_id in self.agents
        }
        
        return observations, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """Execute one step in the environment"""
        # Get current price
        current_price = float(self.data.iloc[self.current_step]['$close'])
        
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
            fee_multiplier = config.get('fee_multiplier', 1.0)
            trading_fee = self.trading_fee * fee_multiplier
            
            # Execute trade
            if abs(action) > 1e-5:  # Non-zero action
                if action > 0:  # Buy
                    max_shares = self.balances[agent_id] / (current_price * (1 + trading_fee))
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
            portfolio_value = self.balances[agent_id] + (self.positions[agent_id] * current_price)
            
            # Calculate reward (strategy-specific)
            reward = self._calculate_reward(agent_id, portfolio_value)
            
            # Store experience in shared buffer
            experience = {
                'agent_id': agent_id,
                'state': self._get_observation(agent_id),
                'action': action,
                'reward': reward,
                'portfolio_value': portfolio_value
            }
            self._add_to_shared_buffer(experience)
            
            # Update return values
            observations[agent_id] = self._get_observation(agent_id)
            rewards[agent_id] = reward
            dones[agent_id] = self.current_step >= len(self.data) - 1
            truncated[agent_id] = False
            infos[agent_id] = {
                'balance': self.balances[agent_id],
                'position': self.positions[agent_id],
                'portfolio_value': portfolio_value
            }
        
        # Move to next step
        self.current_step += 1
        
        return observations, rewards, dones, truncated, infos
    
    def _calculate_reward(self, agent_id: str, portfolio_value: float) -> float:
        """Calculate reward based on agent's strategy"""
        strategy = self.agent_configs[agent_id]['strategy']
        
        if strategy == 'momentum':
            # Reward based on trend following
            return (portfolio_value / self.agent_configs[agent_id]['initial_balance'] - 1) * 100
        
        elif strategy == 'mean_reversion':
            # Reward based on mean reversion opportunities
            window = self.agent_configs[agent_id].get('window', 50)
            mean = np.mean(self.data['$close'].values[self.current_step-window:self.current_step])
            current = self.data['$close'].values[self.current_step]
            deviation = abs(current - mean) / mean
            return (portfolio_value / self.agent_configs[agent_id]['initial_balance'] - 1) * 100 * (1 + deviation)
        
        elif strategy == 'market_making':
            # Reward based on spread capture
            spread = self.data['$high'].values[self.current_step] - self.data['$low'].values[self.current_step]
            volume = self.data['$volume'].values[self.current_step]
            return (portfolio_value / self.agent_configs[agent_id]['initial_balance'] - 1) * 100 * (1 + spread * volume)
        
        return (portfolio_value / self.agent_configs[agent_id]['initial_balance'] - 1) * 100
    
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