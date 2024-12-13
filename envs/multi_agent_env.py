import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Box

class MultiAgentTradingEnv(gym.Env):
    """Multi-agent trading environment where multiple agents can interact with the market"""
    
    def __init__(self, df: pd.DataFrame, 
                 agent_configs: List[Dict],
                 window_size: int = 60,
                 global_trading_fee: float = 0.001):
        super(MultiAgentTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.global_trading_fee = global_trading_fee
        
        # Initialize agents
        self.agents = {}
        self.agent_configs = agent_configs
        
        # Set up observation and action spaces for each agent
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for agent_config in agent_configs:
            agent_id = agent_config['id']
            self.agents[agent_id] = {
                'balance': agent_config.get('initial_balance', 10000.0),
                'position': 0,
                'strategy': agent_config.get('strategy', 'default'),
                'fee_multiplier': agent_config.get('fee_multiplier', 1.0),  # Some agents might have different fees
            }
            
            # Each agent has the same observation and action space
            n_features = 10  # price, volume, position, balance, market_state, etc.
            self.observation_spaces[agent_id] = Box(
                low=-np.inf, high=np.inf, 
                shape=(window_size, n_features), 
                dtype=np.float32
            )
            
            self.action_spaces[agent_id] = Box(
                low=-1, high=1, 
                shape=(1,), 
                dtype=np.float32
            )
        
        self.observation_space = DictSpace({
            agent_id: space for agent_id, space in self.observation_spaces.items()
        })
        self.action_space = DictSpace({
            agent_id: space for agent_id, space in self.action_spaces.items()
        })
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        
        # Reset each agent
        for agent_id, agent_data in self.agents.items():
            agent_config = next(config for config in self.agent_configs if config['id'] == agent_id)
            agent_data['balance'] = agent_config.get('initial_balance', 10000.0)
            agent_data['position'] = 0
            agent_data['trades'] = []
        
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents.keys()
        }
        
        return observations, {}
    
    def step(self, actions: Dict[str, float]):
        """Execute one step in the environment for all agents"""
        # Get current price
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute trades for each agent
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            fee_multiplier = agent['fee_multiplier']
            effective_fee = self.global_trading_fee * fee_multiplier
            
            if action > 0:  # Buy
                shares_to_buy = (agent['balance'] * abs(action)) / current_price
                cost = shares_to_buy * current_price * (1 + effective_fee)
                if cost <= agent['balance']:
                    agent['position'] += shares_to_buy
                    agent['balance'] -= cost
                    agent['trades'].append(('buy', shares_to_buy, current_price))
            
            elif action < 0:  # Sell
                shares_to_sell = agent['position'] * abs(action)
                revenue = shares_to_sell * current_price * (1 - effective_fee)
                agent['position'] -= shares_to_sell
                agent['balance'] += revenue
                agent['trades'].append(('sell', shares_to_sell, current_price))
        
        # Move to next step
        self.current_step += 1
        
        # Calculate rewards and get observations for each agent
        observations = {}
        rewards = {}
        infos = {}
        
        for agent_id, agent in self.agents.items():
            # Calculate portfolio value and reward
            portfolio_value = agent['balance'] + (agent['position'] * current_price)
            prev_portfolio_value = agent['balance'] + (agent['position'] * self.df.iloc[self.current_step-1]['close'])
            reward = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
            observations[agent_id] = self._get_observation(agent_id)
            rewards[agent_id] = reward
            infos[agent_id] = {
                'portfolio_value': portfolio_value,
                'position': agent['position'],
                'balance': agent['balance']
            }
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        dones = {agent_id: done for agent_id in self.agents.keys()}
        
        return observations, rewards, dones, False, infos
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Construct observation for a specific agent"""
        # Get the price data for the current window
        df_window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # Normalize the data
        price_mean = df_window['close'].mean()
        price_std = df_window['close'].std()
        volume_mean = df_window['volume'].mean()
        volume_std = df_window['volume'].std()
        
        # Get agent-specific data
        agent = self.agents[agent_id]
        
        # Construct features
        obs = np.array([
            (df_window['open'] - price_mean) / price_std,
            (df_window['high'] - price_mean) / price_std,
            (df_window['low'] - price_mean) / price_std,
            (df_window['close'] - price_mean) / price_std,
            (df_window['volume'] - volume_mean) / volume_std,
            df_window['close'].pct_change().fillna(0),  # Returns
            df_window['volume'].pct_change().fillna(0),  # Volume change
            [agent['position']] * len(df_window),  # Current position
            [agent['balance'] / self.agent_configs[0]['initial_balance']] * len(df_window),  # Normalized balance
            [(agent['balance'] + agent['position'] * df_window['close'].iloc[-1]) / 
             self.agent_configs[0]['initial_balance']] * len(df_window)  # Normalized portfolio value
        ]).T
        
        return obs.astype(np.float32)