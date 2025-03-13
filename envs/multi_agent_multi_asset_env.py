"""
Multi-Agent Multi-Asset Trading Environment for Reinforcement Learning.

This module provides a unified environment that supports both multiple agents and
multiple assets simultaneously. It combines the functionality of MultiAssetTradingEnv
and MultiAgentTradingEnv to create a comprehensive trading simulation environment.

Features:
- Supports multiple agents trading multiple assets
- Flexible agent-to-asset assignment (all-to-all or specific assignments)
- Shared capital pool option for collaborative multi-agent scenarios
- Expanded observation space with asset-specific and agent-specific components
- Rich information dictionary for detailed performance tracking
- Compatible with standard multi-agent reinforcement learning algorithms

Implementation Notes:
- Extends MultiAssetTradingEnv with multi-agent capabilities
- Uses dictionary-based observation and action spaces following multi-agent gym conventions
- Implements proper reward attribution per agent
- Supports both independent and shared capital modes
- Handles agent priority for action execution order

Recent Changes:
- Initial implementation combining multi-asset and multi-agent functionality
- Added agent-asset assignment mechanism
- Implemented shared capital pool with reallocation logic
- Added support for agent communication through expanded observations
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from collections import defaultdict

# Import from parent environments
from .multi_asset_env import MultiAssetTradingEnv
from .multi_agent_env import MultiAgentTradingEnv
from .risk_manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)

class MultiAgentMultiAssetEnv(gym.Env):
    """
    Multi-Agent Multi-Asset Trading Environment.
    
    This environment supports multiple agents trading multiple assets simultaneously.
    It combines functionality from both MultiAssetTradingEnv and MultiAgentTradingEnv.
    
    Features:
    - Supports multiple agents trading multiple assets
    - Flexible agent-to-asset assignment
    - Shared or independent capital pools
    - Dictionary-based observation and action spaces
    - Rich information tracking per agent and asset
    - Support for agent-specific reward shaping
    
    Implementation Notes:
    - Uses gym.spaces.Dict for observation and action spaces
    - Tracks positions and portfolio values per agent
    - Implements proper step sequencing based on agent priority
    - Supports both portfolio weights and discrete amount action types
    - Handles agent-specific observation generation
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        agent_configs: List[Dict],
        window_size: int = 60,
        trading_fee: float = 0.001,
        action_type: str = "portfolio_weights",
        shared_capital: bool = True,
        capital_reallocation_freq: int = 20,
        risk_config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the multi-agent multi-asset trading environment.
        
        Args:
            data: DataFrame with OHLCV data for multiple assets (should have 'asset' column)
            agent_configs: List of agent configuration dictionaries
            window_size: Number of time steps to include in the observation window
            trading_fee: Trading fee as a fraction of the trade value
            action_type: Type of action space ('portfolio_weights' or 'discrete_amount')
            shared_capital: Whether agents share a capital pool or have independent capital
            capital_reallocation_freq: Frequency (in steps) of capital reallocation when using shared_capital
            risk_config_path: Path to risk configuration file
            device: Device to use for tensor operations ('cuda' or 'cpu')
        """
        self.window_size = window_size
        self.trading_fee = trading_fee
        self.action_type = action_type
        self.shared_capital = shared_capital
        self.capital_reallocation_freq = capital_reallocation_freq
        self.device = device
        
        # Process agent configurations
        self.agent_configs = {cfg["id"]: cfg for cfg in agent_configs}
        self.agents = list(self.agent_configs.keys())
        
        # Extract assets from data
        if isinstance(data, pd.DataFrame):
            if 'asset' in data.columns:
                # Data contains multiple assets with 'asset' column
                self.assets = data['asset'].unique().tolist()
                # Convert to dictionary of DataFrames per asset
                self.dfs = {asset: data[data['asset'] == asset].reset_index(drop=True) for asset in self.assets}
            else:
                # Single asset data
                self.assets = ['default']
                self.dfs = {'default': data}
        elif isinstance(data, dict):
            # Already in dictionary format
            self.dfs = data
            self.assets = list(self.dfs.keys())
        else:
            raise ValueError("Data must be either a DataFrame with 'asset' column or a dictionary of DataFrames")
        
        # Map agents to assets (if specified in config)
        self.agent_assets = {}
        for agent_id, cfg in self.agent_configs.items():
            assigned_assets = cfg.get("assigned_assets", None)
            if assigned_assets:
                # Ensure all assigned assets exist
                invalid_assets = [a for a in assigned_assets if a not in self.assets]
                if invalid_assets:
                    raise ValueError(f"Agent '{agent_id}' assigned to non-existent assets: {invalid_assets}")
                self.agent_assets[agent_id] = assigned_assets
            else:
                # If not specified, agent can trade all assets
                self.agent_assets[agent_id] = self.assets
        
        # Initialize risk manager if config provided
        self.risk_manager = None
        if risk_config_path:
            risk_config = RiskConfig.from_yaml(risk_config_path)
            self.risk_manager = RiskManager(risk_config)
        
        # Create a MultiAssetTradingEnv for each agent (if using independent capital)
        if not self.shared_capital:
            self.agent_envs = {}
            for agent_id, cfg in self.agent_configs.items():
                # Only include assigned assets for this agent
                agent_assets = self.agent_assets[agent_id]
                agent_data = {asset: self.dfs[asset] for asset in agent_assets}
                
                # Create environment
                agent_initial_balance = cfg.get("initial_balance", 10000.0)
                self.agent_envs[agent_id] = MultiAssetTradingEnv(
                    dfs=agent_data,
                    window_size=self.window_size,
                    initial_balance=agent_initial_balance,
                    trading_fee=self.trading_fee,
                    action_type=self.action_type
                )
        else:
            # Initialize shared environment variables
            self.current_step = 0
            self.max_steps = min(len(df) for df in self.dfs.values()) - self.window_size - 1
            
            # Initialize portfolios for all agents
            self.initial_balance = sum(
                cfg.get("initial_balance", 10000.0) for cfg in self.agent_configs.values()
            )
            self.total_capital = self.initial_balance
            
            # Initialize agent balances, positions, and portfolios
            self.agent_balances = {}
            self.agent_positions = {}
            self.agent_portfolio_values = {}
            self.agent_cumulative_rewards = {}
            self.agent_metrics = {}
            
            # Initialize capital allocations based on agent configs
            self.capital_allocations = {}
            total_percentage = sum(
                cfg.get("initial_capital_percentage", 1.0) for cfg in self.agent_configs.values()
            )
            for agent_id, cfg in self.agent_configs.items():
                percentage = cfg.get("initial_capital_percentage", 1.0)
                normalized_percentage = percentage / total_percentage
                self.capital_allocations[agent_id] = self.initial_balance * normalized_percentage
            
            # Initialize agent-specific asset data
            self.agent_observations = {}
            
            # Initialize asset prices
            self.prices = {asset: self.dfs[asset]['$close'].iloc[self.window_size] for asset in self.assets}
        
        # Define action and observation spaces
        self._define_action_spaces()
        self._define_observation_spaces()
        
        logger.info(f"Initialized MultiAgentMultiAssetEnv with {len(self.agents)} agents and {len(self.assets)} assets")
        logger.info(f"Using {'shared' if self.shared_capital else 'independent'} capital, action_type={self.action_type}")
        
        # Reset environment
        self.reset()
    
    def _define_action_spaces(self):
        """Define action spaces for each agent based on their assigned assets."""
        self.action_spaces = {}
        
        for agent_id in self.agents:
            agent_assets = self.agent_assets[agent_id]
            num_assets = len(agent_assets)
            
            if self.action_type == "portfolio_weights":
                # Portfolio weights action space (including cash)
                # Sum of weights must equal 1, all weights between 0 and 1
                self.action_spaces[agent_id] = gym.spaces.Box(
                    low=0.0, high=1.0, shape=(num_assets,), dtype=np.float32
                )
            elif self.action_type == "discrete_amount":
                # Discrete amount action space (-1 to 1 for each asset)
                self.action_spaces[agent_id] = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(num_assets,), dtype=np.float32
                )
            else:
                raise ValueError(f"Unsupported action type: {self.action_type}")
    
    def _define_observation_spaces(self):
        """Define observation spaces for each agent based on their assigned assets."""
        self.observation_spaces = {}
        
        for agent_id in self.agents:
            agent_assets = self.agent_assets[agent_id]
            
            # Calculate features per asset
            features_per_asset = None
            if self.shared_capital:
                # Create a sample observation to determine shape
                sample_dfs = {asset: self.dfs[asset].iloc[:self.window_size] for asset in agent_assets}
                sample_obs = self._get_agent_observation(agent_id, sample_dfs)
                self.observation_spaces[agent_id] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
                )
            else:
                # Use the observation space from agent's environment
                self.observation_spaces[agent_id] = self.agent_envs[agent_id].observation_space
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observations: Dictionary of initial observations for each agent
            info: Dictionary of additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Different reset procedure based on capital mode
        if self.shared_capital:
            self.current_step = 0
            
            # Reset agent-specific variables
            for agent_id in self.agents:
                # Reset balance to initial allocation
                self.agent_balances[agent_id] = self.capital_allocations[agent_id]
                
                # Reset positions to zero for all assets
                self.agent_positions[agent_id] = {asset: 0.0 for asset in self.agent_assets[agent_id]}
                
                # Reset portfolio value to initial balance
                self.agent_portfolio_values[agent_id] = self.agent_balances[agent_id]
                
                # Reset cumulative rewards
                self.agent_cumulative_rewards[agent_id] = 0.0
                
                # Reset metrics
                self.agent_metrics[agent_id] = {
                    "trades": 0,
                    "profitable_trades": 0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "highest_portfolio_value": self.agent_balances[agent_id]
                }
            
            # Update current asset prices
            self.prices = {asset: self.dfs[asset]['$close'].iloc[self.window_size] for asset in self.assets}
            
            # Create observations for each agent
            observations = {}
            for agent_id in self.agents:
                agent_assets = self.agent_assets[agent_id]
                agent_dfs = {asset: self.dfs[asset].iloc[:self.window_size+1] for asset in agent_assets}
                observations[agent_id] = self._get_agent_observation(agent_id, agent_dfs)
            
            # Reset risk manager if available
            if self.risk_manager:
                self.risk_manager.reset()
            
            # Store observations for next step
            self.agent_observations = observations
            
            return observations, {}
        else:
            # Reset individual environments
            observations = {}
            infos = {}
            
            for agent_id in self.agents:
                obs, info = self.agent_envs[agent_id].reset(seed=seed)
                observations[agent_id] = obs
                infos[agent_id] = info
            
            return observations, infos
    
    def _get_agent_observation(self, agent_id, agent_dfs):
        """
        Generate observation for a specific agent.
        
        Args:
            agent_id: ID of the agent
            agent_dfs: Dictionary of DataFrames for assets assigned to this agent
            
        Returns:
            observation: Numpy array containing the agent's observation
        """
        # Convert agent's DataFrames to observation
        # Similar to MultiAssetTradingEnv._get_observation but specific to this agent
        
        # Get time window for each asset
        window_data = {}
        for asset, df in agent_dfs.items():
            # Window starts at current_step and extends window_size steps
            start_idx = self.current_step
            end_idx = start_idx + self.window_size
            window_data[asset] = df.iloc[start_idx:end_idx].copy()
        
        # Extract features (OHLCV) for each asset
        feature_columns = ['$open', '$high', '$low', '$close', '$volume']
        observations = []
        
        for asset, df in window_data.items():
            # Extract price and volume data
            asset_data = df[feature_columns].values
            
            # Normalize data
            asset_data = (asset_data - np.mean(asset_data, axis=0)) / (np.std(asset_data, axis=0) + 1e-8)
            
            # Add asset positions if available
            if self.shared_capital and asset in self.agent_positions.get(agent_id, {}):
                position = self.agent_positions[agent_id][asset]
                position_value = position * self.prices[asset]
                position_percentage = position_value / self.agent_portfolio_values[agent_id] if self.agent_portfolio_values[agent_id] > 0 else 0
                
                # Add position information as additional feature
                position_info = np.ones((len(df), 1)) * position_percentage
                asset_data = np.hstack([asset_data, position_info])
            
            observations.append(asset_data)
        
        # Combine all asset observations
        if observations:
            combined_obs = np.hstack(observations)
        else:
            # Fallback for empty observations
            combined_obs = np.zeros((self.window_size, 5))  # Basic OHLCV shape
        
        return combined_obs.astype(np.float32)
    
    def step(self, actions):
        """
        Take actions in the environment.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            dones: Dictionary of done flags for each agent
            truncated: Dictionary of truncated flags for each agent
            infos: Dictionary of additional information for each agent
        """
        # Validate actions
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                raise ValueError(f"Unknown agent: {agent_id}")
            
            expected_shape = (len(self.agent_assets[agent_id]),)
            if isinstance(action, np.ndarray) and action.shape != expected_shape:
                raise ValueError(f"Action shape {action.shape} for agent {agent_id} doesn't match expected shape {expected_shape}")
        
        # Different step procedure based on capital mode
        if self.shared_capital:
            return self._step_shared_capital(actions)
        else:
            return self._step_independent_capital(actions)
    
    def _step_shared_capital(self, actions):
        """
        Handle step logic for shared capital mode.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observations, rewards, dones, truncated, infos
        """
        # Store portfolio values before actions for reward calculation
        prev_portfolio_values = self.agent_portfolio_values.copy()
        
        # Execute actions for each agent based on priority
        sorted_agents = sorted(
            self.agents,
            key=lambda a_id: self.agent_configs[a_id].get("priority", 1),
            reverse=True  # Higher priority first
        )
        
        # Process actions for each agent
        for agent_id in sorted_agents:
            if agent_id not in actions:
                continue
                
            action = actions[agent_id]
            agent_assets = self.agent_assets[agent_id]
            
            if self.action_type == "portfolio_weights":
                self._process_portfolio_weights_action(agent_id, action, agent_assets)
            elif self.action_type == "discrete_amount":
                self._process_discrete_amount_action(agent_id, action, agent_assets)
        
        # Advance time step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Update prices for all assets
        self.prices = {asset: self.dfs[asset]['$close'].iloc[self.current_step + self.window_size] 
                      for asset in self.assets}
        
        # Update portfolio values for all agents
        self._update_agent_portfolio_values()
        
        # Calculate rewards for each agent
        rewards = {}
        for agent_id in self.agents:
            rewards[agent_id] = self._calculate_agent_reward(
                agent_id, prev_portfolio_values[agent_id]
            )
            self.agent_cumulative_rewards[agent_id] += rewards[agent_id]
        
        # Reallocate capital if needed and using shared capital
        if self.shared_capital and self.current_step % self.capital_reallocation_freq == 0:
            self._reallocate_capital()
        
        # Create observations for each agent
        observations = {}
        for agent_id in self.agents:
            agent_assets = self.agent_assets[agent_id]
            agent_dfs = {asset: self.dfs[asset].iloc[self.current_step:self.current_step+self.window_size+1] 
                        for asset in agent_assets}
            observations[agent_id] = self._get_agent_observation(agent_id, agent_dfs)
        
        # Store observations for next step
        self.agent_observations = observations
        
        # Prepare done, truncated, and info dictionaries
        dones = {agent_id: done for agent_id in self.agents}
        truncated = {agent_id: False for agent_id in self.agents}
        
        # Build info dictionary with agent-specific metrics
        infos = {}
        for agent_id in self.agents:
            infos[agent_id] = {
                "portfolio_value": self.agent_portfolio_values[agent_id],
                "balance": self.agent_balances[agent_id],
                "positions": self.agent_positions[agent_id].copy(),
                "cumulative_reward": self.agent_cumulative_rewards[agent_id],
                "metrics": self.agent_metrics[agent_id],
                "prices": {asset: self.prices[asset] for asset in self.agent_assets[agent_id]}
            }
        
        return observations, rewards, dones, truncated, infos
    
    def _step_independent_capital(self, actions):
        """
        Handle step logic for independent capital mode.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observations, rewards, dones, truncated, infos
        """
        # Step through each agent's environment
        observations = {}
        rewards = {}
        dones = {}
        truncated = {}
        infos = {}
        
        for agent_id in self.agents:
            if agent_id not in actions:
                # Skip agents without actions
                continue
            
            # Get action for this agent
            action = actions[agent_id]
            
            # Step the agent's environment
            obs, reward, done, trunc, info = self.agent_envs[agent_id].step(action)
            
            observations[agent_id] = obs
            rewards[agent_id] = reward
            dones[agent_id] = done
            truncated[agent_id] = trunc
            infos[agent_id] = info
        
        # Check if all environments are done
        all_done = all(dones.values())
        if all_done:
            # Set all environments to done
            for agent_id in self.agents:
                dones[agent_id] = True
        
        return observations, rewards, dones, truncated, infos
    
    def _process_portfolio_weights_action(self, agent_id, action, agent_assets):
        """
        Process portfolio weights action for an agent.
        
        Args:
            agent_id: ID of the agent
            action: Portfolio weights action
            agent_assets: List of assets assigned to this agent
        """
        # Normalize weights to sum to 1.0
        weights = action / (np.sum(action) + 1e-8)
        
        # Get portfolio value
        portfolio_value = self.agent_portfolio_values[agent_id]
        
        # Calculate target position for each asset
        target_positions = {}
        for i, asset in enumerate(agent_assets):
            weight = weights[i]
            price = self.prices[asset]
            
            if price > 0:
                target_positions[asset] = (portfolio_value * weight) / price
            else:
                target_positions[asset] = 0.0
        
        # Execute trades to reach target positions
        for asset, target_position in target_positions.items():
            current_position = self.agent_positions[agent_id].get(asset, 0.0)
            position_change = target_position - current_position
            
            if abs(position_change) > 1e-8:  # Only trade if significant change
                self._execute_trade(agent_id, asset, position_change)
    
    def _process_discrete_amount_action(self, agent_id, action, agent_assets):
        """
        Process discrete amount action for an agent.
        
        Args:
            agent_id: ID of the agent
            action: Discrete amount action
            agent_assets: List of assets assigned to this agent
        """
        # Process each asset action
        for i, asset in enumerate(agent_assets):
            if i >= len(action):
                continue
                
            # Get action value (-1 to +1)
            action_value = action[i]
            
            # Calculate position change based on action value
            if abs(action_value) < 0.1:
                # Small action, no trade
                continue
                
            # Get available capital for this agent
            available_capital = self.agent_balances[agent_id]
            
            # Calculate position change
            max_trade_size = available_capital / self.prices[asset] if self.prices[asset] > 0 else 0
            position_change = action_value * max_trade_size
            
            # Execute trade
            if abs(position_change) > 1e-8:  # Only trade if significant change
                self._execute_trade(agent_id, asset, position_change)
    
    def _execute_trade(self, agent_id, asset, position_change):
        """
        Execute a trade for a specific agent and asset.
        
        Args:
            agent_id: ID of the agent
            asset: Asset to trade
            position_change: Change in position (positive for buy, negative for sell)
            
        Returns:
            bool: Whether the trade was executed successfully
        """
        # Get current price
        price = self.prices[asset]
        if price <= 0:
            return False
        
        # Get current position and balance
        current_position = self.agent_positions[agent_id].get(asset, 0.0)
        balance = self.agent_balances[agent_id]
        
        # Calculate trade value and fees
        trade_value = abs(position_change) * price
        fees = trade_value * self.trading_fee
        
        if position_change > 0:
            # Buying
            total_cost = trade_value + fees
            if total_cost > balance:
                # Scale back position change based on available balance
                scaled_position_change = (balance - fees) / price
                if scaled_position_change <= 0:
                    return False
                position_change = scaled_position_change
                trade_value = position_change * price
                fees = trade_value * self.trading_fee
            
            # Update balance and position
            self.agent_balances[agent_id] -= (trade_value + fees)
            self.agent_positions[agent_id][asset] = current_position + position_change
            
            # Update metrics
            self.agent_metrics[agent_id]["trades"] += 1
        
        elif position_change < 0:
            # Selling
            if abs(position_change) > abs(current_position):
                # Can't sell more than we have
                position_change = -abs(current_position)
                trade_value = abs(position_change) * price
                fees = trade_value * self.trading_fee
            
            # Calculate realized P&L
            avg_entry_price = 0.0  # Would need to track this for accurate P&L
            realized_pnl = (price - avg_entry_price) * abs(position_change) - fees
            
            # Update balance and position
            self.agent_balances[agent_id] += (trade_value - fees)
            self.agent_positions[agent_id][asset] = current_position + position_change
            
            # Update metrics
            self.agent_metrics[agent_id]["trades"] += 1
            self.agent_metrics[agent_id]["total_pnl"] += realized_pnl
            if realized_pnl > 0:
                self.agent_metrics[agent_id]["profitable_trades"] += 1
        
        return True
    
    def _update_agent_portfolio_values(self):
        """Update portfolio values for all agents."""
        for agent_id in self.agents:
            # Calculate position values
            position_value = 0.0
            for asset, position in self.agent_positions[agent_id].items():
                price = self.prices.get(asset, 0.0)
                position_value += position * price
            
            # Calculate total portfolio value
            portfolio_value = self.agent_balances[agent_id] + position_value
            self.agent_portfolio_values[agent_id] = portfolio_value
            
            # Update metrics
            metrics = self.agent_metrics[agent_id]
            if portfolio_value > metrics["highest_portfolio_value"]:
                metrics["highest_portfolio_value"] = portfolio_value
            
            # Calculate drawdown
            drawdown = 1 - (portfolio_value / metrics["highest_portfolio_value"]) if metrics["highest_portfolio_value"] > 0 else 0
            if drawdown > metrics["max_drawdown"]:
                metrics["max_drawdown"] = drawdown
    
    def _calculate_agent_reward(self, agent_id, prev_portfolio_value):
        """
        Calculate reward for an agent.
        
        Args:
            agent_id: ID of the agent
            prev_portfolio_value: Previous portfolio value for this agent
            
        Returns:
            float: Reward value
        """
        current_value = self.agent_portfolio_values[agent_id]
        
        # Basic reward is relative change in portfolio value
        if prev_portfolio_value > 0:
            reward = (current_value / prev_portfolio_value) - 1.0
        else:
            reward = 0.0
        
        # Apply sharpe ratio adjustment if we have enough history
        # (not implemented here for brevity)
        
        # Apply drawdown penalty
        drawdown = self.agent_metrics[agent_id]["max_drawdown"]
        if drawdown > 0.1:  # Penalty for drawdowns over 10%
            drawdown_penalty = (drawdown - 0.1) * 10
            reward -= drawdown_penalty
        
        return reward
    
    def _reallocate_capital(self):
        """Reallocate capital among agents based on performance."""
        # Calculate performance metrics for each agent
        performance_metrics = {}
        for agent_id in self.agents:
            # Use exponential moving average of returns as performance metric
            cumulative_reward = self.agent_cumulative_rewards[agent_id]
            drawdown = self.agent_metrics[agent_id]["max_drawdown"]
            sharpe = 0.0  # Would calculate Sharpe ratio with proper returns history
            
            # Simple performance metric: returns adjusted for drawdown
            performance = cumulative_reward * (1 - drawdown)
            performance_metrics[agent_id] = performance
        
        # Calculate total performance (adjusted to be positive)
        performance_values = list(performance_metrics.values())
        min_perf = min(performance_values) if performance_values else 0
        adjusted_metrics = {
            a: performance_metrics[a] - min_perf + 0.1 for a in self.agents
        }
        
        # Calculate new allocation weights
        total_adjusted = sum(adjusted_metrics.values())
        new_weights = {
            a: adjusted_metrics[a] / total_adjusted for a in self.agents
        } if total_adjusted > 0 else {a: 1.0 / len(self.agents) for a in self.agents}
        
        # Calculate total portfolio value across all agents
        total_portfolio_value = sum(self.agent_portfolio_values.values())
        
        # Calculate new allocations
        new_allocations = {
            a: total_portfolio_value * new_weights[a] for a in self.agents
        }
        
        # Adjust positions to reflect new allocations
        for agent_id in self.agents:
            current_value = self.agent_portfolio_values[agent_id]
            target_value = new_allocations[agent_id]
            
            # Skip if change is minimal
            if abs(target_value - current_value) / (current_value + 1e-8) < 0.05:
                continue
            
            # Calculate value to transfer
            value_change = target_value - current_value
            
            # Update capital allocations
            self.capital_allocations[agent_id] = target_value
            
            # Adjust balance (simplification - in reality would need to liquidate positions)
            self.agent_balances[agent_id] += value_change
            self.agent_portfolio_values[agent_id] = target_value
            
            logger.info(f"Reallocated capital for agent {agent_id}: {current_value:.2f} -> {target_value:.2f}")
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendering result based on mode
        """
        if mode == 'human':
            return self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_human(self):
        """Generate human-readable string representation."""
        output = [f"Step: {self.current_step}"]
        
        # Add portfolio values
        output.append("Portfolio Values:")
        for agent_id in self.agents:
            output.append(f"  {agent_id}: ${self.agent_portfolio_values[agent_id]:.2f}")
        
        # Add positions
        output.append("Positions:")
        for agent_id in self.agents:
            positions_str = ", ".join([
                f"{asset}: {position:.4f} (${position * self.prices.get(asset, 0):.2f})"
                for asset, position in self.agent_positions[agent_id].items()
                if abs(position) > 0
            ])
            output.append(f"  {agent_id}: {positions_str}")
        
        # Add balances
        output.append("Cash Balances:")
        for agent_id in self.agents:
            output.append(f"  {agent_id}: ${self.agent_balances[agent_id]:.2f}")
        
        # Add current prices
        output.append("Asset Prices:")
        for asset, price in self.prices.items():
            output.append(f"  {asset}: ${price:.2f}")
        
        # Add performance metrics
        output.append("Performance Metrics:")
        for agent_id in self.agents:
            metrics = self.agent_metrics[agent_id]
            win_rate = metrics["profitable_trades"] / metrics["trades"] if metrics["trades"] > 0 else 0
            output.append(f"  {agent_id}: Win Rate: {win_rate:.2%}, Max DD: {metrics['max_drawdown']:.2%}, PnL: ${metrics['total_pnl']:.2f}")
        
        return "\n".join(output)
    
    def _render_rgb(self):
        """Generate RGB array representation (not implemented)."""
        # Placeholder - would implement plotting logic here
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        if not self.shared_capital:
            for env in self.agent_envs.values():
                env.close() 