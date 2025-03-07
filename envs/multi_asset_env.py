"""Multi-Asset Trading Environment for Reinforcement Learning.

This module provides a gym-compatible environment for trading multiple assets simultaneously.
It extends the single-asset environment with expanded observation space.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiAssetTradingEnv(gym.Env):
    """Multi-Asset Trading Environment for Reinforcement Learning.
    
    Features:
    - Handles multiple assets simultaneously
    - Expanded observation space with price, volume, indicators for each asset
    - Flexible position tracking per asset
    - Shared capital pool across assets
    - Customizable reward functions
    - Support for CNN/LSTM compatible observation formats
    
    Implementation Notes:
    - Uses a unified DataFrame with multiple assets
    - Properly normalizes observations across different price scales
    - Tracks positions and portfolio values across assets
    - Supports window-based observations for time series
    
    Recent Changes:
    - Initial implementation of multi-asset observation space
    - Added position tracking for multiple assets
    - Implemented shared capital pool
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        df: pd.DataFrame = None,
        assets: List[str] = None,
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
        window_size: int = 30,
        max_position_size: float = 1.0,
        reward_function: str = 'sharpe',  # 'sharpe', 'returns', 'log_returns'
        normalize_observations: bool = True,
        normalization_method: str = 'zscore',  # 'zscore', 'minmax', 'log'
        observation_features: List[str] = None,
        add_position_info: bool = True,
        observation_dtype: np.dtype = np.float32,
        format_3d: bool = False,  # If True, returns 3D observation for CNN/LSTM
        action_type: str = 'discrete_amount',  # 'discrete_amount', 'portfolio_weights', 'discrete_signal'
        portfolio_constraints: Optional[Dict] = None,
        allow_short: bool = False,
        rebalance_freq: int = 1,  # Rebalance every N steps
    ):
        """Initialize Multi-Asset Trading Environment.
        
        Args:
            df: Unified DataFrame with multi-asset data (columns prefixed with asset names)
            assets: List of asset identifiers (e.g., ['BTC', 'ETH', 'SOL'])
            initial_balance: Starting account balance
            trading_fee: Fee applied to transactions (as fraction)
            window_size: Number of time steps in observation window
            max_position_size: Maximum allowed position size (as fraction of balance)
            reward_function: Method to calculate rewards
            normalize_observations: Whether to normalize observations
            normalization_method: Method for normalizing observations
            observation_features: List of features to include in observations
            add_position_info: Whether to include position information in observations
            observation_dtype: Data type for observations
            format_3d: If True, formats observation for CNN/LSTM (3D tensor)
            action_type: Type of action space to use:
                - 'discrete_amount': Continuous values that represent position size changes
                - 'portfolio_weights': Continuous values that represent target portfolio weights
                - 'discrete_signal': Discrete buy/hold/sell signals
            portfolio_constraints: Dictionary of constraints for portfolio weights:
                - 'sum_to_one': Whether weights must sum to 1.0
                - 'max_weight': Maximum weight for any asset
                - 'min_weight': Minimum weight for any asset
            allow_short: Whether to allow short positions
            rebalance_freq: Rebalance portfolio every N steps (for portfolio_weights)
        
        Example:
            >>> env = MultiAssetTradingEnv(
            ...     df=multi_asset_df,
            ...     assets=['BTC', 'ETH', 'SOL'],
            ...     window_size=30,
            ...     normalize_observations=True,
            ...     action_type='portfolio_weights',
            ...     portfolio_constraints={'sum_to_one': True, 'max_weight': 0.5}
            ... )
            >>> obs = env.reset()
            >>> action = [0.3, 0.3, 0.4]  # 30% BTC, 30% ETH, 40% SOL
            >>> obs, reward, done, info = env.step(action)
        """
        super(MultiAssetTradingEnv, self).__init__()
        
        # Validate and store data
        self.df = df
        self.assets = assets or []
        
        # Extract asset names from DataFrame if not provided
        if df is not None and not assets:
            self.assets = list(set([col.split('_')[0] for col in df.columns if '_' in col]))
            logger.info(f"Extracted assets from DataFrame: {self.assets}")
        
        # Trading parameters
        self.initial_balance = float(initial_balance)
        self.trading_fee = float(trading_fee)
        self.window_size = int(window_size)
        self.max_position_size = float(max_position_size)
        self.reward_function = reward_function
        
        # Observation parameters
        self.normalize_observations = normalize_observations
        self.normalization_method = normalization_method
        self.observation_dtype = observation_dtype
        self.format_3d = format_3d
        self.add_position_info = add_position_info
        
        # Action parameters
        self.action_type = action_type
        self.allow_short = allow_short
        self.rebalance_freq = rebalance_freq
        self.steps_since_rebalance = 0
        
        # Portfolio constraints
        if portfolio_constraints is None:
            self.portfolio_constraints = {
                'sum_to_one': True,
                'max_weight': 1.0,
                'min_weight': 0.0 if not allow_short else -1.0
            }
        else:
            self.portfolio_constraints = portfolio_constraints
            # Default values if not specified
            if 'sum_to_one' not in self.portfolio_constraints:
                self.portfolio_constraints['sum_to_one'] = True
            if 'max_weight' not in self.portfolio_constraints:
                self.portfolio_constraints['max_weight'] = 1.0
            if 'min_weight' not in self.portfolio_constraints:
                self.portfolio_constraints['min_weight'] = 0.0 if not allow_short else -1.0
        
        # Define default observation features if not provided
        if observation_features is None:
            self.observation_features = ['$close', '$open', '$high', '$low', '$volume']
        else:
            self.observation_features = observation_features
        
        # Calculate feature dimensions
        self.n_assets = len(self.assets)
        self.n_features_per_asset = len(self.observation_features)
        
        # Position information adds 3 features per asset: position size, avg entry price, unrealized PnL
        self.position_features = 3 if add_position_info else 0
        
        # Add global features (e.g., available balance, total portfolio value)
        self.global_features = 2
        
        # Calculate total features per timestep
        self.total_features = (
            self.n_assets * self.n_features_per_asset +  # Asset price/volume features
            self.n_assets * self.position_features +     # Position info
            self.global_features                         # Global portfolio info
        )
        
        # Initialize state - we need to do this before defining observation_space
        # because the actual observation dimensions may differ from our calculations
        self._initialize_state()
        
        # Extract asset features and create a sample observation to determine exact dimensions
        if df is not None:
            try:
                # Extract features and normalize if needed
                self.asset_dfs = self._extract_features()
                if self.normalize_observations:
                    self.asset_dfs = self._normalize_observations(self.asset_dfs)
                
                # Create a sample observation
                sample_obs = self._get_sample_observation()
                
                # Use the sample observation to define the observation space dimensions
                if self.format_3d:
                    self.observation_space = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=sample_obs.shape,
                        dtype=self.observation_dtype
                    )
                else:
                    self.observation_space = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=sample_obs.shape,
                        dtype=self.observation_dtype
                    )
                
                logger.info(f"Observation space shape: {self.observation_space.shape}")
            except Exception as e:
                logger.error(f"Error creating sample observation: {e}")
                # Define fallback observation space
                self._define_fallback_observation_space()
        else:
            # Define fallback observation space
            self._define_fallback_observation_space()
        
        # Define the action space based on the action type
        self._define_action_space()
        
        # Initialize properly
        self.reset()
    
    def _initialize_state(self):
        """Initialize environment state"""
        # Reset account state
        self.balance = float(self.initial_balance)
        self.positions = {asset: 0.0 for asset in self.assets}
        self.avg_entry_prices = {asset: 0.0 for asset in self.assets}
        self.prices = {asset: 0.0 for asset in self.assets}
        self.portfolio_value = self.balance
        self.transactions = []
        self.portfolio_history = []
        self.current_step = self.window_size  # Start after window_size to have enough history
        self.steps_since_rebalance = 0
        self.current_weights = {asset: 0.0 for asset in self.assets}
        self.target_weights = {asset: 0.0 for asset in self.assets}
    
    def _define_fallback_observation_space(self):
        """Define fallback observation space when we can't create a sample observation."""
        if self.format_3d:
            # Use estimated 3D shape
            expected_features = self.n_features_per_asset + self.position_features + self.global_features
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.n_assets, expected_features),
                dtype=self.observation_dtype
            )
        else:
            # Use estimated 2D shape
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.total_features),
                dtype=self.observation_dtype
            )
        
        logger.warning(f"Using fallback observation space shape: {self.observation_space.shape}")
    
    def _get_sample_observation(self) -> np.ndarray:
        """Create a sample observation to determine the exact shape.
        
        Returns:
            Sample observation array
        """
        # Prepare observation windows
        observation_windows = {}
        
        # Assume we have at least window_size data points
        for asset, df in self.asset_dfs.items():
            if len(df) < self.window_size:
                raise ValueError(f"Not enough data for {asset}: {len(df)} < {self.window_size}")
                
            # Take the first window_size rows for the sample
            observation_windows[asset] = df.iloc[:self.window_size].copy()
        
        # Add position information
        if self.add_position_info:
            for asset in self.assets:
                if asset in observation_windows:
                    # Add position columns with zeros
                    observation_windows[asset]['position_size'] = 0.0
                    observation_windows[asset]['avg_entry_price'] = 0.0
                    observation_windows[asset]['unrealized_pnl'] = 0.0
        
        # Add global features
        for asset in self.assets:
            if asset in observation_windows:
                observation_windows[asset]['available_balance_pct'] = 1.0
                observation_windows[asset]['portfolio_value_pct'] = 1.0
        
        # Format the observation
        if self.format_3d:
            # Create 3D observation
            expected_features = self.n_features_per_asset
            if self.add_position_info:
                expected_features += 3  # position size, avg entry price, unrealized PnL
            expected_features += 2  # global features
            
            sample_obs = np.zeros((
                self.window_size,
                self.n_assets,
                expected_features
            ), dtype=self.observation_dtype)
            
            for i, asset in enumerate(self.assets):
                if asset in observation_windows:
                    df = observation_windows[asset]
                    
                    # Ensure correct number of columns
                    if len(df.columns) != expected_features:
                        if len(df.columns) < expected_features:
                            for j in range(len(df.columns), expected_features):
                                df[f"padding_{j}"] = 0.0
                        df = df.iloc[:, :expected_features]
                    
                    sample_obs[:, i, :] = df.values
        else:
            # Calculate total feature count from observation windows
            total_feature_count = sum(len(df.columns) for df in observation_windows.values())
            
            # Create 2D observation
            sample_obs = np.zeros((self.window_size, total_feature_count), dtype=self.observation_dtype)
            
            # Fill with sample data
            feature_idx = 0
            for asset in self.assets:
                if asset in observation_windows:
                    asset_df = observation_windows[asset]
                    n_cols = asset_df.shape[1]
                    
                    # Ensure we don't exceed array bounds
                    if feature_idx + n_cols <= sample_obs.shape[1]:
                        sample_obs[:, feature_idx:feature_idx+n_cols] = asset_df.values
                        feature_idx += n_cols
        
        return sample_obs
    
    def _extract_features(self) -> Dict[str, pd.DataFrame]:
        """Extract feature DataFrames for each asset.
        
        Returns:
            Dictionary mapping asset names to their feature DataFrames
        """
        if self.df is None:
            raise ValueError("DataFrame is not set. Please provide data to the environment.")
        
        asset_dfs = {}
        
        for asset in self.assets:
            # Get columns for this asset
            asset_cols = [col for col in self.df.columns if col.startswith(f"{asset}_")]
            
            if not asset_cols:
                logger.warning(f"No columns found for asset {asset}")
                continue
                
            # Extract requested features
            feature_cols = []
            for feature in self.observation_features:
                matching_cols = [col for col in asset_cols if feature in col]
                if matching_cols:
                    feature_cols.extend(matching_cols)
            
            if not feature_cols:
                logger.warning(f"No matching features found for asset {asset}")
                continue
                
            # Create DataFrame with just this asset's features
            asset_dfs[asset] = self.df[feature_cols].copy()
            
            # Rename columns to strip asset prefix for easier access
            asset_dfs[asset].columns = [col.split('_', 1)[1] for col in asset_dfs[asset].columns]
            
            # Verify that required features are present
            missing_features = [feature for feature in self.observation_features if feature not in asset_dfs[asset].columns]
            if missing_features:
                logger.warning(f"Missing features for asset {asset}: {missing_features}")
            
            logger.info(f"Extracted {len(feature_cols)} features for asset {asset}: {list(asset_dfs[asset].columns)}")
        
        # Report on any assets that were missing
        missing_assets = set(self.assets) - set(asset_dfs.keys())
        if missing_assets:
            logger.warning(f"Could not extract features for assets: {missing_assets}")
        
        return asset_dfs
    
    def _normalize_observations(self, asset_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalize observations using specified method.
        
        Args:
            asset_dfs: Dictionary of asset DataFrames
            
        Returns:
            Dictionary of normalized asset DataFrames
        """
        if not self.normalize_observations:
            return asset_dfs
        
        normalized_dfs = {}
        
        for asset, df in asset_dfs.items():
            # Create a copy to avoid modifying original
            norm_df = df.copy()
            
            # Apply normalization method
            if self.normalization_method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                for col in norm_df.columns:
                    if norm_df[col].std() > 0:  # Avoid division by zero
                        norm_df[col] = (norm_df[col] - norm_df[col].mean()) / norm_df[col].std()
            
            elif self.normalization_method == 'minmax':
                # Min-max normalization (0 to 1 range)
                for col in norm_df.columns:
                    min_val = norm_df[col].min()
                    max_val = norm_df[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
            
            elif self.normalization_method == 'log':
                # Log normalization
                for col in norm_df.columns:
                    if (norm_df[col] > 0).all():  # Check for positive values
                        norm_df[col] = np.log(norm_df[col])
            
            elif self.normalization_method == 'percent_change':
                # Percent change from first value
                for col in norm_df.columns:
                    if norm_df[col].iloc[0] != 0:  # Avoid division by zero
                        norm_df[col] = norm_df[col].pct_change().fillna(0)
            
            else:
                logger.warning(f"Unknown normalization method: {self.normalization_method}")
            
            normalized_dfs[asset] = norm_df
        
        return normalized_dfs
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on window of data and current positions.
        
        Returns:
            Observation as numpy array with shape based on format_3d setting
        """
        # Check if we have enough data points
        if self.current_step < self.window_size:
            raise ValueError(f"Not enough data points. Current step: {self.current_step}, Window size: {self.window_size}")
        
        # Extract windows for each asset
        observation_windows = {}
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        for asset, df in self.asset_dfs.items():
            observation_windows[asset] = df.iloc[start_idx:end_idx].copy()
        
        # Prepare position information if required
        if self.add_position_info:
            for asset in self.assets:
                if asset not in observation_windows:
                    continue
                    
                # Current position size as percentage of portfolio value
                position_size_pct = self.positions[asset] * self.prices[asset] / self.portfolio_value if self.portfolio_value > 0 else 0
                
                # Add position information columns
                observation_windows[asset]['position_size'] = position_size_pct
                observation_windows[asset]['avg_entry_price'] = self.avg_entry_prices[asset]
                
                # Calculate unrealized PnL as percentage
                if self.positions[asset] != 0 and self.avg_entry_prices[asset] > 0:
                    unrealized_pnl_pct = (self.prices[asset] / self.avg_entry_prices[asset] - 1) * np.sign(self.positions[asset])
                else:
                    unrealized_pnl_pct = 0
                
                observation_windows[asset]['unrealized_pnl'] = unrealized_pnl_pct
        
        # Add global portfolio information to each asset's window
        # This is redundant across assets but ensures each asset has access to the same global info
        for asset in self.assets:
            if asset not in observation_windows:
                continue
                
            # Available balance as percentage of portfolio value
            observation_windows[asset]['available_balance_pct'] = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
            
            # Total portfolio value normalized to initial balance
            observation_windows[asset]['portfolio_value_pct'] = self.portfolio_value / self.initial_balance
        
        # Format the observation based on settings
        if self.format_3d:
            # Create 3D observation: [window_size, n_assets, features_per_asset]
            expected_features = self.n_features_per_asset + (3 if self.add_position_info else 0) + 2  # Price + position + global
            obs_3d = np.zeros((
                self.window_size,
                self.n_assets,
                expected_features
            ), dtype=self.observation_dtype)
            
            for i, asset in enumerate(self.assets):
                if asset in observation_windows:
                    # Ensure the dataframe has the expected number of columns
                    df = observation_windows[asset]
                    if len(df.columns) != expected_features:
                        logger.warning(f"Asset {asset} has {len(df.columns)} features, expected {expected_features}")
                        # Pad or truncate columns if necessary
                        if len(df.columns) < expected_features:
                            for j in range(len(df.columns), expected_features):
                                df[f'padding_{j}'] = 0
                        df = df.iloc[:, :expected_features]
                    
                    obs_3d[:, i, :] = df.values
            
            return obs_3d
        else:
            # Create 2D observation: [window_size, total_features]
            # Calculate total feature size based on actual dataframes
            total_feature_count = sum(len(df.columns) for df in observation_windows.values())
            
            # Initialize the 2D array with proper dimensions
            obs_2d = np.zeros((self.window_size, total_feature_count), dtype=self.observation_dtype)
            
            # Fill the array with each asset's features
            feature_idx = 0
            for asset in self.assets:
                if asset in observation_windows:
                    asset_df = observation_windows[asset]
                    n_cols = asset_df.shape[1]
                    
                    if feature_idx + n_cols <= obs_2d.shape[1]:
                        obs_2d[:, feature_idx:feature_idx+n_cols] = asset_df.values
                        feature_idx += n_cols
                    else:
                        logger.error(f"Feature index {feature_idx}+{n_cols} exceeds observation shape {obs_2d.shape}")
            
            return obs_2d
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Reset internal state
        self._initialize_state()
        
        # Extract asset features - check if dataframe exists
        if self.df is None or self.df.empty:
            raise ValueError("No data provided to environment")
            
        self.asset_dfs = self._extract_features()
        
        # Check if we have enough data for all assets
        if not self.asset_dfs:
            raise ValueError("Failed to extract any asset data. Check asset names and DataFrame columns.")
            
        # Check if we have enough data points for the window size
        for asset, df in self.asset_dfs.items():
            if len(df) <= self.window_size:
                raise ValueError(f"Not enough data points for {asset}. Got {len(df)}, need at least {self.window_size+1}")
        
        # Normalize if required
        if self.normalize_observations:
            self.asset_dfs = self._normalize_observations(self.asset_dfs)
        
        # Set initial prices
        self.prices = {}
        for asset in self.assets:
            if asset in self.asset_dfs and '$close' in self.asset_dfs[asset].columns:
                self.prices[asset] = self.asset_dfs[asset]['$close'].iloc[self.current_step]
            else:
                self.prices[asset] = 0.0
                logger.warning(f"Could not find close price for {asset}, setting to 0")
        
        # Get initial observation
        try:
            observation = self._get_observation()
        except Exception as e:
            logger.error(f"Error getting initial observation: {e}")
            # Create a placeholder observation with the correct shape
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_dtype)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'step': self.current_step,
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action in the environment.
        
        Args:
            action: Action array format depends on action_type:
                - 'discrete_amount': Values between -1 and 1 for each asset
                - 'portfolio_weights': Target portfolio weights for each asset
                - 'discrete_signal': Discrete signals (0: Sell, 1: Hold, 2: Buy)
            
        Returns:
            observation: Current observation
            reward: Reward from the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Validate action shape
        expected_shape = (self.n_assets,)
        if isinstance(action, np.ndarray) and action.shape != expected_shape:
            raise ValueError(f"Action shape {action.shape} doesn't match expected shape {expected_shape}")
        
        # Store portfolio value before action
        prev_portfolio_value = self.portfolio_value
        
        # Process action based on action type
        if self.action_type == 'discrete_amount':
            self._process_discrete_amount_action(action)
        elif self.action_type == 'portfolio_weights':
            # Only rebalance at specified frequency
            self.steps_since_rebalance += 1
            if self.steps_since_rebalance >= self.rebalance_freq:
                self._process_portfolio_weights_action(action)
                self.steps_since_rebalance = 0
        elif self.action_type == 'discrete_signal':
            self._process_discrete_signal_action(action)
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Update prices for next step if not done
        if not done:
            for asset in self.assets:
                if asset in self.asset_dfs and '$close' in self.asset_dfs[asset].columns:
                    self.prices[asset] = self.asset_dfs[asset]['$close'].iloc[self.current_step]
        
        # Calculate portfolio value and current weights
        self._update_portfolio_value()
        self._update_current_weights()
        
        # Record portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'weights': self.current_weights.copy(),
        })
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)
        
        # Get new observation
        try:
            observation = self._get_observation()
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_dtype)
            done = True
        
        # Prepare info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_change': self.portfolio_value / prev_portfolio_value - 1,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'prices': self.prices.copy(),
            'weights': self.current_weights.copy(),
            'step': self.current_step,
        }
        
        return observation, reward, done, False, info
    
    def _execute_trade(self, asset: str, position_change: float) -> bool:
        """Execute a trade for the given asset.
        
        Args:
            asset: Asset to trade
            position_change: Amount to change position by (positive for buy, negative for sell)
            
        Returns:
            Whether the trade was successful
        """
        if abs(position_change) < 1e-10:
            return True  # No trade needed
            
        price = self.prices[asset]
        
        if price <= 0:
            logger.warning(f"Invalid price for {asset}: {price}")
            return False
        
        # Calculate cost with trading fee
        cost = abs(position_change * price)
        fee = cost * self.trading_fee
        total_cost = cost + fee
        
        # Check if selling
        if position_change < 0:
            # Ensure we don't sell more than we have
            if abs(position_change) > abs(self.positions[asset]):
                position_change = -abs(self.positions[asset])
                cost = abs(position_change * price)
                fee = cost * self.trading_fee
                total_cost = cost + fee
            
            # Update balance (add proceeds minus fee)
            self.balance += cost - fee
            
            # Update position
            self.positions[asset] += position_change
            
            # Reset average entry price if position becomes 0
            if abs(self.positions[asset]) < 1e-10:
                self.positions[asset] = 0
                self.avg_entry_prices[asset] = 0
            
            # Record transaction
            self.transactions.append({
                'asset': asset,
                'step': self.current_step,
                'type': 'sell',
                'amount': abs(position_change),
                'price': price,
                'cost': cost,
                'fee': fee
            })
            
            return True
            
        # Buying
        else:
            # Check if we have enough balance
            if total_cost > self.balance:
                # Adjust position change to match available balance
                max_affordable = self.balance / (price * (1 + self.trading_fee))
                position_change = max_affordable
                cost = position_change * price
                fee = cost * self.trading_fee
                total_cost = cost + fee
                
                if position_change < 1e-10:
                    return False  # Can't afford any
            
            # Update balance
            self.balance -= total_cost
            
            # Update average entry price
            if self.positions[asset] + position_change > 0:
                self.avg_entry_prices[asset] = (
                    (self.positions[asset] * self.avg_entry_prices[asset]) + (position_change * price)
                ) / (self.positions[asset] + position_change)
            
            # Update position
            self.positions[asset] += position_change
            
            # Record transaction
            self.transactions.append({
                'asset': asset,
                'step': self.current_step,
                'type': 'buy',
                'amount': position_change,
                'price': price,
                'cost': cost,
                'fee': fee
            })
            
            return True
    
    def _update_portfolio_value(self):
        """Update the current portfolio value based on positions and prices."""
        position_value = sum(
            self.positions[asset] * self.prices[asset]
            for asset in self.assets
            if asset in self.prices and self.prices[asset] > 0
        )
        
        self.portfolio_value = self.balance + position_value
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """Calculate reward based on portfolio performance.
        
        Args:
            prev_portfolio_value: Portfolio value from previous step
            
        Returns:
            Calculated reward
        """
        if self.reward_function == 'returns':
            # Simple returns
            return self.portfolio_value / prev_portfolio_value - 1
            
        elif self.reward_function == 'log_returns':
            # Log returns
            return np.log(self.portfolio_value / prev_portfolio_value)
            
        elif self.reward_function == 'sharpe':
            # Approximate Sharpe ratio (needs portfolio history)
            if len(self.portfolio_history) < 10:
                return 0
                
            # Calculate returns
            returns = []
            for i in range(1, min(11, len(self.portfolio_history))):
                prev_value = self.portfolio_history[-i-1]['portfolio_value']
                curr_value = self.portfolio_history[-i]['portfolio_value']
                returns.append(curr_value / prev_value - 1)
                
            returns = np.array(returns)
            
            # Calculate Sharpe ratio
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                return sharpe
            else:
                return 0
                
        else:
            # Default to simple returns
            return self.portfolio_value / prev_portfolio_value - 1
    
    def render(self, mode='human'):
        """Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendering based on the specified mode
        """
        if mode == 'human':
            return self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_human(self) -> str:
        """Render human-readable representation of environment state.
        
        Returns:
            String representation of the environment
        """
        output = []
        output.append(f"Step: {self.current_step}")
        output.append(f"Portfolio Value: ${self.portfolio_value:.2f}")
        output.append(f"Cash Balance: ${self.balance:.2f}")
        
        output.append("\nPositions:")
        for asset in self.assets:
            if abs(self.positions[asset]) > 0:
                position_value = self.positions[asset] * self.prices[asset]
                pct_of_portfolio = position_value / self.portfolio_value * 100
                
                output.append(
                    f"  {asset}: {self.positions[asset]:.6f} @ ${self.avg_entry_prices[asset]:.2f} "
                    f"(Current: ${self.prices[asset]:.2f}, Value: ${position_value:.2f}, "
                    f"{pct_of_portfolio:.1f}% of portfolio)"
                )
        
        return "\n".join(output)
    
    def _render_rgb(self) -> np.ndarray:
        """Render RGB array representation of environment (e.g., chart).
        
        Not implemented yet, would require additional visualization libraries.
        
        Returns:
            RGB array of the rendered environment
        """
        # This would typically use matplotlib to create a visualization
        # For now, we return a simple placeholder
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass

    def _define_action_space(self):
        """Define action space based on action_type."""
        if self.action_type == 'discrete_amount':
            # Continuous values that represent position size changes
            # -1 means sell all, 1 means buy max allowed, 0 means hold
            self.action_space = gym.spaces.Box(
                low=-1.0 if self.allow_short else 0.0,
                high=1.0,
                shape=(self.n_assets,),
                dtype=np.float32
            )
            logger.info(f"Using discrete_amount action space: {self.action_space}")
            
        elif self.action_type == 'portfolio_weights':
            # Continuous values that represent target portfolio weights
            # Each value represents the target weight of the asset in the portfolio
            # Constraints are applied during execution
            self.action_space = gym.spaces.Box(
                low=self.portfolio_constraints['min_weight'],
                high=self.portfolio_constraints['max_weight'],
                shape=(self.n_assets,),
                dtype=np.float32
            )
            logger.info(f"Using portfolio_weights action space: {self.action_space}")
            
        elif self.action_type == 'discrete_signal':
            # Discrete buy/hold/sell signals
            # 0: Sell, 1: Hold, 2: Buy
            # We use multidiscrete to have a separate signal for each asset
            self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)
            logger.info(f"Using discrete_signal action space: {self.action_space}")
            
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    def _update_current_weights(self):
        """Update the current weights of assets in the portfolio."""
        # Calculate asset values
        asset_values = {}
        for asset in self.assets:
            position = self.positions.get(asset, 0.0)
            price = self.prices.get(asset, 0.0)
            asset_values[asset] = position * price
        
        # Calculate total portfolio value
        total_value = self.portfolio_value
        
        # Calculate current weights
        if total_value > 0:
            self.current_weights = {
                asset: value / total_value 
                for asset, value in asset_values.items()
            }
            # Add cash weight
            self.current_weights['cash'] = self.balance / total_value
        else:
            # If portfolio value is 0, set all weights to 0
            self.current_weights = {asset: 0.0 for asset in self.assets}
            self.current_weights['cash'] = 1.0
    
    def _process_discrete_amount_action(self, action):
        """Process action for discrete_amount action type.
        
        Args:
            action: Array of values between -1 and 1 for each asset,
                   where -1 means sell all, 1 means buy max allowed, 0 means hold
        """
        # Process each asset's action
        for i, asset in enumerate(self.assets):
            # Skip if asset not found
            if asset not in self.prices or self.prices[asset] <= 0:
                continue
            
            # Get action value for this asset
            action_value = action[i]
            
            # Calculate maximum position change based on available balance and max position size
            max_position_value = self.balance * self.max_position_size
            max_position_size = max_position_value / self.prices[asset]
            
            # Calculate target position change
            if action_value > 0:  # Buy
                # Scale action to max position size
                target_position_change = action_value * max_position_size
            elif action_value < 0:  # Sell or short
                # Scale action to current position (sell up to all current position)
                current_position = self.positions[asset]
                if not self.allow_short:
                    # Regular sell (can only sell what we have)
                    target_position_change = action_value * current_position
                else:
                    # Short sell (can go negative up to max position size)
                    if current_position > 0:
                        # First sell existing position
                        target_position_change = action_value * current_position
                    else:
                        # Then short more
                        target_position_change = action_value * max_position_size
            else:  # action_value == 0, no change
                target_position_change = 0.0
            
            # Execute the trade
            self._execute_trade(asset, target_position_change)
    
    def _process_portfolio_weights_action(self, action):
        """Process action for portfolio_weights action type.
        
        Args:
            action: Array of target weights for each asset
        """
        # Apply constraints to the target weights
        target_weights = self._apply_weight_constraints(action)
        
        # Store target weights
        for i, asset in enumerate(self.assets):
            self.target_weights[asset] = target_weights[i]
        
        # Calculate required trades to achieve target weights
        self._rebalance_to_target_weights(target_weights)
    
    def _apply_weight_constraints(self, weights):
        """Apply constraints to portfolio weights.
        
        Args:
            weights: Array of weights
            
        Returns:
            Array of constrained weights
        """
        # Convert to numpy array if needed
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        
        # Apply min/max constraints
        min_weight = self.portfolio_constraints['min_weight']
        max_weight = self.portfolio_constraints['max_weight']
        weights = np.clip(weights, min_weight, max_weight)
        
        # Apply sum to one constraint if needed
        if self.portfolio_constraints['sum_to_one']:
            # If sum is not close to 1, normalize
            if not np.isclose(np.sum(weights), 1.0):
                # Handle case where all weights are zero or negative
                if np.sum(np.maximum(weights, 0)) <= 0:
                    # Set equal weights if all are zero or negative
                    weights = np.ones_like(weights) / len(weights)
                else:
                    # Normalize positive weights to sum to 1
                    weights = np.maximum(weights, 0)  # Ensure non-negative
                    weights = weights / np.sum(weights)
        
        return weights
    
    def _rebalance_to_target_weights(self, target_weights):
        """Rebalance portfolio to match target weights.
        
        Args:
            target_weights: Array of target weights
        """
        # Calculate current portfolio value
        portfolio_value = self.portfolio_value
        
        # Calculate target value for each asset
        target_values = {
            asset: weight * portfolio_value
            for asset, weight in zip(self.assets, target_weights)
        }
        
        # Calculate current asset values
        current_values = {
            asset: self.positions.get(asset, 0.0) * self.prices.get(asset, 0.0)
            for asset in self.assets
        }
        
        # Calculate required trades
        for asset in self.assets:
            # Skip if price is invalid
            if asset not in self.prices or self.prices[asset] <= 0:
                continue
            
            current_value = current_values.get(asset, 0.0)
            target_value = target_values.get(asset, 0.0)
            
            # Calculate value difference
            value_diff = target_value - current_value
            
            # Convert to position change
            price = self.prices[asset]
            position_change = value_diff / price
            
            # Execute trade if significant
            if abs(position_change) > 1e-8:
                self._execute_trade(asset, position_change)
    
    def _process_discrete_signal_action(self, action):
        """Process action for discrete_signal action type.
        
        Args:
            action: Array of discrete signals (0: Sell, 1: Hold, 2: Buy)
        """
        for i, asset in enumerate(self.assets):
            # Skip if asset not found
            if asset not in self.prices or self.prices[asset] <= 0:
                continue
            
            # Get signal for this asset (0: Sell, 1: Hold, 2: Buy)
            signal = action[i]
            
            if signal == 0:  # Sell
                # Sell all (or max allowed for short)
                current_position = self.positions[asset]
                if current_position > 0:
                    # Sell entire long position
                    self._execute_trade(asset, -current_position)
                elif self.allow_short:
                    # Short sell up to max position size
                    max_position_value = self.balance * self.max_position_size
                    max_position_size = max_position_value / self.prices[asset]
                    self._execute_trade(asset, -max_position_size)
            
            elif signal == 2:  # Buy
                # Buy max allowed
                max_position_value = self.balance * self.max_position_size
                max_position_size = max_position_value / self.prices[asset]
                self._execute_trade(asset, max_position_size)
            
            # If signal == 1 (Hold), do nothing


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data.utils.multi_asset_data_loader import MultiAssetDataLoader
    
    # Load sample data
    assets = [
        {'symbol': 'BTC/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'BTC'},
        {'symbol': 'ETH/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'ETH'},
    ]
    
    loader = MultiAssetDataLoader(assets=assets, timeframe='1d')
    df = loader.fetch_multi_asset_data('2023-01-01', '2023-01-31')
    
    # Create environment
    env = MultiAssetTradingEnv(
        df=df,
        assets=['BTC', 'ETH'],
        window_size=10,
        normalize_observations=True,
        action_type='portfolio_weights',
        portfolio_constraints={'sum_to_one': True, 'max_weight': 0.5}
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Print observation shape
    print(f"Observation shape: {obs.shape}")
    
    # Take a few random actions
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.6f}")
        print(env._render_human())
        print("-" * 40) 