import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
import collections

logger = logging.getLogger(__name__)


class SingleAssetRLTradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning with risk-oriented reward shaping and realistic frictions.
    
    Features:
    - Risk-adjusted reward calculation (Sharpe ratio proxy)
    - Drawdown penalties in reward function
    - Slippage simulation based on trade size
    - Partial fill simulation
    - Realistic market friction modeling
    - Configurable reward components
    
    Implementation Notes:
    - Uses rolling returns buffer to compute local Sharpe ratio
    - Tracks portfolio peak value for drawdown calculation
    - Applies slippage proportional to trade size and market conditions
    - Simulates partial fills based on requested trade size
    - Provides detailed trade information in info dictionary
    
    Recent Changes:
    - Added risk-adjusted reward calculation with rolling Sharpe ratio
    - Implemented drawdown penalties in reward function
    - Added realistic slippage simulation based on trade size
    - Added partial fill simulation for large orders
    - Enhanced trade cost model with dynamic fee structure
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        window_size: int = 20,
        max_position_size: float = 1.0,
        # Risk reward shaping parameters
        risk_adjusted_reward: bool = True,
        sharpe_lookback: int = 30,
        sharpe_weight: float = 0.5,
        drawdown_penalty: bool = True,
        max_drawdown_penalty_threshold: float = 0.1,
        # Friction parameters
        apply_slippage: bool = True,
        slippage_factor: float = 0.0005,
        partial_fills: bool = True,
        min_fill_rate: float = 0.8,
        volume_slippage_factor: float = 0.1,
    ):
        """Initialize environment

        Args:
            data: DataFrame with OHLCV data (optional)
            initial_capital: Initial account capital
            trading_fee: Trading fee as fraction of trade value
            window_size: Number of time steps to include in state
            max_position_size: Maximum position size as fraction of capital
            risk_adjusted_reward: Whether to use risk-adjusted rewards
            sharpe_lookback: Number of steps to use for rolling Sharpe calculation
            sharpe_weight: Weight of Sharpe ratio in reward calculation (0-1)
            drawdown_penalty: Whether to apply drawdown penalties
            max_drawdown_penalty_threshold: Drawdown threshold for applying penalties
            apply_slippage: Whether to apply slippage to trades
            slippage_factor: Base slippage factor (as fraction of price)
            partial_fills: Whether to simulate partial fills
            min_fill_rate: Minimum fill rate for partial fills (0-1)
            volume_slippage_factor: Factor for volume-based slippage calculation
        """
        super().__init__()

        # Initialize data
        if data is not None:
            # Convert column names if needed
            rename_map = {
                "open": "$open",
                "high": "$high",
                "low": "$low",
                "close": "$close",
                "volume": "$volume",
            }
            self.data = data.rename(
                columns={
                    k: v for k, v in rename_map.items() if k in data.columns
                }
            )

            # Verify required columns
            required_columns = ["$open", "$high", "$low", "$close", "$volume"]
            missing_cols = [
                col for col in required_columns if col not in self.data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}"
                )
        else:
            self.data = None

        # Store parameters
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.max_position_size = max_position_size
        
        # Risk reward parameters
        self.risk_adjusted_reward = risk_adjusted_reward
        self.sharpe_lookback = sharpe_lookback
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.max_drawdown_penalty_threshold = max_drawdown_penalty_threshold
        
        # Friction parameters
        self.apply_slippage = apply_slippage
        self.slippage_factor = slippage_factor
        self.partial_fills = partial_fills
        self.min_fill_rate = min_fill_rate
        self.volume_slippage_factor = volume_slippage_factor

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: OHLCV data for window_size steps
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 5),  # OHLCV
            dtype=np.float32
        )

        # Initialize state variables
        self.current_step = None
        self.current_position = None
        self.current_capital = None
        self.portfolio_value = None
        self.previous_portfolio_value = None  # Added to track previous portfolio value
        self.done = None
        self.trades = []
        
        # Risk tracking variables
        self.returns_buffer = collections.deque(maxlen=sharpe_lookback)
        self.peak_portfolio_value = None
        self.last_trade_size = 0
        self.last_fill_rate = 1.0
        self.last_slippage = 0.0

        logger.info(
            f"Initialized TradingEnvironment with window_size={window_size}, "
            f"initial_capital={initial_capital}, trading_fee={trading_fee}, "
            f"risk_adjusted_reward={risk_adjusted_reward}, apply_slippage={apply_slippage}"
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if self.data is None:
            raise ValueError("No data provided to environment")

        # Reset state variables
        self.current_step = self.window_size  # Start at window_size
        self.current_position = 0.0
        self.current_capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital
        self.done = False
        self.trades = []
        
        # Reset risk tracking variables
        self.returns_buffer.clear()
        self.peak_portfolio_value = self.initial_capital
        self.last_trade_size = 0
        self.last_fill_rate = 1.0
        self.last_slippage = 0.0

        # Get observation using _get_observation which handles padding
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.done:
            raise RuntimeError("Environment is done, call reset() first")

        # Store the portfolio value before taking action
        self.previous_portfolio_value = self._calculate_portfolio_value(self.current_step)

        # Get current price data
        current_price = self.data.iloc[self.current_step]["$close"]
        current_volume = self.data.iloc[self.current_step]["$volume"]
        
        # Calculate target position change
        position_change = float(action[0]) * self.max_position_size
        target_position = self.current_position + position_change
        
        # Apply position limits
        target_position = np.clip(
            target_position, 
            -self.max_position_size,
            self.max_position_size
        )
        
        # Calculate actual position change
        actual_change = target_position - self.current_position
        
        # Reset trade metrics
        self.last_trade_size = 0
        self.last_fill_rate = 1.0
        self.last_slippage = 0.0
        
        # Execute trade if there is a position change
        if abs(actual_change) > 1e-8:  # Small epsilon to handle float precision
            # Apply partial fills if enabled
            requested_change = actual_change
            if self.partial_fills:
                # Larger trades are more likely to be partially filled
                fill_rate = self._calculate_fill_rate(abs(actual_change), current_volume)
                actual_change = actual_change * fill_rate
                self.last_fill_rate = fill_rate
            
            # Apply slippage to price if enabled
            executed_price = current_price
            if self.apply_slippage:
                # Calculate slippage based on order size and volume
                slippage = self._calculate_slippage(actual_change, current_price, current_volume)
                # Slippage is positive for buys (price goes up), negative for sells (price goes down)
                slippage_direction = 1 if actual_change > 0 else -1
                executed_price = current_price * (1 + slippage_direction * slippage)
                self.last_slippage = slippage
            
            # Calculate trade cost
            trade_value = abs(actual_change * executed_price)
            self.last_trade_size = trade_value
            
            # Apply dynamic fee (larger trades might pay different fees)
            fee_rate = self._calculate_dynamic_fee(trade_value)
            trade_cost = trade_value * fee_rate
            
            # Update capital and position
            self.current_capital -= trade_cost
            if actual_change > 0:  # Buy
                self.current_capital -= trade_value
            else:  # Sell
                self.current_capital += trade_value
            
            self.current_position += actual_change  # Update to the partially filled position
            
            # Record trade
            self.trades.append({
                "step": self.current_step,
                "requested_change": requested_change,
                "actual_change": actual_change,
                "fill_rate": self.last_fill_rate,
                "current_price": current_price,
                "executed_price": executed_price,
                "slippage": self.last_slippage,
                "cost": trade_cost,
                "type": "buy" if actual_change > 0 else "sell"
            })

        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data)
        
        # Calculate new portfolio value after action and step
        self.portfolio_value = self._calculate_portfolio_value(self.current_step - 1 if self.done else self.current_step)
        
        # Update peak portfolio value for drawdown calculation
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        # Calculate basic reward (change in portfolio value)
        eps = 1e-8  # Small epsilon to prevent division by zero
        reward_step = (self.portfolio_value - self.previous_portfolio_value) / max(self.previous_portfolio_value, eps)
        
        # Update returns buffer for Sharpe calculation
        self.returns_buffer.append(reward_step)
        
        # Calculate final reward with risk adjustment
        reward = self._calculate_risk_adjusted_reward(reward_step)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.done, False, info

    def _calculate_risk_adjusted_reward(self, basic_reward: float) -> float:
        """
        Calculate risk-adjusted reward incorporating Sharpe ratio and drawdown penalties.
        
        Args:
            basic_reward: The basic reward (change in portfolio value)
            
        Returns:
            float: Risk-adjusted reward
        """
        # Start with basic reward
        final_reward = basic_reward
        
        # Calculate Sharpe component if enabled and we have enough data
        sharpe_component = 0.0
        if self.risk_adjusted_reward and len(self.returns_buffer) > 3:
            mean_return = np.mean(self.returns_buffer)
            std_return = np.std(self.returns_buffer) + 1e-8  # avoid division by zero
            sharpe_proxy = mean_return / std_return
            
            # Avoid extreme values
            sharpe_proxy = np.clip(sharpe_proxy, -10.0, 10.0)
            
            # Mix in the Sharpe component
            if self.sharpe_weight > 0:
                final_reward = (1 - self.sharpe_weight) * basic_reward + self.sharpe_weight * sharpe_proxy
        
        # Calculate drawdown penalty if enabled
        if self.drawdown_penalty and self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            # Apply penalty if drawdown exceeds threshold
            if drawdown > self.max_drawdown_penalty_threshold:
                # Penalty scales with severity of drawdown
                penalty_factor = 1.0 + (drawdown - self.max_drawdown_penalty_threshold) * 10.0
                # Scale the penalty based on the drawer threshold
                penalty = -0.1 * penalty_factor * drawdown
                final_reward += penalty
        
        return final_reward
    
    def _calculate_fill_rate(self, trade_size: float, volume: float) -> float:
        """
        Calculate the fill rate for a trade based on its size and market volume.
        
        Args:
            trade_size: Absolute size of the trade
            volume: Current market volume
            
        Returns:
            float: Fill rate between min_fill_rate and 1.0
        """
        if not self.partial_fills or volume <= 0:
            return 1.0
            
        # Normalize trade size relative to capital
        relative_size = trade_size / self.initial_capital
        
        # Use volume to estimate liquidity - small trades relative to volume are fully filled
        volume_factor = trade_size / (volume + 1e-10)
        
        # More randomness for realism
        randomness = np.random.uniform(0.95, 1.0)
        
        # Calculate fill rate with a minimum
        fill_rate = max(
            self.min_fill_rate,
            (1.0 - relative_size * 0.5) * (1.0 - volume_factor * self.volume_slippage_factor) * randomness
        )
        
        return fill_rate
        
    def _calculate_slippage(self, trade_size: float, price: float, volume: float) -> float:
        """
        Calculate slippage for a trade based on size and market conditions.
        
        Args:
            trade_size: Size of the trade (signed)
            price: Current market price
            volume: Current market volume
            
        Returns:
            float: Slippage as a fraction of price
        """
        if not self.apply_slippage:
            return 0.0
            
        # Base slippage
        base_slippage = self.slippage_factor
        
        # Volume-based component
        if volume > 0:
            volume_component = abs(trade_size * price) / (volume + 1e-10) * self.volume_slippage_factor
        else:
            volume_component = 0.01  # Default to 1% if no volume data
            
        # Random component
        random_component = np.random.normal(0, 0.2 * base_slippage)
        
        # Total slippage (bounded to reasonable values)
        slippage = base_slippage + volume_component + random_component
        return max(0, min(slippage, 0.05))  # Cap at 5%
        
    def _calculate_dynamic_fee(self, trade_value: float) -> float:
        """
        Calculate dynamic trading fee based on trade size.
        
        Args:
            trade_value: Value of the trade
            
        Returns:
            float: Fee rate as a fraction
        """
        # Simple model: larger trades get better rates, up to 50% discount for very large trades
        base_fee = self.trading_fee
        
        # Normalize trade size
        relative_size = min(1.0, trade_value / (self.initial_capital * 0.2))
        
        # Discount for larger trades (up to 50% discount)
        discount = min(0.5, relative_size * 0.5)
        
        return base_fee * (1.0 - discount)

    def _get_observation(self) -> np.ndarray:
        """Get current observation (window of OHLCV data)
        
        Returns:
            np.ndarray: Observation with shape (window_size, 5) containing OHLCV data.
            If current_step < window_size, the observation is padded with the first row's data.
        """
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Handle negative start index with padding
        if start_idx < 0:
            pad_size = abs(start_idx)
            # Get available data up to current step
            partial_data = self.data.iloc[:end_idx]
            
            # If no data available yet, pad with first row
            if len(partial_data) == 0:
                pad_data = pd.DataFrame([self.data.iloc[0]] * self.window_size)
                window_data = pad_data
            else:
                # Pad with first available row
                pad_data = pd.DataFrame([partial_data.iloc[0]] * pad_size)
                window_data = pd.concat([pad_data, partial_data], axis=0)
        else:
            # Normal case: slice the window
            window_data = self.data.iloc[start_idx:end_idx]
        
        # Verify we have exactly window_size rows
        if len(window_data) != self.window_size:
            raise ValueError(
                f"Observation shape mismatch: got {len(window_data)} rows, expected {self.window_size}"
            )
        
        # Create observation array with OHLCV data
        observation = np.column_stack([
            window_data["$open"].values,
            window_data["$high"].values,
            window_data["$low"].values,
            window_data["$close"].values,
            window_data["$volume"].values,
        ]).astype(np.float32)
        
        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get current state information"""
        # Calculate current drawdown
        drawdown = 0.0
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
        # Calculate Sharpe ratio if we have enough data
        sharpe_ratio = 0.0
        if len(self.returns_buffer) > 3:
            mean_return = np.mean(self.returns_buffer)
            std_return = np.std(self.returns_buffer) + 1e-8
            sharpe_ratio = mean_return / std_return
            
        return {
            "step": self.current_step,
            "position": self.current_position,
            "capital": self.current_capital,
            "portfolio_value": self.portfolio_value,
            "previous_portfolio_value": self.previous_portfolio_value,
            "portfolio_change_pct": (self.portfolio_value - self.previous_portfolio_value) / max(self.previous_portfolio_value, 1e-8),
            "total_trades": len(self.trades),
            "peak_portfolio_value": self.peak_portfolio_value,
            "drawdown": drawdown,
            "sharpe_ratio": sharpe_ratio,
            "last_trade_size": self.last_trade_size,
            "last_fill_rate": self.last_fill_rate,
            "last_slippage": self.last_slippage,
        }

    def _calculate_portfolio_value(self, step: int) -> float:
        """Calculate total portfolio value at current step"""
        if step >= len(self.data):
            return self.portfolio_value
            
        current_price = self.data.iloc[step]["$close"]
        position_value = self.current_position * current_price
        return self.current_capital + position_value

    def render(self, mode: str = "human"):
        """
        Render the environment.
        """
        # Implement rendering if needed
        pass

    def close(self):
        """
        Close the environment.
        """
        pass
