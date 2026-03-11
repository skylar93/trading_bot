import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
import collections

from envs.rewards import MultiComponentReward, RewardConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.regime.regime_detector import RegimeDetector

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
        # Friction parameters
        apply_slippage: bool = True,
        slippage_factor: float = 0.0005,
        partial_fills: bool = True,
        min_fill_rate: float = 0.8,
        volume_slippage_factor: float = 0.1,
        # Reward configuration
        reward_config: Optional[RewardConfig] = None,
        # Stability
        min_episode_steps: int = 30,
        reward_scale: float = 1.0,
        # Optional pre-computed sentiment features (Week 13)
        sentiment_data: Optional[pd.DataFrame] = None,
    ):
        """Initialize environment.

        Args:
            data: DataFrame with OHLCV data (optional)
            initial_capital: Initial account capital
            trading_fee: Trading fee as fraction of trade value
            window_size: Number of time steps to include in state
            max_position_size: Maximum position size as fraction of capital
            apply_slippage: Whether to apply slippage to trades
            slippage_factor: Base slippage factor (as fraction of price)
            partial_fills: Whether to simulate partial fills
            min_fill_rate: Minimum fill rate for partial fills (0-1)
            volume_slippage_factor: Factor for volume-based slippage calculation
            reward_config: MultiComponentReward configuration (uses defaults if None)
            min_episode_steps: Minimum number of steps before allowing early termination

        Observations are log-return based, bounded to [-10, 10]:
            col 0: log(open[t] / close[t-1])
            col 1: log(high[t] / close[t-1])
            col 2: log(low[t]  / close[t-1])
            col 3: log(close[t] / close[t-1])
            col 4: log(vol[t]  / mean_vol_in_window)

        If *regime_detector* is provided the observation space is extended to
        (window_size, 8) with the last 3 columns being the current regime
        probabilities [P(low_vol), P(medium_vol), P(high_vol)] broadcast
        across every row in the window.
        """
        super().__init__()
        
        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

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

        # Friction parameters
        self.apply_slippage = apply_slippage
        self.slippage_factor = slippage_factor
        self.partial_fills = partial_fills
        self.min_fill_rate = min_fill_rate
        self.volume_slippage_factor = volume_slippage_factor

        # Stability
        self.min_episode_steps = min_episode_steps

        # Reward function
        self.reward_fn = MultiComponentReward(reward_config or RewardConfig())

        # Regime detector (optional — Week 6)
        self.regime_detector = regime_detector
        self._n_obs_features = 8 if regime_detector is not None else 5

        # Sentiment data (optional, pre-computed and aligned to price data)
        self.sentiment_data = None
        self._n_sentiment = 0
        if sentiment_data is not None:
            if self.data is not None and len(sentiment_data) != len(self.data):
                raise ValueError(
                    f"sentiment_data length ({len(sentiment_data)}) must match "
                    f"data length ({len(self.data)})"
                )
            self.sentiment_data = sentiment_data.reset_index(drop=True)
            self._n_sentiment = 4
        self._n_features = 5 + self._n_sentiment  # OHLCV + optional sentiment

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space: OHLCV [+ sentiment] for window_size steps
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self._n_features),
            dtype=np.float32
        )

        # State variables (set in reset)
        self.current_step = None
        self.current_position = None
        self.current_capital = None
        self.portfolio_value = None
        self.previous_portfolio_value = None
        self.done = None
        self.trades = []
        self.peak_portfolio_value = None
        self.last_trade_size = 0
        self.last_fill_rate = 1.0
        self.last_slippage = 0.0

        logger.info(
            f"Initialized TradingEnvironment with window_size={window_size}, "
            f"initial_capital={initial_capital}, trading_fee={trading_fee}, "
            f"apply_slippage={apply_slippage}"
        )
        
        # STEP 1-A: Check if data is long enough
        if self.data is not None:
            if len(self.data) < self.window_size + 1:
                raise ValueError(
                    f"Data too short ({len(self.data)}) for window_size={self.window_size}. "
                    f"Need at least window_size+1 rows."
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
        self.peak_portfolio_value = self.initial_capital
        self.last_trade_size = 0
        self.last_fill_rate = 1.0
        self.last_slippage = 0.0

        # Reset reward function internal state
        self.reward_fn.reset()

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
        
        # DEBUG: Check for extreme price values
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            self.logger.warning(f"❌ EXTREME PRICE VALUE at step {self.current_step}: price={current_price}")
            current_price = max(0.01, abs(current_price)) if not np.isnan(current_price) else 1.0
            
        # DEBUG: Check for extreme volume values
        if current_volume <= 0 or np.isnan(current_volume) or np.isinf(current_volume):
            self.logger.warning(f"❌ EXTREME VOLUME VALUE at step {self.current_step}: volume={current_volume}")
            current_volume = 1.0
        
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
        
        # DEBUG: Log action details for monitoring
        # --- START: Added Pre-Trade Logging ---
        self.logger.debug(
            f"💰 PRE-TRADE: step={self.current_step}, capital={self.current_capital:.4f}, "
            f"position={self.current_position:.4f}, price={current_price:.4f}, volume={current_volume:.1f}"
        )
        self.logger.debug(
            f"💰 ACTION DETAILS: action={action[0]:.4f}, position_change={position_change:.4f}, "
            f"target_position={target_position:.4f}, actual_change_requested={actual_change:.4f}"
        )
        # --- END: Added Pre-Trade Logging ---
        
        # Reset trade metrics
        self.last_trade_size = 0
        self.last_fill_rate = 1.0
        self.last_slippage = 0.0
        step_trade_cost = 0.0  # accumulated transaction cost this step
        
        # Execute trade if there is a position change
        if abs(actual_change) > 1e-8:  # Small epsilon to handle float precision
            requested_change = actual_change # Store original requested change

            # --- START: 상세 거래 로그 ---
            trade_log_details = {
                "step": self.current_step,
                "capital_before": self.current_capital,
                "position_before": self.current_position,
                "price": current_price,
                "volume": current_volume,
                "requested_change": requested_change,
                "fill_rate": 1.0, # Default value
                "actual_change_after_fill": requested_change, # Default value
                "slippage_rate": 0.0, # Default value
                "executed_price": current_price, # Default value
                "trade_value": 0.0, # Default value
                "fee_rate": self.trading_fee, # Default value
                "trade_cost": 0.0, # Default value
                "capital_change": 0.0, # Default value
                "capital_after": self.current_capital, # Default value
                "position_after": self.current_position # Default value
            }
            # --- END: 상세 거래 로그 ---

            # Apply partial fills if enabled
            if self.partial_fills:
                try:
                    # Larger trades are more likely to be partially filled
                    fill_rate = self._calculate_fill_rate(abs(actual_change), current_volume)
                    
                    # DEBUG: Check for extreme fill rates
                    if fill_rate <= 0 or fill_rate > 1.0 or np.isnan(fill_rate) or np.isinf(fill_rate):
                        self.logger.warning(f"❌ EXTREME FILL RATE: {fill_rate}, using safe default")
                        fill_rate = self.min_fill_rate
                        
                    actual_change = actual_change * fill_rate
                    self.last_fill_rate = fill_rate
                    trade_log_details["fill_rate"] = fill_rate
                    trade_log_details["actual_change_after_fill"] = actual_change
                    
                    # DEBUG: Log fill rate details
                    self.logger.debug(f"📊 FILL RATE: requested={requested_change:.4f}, fill_rate={fill_rate:.4f}, actual={actual_change:.4f}")
                except Exception as e:
                    self.logger.error(f"❌ ERROR in fill rate calculation: {str(e)}")
                    self.last_fill_rate = self.min_fill_rate
                    actual_change = requested_change * self.last_fill_rate # Use requested_change here
                    trade_log_details["fill_rate"] = self.last_fill_rate
                    trade_log_details["actual_change_after_fill"] = actual_change
            else:
                # If not partial fills, actual_change remains requested_change
                trade_log_details["actual_change_after_fill"] = actual_change

            # Apply slippage to price if enabled
            executed_price = current_price
            if self.apply_slippage:
                try:
                    # Calculate slippage based on order size and volume
                    slippage = self._calculate_slippage(actual_change, current_price, current_volume)
                    
                    # DEBUG: Check for extreme slippage
                    if slippage < 0 or slippage > 0.1 or np.isnan(slippage) or np.isinf(slippage):
                        self.logger.warning(f"❌ EXTREME SLIPPAGE: {slippage}, capping at 0.05")
                        slippage = min(max(0, slippage), 0.05)
                        
                    # Slippage is positive for buys (price goes up), negative for sells (price goes down)
                    slippage_direction = 1 if actual_change > 0 else -1
                    executed_price = current_price * (1 + slippage_direction * slippage)
                    self.last_slippage = slippage
                    trade_log_details["slippage_rate"] = slippage
                    trade_log_details["executed_price"] = executed_price
                    
                    # DEBUG: Log slippage details
                    self.logger.debug(f"📉 SLIPPAGE: rate={slippage:.6f}, direction={slippage_direction}, price: {current_price:.4f} -> {executed_price:.4f}")
                except Exception as e:
                    self.logger.error(f"❌ ERROR in slippage calculation: {str(e)}")
                    self.last_slippage = 0.0
                    executed_price = current_price
                    trade_log_details["executed_price"] = executed_price # Update log even on error/no slippage
            else: # 슬리피지 미적용 시
                trade_log_details["executed_price"] = executed_price # 기록 위해 추가

            # Calculate trade cost
            # Use the actual_change after fill rate application
            trade_value = abs(actual_change * executed_price)
            self.last_trade_size = trade_value
            trade_log_details["trade_value"] = trade_value
            
            # Apply dynamic fee (larger trades might pay different fees)
            try:
                fee_rate = self._calculate_dynamic_fee(trade_value)
                trade_cost = trade_value * fee_rate
                
                # DEBUG: Check for extreme fees
                if fee_rate < 0 or fee_rate > 0.05 or np.isnan(fee_rate) or np.isinf(fee_rate):
                    self.logger.warning(f"❌ EXTREME FEE RATE: {fee_rate}, using default")
                    fee_rate = self.trading_fee
                    trade_cost = trade_value * fee_rate
                
                trade_log_details["fee_rate"] = fee_rate
                trade_log_details["trade_cost"] = trade_cost
                    
                # DEBUG: Log fee details
                self.logger.debug(f"💲 TRADE COSTS: value={trade_value:.4f}, fee_rate={fee_rate:.6f}, cost={trade_cost:.4f}")
            except Exception as e:
                self.logger.error(f"❌ ERROR in fee calculation: {str(e)}")
                fee_rate = self.trading_fee
                trade_cost = trade_value * fee_rate
                trade_log_details["fee_rate"] = fee_rate
                trade_log_details["trade_cost"] = trade_cost

            # Update capital and position
            capital_change = -trade_cost
            if actual_change > 0: # Buy
                capital_change -= trade_value
            else: # Sell
                capital_change += trade_value
            
            trade_log_details["capital_change"] = capital_change
            
            self.current_capital += capital_change
            # Use the actual_change after fill rate application
            self.current_position += actual_change
            step_trade_cost = trade_cost  # capture for reward calculation

            trade_log_details["capital_after"] = self.current_capital
            trade_log_details["position_after"] = self.current_position

            # 중요: 상세 거래 로그 출력
            self.logger.info(f"📊 TRADE EXECUTION DETAILS: {trade_log_details}")

            # Check if we should end the episode based on capital
            force_done = False
            if self.current_capital <= 1.0:
                self.logger.warning(f"❌ NEGATIVE OR NEAR-ZERO CAPITAL ({self.current_capital:.4f}); flagging for episode end.")
                force_done = True
            elif self.current_capital > 1e9:
                self.logger.warning(f"❌ EXTREME CAPITAL ({self.current_capital:.2f}); flagging for episode end.")
                force_done = True
            
            # But don't actually end if we haven't reached minimum episode steps
            if force_done and (self.current_step - self.window_size) < self.min_episode_steps:
                self.logger.warning(f"Delaying episode termination until minimum steps {self.min_episode_steps} are reached. Current: {self.current_step - self.window_size}")
                force_done = False
            
            # Apply force_done only if we've reached minimum steps
            if force_done:
                self.done = True
            
            # DEBUG: Check for negative capital (shouldn't happen but could cause issues)
            if self.current_capital < 0:
                self.logger.warning(f"❌ NEGATIVE CAPITAL after trade: {self.current_capital:.4f}")
            
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
        # --- START: Added Portfolio Calc Logging ---
        portfolio_calc_price = self.data.iloc[self.current_step - 1 if self.done else self.current_step]["$close"]
        # Handle potential invalid price during calculation
        if portfolio_calc_price <= 0 or np.isnan(portfolio_calc_price) or np.isinf(portfolio_calc_price):
             self.logger.warning(f"❌ Invalid price used in portfolio calculation at step {self.current_step}: {portfolio_calc_price}. Using 1.0 as fallback.")
             portfolio_calc_price = 1.0

        position_value_calc = self.current_position * portfolio_calc_price
        self.portfolio_value = self.current_capital + position_value_calc
        self.logger.debug(
            f"💰 PORTFOLIO CALC: step={self.current_step}, capital={self.current_capital:.4f}, "
            f"position={self.current_position:.4f}, calc_price={portfolio_calc_price:.4f}, "
            f"pos_value={position_value_calc:.4f}, portfolio_value={self.portfolio_value:.4f}"
        )
        # --- END: Added Portfolio Calc Logging ---

        # --- START: 강화된 포트폴리오 가치 체크 및 즉시 종료 ---
        CRITICAL_LOW_THRESHOLD = 1.0 # 파산 임계값 (초기 자본의 극히 일부)


        FORCE_TERMINATION = False
        TERMINATION_REASON = None
        final_reward_on_termination = 0.0 # Define a default value

        if self.portfolio_value <= CRITICAL_LOW_THRESHOLD:
            self.logger.error(f"💥 CRITICAL PORTFOLIO VALUE at step {self.current_step}: {self.portfolio_value:.4f}. Forcing episode termination.")
            FORCE_TERMINATION = True
            TERMINATION_REASON = "bankruptcy"
            # 포트폴리오 가치를 0 미만으로 두지 않도록 강제
            self.portfolio_value = max(CRITICAL_LOW_THRESHOLD, self.portfolio_value) 
            # 파산 시 큰 패널티 부여
            final_reward_on_termination = -100.0 # 예: -10점 (기존 -1.0보다 크게)
        elif np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value):
            self.logger.error(f"💥 INVALID PORTFOLIO VALUE (NaN/Inf) at step {self.current_step}. Forcing episode termination.")
            FORCE_TERMINATION = True
            TERMINATION_REASON = "nan_inf_portfolio"
            # 이전 값으로 대체하거나, 안전한 값으로 설정 후 종료
            self.portfolio_value = max(CRITICAL_LOW_THRESHOLD, self.previous_portfolio_value) 
            final_reward_on_termination = -5.0 # 예: -5점
        elif self.portfolio_value > 1e10: # 자본이 비정상적으로 커지는 경우도 방지 (기존 1e9에서 더 높은 값으로)
             self.logger.error(f"💥 EXTREME POSITIVE PORTFOLIO VALUE at step {self.current_step}: {self.portfolio_value:.2f}. Forcing termination.")
             FORCE_TERMINATION = True
             TERMINATION_REASON = "extreme_positive_portfolio"
             self.portfolio_value = 1e10 # 상한선 설정
             final_reward_on_termination = -5.0 

        # Episode forced termination check (after min_episode_steps)
        min_steps_elapsed = (self.current_step - self.window_size) >= self.min_episode_steps
        if FORCE_TERMINATION and min_steps_elapsed:
            self.done = True
            observation = self._get_observation()
            info = self._get_info()
            info["early_termination_reason"] = TERMINATION_REASON
            penalty = (
                self.reward_fn.config.bankruptcy_penalty
                if TERMINATION_REASON == "bankruptcy"
                else self.reward_fn.config.nan_inf_penalty
            )
            return observation, penalty, self.done, False, info
        elif FORCE_TERMINATION and not min_steps_elapsed:
            self.logger.warning(
                f"Portfolio issue ({TERMINATION_REASON}) but delaying termination until "
                f"min_steps {self.min_episode_steps} reached."
            )
            if TERMINATION_REASON in ("bankruptcy", "nan_inf_portfolio"):
                self.portfolio_value = max(CRITICAL_LOW_THRESHOLD, self.previous_portfolio_value)
                self.logger.warning(f"Reverted portfolio value to {self.portfolio_value:.4f} temporarily.")


        # --- END: 강화된 포트폴리오 가치 체크 ---

        # Update peak portfolio value (only when valid)
        if np.isfinite(self.portfolio_value) and self.portfolio_value > 0:
            self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        # Compute reward via MultiComponentReward
        reward, reward_components = self.reward_fn.compute(
            portfolio_value=self.portfolio_value,
            prev_portfolio_value=self.previous_portfolio_value,
            peak_portfolio_value=self.peak_portfolio_value,
            trade_cost=step_trade_cost,
        )

        observation = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_components

        # Early termination (capital gone, min_steps already passed)
        if self.done and self.current_step < len(self.data):
            penalty = self.reward_fn.config.bankruptcy_penalty
            info["early_termination"] = True
            return observation, penalty, True, False, info

        if self.current_step % 10 == 0:
            self.logger.info(
                f"🔄 STEP {self.current_step}: portfolio={self.portfolio_value:.2f}, "
                f"position={self.current_position:.4f}, reward={reward:.4f}, "
                f"components={reward_components}"
            )

        return observation, reward, self.done, False, info

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
            
        # Safety checks for inputs
        if trade_size <= 0:
            self.logger.warning("Invalid trade size for fill rate calculation")
            return 1.0
            
        # Normalize trade size relative to capital
        relative_size = trade_size / max(self.initial_capital, 1e-8)
        
        # Use volume to estimate liquidity - small trades relative to volume are fully filled
        volume_factor = trade_size / (volume + 1e-10)
        
        # Reduced randomness for stability (0.98-1.0 instead of 0.95-1.0)
        randomness = np.random.uniform(0.98, 1.0)
        
        # Calculate fill rate with a minimum
        fill_rate_raw = (1.0 - relative_size * 0.5) * (1.0 - volume_factor * self.volume_slippage_factor) * randomness
        
        # More aggressive clipping for stability
        fill_rate = np.clip(fill_rate_raw, self.min_fill_rate, 1.0)
        
        # DEBUG: Log fill rate calculation
        self.logger.debug(
            f"🔢 FILL RATE CALC: relative_size={relative_size:.6f}, "
            f"volume_factor={volume_factor:.6f}, fill_rate={fill_rate:.4f}"
        )
        
        # Safety check for final value
        if fill_rate < self.min_fill_rate or fill_rate > 1.0 or np.isnan(fill_rate) or np.isinf(fill_rate):
            self.logger.warning(f"❌ Invalid fill rate calculated: {fill_rate}, using min_fill_rate")
            fill_rate = self.min_fill_rate
        
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
            
        # Safety checks for inputs
        if price <= 0 or np.isnan(price) or np.isinf(price):
            self.logger.warning(f"❌ Invalid price for slippage calculation: {price}")
            return 0.0
            
        # Base slippage
        base_slippage = self.slippage_factor
        
        # Volume-based component
        volume_component = 0.0
        if volume > 0:
            volume_component = abs(trade_size * price) / (volume + 1e-10) * self.volume_slippage_factor
        else:
            volume_component = 0.01  # Default to 1% if no volume data
            
        # Reduced random component for stability (0.05 instead of 0.2)
        random_component = np.random.normal(0, 0.05 * base_slippage)
        
        # Total slippage (bounded to reasonable values)
        slippage = base_slippage + volume_component + random_component
        
        # DEBUG: Log slippage calculation
        self.logger.debug(
            f"🔢 SLIPPAGE CALC: base={base_slippage:.6f}, "
            f"volume_component={volume_component:.6f}, "
            f"random_component={random_component:.6f}, "
            f"total={slippage:.6f}"
        )
        
        # Ensure slippage is within reasonable bounds
        slippage = max(0, min(slippage, 0.05))  # Cap at 5%
        
        return slippage

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
        """Return log-return-based observation, clipped to [-10, 10].

        Shape (window_size, 5) normally, or (window_size, 8) when a
        regime_detector is attached (+3 regime probability columns).

        Features per timestep (all relative to previous close):
            col 0: log(open[t]  / close[t-1])
            col 1: log(high[t]  / close[t-1])
            col 2: log(low[t]   / close[t-1])
            col 3: log(close[t] / close[t-1])
            col 4: log(vol[t]   / mean_vol_in_window)

        When regime_detector is set (cols 5-7, same value broadcast over all rows):
            col 5: P(low_vol)
            col 6: P(medium_vol)
            col 7: P(high_vol)

        For t=0 in the window, close[t-1] is data[start_idx-1] if available,
        otherwise data[start_idx] itself (giving log-return = 0 for that row).
        """
        eps = 1e-10
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        # reset() guarantees current_step >= window_size, so start_idx >= 0
        window_data = self.data.iloc[start_idx:end_idx]

        close = window_data["$close"].values.astype(np.float64)
        high  = window_data["$high"].values.astype(np.float64)
        low   = window_data["$low"].values.astype(np.float64)
        open_ = window_data["$open"].values.astype(np.float64)
        vol   = window_data["$volume"].values.astype(np.float64)

        # Reference close for the first row in the window
        if start_idx > 0:
            ref_close = float(self.data.iloc[start_idx - 1]["$close"])
        else:
            # Normal case: slice the window
            window_data = self.data.iloc[start_idx:end_idx]
        
        # Verify we have exactly window_size rows
        if len(window_data) != self.window_size:
            self.logger.warning(
                f"Window data length mismatch: got {len(window_data)}, expected {self.window_size}. "
                f"start_idx={start_idx}, end_idx={end_idx}, current_step={self.current_step}. "
                f"Padding to correct length."
            )
            # Force correct length by either padding or truncating
            if len(window_data) < self.window_size:
                # Pad with first row if we don't have enough data
                pad_size = self.window_size - len(window_data)
                pad_row = window_data.iloc[0] if len(window_data) > 0 else self.data.iloc[0]
                pad_data = pd.DataFrame([pad_row] * pad_size)
                window_data = pd.concat([pad_data, window_data], axis=0)
            else:
                # Truncate if we somehow got too much data
                window_data = window_data.iloc[-self.window_size:]
        
        # Create observation array with OHLCV data
        observation = np.column_stack([
            window_data["$open"].values,
            window_data["$high"].values,
            window_data["$low"].values,
            window_data["$close"].values,
            window_data["$volume"].values,
        ]).astype(np.float32)
        
        # Scale OHLCV data if enabled to prevent numerical instability
        if self.scale_ohlcv:
            # Scale price data (OHLC) and volume separately
            observation[:, 0] /= self.price_scale_factor  # $open
            observation[:, 1] /= self.price_scale_factor  # $high
            observation[:, 2] /= self.price_scale_factor  # $low
            observation[:, 3] /= self.price_scale_factor  # $close
            observation[:, 4] /= self.volume_scale_factor  # $volume

        # Append pre-computed sentiment features (columns 5-8) if available
        if self.sentiment_data is not None:
            if start_idx < 0:
                pad_size_s = abs(start_idx)
                partial_sent = self.sentiment_data.iloc[:end_idx]
                if len(partial_sent) == 0:
                    sent_window = np.zeros(
                        (self.window_size, self._n_sentiment), dtype=np.float32
                    )
                else:
                    pad_row_s = partial_sent.iloc[0].values.astype(np.float32)
                    pad_sent = np.tile(pad_row_s, (pad_size_s, 1))
                    sent_window = np.vstack(
                        [pad_sent, partial_sent.values.astype(np.float32)]
                    )
            else:
                sent_window = self.sentiment_data.iloc[start_idx:end_idx].values.astype(
                    np.float32
                )

            # Ensure correct length (mirror OHLCV padding logic)
            if len(sent_window) < self.window_size:
                pad_size_s2 = self.window_size - len(sent_window)
                pad_row_s2 = (
                    sent_window[0]
                    if len(sent_window) > 0
                    else np.zeros(self._n_sentiment, dtype=np.float32)
                )
                sent_window = np.vstack(
                    [np.tile(pad_row_s2, (pad_size_s2, 1)), sent_window]
                )
            elif len(sent_window) > self.window_size:
                sent_window = sent_window[-self.window_size :]

            observation = np.concatenate([observation, sent_window], axis=1)

        # Final safety check to ensure correct shape
        if observation.shape != (self.window_size, self._n_features):
            self.logger.error(
                f"Observation shape wrong after all processing: {observation.shape}, "
                f"expected ({self.window_size}, {self._n_features}). Forcing correct shape."
            )
            # Create a properly shaped array filled with first row data
            if len(observation) > 0 and observation.shape[1] == self._n_features:
                first_row = observation[0]
            else:
                first_row = np.zeros(self._n_features, dtype=np.float32)

            correct_observation = np.tile(first_row, (self.window_size, 1))

            # Copy as much data as possible from original observation
            if len(observation) > 0:
                copy_rows = min(len(observation), self.window_size)
                correct_observation[-copy_rows:] = observation[:copy_rows]

            observation = correct_observation

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get current state information"""
        drawdown = 0.0
        if self.peak_portfolio_value and self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value

        sharpe_ratio = self.reward_fn.get_sharpe_ratio()
            
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
