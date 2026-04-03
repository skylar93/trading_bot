import gymnasium as gym
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional
import logging
import collections

from envs.market_impact import AlmgrenChrissImpact

if TYPE_CHECKING:
    from agents.offline.dt_forecaster import DTForecaster

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
        sharpe_lookback: int = 60,
        sharpe_weight: float = 0.5,
        drawdown_penalty: bool = True,
        max_drawdown_penalty_threshold: float = 0.1,
        # Friction parameters
        apply_slippage: bool = True,
        slippage_factor: float = 0.0005,
        partial_fills: bool = True,
        min_fill_rate: float = 0.8,
        volume_slippage_factor: float = 0.1,
        # Additional stability parameters
        scale_ohlcv: bool = True,
        price_scale_factor: float = 1000.0,
        volume_scale_factor: float = 1e6,
        min_episode_steps: int = 30,
        reward_scale: float = 1.0,
        # Optional pre-computed sentiment features (Week 13)
        sentiment_data: Optional[pd.DataFrame] = None,
        # Market impact model (Week 21 — Almgren-Chriss)
        use_market_impact: bool = False,
        market_impact_model: str = "sqrt",
        market_impact_sigma: float = 0.02,
        market_impact_kappa: float = 0.5,
        market_impact_eta: float = 0.01,
        market_impact_gamma: float = 0.001,
        # Week 22: DTForecaster — inject return predictions into observation
        dt_forecaster: Optional["DTForecaster"] = None,
        # Week 30: optional risk manager for regime-based position sizing
        risk_manager=None,
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
            scale_ohlcv: Whether to scale OHLCV data to prevent numerical instability
            price_scale_factor: Factor to scale price data (OHLC)
            volume_scale_factor: Factor to scale volume data
            min_episode_steps: Minimum number of steps before allowing early termination
            reward_scale: Factor to scale rewards (smaller values = more stable)
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

        # Market impact model (Week 21)
        self.market_impact: Optional[AlmgrenChrissImpact] = None
        if use_market_impact:
            self.market_impact = AlmgrenChrissImpact(
                model=market_impact_model,
                sigma=market_impact_sigma,
                kappa=market_impact_kappa,
                eta=market_impact_eta,
                gamma=market_impact_gamma,
            )
            self.logger.info(
                f"Market impact model enabled: AlmgrenChriss model='{market_impact_model}'"
            )
        
        # Stability parameters
        self.scale_ohlcv = scale_ohlcv
        self.price_scale_factor = price_scale_factor
        self.volume_scale_factor = volume_scale_factor
        self.min_episode_steps = min_episode_steps
        self.reward_scale = reward_scale

        # Week 22: DTForecaster (optional) — adds 3 extra features per timestep
        # Features added: [return_1step, return_5step, confidence]
        self.dt_forecaster = dt_forecaster
        self._n_dt_forecast = 3 if dt_forecaster is not None else 0
        if dt_forecaster is not None:
            self.logger.info(
                "DTForecaster attached — observation will include 3 forecast features"
            )

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
        self._n_features = 5 + self._n_sentiment + self._n_dt_forecast  # OHLCV + optional sentiment + optional DT forecast

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

        # Week 30: regime-based position sizing
        self._risk_manager = risk_manager
        self._regime_probs = None  # 외부에서 set_regime_probs()로 업데이트
        self._entry_price: Optional[float] = None  # Week 37: entry price for stop loss tracking

        logger.info(
            f"Initialized TradingEnvironment with window_size={window_size}, "
            f"initial_capital={initial_capital}, trading_fee={trading_fee}, "
            f"risk_adjusted_reward={risk_adjusted_reward}, apply_slippage={apply_slippage}"
        )
        
        # STEP 1-A: Check if data is long enough
        if self.data is not None:
            if len(self.data) < self.window_size + 1:
                raise ValueError(
                    f"Data too short ({len(self.data)}) for window_size={self.window_size}. "
                    f"Need at least window_size+1 rows."
                )

    def set_regime_probs(self, regime_probs) -> None:
        """Week 30: 외부에서 HMM regime 확률을 주입한다. step()에서 position sizing에 반영."""
        self._regime_probs = regime_probs

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
        self._entry_price = None
        if self._risk_manager is not None:
            self._risk_manager.reset()

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
        raw_action = float(action[0])
        # Week 30: apply regime-based position sizing if available
        if self._risk_manager is not None and self._regime_probs is not None:
            raw_action = self._risk_manager.adjust_for_regime(raw_action, self._regime_probs)
        position_change = raw_action * self.max_position_size
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
        
        # Initialize reward debug information
        reward_debug = {
            "basic_reward": 0.0,
            "sharpe_component": 0.0,
            "drawdown_penalty": 0.0,
            "final_reward": 0.0,
            "portfolio_change": 0.0,
            "pre_portfolio": self.previous_portfolio_value,
            "post_portfolio": 0.0,
        }
        
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
            _prev_position = self.current_position
            self.current_position += actual_change

            # Week 37: update entry price for stop loss tracking
            if abs(_prev_position) < 1e-8 and abs(self.current_position) >= 1e-8:
                self._entry_price = executed_price  # flat → non-flat: new position opened
            elif abs(self.current_position) < 1e-8:
                self._entry_price = None  # position fully closed
            elif (_prev_position > 0) != (self.current_position > 0):
                self._entry_price = executed_price  # direction flipped

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

        # 에피소드 강제 종료 조건 확인 (최소 스텝 이후)
        min_steps_elapsed = (self.current_step - self.window_size) >= self.min_episode_steps
        if FORCE_TERMINATION and min_steps_elapsed:
            self.done = True
            observation = self._get_observation() # 최종 상태 가져오기
            info = self._get_info()
            info["early_termination_reason"] = TERMINATION_REASON
            info["reward_debug"] = reward_debug # Add reward debug info
            info["reward_debug"]["final_reward"] = final_reward_on_termination # Overwrite final reward
            
            # 종료 시 최종 보상을 설정 (위에서 정의한 값 사용)
            # reward_debug 업데이트는 생략하거나 기본값으로 둘 수 있음
            # 주의: 이 return 문은 reward 계산 로직 전에 위치해야 함
            return observation, final_reward_on_termination, self.done, False, info
        elif FORCE_TERMINATION and not min_steps_elapsed:
            self.logger.warning(f"Portfolio value issue detected ({TERMINATION_REASON}) but delaying termination until min_steps {self.min_episode_steps} reached.")
            # 경고는 하지만 일단 진행 (최소 스텝 보장 위해). 단, reward 계산 시 문제 발생 가능성 있음.
            # 이 경우, 아래 reward 계산 로직에서 여전히 문제가 발생할 수 있으므로 주의 필요.
            # 안전하게 하려면, 문제가 발생했을 때 포트폴리오 가치를 이전 값으로 되돌리는 로직 유지
            if TERMINATION_REASON == "bankruptcy" or TERMINATION_REASON == "nan_inf_portfolio":
                 self.portfolio_value = max(CRITICAL_LOW_THRESHOLD, self.previous_portfolio_value)
                 self.logger.warning(f"Reverted portfolio value to {self.portfolio_value:.4f} temporarily.")


        # --- END: 강화된 포트폴리오 가치 체크 ---
        
        # Update peak portfolio value for drawdown calculation
        # 포트폴리오 가치가 유효한 경우에만 peak 업데이트
        if not (np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value) or self.portfolio_value <= 0):
             self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        # --- Week 37: hard risk limit enforcement ---
        _risk_limit_info: Dict[str, Any] = {}
        if self._risk_manager is not None and abs(self.current_position) >= 1e-8:
            _risk_triggered, _risk_reason = self._enforce_risk_limits(current_price)
            if _risk_triggered:
                _risk_limit_info["risk_limit_triggered"] = _risk_reason
                # Position is now 0; recalc portfolio value
                self.portfolio_value = self.current_capital
                if _risk_reason == "max_drawdown":
                    self.done = True

        # Calculate basic reward (change in portfolio value) using log returns with ratio clipping
        eps = 1e-8 # Small epsilon to prevent division by zero
        current_val = max(self.portfolio_value, eps)
        previous_val = max(self.previous_portfolio_value, eps)

        ratio = current_val / previous_val

        # Winsorization/Clipping the ratio before taking log
        # (예: 99% 손실 ~ 100배 이익 범위까지만 허용)
        ratio_clipped = np.clip(ratio, 0.01, 100.0) 

        log_return = np.log(ratio_clipped) # 이제 log_return 값은 극단적으로 튀지 않음

        reward_step_raw = log_return # 스케일링 없이 사용

        # 클리핑 범위는 Winsorization 후의 로그 리턴 범위에 맞게 설정 (예: log(0.01) ~ log(100))
        # 이 범위는 약 -4.6 ~ +4.6 이므로, [-5.0, 5.0] 정도면 충분할 수 있음
        REWARD_CLIP_RANGE = 5.0 

        # Final clipping check (should rarely trigger now)
        if np.isnan(reward_step_raw) or np.isinf(reward_step_raw) or abs(reward_step_raw) > REWARD_CLIP_RANGE:
             self.logger.warning(f"⚠️ CLAMPED LOG RETURN after ratio clipping: {reward_step_raw:.4f}, capping to range [{-REWARD_CLIP_RANGE}, {REWARD_CLIP_RANGE}]")
             if np.isnan(reward_step_raw):
                 reward_step = 0.0
             else:
                 reward_step = np.clip(reward_step_raw, -REWARD_CLIP_RANGE, REWARD_CLIP_RANGE)
        else:
             reward_step = reward_step_raw

        # reward_scale 적용 (이건 유지하거나 필요시 조정)
        reward_step = reward_step * self.reward_scale
            
        # Update reward debug information
        reward_debug["basic_reward"] = reward_step # Store the final scaled and clipped reward
        # Store the unscaled log return for comparison
        reward_debug["portfolio_change"] = log_return 
        reward_debug["post_portfolio"] = self.portfolio_value
        
        # Update returns buffer for Sharpe calculation
        # Use the scaled and clipped reward_step for the buffer
        self.returns_buffer.append(reward_step) 
        
        # Calculate final reward with risk adjustment
        try:
            reward = self._calculate_risk_adjusted_reward(reward_step, reward_debug)
            
            # Apply final tighter reward clipping (±5 instead of ±100)
            if np.isnan(reward) or np.isinf(reward) or abs(reward) > 5.0:
                self.logger.warning(f"❌ FINAL REWARD IS INVALID: {reward}, fallback to [-5, 5]")
                if np.isnan(reward):
                    reward = 0.0
                else:
                    reward = np.clip(reward, -5.0, 5.0)
        except Exception as e:
            self.logger.error(f"❌ ERROR calculating risk-adjusted reward: {str(e)}")
            # Use the adjusted clip range for the fallback reward
            reward = np.clip(reward_step, -REWARD_CLIP_RANGE * self.reward_scale, REWARD_CLIP_RANGE * self.reward_scale) if not np.isnan(reward_step) else 0.0
        
        # Update info with reward debug
        observation = self._get_observation()
        info = self._get_info()
        info["reward_debug"] = reward_debug
        info.update(_risk_limit_info)

        # If we decided self.done = True above, we can forcibly end now
        if self.done and self.current_step < len(self.data):  # Only early termination, not normal end
            # Smaller penalty for capital <= 1.0
            if self.current_capital <= 1.0:
                reward = -1.0  # Reduced from -10.0 to -1.0 for stability
            elif self.portfolio_value < 1.0:
                reward = -0.5  # Reduced from -5.0 to -0.5 for stability

            observation = self._get_observation()
            info = self._get_info()
            info["reward_debug"] = reward_debug
            info["early_termination"] = True  # Add flag for agent to know this was early termination
            info.update(_risk_limit_info)

            return observation, reward, True, False, info

        # DEBUG: Log step summary periodically
        if self.current_step % 10 == 0:
            self.logger.info(
                f"🔄 STEP {self.current_step}: portfolio={self.portfolio_value:.2f}, "
                f"position={self.current_position:.4f}, reward={reward:.4f}"
            )
            self.logger.info(f"📈 REWARD DEBUG: {reward_debug}")

        return observation, reward, self.done, False, info

    def _enforce_risk_limits(self, current_price: float) -> Tuple[bool, str]:
        """Week 37: check risk manager limits and force-close position when triggered.

        Checks (in priority order):
        1. Stop loss (uses entry_price vs current_price)
        2. Trailing stop (uses peak price tracked by risk manager)
        3. Max drawdown (uses peak_portfolio_value vs current portfolio_value)

        Returns:
            (triggered, reason) — reason is 'stop_loss' | 'trailing_stop' | 'max_drawdown' | ''
        """
        if self._risk_manager is None or abs(self.current_position) < 1e-8:
            return False, ""

        # 1. Stop loss
        if self._entry_price is not None:
            if self._risk_manager.check_stop_loss(
                "env", self.current_position, self._entry_price, current_price
            ):
                self._force_close_position(current_price, "stop_loss")
                return True, "stop_loss"

        # 2. Trailing stop
        if self._risk_manager.check_trailing_stop(
            "env", "asset", self.current_position, current_price
        ):
            self._force_close_position(current_price, "trailing_stop")
            return True, "trailing_stop"

        # 3. Max drawdown
        if self._risk_manager.check_max_drawdown(
            "env", self.peak_portfolio_value, self.portfolio_value
        ):
            self._force_close_position(current_price, "max_drawdown")
            return True, "max_drawdown"

        return False, ""

    def _force_close_position(self, current_price: float, reason: str) -> None:
        """Week 37: close entire position at current_price with fee only (no slippage)."""
        trade_value = abs(self.current_position) * current_price
        trade_cost = trade_value * self.trading_fee

        if self.current_position > 0:  # Long: sell
            self.current_capital += trade_value - trade_cost
        else:  # Short: buy back
            self.current_capital -= trade_value + trade_cost

        self.logger.warning(
            f"Risk limit ({reason}) at step {self.current_step}: "
            f"force-closed {self.current_position:.4f} @ {current_price:.4f}, "
            f"capital after={self.current_capital:.2f}"
        )
        self.last_trade_size = trade_value
        self.current_position = 0.0
        self._entry_price = None

    def _calculate_risk_adjusted_reward(self, basic_reward: float, reward_debug: dict = None) -> float:
        """
        Calculate risk-adjusted reward incorporating Sharpe ratio and drawdown penalties.
        
        Args:
            basic_reward: The basic reward (change in portfolio value)
            reward_debug: Optional dictionary to store reward components for debugging
            
        Returns:
            float: Risk-adjusted reward
        """
        # Start with basic reward
        final_reward = basic_reward
        
        # Calculate Sharpe component if enabled and we have enough data
        sharpe_component = 0.0
        if self.risk_adjusted_reward and len(self.returns_buffer) > 3:
            try:
                # Calculate mean and standard deviation of returns
                returns_array = np.array(list(self.returns_buffer))
                
                # Check for NaN/Inf values in returns buffer
                if np.any(np.isnan(returns_array)) or np.any(np.isinf(returns_array)):
                    self.logger.warning(f"❌ NaN/Inf values in returns buffer: {returns_array}")
                    # Clean up the returns array
                    returns_array = np.array([r for r in returns_array if not np.isnan(r) and not np.isinf(r)])
                    if len(returns_array) < 3:
                        # Not enough valid returns, skip Sharpe calculation
                        if reward_debug is not None:
                            reward_debug["sharpe_component"] = 0.0
                        return basic_reward
                
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array) + 1e-8  # avoid division by zero
                
                # DEBUG: Log Sharpe calculation details
                self.logger.debug(f"📊 SHARPE CALC: mean={mean_return:.6f}, std={std_return:.6f}, n={len(returns_array)}")
                
                sharpe_proxy = mean_return / std_return * np.sqrt(252)

                # Avoid extreme values
                sharpe_proxy = np.clip(sharpe_proxy, -10.0, 10.0)
                sharpe_component = sharpe_proxy
                
                # Mix in the Sharpe component
                if self.sharpe_weight > 0:
                    final_reward = (1 - self.sharpe_weight) * basic_reward + self.sharpe_weight * sharpe_proxy
                    
                # DEBUG: Log Sharpe contribution
                self.logger.debug(f"📈 SHARPE CONTRIB: sharpe={sharpe_proxy:.4f}, weight={self.sharpe_weight:.2f}")
                
                # Store in debug info if provided
                if reward_debug is not None:
                    reward_debug["sharpe_component"] = sharpe_component
            except Exception as e:
                self.logger.error(f"❌ ERROR in Sharpe calculation: {str(e)}")
                # Keep the basic reward if there's an error
                if reward_debug is not None:
                    reward_debug["sharpe_component"] = 0.0
        
        # Calculate drawdown penalty if enabled
        drawdown_penalty = 0.0
        if self.drawdown_penalty and self.peak_portfolio_value > 0:
            try:
                drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
                
                # DEBUG: Log drawdown details
                self.logger.debug(f"📉 DRAWDOWN: current={drawdown:.4f}, threshold={self.max_drawdown_penalty_threshold:.4f}")
                
                # Apply penalty if drawdown exceeds threshold
                if drawdown > self.max_drawdown_penalty_threshold:
                    # Penalty scales with severity of drawdown
                    penalty_factor = 1.0 + (drawdown - self.max_drawdown_penalty_threshold) * 10.0
                    # Scale the penalty based on the drawer threshold
                    penalty = -0.1 * penalty_factor * drawdown
                    
                    # Clip penalty to prevent extreme values
                    penalty = np.clip(penalty, -1.0, 0.0)
                    
                    # DEBUG: Log penalty details
                    self.logger.debug(f"⚠️ DRAWDOWN PENALTY: factor={penalty_factor:.4f}, penalty={penalty:.4f}")
                    
                    drawdown_penalty = penalty
                    final_reward += penalty
                    
                # Store in debug info if provided
                if reward_debug is not None:
                    reward_debug["drawdown_penalty"] = drawdown_penalty
            except Exception as e:
                self.logger.error(f"❌ ERROR in drawdown calculation: {str(e)}")
                # Keep the reward without drawdown penalty if there's an error
                if reward_debug is not None:
                    reward_debug["drawdown_penalty"] = 0.0
        
        # Final safety check for reward value
        if np.isnan(final_reward) or np.isinf(final_reward):
            self.logger.warning(f"❌ INVALID FINAL REWARD: {final_reward}, using basic reward")
            final_reward = basic_reward
            
        # Store in debug info if provided
        if reward_debug is not None:
            reward_debug["final_reward"] = final_reward
        
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

        # Delegate to Almgren-Chriss model when enabled (Week 21)
        if self.market_impact is not None:
            return self.market_impact.compute(
                shares=abs(trade_size),
                price=price,
                daily_volume=max(volume, 1e-8),
            )

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
        """Get current observation (window of OHLCV data)
        
        Returns:
            np.ndarray: Observation with shape (window_size, 5) containing OHLCV data.
            If current_step < window_size, the observation is padded with the first row's data.
            
        Note:
            The policy network expects a flattened observation vector of shape (window_size * 5,)
            but we return the original shape (window_size, 5) here since the policy network
            will handle reshaping internally.
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

        # Week 22: Append DTForecaster predictions as 3 extra constant columns
        # Columns: [return_1step, return_5step, confidence] — same value in every row
        # so that downstream 2-D feature extractors (CNN/LSTM/GTrXL) see them at each step.
        if self.dt_forecaster is not None and self._n_dt_forecast > 0:
            try:
                # Use only the base OHLCV window as input to the forecaster
                # (not the augmented observation, to keep inputs stable)
                base_obs = observation[:, :5]  # (window_size, 5)
                pred = self.dt_forecaster.predict(base_obs)
                forecast_row = np.array(
                    [pred["return_1step"], pred["return_5step"], pred["confidence"]],
                    dtype=np.float32,
                )
                # Broadcast to (window_size, 3)
                forecast_cols = np.tile(forecast_row, (self.window_size, 1))
                observation = np.concatenate([observation, forecast_cols], axis=1)
            except Exception as e:
                self.logger.warning(
                    "DTForecaster prediction failed (%s) — filling with zeros", e
                )
                observation = np.concatenate(
                    [observation, np.zeros((self.window_size, 3), dtype=np.float32)],
                    axis=1,
                )

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
