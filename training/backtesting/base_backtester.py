import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from pathlib import Path
from dataclasses import dataclass
from risk_management import create_risk_manager, create_risk_config
from risk_management.backtesting_risk_manager import BacktestingRiskConfig

logger = logging.getLogger(__name__)

class BaseBacktester:
    """
    Unified backtester that can handle both single-asset and multi-asset logic.
    
    Features:
    - Supports both single-asset and multi-asset backtesting
    - Tracks portfolio value, positions, and trade history
    - Handles trading fees
    - Position size limits
    - Basic risk management
    - Performance metrics calculation
    - Cost basis tracking and profit realization
    - Enhanced risk management with stop-loss and trailing stops
    - Forced liquidation when drawdown exceeds threshold
    - Time-based position constraints
    - Improved position tracking for weekend close and max holding period tests
    
    Implementation Notes:
    - For single-asset mode, uses 'default' as the asset key
    - For multi-asset mode, uses asset symbols as keys
    - All prices and position data stored in dictionaries for consistency
    - Handles transaction fees for accurate PnL calculation
    - Implements peak value tracking for drawdown calculation
    - Tracks cost basis per position for accurate profit calculation
    - Stop-loss and trailing stops are checked on each bar
    - Forced liquidation can be triggered by excessive drawdown
    
    Recent Changes:
    - Added support for stop-loss and trailing stops
    - Added forced liquidation when max drawdown is exceeded
    - Added time-based position constraints (weekend close, max holding)
    - Added support for partial fills and slippage simulation
    
    Now configured for FRACTIONAL HOLDING:
    - action in [0,1] => fraction of total portfolio to hold in the asset
    - No short selling (units never go below 0)
    """

    REQUIRED_COLUMNS = {"$open", "$high", "$low", "$close", "$volume"}
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,  # 0.1% trading fee
        max_position: float = 1.0,
        data: pd.DataFrame = None,
        risk_config: Optional[BacktestingRiskConfig] = None,
    ):
        """
        Initialize the backtester with common parameters for both single and multi-asset testing.
        
        Args:
            initial_capital (float): Starting capital for the portfolio
            trading_fee (float): Fee per trade as a fraction of trade value (default: 0.1%)
            max_position (float): (Optional) Maximum fraction to allow holding. 
                                  For example, 1.0 means up to 100% of portfolio in coin.
            data (pd.DataFrame, optional): OHLCV data with columns: $open, $high, $low, $close, $volume
            risk_config (BacktestingRiskConfig, optional): Risk management configuration
            
        Notes:
        - Data columns must be prefixed with '$' (e.g., '$close')
        - For multi-asset data, use asset_$column format (e.g., 'BTC_$close')
        """
        if data is not None:
            # Validate required columns for single-asset mode
            if not any("_$" in col for col in data.columns):  # Single-asset mode
                missing_columns = self.REQUIRED_COLUMNS - set(data.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {list(missing_columns)}")
            
            self.data = data.copy()
            
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.max_position = max_position
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize risk manager if config provided
        self.risk_manager = create_risk_manager("backtesting", risk_config.__dict__ if risk_config else None)
        
        self.reset()

    def reset(self):
        """
        Reset portfolio state to initial conditions.
        Clears all positions, trades, and history.
        
        For single-asset mode:
        - Initializes positions with {'default': {'units': 0.0, 'avg_price': 0.0, 'cost_basis': 0.0}}
        
        For multi-asset mode:
        - Initializes empty positions dictionary
        """
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}
        self.trades: List[Dict] = []
        self.portfolio_history: List[float] = [self.initial_capital]
        self.cash_history: List[float] = [self.initial_capital]
        self.peak_value = self.initial_capital
        self.current_timestamp = None
        
        if self.risk_manager:
            self.risk_manager.reset()

    def update(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        actions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Update portfolio based on current prices and desired actions.
        
        Args:
            timestamp: Current timestamp
            prices: Dictionary of prices for each asset
            actions: Dictionary of desired actions for each asset
            
        Returns:
            Dictionary containing:
            - trades: Dict of trade results by asset
            - portfolio_value: Current portfolio value
            - cash: Current cash
            
        Core backtest update function:
        1. Check for risk signals (stop-loss, trailing-stop, max-drawdown)
        2. Apply liquidation trades if needed
        3. Execute regular trades
        4. Return updated state
        """
        self.current_timestamp = timestamp
        trades = {}
        
        # Check for risk signals (stop-loss, trailing-stop, max-drawdown)
        liquidation_signals = self._check_risk_signals(timestamp, prices)
        
        # Apply liquidation trades if needed
        if liquidation_signals:
            for asset, _ in liquidation_signals:
                self.logger.warning(f"Forced liquidation of {asset} at {timestamp}")
                if asset not in prices:
                    continue
                    
                price = prices[asset]
                
                # Force liquidation by setting action to 0 (close position)
                trade_result = self.execute_trade(
                    timestamp=timestamp,
                    action=0.0,  # Force to 0% allocation (full liquidation)
                    price_data={asset: price},
                    asset=asset,
                    is_forced_liquidation=True
                )
                if trade_result:
                    trades[asset] = trade_result
                    self.logger.debug(f"Liquidation trade result: {trade_result}")
        
        # If we had liquidations, don't process regular actions
        # to avoid conflicts between liquidation and regular trades
        if not liquidation_signals:
            # Apply regular trades
            for asset, action in actions.items():
                # Get price for this asset
                if asset not in prices:
                    self.logger.warning(f"No price data for asset {asset}, skipping trade")
                    continue
                    
                price = prices[asset]
                
                # Process the trade
                is_liquidation = any(s == asset for s, _ in liquidation_signals)
                self.logger.debug(f"Processing trade for {asset}, action={action}, is_liquidation={is_liquidation}")
                
                trade_result = self.execute_trade(
                    timestamp=timestamp,
                    action=action,
                    price_data={asset: price},
                    asset=asset,
                    is_forced_liquidation=is_liquidation
                )
                if trade_result:
                    trades[asset] = trade_result
                    self.logger.debug(f"Trade result: {trade_result}")
        
        # Print positions after update for debugging
        self.logger.debug(f"Positions after update: {self.positions}")
        
        # Calculate portfolio value
        portfolio_value = self.get_portfolio_value(prices)
        
        # 포트폴리오 히스토리 업데이트
        self.portfolio_history.append(portfolio_value)
        self.cash_history.append(self.cash)
        
        # Return the results
        return {
            "trades": trades,
            "portfolio_value": portfolio_value,
            "cash": self.cash
        }
    
    def _check_risk_signals(self, timestamp: pd.Timestamp, prices: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Check for risk management signals that might trigger position adjustments.
        
        Args:
            timestamp: Current timestamp
            prices: Dictionary of current prices
            
        Returns:
            List of tuples (symbol, units_to_liquidate) for positions that need adjustment
        """
        liquidation_signals = []
        
        # Skip risk checks if no risk manager is configured
        if self.risk_manager is None:
            return liquidation_signals
        
        # Create a copy of positions to safely iterate
        position_items = list(self.positions.items())
        
        # 1. Check for forced liquidation signal
        forced_liquidation_result = self.risk_manager.check_forced_liquidation()
        forced_liquidation_triggered = False
        
        if isinstance(forced_liquidation_result, dict):
            # New format: check if allowed is False
            forced_liquidation_triggered = not forced_liquidation_result.get("allowed", True)
        else:
            # Old format: check if result is True
            forced_liquidation_triggered = forced_liquidation_result
            
        if forced_liquidation_triggered:
            self.logger.warning("Forced liquidation triggered due to max drawdown")
            # Add all positions to liquidation signals
            for symbol, pos_dict in position_items:
                if symbol in prices and abs(pos_dict["units"]) > 1e-8:
                    liquidation_signals.append((symbol, -pos_dict["units"]))
            return liquidation_signals  # Return early to ensure all positions are liquidated
                
        # 2. Check for stop-loss triggers
        if self.risk_manager.config.use_stop_loss:
            stop_loss_triggers = self.risk_manager.check_stop_losses(prices, self.positions)
            # Create a copy of items to safely iterate
            stop_loss_items = list(stop_loss_triggers.items())
            for symbol, triggered in stop_loss_items:
                if triggered and symbol in self.positions and abs(self.positions[symbol]["units"]) > 1e-8:
                    # Add to liquidation signals if not already included
                    if not any(s == symbol for s, _ in liquidation_signals):
                        liquidation_signals.append((symbol, -self.positions[symbol]["units"]))
                        self.logger.warning(f"Stop loss triggered for {symbol} at price {prices[symbol]:.2f}")
        
        # 3. Check for weekend close signal
        weekend_close_result = self.risk_manager.check_weekend_close(timestamp)
        weekend_close_triggered = False
        
        if isinstance(weekend_close_result, dict):
            # New format: check if allowed is False
            weekend_close_triggered = not weekend_close_result.get("allowed", True)
        else:
            # Old format: check if result is True
            weekend_close_triggered = weekend_close_result
            
        if weekend_close_triggered:
            # Close all positions if it's Friday end of day
            for symbol, pos_dict in position_items:
                if symbol in prices and abs(pos_dict["units"]) > 1e-8:
                    # Add to liquidation signals if not already included
                    if not any(s == symbol for s, _ in liquidation_signals):
                        liquidation_signals.append((symbol, -pos_dict["units"]))
                        self.logger.info(f"Weekend close for {symbol} at price {prices[symbol]:.2f}")
        
        # 4. Check for max holding period exceeded
        if self.risk_manager.config.max_holding_period_days > 0:
            holding_period_exceeded = self.risk_manager.check_max_holding_period(timestamp, self.positions)
            # Create a copy of items to safely iterate
            holding_period_items = list(holding_period_exceeded.items())
            for symbol, exceeded in holding_period_items:
                if exceeded and symbol in self.positions and abs(self.positions[symbol]["units"]) > 1e-8:
                    # Add to liquidation signals if not already included
                    if not any(s == symbol for s, _ in liquidation_signals):
                        liquidation_signals.append((symbol, -self.positions[symbol]["units"]))
                        self.logger.warning(f"Max holding period exceeded for {symbol}, closing position")
        
        return liquidation_signals
    
    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        action: float,
        price_data: Dict[str, float],
        asset: str = "default",
        is_forced_liquidation: bool = False
    ) -> Dict[str, Any]:
        """
        Fractional Holding version:
        action in [0,1] => fraction of total portfolio to hold in this asset.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            action (float): Desired fraction [0,1] of portfolio to hold in this asset
            price_data (Dict[str, float]): Current prices for this asset
            asset (str): Asset symbol
            is_forced_liquidation (bool): Whether this is a forced liquidation due to risk management
        """
        # Save current position state to detect new positions
        previous_positions = {}
        if asset in self.positions:
            previous_positions[asset] = self.positions[asset].copy()
            
        trade = {
            "timestamp": timestamp,
            "symbol": asset,
            "amount": 0.0,  # float
            "price": 0.0,   # float
            "fee": 0.0,     # float
            "cost": 0.0,    # float
            "revenue": 0.0, # float
            "profit": 0.0,  # float
            "success": False,  # bool
            "reason": "",     # str
            "action": float(action),  # float
            "type": "none",   # str (will be "buy" or "sell")
            "value": 0.0,     # float
            "portfolio_value_before": self.get_portfolio_value(price_data),  # float
            "portfolio_value_after": 0.0,  # float
            "cumulative_pnl": 0.0,  # float
            "cash_after": 0.0,      # float
            "position_units": 0.0,   # float
            "position_value": 0.0,   # float
            "is_forced_liquidation": is_forced_liquidation,
        }

        if asset not in price_data:
            trade["reason"] = "price_not_available"
            trade["success"] = False  # explicit bool
            self.trades.append(trade)
            self.logger.debug("[TRADE_SKIP] %s: price_not_available", asset)
            return trade

        current_price = float(price_data[asset])  # ensure float
        
        # Apply slippage if risk manager is configured
        if self.risk_manager and self.risk_manager.config.slippage_std > 0:
            current_price = self.risk_manager.apply_slippage(current_price)
            
        trade["price"] = current_price
        
        # clamp action in [0,1]
        if action < 0.0:
            action = 0.0
        elif action > 1.0:
            action = 1.0
        
        if asset not in self.positions:
            self.positions[asset] = {
                "units": 0.0,
                "avg_price": 0.0,
                "cost_basis": 0.0
            }

        pos_dict = self.positions[asset]
        old_units = float(pos_dict["units"])  # ensure float
        old_cost_basis = float(pos_dict["cost_basis"])  # ensure float

        # current coin value
        current_coin_value = old_units * current_price
        # total portfolio
        portfolio_value = self.cash + current_coin_value
        
        # Check minimum trade size based on action (as a fraction of portfolio)
        if self.risk_manager and not is_forced_liquidation:
            min_trade_size = self.risk_manager.config.min_trade_size
            # Current position as fraction of portfolio
            current_position_fraction = current_coin_value / portfolio_value if portfolio_value > 0 else 0
            # If the requested change in position is less than min_trade_size, reject
            if abs(action - current_position_fraction) < min_trade_size:
                trade["reason"] = "trade_size_too_small"
                trade["success"] = False  # explicit bool
                self.trades.append(trade)
                return trade
        
        # desired coin value
        target_coin_value = action * portfolio_value
        
        diff_value = target_coin_value - current_coin_value
        if abs(diff_value) < 1e-12:
            trade["reason"] = "trade_size_too_small"
            trade["success"] = False  # explicit bool
            self.trades.append(trade)
            return trade

        trade_amount = diff_value / current_price if abs(current_price) > 1e-12 else 0.0
        if abs(trade_amount) < 1e-12:
            trade["reason"] = "trade_size_too_small"
            trade["success"] = False  # explicit bool
            self.trades.append(trade)
            return trade

        trade_value = abs(diff_value)
        fee = trade_value * self.trading_fee

        # For forced liquidations, skip risk check
        if not is_forced_liquidation and self.risk_manager and abs(trade_amount) > 1e-12:
            risk_check = self.risk_manager.check_trade(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                trade_size=trade_amount,
                price=current_price,
                positions=self.positions,
                asset=asset
            )
            if not risk_check['allowed']:
                trade["success"] = False  # explicit bool
                trade["reason"] = risk_check['reason'] if risk_check['reason'] else "risk_check_failed"
                # Ensure we set the adjusted size to 0 for minimum trade size issues
                if "minimum" in trade["reason"].lower() or "size" in trade["reason"].lower():
                    trade["reason"] = "trade_size_too_small"
                self.trades.append(trade)
                return trade

            # if size adjusted
            if abs(risk_check['adjusted_size'] - trade_amount) > 1e-12:
                trade_amount = float(risk_check['adjusted_size'])  # ensure float
                diff_value = trade_amount * current_price
                trade_value = abs(diff_value)
                fee = trade_value * self.trading_fee
                
                # Check if the adjusted size is now too small (not already caught by RiskManager)
                if abs(trade_amount) < 1e-12:
                    trade["reason"] = "trade_size_too_small"
                    trade["success"] = False  # explicit bool
                    self.trades.append(trade)
                    return trade

        # Only log significant trades (value > 1% of portfolio)
        is_significant = trade_value > (portfolio_value * 0.01)
        
        # Determine if this is a long or short position
        is_long = trade_amount > 0 or (trade_amount == 0 and old_units > 0)

        # BUY
        if diff_value > 0:
            trade["type"] = "buy"  # explicit str
            total_cost = trade_value + fee
            if total_cost > self.cash + 1e-12:
                trade["reason"] = "insufficient_funds"
                trade["success"] = False  # explicit bool
                self.trades.append(trade)
                return trade
            
            new_units = old_units + trade_amount
            new_cost_basis = old_cost_basis + total_cost
            avg_price = new_cost_basis/new_units if new_units>1e-12 else 0.0
            
            self.positions[asset] = {
                "units": float(new_units),  # ensure float
                "avg_price": float(avg_price),  # ensure float
                "cost_basis": float(new_cost_basis),  # ensure float
            }
            self.cash -= total_cost

            trade["amount"] = float(trade_amount)  # ensure float
            trade["value"] = float(trade_value)  # ensure float
            trade["fee"] = float(fee)  # ensure float
            trade["cost"] = float(total_cost)  # ensure float
            trade["revenue"] = 0.0  # explicit float
            trade["profit"] = 0.0  # explicit float
            trade["success"] = True  # explicit bool

            if is_significant:
                self.logger.info(
                    "[BUY] %s: Amount=%.6f, Price=$%.2f, Total=$%.2f, Fee=$%.2f",
                    asset, trade_amount, current_price, total_cost, fee
                )

            # Ensure position start time is tracked for max holding period checks
            if asset in self.positions and self.positions[asset]["units"] > 1e-8 and self.risk_manager is not None:
                # Check if this is a new position or the position was previously closed
                was_no_position = asset not in self.risk_manager.position_start_times or asset not in previous_positions
                was_closed_position = asset in previous_positions and abs(previous_positions[asset]["units"]) < 1e-8
                
                self.logger.debug(f"Position start time check: asset={asset}, was_no_position={was_no_position}, was_closed_position={was_closed_position}")
                self.logger.debug(f"Previous positions: {previous_positions}")
                self.logger.debug(f"Current positions: {self.positions}")
                self.logger.debug(f"Position start times before: {self.risk_manager.position_start_times}")
                
                if was_no_position or was_closed_position:
                    self.risk_manager.position_start_times[asset] = timestamp
                    self.logger.debug(f"Setting position start time for {asset}: {timestamp}")
                    
                self.logger.debug(f"Position start times after: {self.risk_manager.position_start_times}")

        else:
            # SELL
            trade["type"] = "sell"  # explicit str
            sell_amount = abs(trade_amount)
            if sell_amount > old_units+1e-12:
                sell_amount = old_units
                diff_value = sell_amount*current_price
                trade_value = abs(diff_value)
                fee = trade_value*self.trading_fee

            fraction = sell_amount/old_units if old_units>1e-12 else 1.0
            cost_portion = old_cost_basis*fraction
            revenue = trade_value
            realized_profit = revenue - cost_portion - fee

            new_units = old_units - sell_amount
            if new_units>1e-12:
                new_cost_basis = old_cost_basis - cost_portion
                self.positions[asset] = {
                    "units": float(new_units),  # ensure float
                    "avg_price": float(pos_dict["avg_price"]),  # ensure float
                    "cost_basis": float(new_cost_basis),  # ensure float
                }
            else:
                del self.positions[asset]

            self.cash += (revenue - fee)

            trade["amount"] = float(-sell_amount)  # ensure float
            trade["value"] = float(revenue)  # ensure float
            trade["fee"] = float(fee)  # ensure float
            trade["cost"] = float(cost_portion)  # ensure float
            trade["revenue"] = float(revenue)  # ensure float
            trade["profit"] = float(realized_profit)  # ensure float
            trade["success"] = True  # explicit bool

            if is_significant:
                log_prefix = "[FORCED LIQUIDATION]" if is_forced_liquidation else "[SELL]"
                self.logger.info(
                    "%s %s: Amount=%.6f, Price=$%.2f, Revenue=$%.2f, Profit=$%.2f",
                    log_prefix, asset, sell_amount, current_price, revenue, realized_profit
                )
        
        # If trade was successful, update portfolio value after and related metrics
        if trade["success"]:  # using bool
            updated_portfolio_value = self.get_portfolio_value(price_data)
            trade["portfolio_value_after"] = float(updated_portfolio_value)  # ensure float
            trade["cumulative_pnl"] = float(updated_portfolio_value - self.initial_capital)  # ensure float
            trade["cash_after"] = float(self.cash)  # ensure float
            
            # Track position details
            if asset in self.positions:
                pos = self.positions[asset]
                trade["position_units"] = float(pos["units"])  # ensure float
                trade["position_value"] = float(pos["units"] * current_price)  # ensure float
            else:
                trade["position_units"] = 0.0  # explicit float
                trade["position_value"] = 0.0  # explicit float
            
            # risk manager post-update
            if self.risk_manager:
                units = float(self.positions[asset]["units"]) if asset in self.positions else 0.0
                entry_price = float(self.positions[asset]["avg_price"]) if asset in self.positions else current_price
                
                # Calculate stop price based on entry price and stop loss threshold
                stop_price = None
                if self.risk_manager.config.use_stop_loss:
                    # Use a fixed value for stop loss threshold to avoid property issues
                    stop_loss_threshold = 0.02  # Default value
                    
                    if units > 0:  # Long position
                        stop_price = entry_price * (1 - stop_loss_threshold)
                    elif units < 0:  # Short position
                        stop_price = entry_price * (1 + stop_loss_threshold)
                
                self.risk_manager.update_after_trade(
                    timestamp=timestamp,
                    asset=asset,
                    entry_price=entry_price,
                    position_size=units,
                    stop_price=stop_price,
                    trailing=self.risk_manager.config.use_trailing_stop
                )
                
                # Ensure position start time is tracked for max holding period checks
                if asset in self.positions and self.positions[asset]["units"] > 1e-8 and self.risk_manager is not None:
                    # Check if this is a new position or the position was previously closed
                    was_no_position = asset not in self.risk_manager.position_start_times or asset not in previous_positions
                    was_closed_position = asset in previous_positions and abs(previous_positions[asset]["units"]) < 1e-8
                    
                    if was_no_position or was_closed_position:
                        self.risk_manager.position_start_times[asset] = timestamp
                        self.logger.debug(f"Setting position start time for {asset}: {timestamp}")
        else:
            # For unsuccessful trades, still record actual portfolio value
            updated_portfolio_value = self.get_portfolio_value(price_data)
            trade["portfolio_value_after"] = float(updated_portfolio_value)  # ensure float
            trade["cumulative_pnl"] = float(updated_portfolio_value - self.initial_capital)  # ensure float
            trade["cash_after"] = float(self.cash)  # ensure float
            trade["position_units"] = float(self.positions[asset]["units"]) if asset in self.positions else 0.0  # ensure float
            trade["position_value"] = float(self.positions[asset]["units"] * current_price) if asset in self.positions else 0.0  # ensure float

        self.trades.append(trade)

        return trade
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and all positions.
        """
        position_value = sum(
            self.positions[asset]["units"] * price
            for asset, price in prices.items()
            if asset in self.positions
        )
        portfolio_value = self.cash + position_value
        
        if self.risk_manager:
            self.risk_manager.update_portfolio_value(portfolio_value)
            
            # Initialize update counter if not exists
            if not hasattr(self, '_update_counter'):
                self._update_counter = 0
                self._last_logged_value = portfolio_value
            
            self._update_counter += 1
            
            # Log on significant events:
            # 1. Significant drawdown (>5%)
            # 2. Significant value change (>2%)
            # 3. Every 100 updates
            current_drawdown = 0
            if self.risk_manager.peak_value is not None and self.risk_manager.peak_value > 0:
                current_drawdown = (self.risk_manager.peak_value - portfolio_value) / self.risk_manager.peak_value
                
            should_log = (
                current_drawdown > 0.05 or  # Significant drawdown
                abs(portfolio_value - self._last_logged_value) / self._last_logged_value > 0.02 or  # Significant value change
                self._update_counter % 100 == 0  # Periodic update
            )
            
            if should_log:
                if current_drawdown > 0.05:
                    self.logger.warning(
                        "High drawdown: %.2f%% (Portfolio: $%.2f, Peak: $%.2f)",
                        current_drawdown * 100,
                        portfolio_value,
                        self.risk_manager.peak_value
                    )
                else:
                    self.logger.info(
                        "Portfolio Update - Value: $%.2f (%.2f%% change), Drawdown: %.2f%%",
                        portfolio_value,
                        ((portfolio_value - self._last_logged_value) / self._last_logged_value) * 100,
                        current_drawdown * 100
                    )
                self._last_logged_value = portfolio_value
                
        return portfolio_value

    def run(
        self,
        strategy: Any,
        window_size: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run full backtest using OHLCV data (from constructor).
        Features:
        - Handles NaNs
        - Progress reporting
        - Returns metrics
        
        Basic usage:
        ```
        backtester = BaseBacktester(data=df)  # df has $open,$high,$low,$close,$volume
        metrics = backtester.run(strategy)
        print(metrics["total_return"], metrics["sharpe_ratio"])
        ```
        """
        if self.data is None:
            raise ValueError("No data provided. Set data in constructor or load_data()")

        try:
            # Reset before running
            self.reset()
            
            # Process one bar at a time
            for i in range(window_size - 1, len(self.data)):
                timestamp = self.data.index[i]
                window_start = i - window_size + 1
                window_data = self.data.iloc[window_start : i + 1]
                
                # 1) Get action from strategy (0.0 to 1.0 for long-only)
                action = None
                current_data = None
                
                try:
                    # For basic strategies
                    if hasattr(strategy, "get_action"):
                        # 에이전트 유형에 따른 입력 데이터 변환
                        agent_name = getattr(strategy, "__class__").__name__
                        is_ppo_agent = "PPO" in agent_name
                        
                        # PPO 에이전트에 대한 특별 처리
                        if is_ppo_agent:
                            # PPO 에이전트는 10차원 입력을 기대합니다
                            # OHLCV 데이터를 flatten하여 10개의 특성으로 변환
                            feature_cols = ["$open", "$high", "$low", "$close", "$volume"]
                            # 마지막 2개 시간 단계의 데이터만 사용 (10개 특성)
                            flat_data = np.array([])
                            for col in feature_cols:
                                flat_data = np.append(flat_data, window_data[col].values[-2:])
                            
                            # 올바른 차원을 가진 배열로 변환
                            if len(flat_data) != 10:
                                # 필요한 경우 크기 조정
                                flat_data = np.pad(flat_data[:10], (0, max(0, 10 - len(flat_data))), 'constant')
                                
                            action = strategy.get_action(flat_data)
                        else:
                            action = strategy.get_action(window_data)
                    else:
                        # For RL models - they use state dict
                        current_data = {col: window_data[col].values for col in window_data.columns}
                        action = strategy.predict(current_data)[0]
                except Exception as e:
                    self.logger.error(
                        f"Error getting action from strategy: {str(e)}"
                    )
                    continue

                # Skip if we couldn't get a valid action
                if action is None:
                    continue

                # 2) Prepare prices and actions dict for update()
                # Use window_data instead of current_data if current_data is None
                if current_data is None:
                    prices = {"default": float(window_data["$close"].iloc[-1])}
                else:
                    # Ensure we're using a scalar value, not a numpy array
                    close_value = current_data["$close"][-1]
                    # Convert to native Python float if it's a numpy type
                    if hasattr(close_value, 'item'):
                        close_value = close_value.item()
                    prices = {"default": float(close_value)}
                
                actions = {"default": float(action) if hasattr(action, 'item') else float(action)}

                # 3) Call update() once per bar
                update_result = self.update(
                    timestamp=timestamp,
                    prices=prices,
                    actions=actions
                )

                if verbose and i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(self.data)} bars processed")

        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise

        # final metrics
        metrics = self._calculate_metrics()

        # align length
        expected_length = len(self.data) - window_size + 1
        if len(self.portfolio_history) < expected_length:
            last_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_capital
            self.portfolio_history.extend([last_value]*(expected_length - len(self.portfolio_history)))
        elif len(self.portfolio_history) > expected_length:
            self.portfolio_history = self.portfolio_history[:expected_length]

        return {
            "metrics": metrics,
            "trades": self.trades,
            "portfolio_values": self.portfolio_history,
            "timestamps": self.data.index[window_size - 1 :].tolist(),
        }

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        
        Returns:
            Dict[str, float] containing:
            - total_return: Total return as decimal (e.g., -0.0864 means -8.64%)
            - sharpe_ratio: Annualized Sharpe ratio (not percentage)
            - sortino_ratio: Annualized Sortino ratio (not percentage)
            - max_drawdown: Maximum drawdown as decimal (e.g., 0.152 means 15.2%)
            - total_trades: Number of successful trades (count)
            - win_rate: Win rate as decimal (e.g., 0.345 means 34.5%)
            - final_balance: Final cash balance (absolute value)
            - final_portfolio_value: Final total portfolio value (absolute value)
            - successful_trades: Number of successfully executed trades (count)
            - total_trade_attempts: Total number of trade attempts (count)
        """
        try:
            values = np.array(self.portfolio_history)
            if len(values) < 2:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'final_balance': self.cash,
                    'final_portfolio_value': self.cash,
                    'successful_trades': 0,
                    'total_trade_attempts': len(self.trades)
                }

            returns = np.diff(values) / values[:-1]  # returns as decimals
            
            # Calculate total return as decimal
            total_return = (values[-1] / values[0]) - 1  # decimal form (e.g., -0.0864)
            
            # Calculate Sharpe ratio (this is a ratio, not a percentage)
            if len(returns) > 1 and np.std(returns)>0:
                sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
            else:
                sharpe_ratio = 0.0
            
            # Calculate Sortino ratio (this is a ratio, not a percentage)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns)>0:
                sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
            else:
                sortino_ratio = 0.0
            
            # Calculate max drawdown as decimal
            peak = values[0]
            max_dd = 0.0  # decimal form (e.g., 0.152)
            for val in values[1:]:
                if val>peak:
                    peak=val
                dd = (peak-val)/peak
                max_dd = max(max_dd, dd)
            
            # Calculate win rate with detailed debugging
            successful_trades = []
            profitable_trades = 0
            total_trades = 0
            sell_trades = 0
            buy_trades = 0
            
            print("\n=== Starting Win Rate Calculation ===")
            print(f"Total trades to analyze: {len(self.trades)}")
            
            # Let's examine first few trades in detail
            for idx, t in enumerate(self.trades[:5]):  # Look at first 5 trades
                print(f"\nTrade {idx+1} Details:")
                print(f"Raw success: {t.get('success')} (type: {type(t.get('success'))})")
                print(f"Raw type: {t.get('type')} (type: {type(t.get('type'))})")
                print(f"Raw profit: {t.get('profit')} (type: {type(t.get('profit'))})")
            
            # Now process all trades
            for idx, t in enumerate(self.trades):
                # Get raw values first for debugging
                raw_success = t.get("success")
                raw_type = t.get("type")
                raw_profit = t.get("profit")
                
                # Now process them
                success = (isinstance(raw_success, bool) and raw_success) or \
                         (isinstance(raw_success, str) and raw_success.lower() == "true")
                
                trade_type = str(raw_type).strip().lower() if raw_type else "none"
                
                try:
                    profit = float(str(raw_profit).replace("$", "").replace(",", "").strip()) \
                            if raw_profit is not None else 0.0
                except (ValueError, TypeError):
                    profit = 0.0
                
                if success:
                    total_trades += 1
                    if trade_type == "sell":
                        sell_trades += 1
                        if profit > 0:
                            profitable_trades += 1
                            if idx < 5:  # Print details for first 5 profitable trades
                                print(f"\nFound profitable sell trade #{idx+1}:")
                                print(f"  Processed success: {success}")
                                print(f"  Processed type: {trade_type}")
                                print(f"  Processed profit: {profit}")
                    elif trade_type == "buy":
                        buy_trades += 1
                        portfolio_before = float(t.get("portfolio_value_before", 0))
                        portfolio_after = float(t.get("portfolio_value_after", 0))
                        if portfolio_after > portfolio_before:
                            profitable_trades += 1
            
            # Calculate win rate
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
            
            print("\n=== Win Rate Calculation Summary ===")
            print(f"Total trades processed: {len(self.trades)}")
            print(f"Successful trades: {total_trades}")
            print(f"Buy trades: {buy_trades}")
            print(f"Sell trades: {sell_trades}")
            print(f"Profitable trades: {profitable_trades}")
            print(f"Final win rate: {win_rate:.1%}")
            
            metrics = {
                'total_return': (values[-1] / values[0]) - 1,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_dd,
                'total_trades': total_trades,
                'win_rate': win_rate,  # This is now correctly stored
                'final_balance': self.cash,
                'final_portfolio_value': values[-1],
                'successful_trades': total_trades,
                'total_trade_attempts': len(self.trades)
            }
            
            # Log the metrics we're returning
            print("\nReturning metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
                
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'final_balance': self.cash,
                'final_portfolio_value': self.cash,
                'successful_trades': 0,
                'total_trade_attempts': len(self.trades)
            }

    def get_returns(self) -> pd.Series:
        """
        Calculate returns series.
        """
        returns = pd.Series(self.portfolio_history).pct_change()
        if self.current_timestamp is not None:
            returns.index = pd.date_range(
                start=self.current_timestamp - pd.Timedelta(days=len(returns) - 1),
                end=self.current_timestamp,
                periods=len(returns),
            )
        return returns
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Get position value history for each asset.
        
        Returns:
            pd.DataFrame: DataFrame containing position history with timestamps and values
        """
        position_values = []
        if self.current_timestamp is not None:
            timestamps = pd.date_range(
                start=self.current_timestamp - pd.Timedelta(days=len(self.portfolio_history) - 1),
                end=self.current_timestamp,
                periods=len(self.portfolio_history),
            )
        else:
            timestamps = range(len(self.portfolio_history))
        
        for timestamp, portfolio_value in zip(timestamps, self.portfolio_history):
            row = {'timestamp': timestamp, 'total': portfolio_value}
            # Make a copy of the positions dictionary to avoid modification during iteration
            positions_copy = self.positions.copy()
            for asset, pos in positions_copy.items():
                row[f"{asset}_units"] = pos["units"]
                row[f"{asset}_value"] = pos["units"] * pos["avg_price"]
            position_values.append(row)
        
        return pd.DataFrame(position_values)

    def run_scenario(
        self,
        strategy: Any,
        scenario_type: str,
        window_size: int = 20,
        verbose: bool = True,
        **scenario_params
    ) -> Dict[str, Any]:
        """
        Run backtest with a specific scenario (flash_crash, low_liquidity).
        
        Args:
            strategy: Strategy object with get_action() method
            scenario_type: Type of scenario ('flash_crash' or 'low_liquidity')
            window_size: Size of window for strategy
            verbose: Whether to print progress
            scenario_params: Parameters for scenario generation
            
        Returns:
            Dict containing results and metrics
        """
        from .scenario import (
            generate_flash_crash_data,
            generate_low_liquidity_data,
            calculate_flash_crash_metrics,
            calculate_low_liquidity_metrics
        )
        
        # Generate scenario data
        if scenario_type == "flash_crash":
            self.data = generate_flash_crash_data(**scenario_params)
            metric_fn = calculate_flash_crash_metrics
        elif scenario_type == "low_liquidity":
            self.data = generate_low_liquidity_data(**scenario_params)
            metric_fn = calculate_low_liquidity_metrics
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Reset before running the scenario
        self.reset()
        
        # Run the backtest
        results = self.run(strategy, window_size, verbose)
        
        # Calculate scenario-specific metrics
        results["scenario_metrics"] = metric_fn(results)
        results["scenario_type"] = scenario_type
        results["scenario_params"] = scenario_params
        
        return results

    def plot_scenario_results(
        self,
        results: Dict,
        save_path: str = None
    ):
        """Plot results with scenario-specific annotations."""
        from .scenario import plot_scenario_results
        plot_scenario_results(
            results=results,
            scenario_type=results["scenario_type"],
            save_path=save_path
        )

    def save_scenario_results(
        self,
        results: Dict,
        save_dir: str
    ):
        """Save scenario-specific results and plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save standard results
        self.save_results(results, save_dir)

        # Save scenario-specific metrics
        scenario_df = pd.DataFrame([results["scenario_metrics"]])
        scenario_df.to_csv(
            save_dir / f"{results['scenario_type']}_metrics.csv",
            index=False
        )

        # Save enhanced plots
        self.plot_scenario_results(
            results,
            save_path=str(save_dir / f"{results['scenario_type']}_plot.png")
        )

        logger.info(f"Scenario results saved to {save_dir}")

    def save_results(self, results: Dict, save_dir: Path):
        """Helper to save backtest results (trades, portfolio, etc.) to CSV."""
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results["trades"]).to_csv(save_dir / "trades.csv", index=False)
        pd.DataFrame({"portfolio_values": results["portfolio_values"]}).to_csv(save_dir / "portfolio.csv", index=False)
        pd.DataFrame({"timestamps": results["timestamps"]}).to_csv(save_dir / "timestamps.csv", index=False)
        pd.DataFrame([results["metrics"]]).to_csv(save_dir / "metrics.csv", index=False)
        logger.info(f"Backtest results saved to {save_dir}")
