from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from .backtest_engine import BacktestEngine
from risk.risk_manager import RiskConfig

class ExperimentalBacktester(BacktestEngine):
    """Experimental backtester with improved position management and PnL calculation."""
    
    def __init__(self, data: pd.DataFrame, risk_config: RiskConfig, initial_balance: float = 10000.0):
        super().__init__(
            initial_capital=initial_balance,
            transaction_cost=0.001,  # Default trading fee
            max_position=risk_config.max_position_size
        )
        self.data = data
        self.risk_config = risk_config
        self.entry_prices: Dict[str, float] = {}  # Track entry prices per asset
        self.positions: Dict[str, float] = {}     # Track positions per asset
        self.trading_fee = 0.001  # For compatibility with tests
        self.trade_sizes: Dict[str, Dict[float, float]] = {}  # Track trade sizes and prices for entry price calculation
    
    def _calculate_trade_revenue(self, asset: str, trade_size: float, current_price: float) -> Dict[str, float]:
        """Calculate trade revenue and PnL for position closure."""
        old_pos = self.positions.get(asset, 0)
        entry_price = self.entry_prices.get(asset, current_price)
        
        if trade_size < 0 and old_pos > 0:  # Closing long position
            # Calculate actual closure size
            close_size = min(abs(trade_size), old_pos)
            
            # Calculate gross revenue and fees
            gross_revenue = close_size * current_price
            fee = gross_revenue * self.trading_fee
            net_revenue = gross_revenue - fee
            
            # Calculate realized PnL
            realized_pnl = close_size * (current_price - entry_price) - fee
            
            return {
                "revenue": net_revenue,
                "pnl": realized_pnl,
                "close_size": close_size
            }
        
        # For new positions or position increases
        trade_value = abs(trade_size) * current_price
        fee = trade_value * self.trading_fee
        
        if trade_size > 0:  # Opening/increasing long position
            return {
                "cost": trade_value + fee,
                "pnl": -fee,
                "close_size": 0
            }
        else:  # Opening short position (if allowed)
            return {
                "revenue": trade_value - fee,
                "pnl": -fee,
                "close_size": 0
            }
    
    def execute_trade(self, timestamp: pd.Timestamp, action: float, price_data: Dict[str, float]) -> Dict[str, Any]:
        """Execute trade with improved position management and PnL calculation."""
        print(f"\nExecuting trade at {timestamp}:")
        print(f"Action: {action}")
        print(f"Price data: {price_data}")
        
        if abs(action) < self.risk_config.min_trade_size:
            return {
                "timestamp": timestamp,
                "action": "skip",
                "reason": "action too small",
                "position": sum(self.positions.values())
            }
        
        current_price = price_data.get("$close")
        if current_price is None:
            current_price = price_data.get("close")
        if current_price is None:
            raise ValueError("Price data must contain '$close' or 'close'")
        
        print(f"Current price: {current_price}")
        
        # Calculate portfolio value and maximum allowed position
        portfolio_value = self.get_portfolio_value({"default": current_price})
        max_position_value = portfolio_value * self.risk_config.max_position_size * 0.98  # Add 2% buffer
        max_position_size = max_position_value / current_price
        
        print(f"Portfolio value: {portfolio_value}")
        print(f"Max position value: {max_position_value}")
        print(f"Max position size: {max_position_size}")
        
        # For position closure, use the actual position size
        asset = "default"
        current_position = self.positions.get(asset, 0)
        
        if action < 0 and current_position > 0:  # Closing long position
            # Calculate closure size based on current position
            # action = -0.25 means close 50% of current position
            close_size = current_position * (2 * abs(action))
            trade_size = -close_size
            
            # For complete closure (action <= -0.5), close entire position
            if abs(action) >= 0.5:
                trade_size = -current_position
        else:  # Opening new position or adding to position
            # Calculate target position size based on action
            # action = 0.5 means target 50% of max_position_size
            target_size = max_position_size * abs(action)  # Use abs(action) to handle negative actions
            
            if action > 0:  # Buy
                # Calculate additional size to buy
                trade_size = target_size
            else:  # Sell
                trade_size = -target_size
            
            # Ensure we don't exceed position limits
            new_position = current_position + trade_size
            new_position_value = new_position * current_price
            if new_position_value > max_position_value:
                trade_size = (max_position_value / current_price) - current_position
        
        print(f"Trade size: {trade_size}")
        
        # Calculate trade details
        trade_details = self._calculate_trade_revenue(asset, trade_size, current_price)
        
        # Prepare trade result
        trade_result = {
            "timestamp": timestamp,
            "type": "buy" if trade_size > 0 else "sell",
            "price": current_price,
            "size": abs(trade_size),
            "position": current_position,  # Store current position before update
            "balance": self.cash,
            "portfolio_value": self.get_portfolio_value({"default": current_price}),
            "entry_price": self.entry_prices.get(asset, current_price)  # Add entry price to result
        }
        
        # Update positions and entry prices
        if trade_size < 0 and current_position > 0:  # Position closure
            close_size = min(abs(trade_size), current_position)
            new_position = current_position - close_size
            
            if abs(new_position) < 1e-8 or abs(action) >= 0.5:  # Complete closure
                del self.positions[asset]
                del self.entry_prices[asset]
                if asset in self.trade_sizes:
                    del self.trade_sizes[asset]
            else:  # Partial closure
                self.positions[asset] = new_position
        elif trade_size > 0:  # Opening/increasing long position
            new_position = current_position + trade_size
            self.positions[asset] = new_position
            
            # Track trade size and price for entry price calculation
            if asset not in self.trade_sizes:
                self.trade_sizes[asset] = {}
            self.trade_sizes[asset][current_price] = trade_size
            
            print(f"\nTrade details:")
            print(f"Current position: {current_position}")
            print(f"New position: {new_position}")
            print(f"Trade sizes: {self.trade_sizes[asset]}")
            
            # Calculate weighted average entry price
            total_size = sum(size for size in self.trade_sizes[asset].values())
            weighted_sum = sum(price * size for price, size in self.trade_sizes[asset].items())
            self.entry_prices[asset] = weighted_sum / total_size
            
            print(f"Total size: {total_size}")
            print(f"Weighted sum: {weighted_sum}")
            print(f"New entry price: {self.entry_prices[asset]}")
        
        # Update cash balance
        if "cost" in trade_details:
            self.cash -= trade_details["cost"]
        if "revenue" in trade_details:
            self.cash += trade_details["revenue"]
        
        # Add revenue/cost and PnL
        trade_result.update({k: v for k, v in trade_details.items() 
                           if k in ["revenue", "cost", "pnl"]})
        
        # Update position in result after all changes
        trade_result["position"] = self.positions.get(asset, 0)
        
        # Record trade
        self.trades.append(trade_result)
        
        return trade_result 