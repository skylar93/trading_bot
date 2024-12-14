from typing import Dict, List, Tuple, Optional
import logging
import asyncio
import numpy as np
import pandas as pd
from gymnasium import spaces
from datetime import datetime
from collections import deque
from data.utils.websocket_loader import WebSocketLoader

logger = logging.getLogger(__name__)

class PaperTradingEnvironment:
    """Paper Trading Environment for live testing without real money"""
    
    def __init__(self, symbol: str = "BTC/USDT", initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001, window_size: int = 60,
                 max_position_size: float = 1.0, stop_loss_pct: float = 0.02):
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        
        # Risk parameters
        self.max_position_size = max_position_size  # Maximum position size as fraction of portfolio
        self.stop_loss_pct = stop_loss_pct  # Stop loss percentage
        
        # Initialize WebSocket
        self.websocket = WebSocketLoader(symbol=symbol)
        self.websocket.add_callback(self._on_market_update)
        
        # Trading state
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.trade_history = []
        self.stop_loss_price = 0
        
        # Performance tracking
        self._portfolio_values = deque(maxlen=10000)
        self._trade_returns = deque(maxlen=10000)
        self._timestamps = deque(maxlen=10000)
        self.peak_value = initial_balance
        self.max_drawdown = 0
        
        # Market state
        self._market_data = deque(maxlen=window_size)
        self._latest_price = 0
        self._latest_orderbook = None
    
    async def start(self):
        """Start paper trading"""
        await self.websocket.start()
    
    async def stop(self):
        """Stop paper trading"""
        await self.websocket.stop()
    
    async def _on_market_update(self, data: Dict):
        """Handle market data updates"""
        if data['type'] == 'ticker':
            self._latest_price = float(data['data']['close'])
            self._market_data.append(data['data'])
            await self._check_stop_loss()
            
            # Update metrics
            timestamp = datetime.now()
            portfolio_value = self.get_portfolio_value()
            self._update_metrics(portfolio_value, timestamp)
            
        elif data['type'] == 'orderbook':
            self._latest_orderbook = data['data']
    
    async def _check_stop_loss(self):
        """Check and execute stop loss if needed"""
        if not self.position or not self.stop_loss_price:
            return
            
        if self.position > 0 and self._latest_price <= self.stop_loss_price:
            # Long position stop loss
            await self.execute_order(-1.0)  # Full close
            
        elif self.position < 0 and self._latest_price >= self.stop_loss_price:
            # Short position stop loss
            await self.execute_order(1.0)  # Full close
    
    def _update_metrics(self, portfolio_value: float, timestamp: datetime):
        """Update performance metrics"""
        self.peak_value = max(self.peak_value, portfolio_value)
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self._portfolio_values.append(portfolio_value)
        self._timestamps.append(timestamp)
        
        if len(self._portfolio_values) > 1:
            returns = (portfolio_value - self._portfolio_values[-2]) / self._portfolio_values[-2]
            self._trade_returns.append(returns)
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if not self._latest_price:
            return self.balance
        return self.balance + (self.position * self._latest_price)
    
    async def execute_order(self, action: float) -> Dict:
        """
        Execute a paper trade
        
        Args:
            action: Value between -1 and 1 indicating the trading action
                   -1: full sell, 0: hold, 1: full buy
        """
        if not self._latest_price or abs(action) > 1:
            return {
                'success': False,
                'error': 'Invalid price or action'
            }
        
        current_portfolio = self.get_portfolio_value()
        execution_price = self._latest_price
        
        # Calculate order size based on max_position_size
        max_order_value = current_portfolio * self.max_position_size
        
        if action > 0:  # Buy
            max_shares = max_order_value / execution_price
            shares_to_buy = max_shares * abs(action)
            cost = shares_to_buy * execution_price * (1 + self.transaction_fee)
            
            if cost <= self.balance:
                self.position += shares_to_buy
                self.balance -= cost
                self.entry_price = execution_price
                self.stop_loss_price = execution_price * (1 - self.stop_loss_pct)
                
                trade_info = {
                    'type': 'buy',
                    'shares': shares_to_buy,
                    'price': execution_price,
                    'cost': cost,
                    'timestamp': datetime.now(),
                    'portfolio_value': self.get_portfolio_value()
                }
                self.trades.append(trade_info)
                return {'success': True, 'trade': trade_info}
                
        elif action < 0:  # Sell
            shares_to_sell = self.position * abs(action)
            revenue = shares_to_sell * execution_price * (1 - self.transaction_fee)
            
            self.position -= shares_to_sell
            self.balance += revenue
            
            # Reset stop loss if position closed
            if self.position == 0:
                self.stop_loss_price = 0
            else:
                self.stop_loss_price = execution_price * (1 + self.stop_loss_pct)
            
            trade_info = {
                'type': 'sell',
                'shares': shares_to_sell,
                'price': execution_price,
                'revenue': revenue,
                'timestamp': datetime.now(),
                'portfolio_value': self.get_portfolio_value()
            }
            self.trades.append(trade_info)
            
            # Calculate trade P&L if position closed
            if self.position == 0:
                pnl = revenue - (shares_to_sell * self.entry_price)
                self.trade_history.append({
                    'entry_price': self.entry_price,
                    'exit_price': execution_price,
                    'shares': shares_to_sell,
                    'pnl': pnl,
                    'return': pnl / (shares_to_sell * self.entry_price)
                })
            
            return {'success': True, 'trade': trade_info}
            
        return {'success': False, 'error': 'No trade executed'}
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        if len(self._portfolio_values) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': self.max_drawdown,
                'win_rate': 0,
                'trade_count': len(self.trade_history)
            }
        
        # Calculate metrics
        total_return = (self._portfolio_values[-1] - self.initial_balance) / self.initial_balance
        
        returns_array = np.array(list(self._trade_returns))
        sharpe_ratio = 0
        if len(returns_array) > 0 and returns_array.std() > 0:
            sharpe_ratio = np.sqrt(365) * (returns_array.mean() / returns_array.std())
        
        win_rate = 0
        if self.trade_history:
            winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            win_rate = winning_trades / len(self.trade_history)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'trade_count': len(self.trade_history),
            'current_position': self.position,
            'current_balance': self.balance,
            'portfolio_value': self.get_portfolio_value(),
            'latest_price': self._latest_price
        }
    
    def get_current_data(self) -> pd.DataFrame:
        """Get current market data window"""
        return pd.DataFrame(list(self._market_data))

# Example usage
async def main():
    # Create paper trading environment
    env = PaperTradingEnvironment(
        symbol="BTC/USDT",
        initial_balance=10000.0,
        max_position_size=0.5,  # Max 50% of portfolio per position
        stop_loss_pct=0.02  # 2% stop loss
    )
    
    try:
        # Start market data stream
        await env.start()
        
        # Simulate some trades
        print("Initial metrics:", env.get_metrics())
        
        # Buy 25% of max position
        result = await env.execute_order(0.25)
        print("Buy order result:", result)
        print("After buy metrics:", env.get_metrics())
        
        # Wait for some price updates
        await asyncio.sleep(10)
        
        # Sell the position
        result = await env.execute_order(-1.0)
        print("Sell order result:", result)
        print("Final metrics:", env.get_metrics())
        
    finally:
        await env.stop()

if __name__ == "__main__":
    asyncio.run(main())