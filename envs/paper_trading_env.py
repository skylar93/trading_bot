from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
import asyncio
import numpy as np
import pandas as pd
from gymnasium import spaces
from datetime import datetime
from collections import deque
from data.utils.websocket_loader import WebSocketLoader
import os

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"
    ICEBERG = "iceberg"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class Order:
    def __init__(self, 
                 order_type: OrderType,
                 side: str,
                 quantity: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 trailing_pct: Optional[float] = None,
                 time_in_force: str = "GTC",
                 expire_time: Optional[datetime] = None,
                 iceberg_qty: Optional[float] = None):
        self.id = str(int(datetime.now().timestamp() * 1000))  # Unique order ID
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.trailing_pct = trailing_pct
        self.time_in_force = time_in_force
        self.expire_time = expire_time
        self.iceberg_qty = iceberg_qty
        self.status = OrderStatus.PENDING
        self.filled_qty = 0
        self.filled_price = 0
        self.timestamp = datetime.now()
        
        # For trailing stop orders
        if order_type == OrderType.TRAILING_STOP:
            # Initialize trailing price to current price if available
            self.trailing_price = price if price else 0.0
        else:
            self.trailing_price = stop_price if stop_price else price if price else None
        
        # For OCO orders
        self.linked_order = None
        
        # For iceberg orders
        self.visible_qty = min(iceberg_qty, quantity) if iceberg_qty else quantity
        self.executed_qty = 0
        
        # For stop-limit orders
        self.triggered = False if order_type == OrderType.STOP_LIMIT else None

class PaperTradingEnvironment:
    """Paper Trading Environment for live testing without real money"""
    
    def __init__(self, 
                 symbol: str,
                 initial_balance: float = 10000.0,
                 trading_fee: float = 0.001,
                 window_size: int = 20,
                 test_mode: bool = False,
                 max_position_size: float = 0.5):
        """Initialize paper trading environment"""
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.test_mode = test_mode
        self.max_position_size = max_position_size
        
        # Trading state
        self.position = 0.0
        self.entry_price = 0.0
        self._latest_price = 0.0
        self.stop_loss_price = 0.0
        self.stop_loss_pct = 0.02  # 2% stop loss
        
        # Order management
        self.active_orders = []
        self.filled_orders = []
        self.cancelled_orders = []
        self.expired_orders = []
        
        # Trade history
        self.trades = []
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Market data buffer
        self.price_history = deque(maxlen=window_size)
        
    async def initialize(self):
        """Initialize environment"""
        if not self.test_mode:
            self.ws_loader = WebSocketLoader(self.symbol)
            await self.ws_loader.connect()
        self.logger.info(f"Initialized {self.__class__.__name__} for {self.symbol}")
        
    async def cleanup(self):
        """Cleanup resources"""
        if not self.test_mode and hasattr(self, 'ws_loader'):
            await self.ws_loader.disconnect()
        
    async def update_market_data(self, data: Dict):
        """Update market data"""
        if 'type' in data and data['type'] == 'ticker':
            self._latest_price = float(data['data']['close'])
            self.price_history.append(self._latest_price)
            self.logger.debug(f"Updated market price to {self._latest_price}")
            
            # Check stop loss
            if self.position != 0 and self.stop_loss_price != 0:
                if (self.position > 0 and self._latest_price <= self.stop_loss_price) or \
                   (self.position < 0 and self._latest_price >= self.stop_loss_price):
                    await self.execute_order(-1)  # Close position
            
            # Check and execute limit orders
            await self._check_limit_orders()
            
            # Update trailing prices for trailing stop orders
            for order in self.active_orders:
                if order.order_type == OrderType.TRAILING_STOP:
                    if order.side == "sell":
                        if self._latest_price > order.trailing_price:
                            order.trailing_price = self._latest_price
                            self.logger.debug(f"Updated sell trailing price to {order.trailing_price}")
                    else:  # buy
                        if self._latest_price < order.trailing_price:
                            order.trailing_price = self._latest_price
                            self.logger.debug(f"Updated buy trailing price to {order.trailing_price}")
                            
    async def place_order(self, order: Order) -> Dict:
        """Place a new order"""
        try:
            if not self._latest_price:
                await self.update_market_data({
                    'type': 'ticker',
                    'data': {'close': 50000.0}  # Default price for testing
                })
                if not self._latest_price:
                    return {'success': False, 'message': 'No market price available'}
            
            if order.order_type == OrderType.MARKET:
                price = self._latest_price
                if order.side == "buy":
                    cost = order.quantity * price * (1 + self.trading_fee)
                    if cost <= self.balance:
                        self.balance -= cost
                        self.position += order.quantity
                        self.entry_price = price
                        self.stop_loss_price = price * (1 - self.stop_loss_pct)
                        order.status = OrderStatus.FILLED
                        order.filled_qty = order.quantity
                        order.filled_price = price
                        self.filled_orders.append(order)
                        return {'success': True, 'message': f'Market buy executed at {price}'}
                else:  # sell
                    if order.quantity <= self.position:
                        revenue = order.quantity * price * (1 - self.trading_fee)
                        self.balance += revenue
                        self.position -= order.quantity
                        if abs(self.position) < 1e-8:  # Close enough to zero
                            self.position = 0
                            self.stop_loss_price = 0
                        else:
                            self.stop_loss_price = price * (1 + self.stop_loss_pct)
                        order.status = OrderStatus.FILLED
                        order.filled_qty = order.quantity
                        order.filled_price = price
                        self.filled_orders.append(order)
                        return {'success': True, 'message': f'Market sell executed at {price}'}
                return {'success': False, 'message': 'Insufficient balance/position'}
            else:
                # For non-market orders, just add to active orders
                if order.order_type == OrderType.TRAILING_STOP:
                    order.trailing_price = self._latest_price
                    self.logger.info(f"Initialized trailing price to {order.trailing_price}")
                # For stop-limit orders, set initial trailing price
                elif order.order_type == OrderType.STOP_LIMIT:
                    order.trailing_price = order.stop_price
                
                self.active_orders.append(order)
                return {'success': True, 'message': f'Order placed: {order.order_type.value}'}
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return {'success': False, 'message': str(e)}
            
    async def _check_limit_orders(self):
        """Check and execute limit orders"""
        if not self._latest_price:
            return
            
        for order in self.active_orders[:]:
            if order.status != OrderStatus.PENDING:
                continue
                
            execute_order = False
            execute_quantity = order.quantity
            
            if order.order_type == OrderType.LIMIT:
                # For buy limit orders, execute when price falls below limit price
                if order.side == "buy" and self._latest_price <= order.price:
                    execute_order = True
                    self.logger.info(f"Buy limit order executed at {self._latest_price}")
                # For sell limit orders, execute when price rises above limit price
                elif order.side == "sell" and self._latest_price >= order.price:
                    execute_order = True
                    self.logger.info(f"Sell limit order executed at {self._latest_price}")
                    
            elif order.order_type == OrderType.STOP_LIMIT:
                # For sell stop-limit:
                # 1. Trigger when price rises above stop price
                # 2. Execute when price rises to limit price after trigger
                if order.side == "sell":
                    if not order.triggered and self._latest_price >= order.stop_price:
                        order.triggered = True
                        self.logger.info(f"Sell stop-limit order triggered at {self._latest_price}")
                    elif order.triggered and self._latest_price >= order.price:
                        execute_order = True
                        self.logger.info(f"Sell stop-limit order executed at {self._latest_price}")
                # For buy stop-limit:
                # 1. Trigger when price rises above stop price
                # 2. Execute when price falls to limit price after trigger
                else:  # buy
                    if not order.triggered and self._latest_price >= order.stop_price:
                        order.triggered = True
                        self.logger.info(f"Buy stop-limit order triggered at {self._latest_price}")
                    elif order.triggered and self._latest_price <= order.price:
                        execute_order = True
                        self.logger.info(f"Buy stop-limit order executed at {self._latest_price}")
                        
            elif order.order_type == OrderType.TRAILING_STOP:
                # Update trailing price
                if order.side == "sell":
                    # For sell orders, trailing price moves up with the market
                    if self._latest_price > order.trailing_price:
                        order.trailing_price = self._latest_price
                        self.logger.info(f"Updated sell trailing price to {order.trailing_price}")
                    # Execute when price falls below trailing price by trailing percentage
                    elif self._latest_price <= order.trailing_price * (1 - order.trailing_pct):
                        execute_order = True
                        self.logger.info(f"Trailing stop sell executed at {self._latest_price}")
                else:  # buy
                    # For buy orders, trailing price moves down with the market
                    if self._latest_price < order.trailing_price:
                        order.trailing_price = self._latest_price
                        self.logger.info(f"Updated buy trailing price to {order.trailing_price}")
                    # Execute when price rises above trailing price by trailing percentage
                    elif self._latest_price >= order.trailing_price * (1 + order.trailing_pct):
                        execute_order = True
                        self.logger.info(f"Trailing stop buy executed at {self._latest_price}")
                        
            elif order.order_type == OrderType.ICEBERG:
                # For buy iceberg orders, execute when price falls below limit price
                if order.side == "buy" and self._latest_price <= order.price:
                    execute_order = True
                    execute_quantity = min(order.iceberg_qty, order.quantity - order.executed_qty)
                    self.logger.info(f"Buy iceberg order partial execution: {execute_quantity} at {self._latest_price}")
                # For sell iceberg orders, execute when price rises above limit price
                elif order.side == "sell" and self._latest_price >= order.price:
                    execute_order = True
                    execute_quantity = min(order.iceberg_qty, order.quantity - order.executed_qty)
                    self.logger.info(f"Sell iceberg order partial execution: {execute_quantity} at {self._latest_price}")
            
            if execute_order:
                try:
                    # For iceberg orders, execute partial fills
                    if order.order_type == OrderType.ICEBERG:
                        # Create a new order for the partial fill
                        partial_order = Order(
                            order_type=OrderType.LIMIT,
                            side=order.side,
                            quantity=execute_quantity,
                            price=order.price
                        )
                        await self._execute_order_fill(partial_order)
                        order.executed_qty += execute_quantity
                        
                        # If the entire quantity is executed, mark the original order as filled
                        if order.executed_qty >= order.quantity:
                            order.status = OrderStatus.FILLED
                            if order in self.active_orders:
                                self.active_orders.remove(order)
                                self.filled_orders.append(order)
                    else:
                        await self._execute_order_fill(order)
                        if order in self.active_orders:
                            self.active_orders.remove(order)
                            self.filled_orders.append(order)
                    
                    self.logger.info(f"Order executed: {order.order_type.value} {order.side} at {self._latest_price}")
                except Exception as e:
                    self.logger.error(f"Error executing order: {str(e)}")
                    continue
                
    async def _execute_order_fill(self, order: Order):
        """Execute order fill"""
        try:
            # For test mode, use exact price
            if self.test_mode:
                # For limit orders, use limit price
                if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                    price = order.price
                else:
                    price = self._latest_price
            else:
                # For limit orders, use limit price with slippage
                if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                    price = self._simulate_slippage(order.price)
                else:
                    price = self._simulate_slippage(self._latest_price)
            
            if order.side == "buy":
                cost = order.quantity * price * (1 + self.trading_fee)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += order.quantity
                    self.entry_price = price
                    self.stop_loss_price = price * (1 - self.stop_loss_pct)
                    order.status = OrderStatus.FILLED
                    order.filled_qty = order.quantity
                    order.filled_price = price
                    if order in self.active_orders:
                        self.active_orders.remove(order)
                    self.filled_orders.append(order)
                    self.logger.info(f"Order filled: {order.order_type.value} buy {order.quantity} @ {price}")
                else:
                    self.logger.warning(f"Insufficient balance for buy order: {cost} > {self.balance}")
            else:  # sell
                if order.quantity <= self.position:
                    revenue = order.quantity * price * (1 - self.trading_fee)
                    self.balance += revenue
                    self.position -= order.quantity
                    if abs(self.position) < 1e-8:  # Close enough to zero
                        self.position = 0
                        self.stop_loss_price = 0
                    else:
                        self.stop_loss_price = price * (1 + self.stop_loss_pct)
                    order.status = OrderStatus.FILLED
                    order.filled_qty = order.quantity
                    order.filled_price = price
                    if order in self.active_orders:
                        self.active_orders.remove(order)
                    self.filled_orders.append(order)
                    self.logger.info(f"Order filled: {order.order_type.value} sell {order.quantity} @ {price}")
                else:
                    self.logger.warning(f"Insufficient position for sell order: {order.quantity} > {self.position}")
        except Exception as e:
            self.logger.error(f"Error executing order fill: {str(e)}")
            raise
            
    def _simulate_slippage(self, price: float) -> float:
        """Simulate price slippage"""
        if self.test_mode:
            return price
        slippage = np.random.normal(0, 0.0005)  # 0.05% standard deviation
        return price * (1 + slippage)
        
    async def get_active_orders(self) -> List[Order]:
        """Get list of active orders"""
        return self.active_orders
        
    async def get_filled_orders(self) -> List[Order]:
        """Get list of filled orders"""
        return self.filled_orders
        
    async def get_cancelled_orders(self) -> List[Order]:
        """Get list of cancelled orders"""
        return self.cancelled_orders
        
    async def get_expired_orders(self) -> List[Order]:
        """Get list of expired orders"""
        return self.expired_orders

    async def calculate_position_size(self, risk_fraction: float) -> float:
        """Calculate position size based on risk fraction"""
        if not self._latest_price:
            return 0.0
        max_position_value = self.balance * risk_fraction
        return max_position_value / self._latest_price

    async def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        portfolio_value = self.balance + self.position * self._latest_price
        return {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'entry_price': self.entry_price,
            'current_price': self._latest_price,
            'unrealized_pnl': (self._latest_price - self.entry_price) * self.position if self.position != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown()
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.price_history) < 2:
            return 0.0
        peaks = pd.Series(self.price_history).expanding(min_periods=1).max()
        drawdowns = (pd.Series(self.price_history) - peaks) / peaks
        return abs(float(drawdowns.min()))

    async def log_info(self, message: str):
        """Log information message"""
        self.logger.info(message)

    async def check_expired_orders(self):
        """Check and handle expired orders"""
        current_time = datetime.now()
        expired_count = 0
        
        for order in self.active_orders[:]:  # Use slice copy to safely modify list during iteration
            if order.expire_time and current_time > order.expire_time:
                order.status = OrderStatus.EXPIRED
                self.active_orders.remove(order)
                self.expired_orders.append(order)
                expired_count += 1
                self.logger.info(f"Order {order.id} expired")
        
        return expired_count

    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel an active order"""
        for order in self.active_orders[:]:  # Use slice copy to safely modify list during iteration
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                self.active_orders.remove(order)
                self.cancelled_orders.append(order)
                self.logger.info(f"Order {order_id} cancelled")
                return {'success': True, 'message': f'Order {order_id} cancelled'}
        return {'success': False, 'message': f'Order {order_id} not found or already processed'}

    @property
    def current_price(self) -> float:
        """Get current market price"""
        return self._latest_price

    def simulate_slippage(self, price: float) -> float:
        """Simulate price slippage"""
        if self.test_mode:
            return price
        slippage = np.random.normal(0, 0.0005)  # 0.05% standard deviation
        return price * (1 + slippage)

    async def execute_order(self, size_fraction: float) -> Dict:
        """Execute market order with given size fraction"""
        if not self._latest_price:
            return {'success': False, 'message': 'No market price available'}
            
        if size_fraction > 0:  # Buy
            quantity = abs(size_fraction) * self.balance / self._latest_price
            order = Order(
                order_type=OrderType.MARKET,
                side="buy",
                quantity=quantity
            )
        else:  # Sell
            if self.position == 0:
                return {'success': False, 'message': 'No position to sell'}
            quantity = abs(size_fraction) * self.position
            order = Order(
                order_type=OrderType.MARKET,
                side="sell",
                quantity=quantity
            )
            
        result = await self.place_order(order)
        return result

# Example usage
async def main():
    # Create paper trading environment
    env = PaperTradingEnvironment(
        symbol="BTC/USDT",
        initial_balance=10000.0,
        max_position_size=0.5,  # Max 50% of portfolio per position
        stop_loss_pct=0.02,  # 2% stop loss
        slippage_std=0.0005,  # Standard deviation for slippage simulation
        risk_free_rate=0.02,  # For Sharpe ratio calculation
        max_leverage=1.0,  # Max leverage
        test_mode=True  # Test mode
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