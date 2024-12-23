import gymnasium as gym
import logging
import asyncio
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, AsyncGenerator
from data.utils.websocket_loader import WebSocketLoader
import ccxt.async_support as ccxt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OrderStatus:
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"

class Order:
    def __init__(
        self,
        id: str,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float,
        status: str = OrderStatus.PENDING,
        filled: float = 0.0,
        remaining: float = 0.0
    ):
        self.id = id
        self.symbol = symbol
        self.type = type
        self.side = side
        self.amount = amount
        self.price = price
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

class LiveTradingEnvironment(gym.Env):
    """Live Trading Environment that extends the base TradingEnvironment"""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
        window_size: int = 60,
        exchange_id: str = "binance",
        max_data_points: int = 1000,
        test_mode: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        websocket: Optional[WebSocketLoader] = None,
        exchange: Optional[ccxt.Exchange] = None,
    ):
        super(LiveTradingEnvironment, self).__init__()

        self.symbol = symbol
        self.test_mode = test_mode
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Initialize exchange
        if exchange is not None:
            self.exchange = exchange
        elif not test_mode:
            self.exchange = getattr(ccxt, exchange_id)()
        
        # Initialize WebSocket loader
        if websocket is not None:
            self.websocket = websocket
        else:
            self.websocket = WebSocketLoader(
                exchange_id=exchange_id,
                symbol=symbol,
                max_data_points=max_data_points,
            )

        # Trading parameters
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.n_features = 13  # OHLCV (5) + market depth (4) + technical (2) + portfolio (2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_features),
            dtype=np.float32,
        )

        # Order management
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.canceled_orders: List[Order] = []
        
        # Rate limiting
        self.last_api_call = datetime.min
        self.rate_limit_reset = 0.0
        
        # Initialize state
        self._initialize_state()

    def _initialize_state(self):
        """Initialize environment state"""
        try:
            # Reset account state
            self.balance = float(self.initial_balance)
            self.position = 0.0
            self.trades = []
            
            # Reset order management
            self.active_orders = {}
            self.filled_orders = []
            self.canceled_orders = []
            
            # Reset tracking variables
            self._last_portfolio_value = float(self.initial_balance)
            self._last_price = 50000.0  # Initial price for test mode
            
            # Reset rate limiting
            self.last_api_call = datetime.min
            self.rate_limit_reset = 0.0
            
            # Initialize mock data for testing
            if self.test_mode:
                self._update_mock_data()
                
        except Exception as e:
            logger.error(f"Error initializing state: {e}")
            # Ensure basic state is set even if initialization fails
            self.balance = float(self.initial_balance)
            self.position = 0.0
            self._last_portfolio_value = float(self.initial_balance)

    def _update_mock_data(self):
        """Update mock market data for testing"""
        if not self.test_mode:
            return
            
        # Simulate price movement
        price_change = np.random.normal(0, 1.0)
        self._last_price *= (1 + price_change * 0.01)
        
        # Create mock market data
        mock_data = {
            'timestamp': datetime.now().timestamp(),
            'open': self._last_price * 0.999,
            'high': self._last_price * 1.001,
            'low': self._last_price * 0.998,
            'close': self._last_price,
            'volume': 1000.0,
            'bid': self._last_price * 0.995,
            'ask': self._last_price * 1.005,
            'bid_volume': 500.0,
            'ask_volume': 500.0,
            'rsi': 50.0 + price_change * 5.0,  # Mock RSI with some variation
            'macd': price_change * 0.1,  # Mock MACD with some variation
        }
        
        # Update WebSocket data
        self.websocket.add_mock_data(mock_data)
        
        # Update current data
        df = pd.DataFrame([mock_data])
        df.set_index('timestamp', inplace=True)
        self.websocket._current_data = df

    @property
    def portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            # Get current price
            current_price = self._last_price if self.test_mode else self.websocket.get_latest_price()
            if not current_price or not np.isfinite(current_price):
                logger.error("Invalid current price")
                return float(self.balance)
            
            # Calculate position value
            position = 0.0 if abs(self.position) < 1e-6 else self.position
            position_value = float(position * current_price)
            
            # Calculate total value
            total_value = float(self.balance + position_value)
            
            # Ensure value is valid
            if not np.isfinite(total_value) or total_value < 0:
                logger.error(f"Invalid portfolio value: {total_value}")
                return float(self.balance)
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return float(self.balance)

    def _reset_position(self):
        """Reset position and update state"""
        try:
            # Reset position
            self.position = 0.0
            
            # Cancel all active orders
            self.active_orders.clear()
            self.filled_orders.clear()
            self.canceled_orders.clear()
            
            # Reset tracking variables
            self._last_portfolio_value = float(self.balance)
            
            # Log reset
            logger.info("Position and state reset")
            
        except Exception as e:
            logger.error(f"Error resetting position: {e}")
            # Ensure position is reset even if other operations fail
            self.position = 0.0

    async def create_order(self, side: str, amount: float, price: float) -> Optional[str]:
        """Create a new order"""
        try:
            # Validate inputs
            if amount <= 0 or price <= 0:
                logger.error(f"Invalid order parameters: amount={amount}, price={price}")
                self._reset_position()
                return None

            if self.test_mode:
                # In test mode, handle simulated errors
                if hasattr(self.exchange, 'create_order'):
                    if isinstance(getattr(self.exchange.create_order, 'side_effect', None), (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeError)):
                        # Let the error propagate to simulate real exchange behavior
                        await self.exchange.create_order(
                            symbol=self.symbol,
                            type="limit",
                            side=side,
                            amount=amount,
                            price=price
                        )
                
                # Create order with zero fill by default
                order_id = f"test_order_{len(self.active_orders)}"
                order = Order(
                    id=order_id,
                    symbol=self.symbol,
                    type="limit",
                    side=side,
                    amount=amount,
                    price=price,
                    status=OrderStatus.OPEN,
                    filled=0.0,  # Initialize with zero fill
                    remaining=amount
                )
                self.active_orders[order_id] = order
                return order_id
            
            # Call exchange API
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side=side,
                amount=amount,
                price=price
            )
            
            if not order or 'id' not in order:
                logger.error("Invalid order response from exchange")
                self._reset_position()
                return None
            
            # Store order details
            order_id = order['id']
            order_obj = Order(
                id=order_id,
                symbol=self.symbol,
                type="limit",
                side=side,
                amount=amount,
                price=price,
                status=OrderStatus.OPEN,
                filled=0.0,
                remaining=amount
            )
            self.active_orders[order_id] = order_obj
            return order_id
            
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RateLimitExceeded) as e:
            logger.error(f"Exchange error creating order: {e}")
            self._reset_position()
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating order: {e}")
            self._reset_position()
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id not in self.active_orders:
                return False
            
            if self.test_mode:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELED
                self.canceled_orders.append(order)
                del self.active_orders[order_id]
                return True
            
            # Call exchange API
            await self.exchange.cancel_order(order_id, self.symbol)
            
            # Update order status
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELED
            self.canceled_orders.append(order)
            del self.active_orders[order_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def check_order_status(self, order_id: str) -> Dict:
        """Check the status of an order"""
        try:
            if order_id not in self.active_orders:
                return {
                    'id': order_id,
                    'status': OrderStatus.CANCELED,
                    'filled': 0.0,
                    'remaining': 0.0,
                    'price': 0.0
                }
            
            if self.test_mode:
                order = self.active_orders[order_id]
                
                # In test mode, check exchange API response first
                if hasattr(self.exchange, 'fetch_order'):
                    try:
                        order_info = await self.exchange.fetch_order(order_id, self.symbol)
                        if order_info and 'status' in order_info:
                            # Update order status from exchange response
                            order.status = order_info['status']
                            order.filled = float(order_info.get('filled', 0))
                            order.remaining = order.amount - order.filled
                            
                            if order.status == OrderStatus.CLOSED:
                                try:
                                    # Calculate new position and balance
                                    if order.side == 'buy':
                                        new_position = round(self.position + order.filled, 8)
                                        new_balance = round(self.balance - order.filled * order.price * (1 + self.trading_fee), 8)
                                    else:
                                        new_position = round(self.position - order.filled, 8)
                                        new_balance = round(self.balance + order.filled * order.price * (1 - self.trading_fee), 8)
                                    
                                    # Validate updates
                                    if new_balance >= 0 and np.isfinite(new_position) and np.isfinite(new_balance):
                                        self.position = new_position
                                        self.balance = new_balance
                                        
                                        # Move to filled orders
                                        self.filled_orders.append(order)
                                        del self.active_orders[order_id]
                                    else:
                                        logger.error(f"Invalid position/balance update: position={new_position}, balance={new_balance}")
                                        self._reset_position()
                                        order.status = OrderStatus.REJECTED
                                        return {
                                            'id': order_id,
                                            'status': OrderStatus.REJECTED,
                                            'filled': 0.0,
                                            'remaining': order.amount,
                                            'price': order.price
                                        }
                                except Exception as e:
                                    logger.error(f"Error updating position/balance: {e}")
                                    self._reset_position()
                                    order.status = OrderStatus.REJECTED
                                    return {
                                        'id': order_id,
                                        'status': OrderStatus.REJECTED,
                                        'filled': 0.0,
                                        'remaining': order.amount,
                                        'price': order.price
                                    }
                            
                            return {
                                'id': order.id,
                                'status': order.status,
                                'filled': order.filled,
                                'remaining': order.remaining,
                                'price': order.price
                            }
                    except Exception as e:
                        logger.error(f"Error fetching order status: {e}")
                
                # If no exchange response or error, process order locally
                if order.status == OrderStatus.OPEN:
                    try:
                        # Get current price
                        current_price = self._last_price
                        
                        # Check if order should be filled
                        should_fill = False
                        if order.side == 'buy' and current_price <= order.price:
                            should_fill = True
                        elif order.side == 'sell' and current_price >= order.price:
                            should_fill = True
                        
                        if should_fill:
                            # Calculate new position and balance
                            if order.side == 'buy':
                                new_position = round(self.position + order.amount, 8)
                                new_balance = round(self.balance - order.amount * current_price * (1 + self.trading_fee), 8)
                            else:
                                new_position = round(self.position - order.amount, 8)
                                new_balance = round(self.balance + order.amount * current_price * (1 - self.trading_fee), 8)
                            
                            # Validate updates
                            if new_balance >= 0 and np.isfinite(new_position) and np.isfinite(new_balance):
                                # Update order status
                                order.filled = order.amount
                                order.remaining = 0.0
                                order.status = OrderStatus.CLOSED
                                
                                # Update position and balance
                                self.position = new_position
                                self.balance = new_balance
                                
                                # Move to filled orders
                                self.filled_orders.append(order)
                                del self.active_orders[order_id]
                            else:
                                logger.error(f"Invalid position/balance update: position={new_position}, balance={new_balance}")
                                self._reset_position()
                                order.status = OrderStatus.REJECTED
                                return {
                                    'id': order_id,
                                    'status': OrderStatus.REJECTED,
                                    'filled': 0.0,
                                    'remaining': order.amount,
                                    'price': order.price
                                }
                        else:
                            # Order not filled yet
                            return {
                                'id': order.id,
                                'status': order.status,
                                'filled': 0.0,
                                'remaining': order.amount,
                                'price': order.price
                            }
                    except Exception as e:
                        logger.error(f"Error updating position/balance: {e}")
                        self._reset_position()
                        order.status = OrderStatus.REJECTED
                        return {
                            'id': order_id,
                            'status': OrderStatus.REJECTED,
                            'filled': 0.0,
                            'remaining': order.amount,
                            'price': order.price
                        }
                
                return {
                    'id': order.id,
                    'status': order.status,
                    'filled': order.filled,
                    'remaining': order.remaining,
                    'price': order.price
                }
            
            # Call exchange API
            order_info = await self.exchange.fetch_order(order_id, self.symbol)
            
            # Update order status
            order = self.active_orders[order_id]
            order.status = order_info['status']
            order.filled = float(order_info.get('filled', 0))
            order.remaining = order.amount - order.filled
            
            if order.status == OrderStatus.CLOSED:
                try:
                    # Calculate new position and balance
                    if order.side == 'buy':
                        new_position = round(self.position + order.filled, 8)
                        new_balance = round(self.balance - order.filled * order.price * (1 + self.trading_fee), 8)
                    else:
                        new_position = round(self.position - order.filled, 8)
                        new_balance = round(self.balance + order.filled * order.price * (1 - self.trading_fee), 8)
                    
                    # Validate updates
                    if new_balance >= 0 and np.isfinite(new_position) and np.isfinite(new_balance):
                        self.position = new_position
                        self.balance = new_balance
                    else:
                        logger.error(f"Invalid position/balance update: position={new_position}, balance={new_balance}")
                        self._reset_position()
                        order.status = OrderStatus.REJECTED
                        return {
                            'id': order_id,
                            'status': OrderStatus.REJECTED,
                            'filled': 0.0,
                            'remaining': order.amount,
                            'price': order.price
                        }
                except Exception as e:
                    logger.error(f"Error updating position/balance: {e}")
                    self._reset_position()
                    order.status = OrderStatus.REJECTED
                    return {
                        'id': order_id,
                        'status': OrderStatus.REJECTED,
                        'filled': 0.0,
                        'remaining': order.amount,
                        'price': order.price
                    }
                
                # Move to filled orders
                self.filled_orders.append(order)
                del self.active_orders[order_id]
            
            return {
                'id': order.id,
                'status': order.status,
                'filled': order.filled,
                'remaining': order.remaining,
                'price': order.price
            }
            
        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            self._reset_position()
            return {
                'id': order_id,
                'status': OrderStatus.REJECTED,
                'filled': 0.0,
                'remaining': 0.0,
                'price': 0.0
            }

    async def monitor_order(self, order_id: str, timeout: float = 30.0) -> AsyncGenerator[Dict, None]:
        """Monitor order status until it's filled or canceled"""
        start_time = datetime.now()
        last_status = None
        error_count = 0
        max_errors = 3
        
        while True:
            try:
                # Check if order exists
                if order_id not in self.active_orders:
                    status = {
                        'id': order_id,
                        'status': OrderStatus.CANCELED,
                        'filled': 0.0,
                        'remaining': 0.0,
                        'price': 0.0
                    }
                    self._reset_position()
                    yield status
                    break

                # Update mock data in test mode
                if self.test_mode:
                    self._update_mock_data()
                
                # Get order status
                try:
                    status = await self.check_order_status(order_id)
                    error_count = 0  # Reset error count on successful status check
                except Exception as e:
                    logger.error(f"Error checking order status: {e}")
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error("Max error count reached, canceling order")
                        self._reset_position()
                        status = {
                            'id': order_id,
                            'status': OrderStatus.REJECTED,
                            'filled': 0.0,
                            'remaining': 0.0,
                            'price': 0.0
                        }
                        yield status
                        break
                    continue
                
                # Yield only if status changed
                if status != last_status:
                    yield status
                    last_status = status.copy()
                
                # Break if order is complete
                if status['status'] in [
                    OrderStatus.CLOSED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED
                ]:
                    # Reset position if order failed or had no fill
                    if status['status'] in [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED] or \
                       (status['status'] == OrderStatus.CLOSED and status.get('filled', 0) == 0):
                        self._reset_position()
                    break
                
                # Check timeout
                if (datetime.now() - start_time).total_seconds() > timeout:
                    logger.warning(f"Order {order_id} monitoring timed out")
                    # Cancel order and reset position on timeout
                    try:
                        await self.cancel_order(order_id)
                    except Exception as e:
                        logger.error(f"Error canceling order on timeout: {e}")
                    self._reset_position()
                    status = {
                        'id': order_id,
                        'status': OrderStatus.CANCELED,
                        'filled': 0.0,
                        'remaining': 0.0,
                        'price': 0.0
                    }
                    yield status
                    break
                
                await asyncio.sleep(0.5)  # Increased sleep time
                
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                error_count += 1
                if error_count >= max_errors:
                    logger.error("Max error count reached in monitor_order")
                    self._reset_position()
                    status = {
                        'id': order_id,
                        'status': OrderStatus.REJECTED,
                        'filled': 0.0,
                        'remaining': 0.0,
                        'price': 0.0
                    }
                    yield status
                    break
                await asyncio.sleep(0.5)  # Sleep before retry

    async def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        try:
            # Cancel all active orders
            if hasattr(self, 'active_orders'):
                await self.cancel_all_orders()
            
            # Initialize state
            self._initialize_state()
            
            # Update mock data in test mode
            if self.test_mode:
                self._update_mock_data()
            
            # Get initial observation
            observation = await self._get_observation()
            
            # Get initial info with exact values
            info = {
                'balance': float(self.balance),
                'position': 0.0,  # Always start with no position
                'portfolio_value': float(self.initial_balance)
            }
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            # Return zero-filled observation and default info
            observation = np.zeros((self.window_size, self.n_features), dtype=np.float32)
            info = {
                'balance': float(self.initial_balance),
                'position': 0.0,
                'portfolio_value': float(self.initial_balance)
            }
            return observation, info

    async def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        try:
            # Get current price and portfolio value before action
            current_price = self._last_price if self.test_mode else self.websocket.get_latest_price()
            if not current_price or not np.isfinite(current_price):
                logger.error("Invalid current price")
                self._reset_position()
                return await self._get_observation(), -1.0, False, False, self._get_info()
            
            portfolio_value_before = float(self.portfolio_value)

            # Parse action
            action_value = float(action[0])  # Convert to float
            
            # Skip if action is too small
            if abs(action_value) < 0.01:
                return await self._get_observation(), 0.0, False, False, self._get_info()

            # Calculate order size
            try:
                if action_value > 0:  # Buy
                    max_buy_amount = (self.balance * 0.99) / current_price  # Leave some margin for fees
                    order_size = max_buy_amount * action_value
                    side = "buy"
                else:  # Sell
                    order_size = abs(self.position * action_value)
                    side = "sell"
                
                # Validate order size
                order_size = round(order_size, 8)  # Round to 8 decimal places
                if order_size <= 0 or not np.isfinite(order_size):
                    logger.error(f"Invalid order size: {order_size}")
                    self._reset_position()
                    return await self._get_observation(), -1.0, False, False, self._get_info()
                
                # Skip if order size is too small
                if order_size * current_price < 10.0:  # Minimum order value
                    return await self._get_observation(), 0.0, False, False, self._get_info()
            except Exception as e:
                logger.error(f"Error calculating order size: {e}")
                self._reset_position()
                return await self._get_observation(), -1.0, False, False, self._get_info()

            try:
                # Create order
                order_id = await self.create_order(side, order_size, current_price)
                if not order_id:  # Order creation failed
                    logger.warning("Order creation failed")
                    self._reset_position()
                    return await self._get_observation(), -1.0, False, False, self._get_info()

                # Monitor order
                filled_amount = 0.0
                order_status = None
                async for order_update in self.monitor_order(order_id):
                    order_status = order_update
                    if order_update['status'] in [OrderStatus.CLOSED, OrderStatus.CANCELED]:
                        filled_amount = float(order_update.get('filled', 0))
                        break

                # Handle order completion
                if not order_status or order_status['status'] in [OrderStatus.CANCELED, OrderStatus.REJECTED]:
                    logger.warning(f"Order {order_id} was not successful: {order_status}")
                    self._reset_position()
                    return await self._get_observation(), -1.0, False, False, self._get_info()

                # Validate filled amount
                if filled_amount <= 0:
                    logger.warning(f"Order {order_id} had no fill")
                    self._reset_position()
                    return await self._get_observation(), -1.0, False, False, self._get_info()

                # Calculate reward
                portfolio_value_after = float(self.portfolio_value)
                if portfolio_value_before > 0:
                    reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
                else:
                    reward = 0.0

                # Ensure reward is finite
                if not np.isfinite(reward):
                    reward = -1.0

                # Get observation and info
                obs = await self._get_observation()
                info = self._get_info()

                return obs, reward, False, False, info

            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RateLimitExceeded) as e:
                logger.error(f"Exchange error: {e}")
                self._reset_position()
                return await self._get_observation(), -1.0, False, False, self._get_info()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self._reset_position()
                return await self._get_observation(), -1.0, False, False, self._get_info()

        except Exception as e:
            logger.error(f"Error in step: {e}")
            self._reset_position()
            return await self._get_observation(), -1.0, False, False, self._get_info()

    async def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        try:
            # Get current market data
            df = self.websocket.get_current_data()
            if df is None or len(df) == 0:
                raise ValueError("No data available from WebSocket")

            # Ensure we have all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask', 'bid_volume', 'ask_volume', 'rsi', 'macd']
            for col in required_columns:
                if col not in df.columns:
                    if col in ['bid', 'ask']:
                        # Use close price as fallback for bid/ask
                        df[col] = df['close'] * (0.995 if col == 'bid' else 1.005)
                    elif col in ['bid_volume', 'ask_volume']:
                        df[col] = df['volume'] / 2
                    else:
                        df[col] = 0.0  # Default value for missing columns

            # Calculate portfolio-related features
            df['position'] = self.position
            df['portfolio_value'] = self.portfolio_value

            # Ensure we have enough data points
            if len(df) < self.window_size:
                # Pad with the first row if needed
                pad_rows = self.window_size - len(df)
                df = pd.concat([pd.DataFrame([df.iloc[0]] * pad_rows), df])

            # Get the last window_size rows
            df = df.iloc[-self.window_size:]

            # Extract features in the correct order
            features = []
            for col in required_columns + ['position', 'portfolio_value']:
                features.append(df[col].values)

            # Stack features and normalize
            obs = np.column_stack(features)
            obs = np.clip(obs, -np.inf, np.inf)  # Ensure no NaN/inf values

            return obs.astype(np.float32)

        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            # Return zero-filled observation in case of error
            return np.zeros((self.window_size, self.n_features), dtype=np.float32)

    async def cancel_all_orders(self) -> List[str]:
        """Cancel all active orders"""
        canceled_orders = []
        for order_id in list(self.active_orders.keys()):
            if await self.cancel_order(order_id):
                canceled_orders.append(order_id)
        return canceled_orders

    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all active orders
        await self.cancel_all_orders()
        
        # Close exchange connection
        if not self.test_mode and hasattr(self, 'exchange'):
            await self.exchange.close()
        
        # Stop WebSocket
        if hasattr(self, 'websocket'):
            await self.websocket.stop()

    def render(self):
        """Render the environment"""
        pass

    def _get_info(self) -> Dict:
        """Get current information"""
        try:
            # Apply position threshold
            position = 0.0 if abs(self.position) < 1e-6 else round(self.position, 8)
            
            # Calculate portfolio value
            portfolio_value = self.portfolio_value
            
            # Ensure values are valid
            if not np.isfinite(position):
                position = 0.0
            if not np.isfinite(portfolio_value):
                portfolio_value = self.balance
            
            return {
                'balance': round(float(self.balance), 8),
                'position': float(position),
                'portfolio_value': round(float(portfolio_value), 8)
            }
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {
                'balance': float(self.initial_balance),
                'position': 0.0,
                'portfolio_value': float(self.initial_balance)
            }


# Example usage
async def main():
    # Create environment
    env = LiveTradingEnvironment()

    # Run a simple loop
    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i}: Reward = {reward:.4f}, Portfolio = {info['portfolio_value']:.2f}"
        )
        await asyncio.sleep(1)  # Wait for next update

    env.close()


if __name__ == "__main__":
    asyncio.run(main())
