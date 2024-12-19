import gymnasium as gym
import logging
import asyncio
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
from data.utils.websocket_loader import WebSocketLoader

logger = logging.getLogger(__name__)

class LiveTradingEnvironment(gym.Env):
    """Live Trading Environment that extends the base TradingEnvironment"""
    
    def __init__(self, symbol: str = "BTC/USDT", initial_balance: float = 10000.0,
                 trading_fee: float = 0.001, window_size: int = 60,
                 exchange_id: str = "binance", max_data_points: int = 1000):
        super(LiveTradingEnvironment, self).__init__()

        # Initialize WebSocket loader
        self.websocket = WebSocketLoader(
            exchange_id=exchange_id,
            symbol=symbol,
            max_data_points=max_data_points
        )
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size
        
        # Action space: continuous action between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: OHLCV data + technical indicators + market depth
        n_features = 12  # price, volume, position, balance, market depth
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, n_features), dtype=np.float32
        )
        
        # Internal state
        self.reset()
        
        # Start WebSocket
        asyncio.create_task(self._start_websocket())
    
    async def _start_websocket(self):
        """Start WebSocket connection"""
        await self.websocket.start()
    
    @property
    def portfolio_value(self):
        """Calculate current portfolio value"""
        current_price = self._get_current_price()
        return self.balance + (self.position * current_price)
    
    def _get_current_price(self) -> float:
        """Get current market price"""
        if self.websocket._latest_ticker:
            return float(self.websocket._latest_ticker['last'])
        return 0.0
    
    def _get_market_depth(self) -> Tuple[float, float, float, float]:
        """Get market depth information"""
        if not self.websocket._latest_orderbook:
            return 0.0, 0.0, 0.0, 0.0
            
        book = self.websocket._latest_orderbook
        
        # Best bid/ask
        best_bid = float(book['bids'][0][0]) if book['bids'] else 0.0
        best_ask = float(book['asks'][0][0]) if book['asks'] else 0.0
        
        # Volume at best bid/ask
        bid_volume = float(book['bids'][0][1]) if book['bids'] else 0.0
        ask_volume = float(book['asks'][0][1]) if book['asks'] else 0.0
        
        return best_bid, best_ask, bid_volume, ask_volume
    
    async def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self._last_portfolio_value = self.initial_balance
        
        # Wait for initial data
        max_retries = 10
        retry_count = 0
        while len(self.websocket.get_current_data()) < self.window_size:
            logger.info("Waiting for initial data...")
            await asyncio.sleep(1)
            retry_count += 1
            if retry_count >= max_retries:
                logger.warning("Timeout waiting for initial data")
                break
        
        return self._get_observation(), {}
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        # Get current price and market depth
        current_price = self._get_current_price()
        best_bid, best_ask, bid_volume, ask_volume = self._get_market_depth()
        
        if current_price <= 0:
            logger.warning("Invalid price, skipping action")
            return self._get_observation(), 0, False, False, {}
        
        # Execute trade
        if action > 0:  # Buy
            execution_price = best_ask or current_price
            shares_to_buy = (self.balance * abs(action)) / execution_price
            cost = shares_to_buy * execution_price * (1 + self.trading_fee)
            
            if cost <= self.balance:
                self.position += shares_to_buy
                self.balance -= cost
                self.trades.append(('buy', shares_to_buy, execution_price))
        
        elif action < 0:  # Sell
            execution_price = best_bid or current_price
            shares_to_sell = self.position * abs(action)
            revenue = shares_to_sell * execution_price * (1 - self.trading_fee)
            
            self.position -= shares_to_sell
            self.balance += revenue
            self.trades.append(('sell', shares_to_sell, execution_price))
        
        # Calculate reward (change in portfolio value)
        new_portfolio_value = self.portfolio_value
        reward = (new_portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
        self._last_portfolio_value = new_portfolio_value
        
        info = {
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'current_price': current_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
        }
        
        return self._get_observation(), reward, False, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation"""
        # Get the latest data window
        df = self.websocket.get_current_data().tail(self.window_size)
        
        if len(df) < self.window_size:
            logger.warning(f"Not enough data points: {len(df)} < {self.window_size}")
            # Pad with zeros if necessary
            padding = pd.DataFrame(
                np.zeros((self.window_size - len(df), len(df.columns))),
                columns=df.columns
            )
            df = pd.concat([padding, df])
        
        # Normalize the data
        price_mean = df['close'].mean()
        price_std = df['close'].std() or 1.0  # Avoid division by zero
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std() or 1.0
        
        # Get market depth
        best_bid, best_ask, bid_volume, ask_volume = self._get_market_depth()
        
        # Construct features
        features = []
        # Price features
        for col in ['open', 'high', 'low', 'close']:
            features.append((df[col] - price_mean) / price_std)
        # Volume
        features.append((df['volume'] - volume_mean) / volume_std)
        # Returns and changes
        features.append(df['close'].pct_change().fillna(0))
        features.append(df['volume'].pct_change().fillna(0))
        
        # Market depth features
        features.append(pd.Series([best_bid / price_mean if best_bid else 0] * len(df)))
        features.append(pd.Series([best_ask / price_mean if best_ask else 0] * len(df)))
        features.append(pd.Series([bid_volume / volume_mean if bid_volume else 0] * len(df)))
        features.append(pd.Series([ask_volume / volume_mean if ask_volume else 0] * len(df)))
        
        # Portfolio info
        features.append(pd.Series([self.position] * len(df)))
        features.append(pd.Series([self.balance / self.initial_balance] * len(df)))
        
        obs = np.array(features).T
        return obs.astype(np.float32)
    
    def render(self):
        """Render the environment"""
        # Can be implemented for visualization if needed
        pass
    
    def close(self):
        """Close the environment"""
        asyncio.create_task(self.websocket.stop())

# Example usage
async def main():
    # Create environment
    env = LiveTradingEnvironment()
    
    # Run a simple loop
    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward = {reward:.4f}, Portfolio = {info['portfolio_value']:.2f}")
        await asyncio.sleep(1)  # Wait for next update
    
    env.close()

if __name__ == "__main__":
    asyncio.run(main())