"""
Data streaming utilities for the Trading Bot UI
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import asyncio

logger = logging.getLogger(__name__)

class DataStream:
    """Real-time market data stream manager"""
    
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_buffer = []
        self.max_buffer_size = 1000
        self.is_running = False
        self.last_update = None
        
        # For test mode
        self.test_mode = True
        self.test_price = 100.0
        self.test_timestamp = datetime.now() - timedelta(hours=2)  # Start from 2 hours ago
        
        # Initialize with some historical data
        self._initialize_test_data()
    
    def _initialize_test_data(self):
        """Initialize buffer with some historical data"""
        if self.test_mode:
            # Generate initial trend direction
            self.trend = np.random.choice([-1, 1])
            self.trend_strength = np.random.uniform(0.0001, 0.0003)
            self.volatility = 0.0002
            
            # Generate more initial data points
            for _ in range(200):  # Start with 200 historical points
                self.data_buffer.append(self._generate_test_data())
    
    def _generate_test_data(self) -> Dict:
        """Generate test market data with more realistic price movements"""
        # Occasionally change trend
        if np.random.random() < 0.05:  # 5% chance to change trend
            self.trend = -self.trend
            self.trend_strength = np.random.uniform(0.0001, 0.0003)
        
        # Add trend movement
        trend_movement = self.trend * self.trend_strength
        
        # Add random movement
        random_movement = np.random.normal(0, self.volatility)
        
        # Add periodic movement (simulate market cycles)
        time_factor = len(self.data_buffer) / 100
        periodic_movement = 0.0001 * np.sin(time_factor * np.pi)
        
        # Combine movements
        total_movement = trend_movement + random_movement + periodic_movement
        self.test_price *= (1 + total_movement)
        
        # Generate OHLCV data with more variation
        base_price = self.test_price
        price_volatility = self.volatility * base_price
        
        open_price = base_price * (1 + np.random.normal(0, 0.0001))
        high_price = max(base_price, open_price) + abs(np.random.normal(0, price_volatility))
        low_price = min(base_price, open_price) - abs(np.random.normal(0, price_volatility))
        close_price = base_price
        
        # Generate volume with trends
        base_volume = 1000
        volume_volatility = np.random.randint(-300, 700)
        volume = max(100, base_volume + volume_volatility)  # Ensure minimum volume
        
        # If price moved significantly, increase volume
        if abs(total_movement) > self.volatility * 2:
            volume *= 1.5
        
        # Move timestamp forward based on timeframe
        minutes_delta = 1
        if self.timeframe == "5m":
            minutes_delta = 5
        elif self.timeframe == "15m":
            minutes_delta = 15
        elif self.timeframe == "1h":
            minutes_delta = 60
        elif self.timeframe == "4h":
            minutes_delta = 240
        
        self.test_timestamp += timedelta(minutes=minutes_delta)
        
        return {
            "timestamp": self.test_timestamp,
            "$open": open_price,
            "$high": high_price,
            "$low": low_price,
            "$close": close_price,
            "$volume": volume
        }
    
    async def start(self):
        """Start data streaming"""
        try:
            self.is_running = True
            logger.info(f"Started data stream for {self.symbol}")
            
            while self.is_running:
                if self.test_mode:
                    new_data = self._generate_test_data()
                else:
                    # TODO: Implement real market data fetching
                    new_data = None
                
                if new_data:
                    self.data_buffer.append(new_data)
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer.pop(0)
                    self.last_update = datetime.now()
                
                # Adjust sleep time based on timeframe
                sleep_time = 0.5  # Default for 1m
                if self.timeframe == "5m":
                    sleep_time = 2.5
                elif self.timeframe == "15m":
                    sleep_time = 7.5
                elif self.timeframe == "1h":
                    sleep_time = 30
                elif self.timeframe == "4h":
                    sleep_time = 120
                
                await asyncio.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in data stream: {str(e)}", exc_info=True)
            self.is_running = False
    
    def stop(self):
        """Stop data streaming"""
        self.is_running = False
        logger.info("Stopped data stream")
    
    def get_current_data(self, lookback: int = 100) -> pd.DataFrame:
        """Get current market data as DataFrame"""
        try:
            if not self.data_buffer:
                return pd.DataFrame()
            
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffer)
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
            
            # Sort index to ensure correct order
            df.sort_index(inplace=True)
            
            # Return the last 'lookback' number of rows
            return df.tail(lookback)
        
        except Exception as e:
            logger.error(f"Error getting current data: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest price"""
        try:
            if not self.data_buffer:
                return None
            return self.data_buffer[-1]["$close"]
        except Exception as e:
            logger.error(f"Error getting latest price: {str(e)}", exc_info=True)
            return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators"""
        try:
            indicators = {}
            
            # Simple Moving Averages
            indicators["SMA20"] = data["$close"].rolling(window=20).mean()
            indicators["SMA50"] = data["$close"].rolling(window=50).mean()
            
            # Exponential Moving Averages
            indicators["EMA20"] = data["$close"].ewm(span=20, adjust=False).mean()
            indicators["EMA50"] = data["$close"].ewm(span=50, adjust=False).mean()
            
            # RSI
            delta = data["$close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators["RSI"] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma20 = data["$close"].rolling(window=20).mean()
            std20 = data["$close"].rolling(window=20).std()
            indicators["BB_upper"] = sma20 + (std20 * 2)
            indicators["BB_lower"] = sma20 - (std20 * 2)
            
            return indicators
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return {}
