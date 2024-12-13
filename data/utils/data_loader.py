import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for cryptocurrency data using ccxt"""
    
    def __init__(self, exchange_id: str = "binance", symbol: str = "BTC/USDT",
                 timeframe: str = "1h", cache_dir: Optional[str] = "data/raw"):
        """
        Initialize DataLoader
        
        Args:
            exchange_id: Exchange name (e.g., "binance")
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1m", "1h", "1d")
            cache_dir: Directory to cache downloaded data
        """
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, start_date: str, end_date: str) -> Path:
        """Get cache file path for the given date range"""
        symbol_clean = self.symbol.replace('/', '_')
        return self.cache_dir / f"{symbol_clean}_{self.timeframe}_{start_date}_{end_date}.csv"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        try:
            if cache_path.exists():
                df = pd.read_csv(cache_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info(f"Loaded data from cache: {cache_path}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache"""
        try:
            df.to_csv(cache_path)
            logger.info(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def fetch_data(self, start_date: str, end_date: str = None, 
                  use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: current date)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Try loading from cache first
            if use_cache and self.cache_dir:
                cache_path = self._get_cache_path(start_date, end_date)
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    return cached_data
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            all_data = []
            current_timestamp = start_timestamp
            
            # Fetch data in chunks
            while current_timestamp < end_timestamp:
                logger.info(f"Fetching data from {datetime.fromtimestamp(current_timestamp/1000)}")
                
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        since=current_timestamp,
                        limit=1000  # Maximum allowed by most exchanges
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_data.extend(ohlcv)
                    
                    # Update timestamp for next iteration
                    current_timestamp = ohlcv[-1][0] + 1
                    
                except Exception as e:
                    logger.error(f"Error fetching chunk: {str(e)}")
                    # Add delay before next attempt
                    from time import sleep
                    sleep(1)
            
            if not all_data:
                raise ValueError("No data fetched")
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Sort index
            df = df.sort_index()
            
            # Add basic sanity checks
            if df.empty:
                raise ValueError("Empty dataframe")
            
            if df.index.duplicated().any():
                logger.warning("Found duplicate timestamps, keeping last occurrence")
                df = df[~df.index.duplicated(keep='last')]
            
            if (df.index.to_series().diff() <= timedelta(0)).any():
                raise ValueError("Timestamps not strictly increasing")
            
            if (df < 0).any().any():
                raise ValueError("Found negative values in data")
            
            # Save to cache if enabled
            if use_cache and self.cache_dir:
                self._save_to_cache(df, self._get_cache_path(start_date, end_date))
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def get_latest_data(self, lookback_days: int = 1) -> pd.DataFrame:
        """
        Fetch most recent data
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with recent OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        return self.fetch_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            use_cache=False  # Don't use cache for latest data
        )

def test_data_loader():
    """Quick test of DataLoader functionality"""
    loader = DataLoader()
    
    # Test basic functionality
    df = loader.fetch_data('2023-01-01', '2023-01-02')
    print("\nFetched data shape:", df.shape)
    print("\nSample data:")
    print(df.head())
    
    # Test latest data functionality
    df_latest = loader.get_latest_data(lookback_days=1)
    print("\nLatest data shape:", df_latest.shape)
    print("\nLatest data sample:")
    print(df_latest.head())

if __name__ == "__main__":
    test_data_loader()