import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils.data_loader import DataLoader
import pandas as pd

def test_data_fetching():
    """Test basic data fetching functionality"""
    try:
        # Initialize the data loader
        loader = DataLoader('binance')
        
        # Fetch some BTC data
        btc_data = loader.fetch_ohlcv('BTC/USDT', '1h', limit=10)
        
        # Basic validations
        assert isinstance(btc_data, pd.DataFrame), "Result should be a DataFrame"
        assert len(btc_data) > 0, "DataFrame should not be empty"
        assert all(col in btc_data.columns for col in ['open', 'high', 'low', 'close', 'volume']), \
            "DataFrame should have OHLCV columns"
            
        print("\n=== Test Results ===")
        print(f"Data Shape: {btc_data.shape}")
        print("\nFirst few rows:")
        print(btc_data.head())
        print("\nTest passed successfully! ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_data_fetching()