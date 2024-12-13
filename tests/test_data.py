import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from data.utils.data_loader import DataLoader
from data.utils.qlib_processor import QlibProcessor
from data.utils.feature_generator import FeatureGenerator, FeatureConfig

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'timestamp': dates,
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def mock_exchange():
    """Create mock exchange"""
    mock = MagicMock()
    mock.fetch_ohlcv.return_value = [
        [1640995200000, 100.0, 105.0, 95.0, 101.0, 1000.0],  # 2022-01-01
        [1641081600000, 101.0, 106.0, 96.0, 102.0, 1100.0],  # 2022-01-02
        [1641168000000, 102.0, 107.0, 97.0, 103.0, 1200.0],  # 2022-01-03
    ]
    return mock

@pytest.fixture
def data_loader(mock_exchange):
    """Create DataLoader instance with mock exchange"""
    with patch('ccxt.binance', return_value=mock_exchange):
        return DataLoader('binance')

@pytest.fixture
def qlib_processor(tmp_path):
    """Create QlibProcessor instance with temporary path"""
    return QlibProcessor(data_path=str(tmp_path / 'qlib_data'))

@pytest.fixture
def feature_generator():
    """Create FeatureGenerator instance"""
    return FeatureGenerator()

class TestDataLoader:
    def test_initialization(self, data_loader, mock_exchange):
        """Test DataLoader initialization"""
        assert data_loader.exchange == mock_exchange
    
    def test_fetch_ohlcv(self, data_loader):
        """Test fetching OHLCV data"""
        df = data_loader.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=3)
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check data validity
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        assert (df['volume'] >= 0).all()
        assert len(df) == 3

class TestQlibProcessor:
    def test_process_raw_data(self, qlib_processor, sample_ohlcv_data):
        """Test processing raw data to Qlib format"""
        processed_df = qlib_processor.process_raw_data(sample_ohlcv_data, 'BTC/USDT')
        
        # Check required columns
        required_columns = ['datetime', 'instrument', '$open', '$high', '$low', '$close', 
                          '$volume', '$amount', '$factor']
        assert all(col in processed_df.columns for col in required_columns)
        
        # Check instrument name
        assert (processed_df['instrument'] == 'BTCUSDT').all()
        
        # Check calculations
        assert (processed_df['$amount'] == 
               processed_df['$close'] * processed_df['$volume']).all()
        assert (processed_df['$factor'] == 1.0).all()

class TestFeatureGenerator:
    def test_generate_features(self, feature_generator, sample_ohlcv_data):
        """Test feature generation"""
        # Prepare data in Qlib format
        df = sample_ohlcv_data.copy()
        df.columns = ['$' + col for col in df.columns]
        
        # Create feature configs
        configs = [
            FeatureConfig("RSI", {"window": 2}),
            FeatureConfig("BB", {"window": 2}),
            FeatureConfig("Volume_Features", {"window": 2}),
            FeatureConfig("Price_Features")
        ]
        
        # Generate features
        features_df = feature_generator.generate_features(df, configs)
        
        # Check basic feature properties
        assert not features_df.empty
        assert not features_df.isnull().all().any()  # No columns should be all null
        
        # Check specific features
        assert 'RSI' in features_df.columns
        assert 'BB_upper' in features_df.columns
        assert 'BB_lower' in features_df.columns
        assert 'Volume_SMA' in features_df.columns
        assert 'Daily_return' in features_df.columns

def test_full_pipeline(data_loader, qlib_processor, feature_generator):
    """Test entire data pipeline"""
    # 1. Fetch data
    raw_data = data_loader.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=10)
    assert not raw_data.empty
    assert len(raw_data) == 3  # Mock data has 3 entries
    
    # 2. Process to Qlib format
    processed_data = qlib_processor.process_raw_data(raw_data, 'BTC/USDT')
    assert not processed_data.empty
    assert len(processed_data) == 3
    
    # 3. Generate features
    configs = [
        FeatureConfig("RSI", {"window": 2}),
        FeatureConfig("BB", {"window": 2}),
        FeatureConfig("Volume_Features", {"window": 2}),
        FeatureConfig("Price_Features")
    ]
    features_df = feature_generator.generate_features(processed_data, configs)
    
    # Check final output
    assert not features_df.empty
    assert len(features_df) == 3
    # Features will have some NaN values due to rolling windows, but not all NaN
    assert not features_df.isnull().all().any()  # No columns should be all null