import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta

from data.utils.data_loader import DataLoader
from data.utils.feature_generator import FeatureGenerator

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 100 + 1000,
        'high': np.random.randn(len(dates)) * 100 + 1000,
        'low': np.random.randn(len(dates)) * 100 + 1000,
        'close': np.random.randn(len(dates)) * 100 + 1000,
        'volume': np.abs(np.random.randn(len(dates)) * 1000 + 5000)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    return df

def test_data_loader():
    """Test DataLoader functionality"""
    loader = DataLoader(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h"
    )
    
    # Test data fetching for a small time window
    df = loader.fetch_data('2023-01-01', '2023-01-02')
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Test data validation
    assert not df.isnull().any().any()
    assert (df['high'] >= df['low']).all()
    assert (df['volume'] >= 0).all()

def test_feature_generator(sample_data):
    """Test FeatureGenerator functionality"""
    generator = FeatureGenerator()
    
    # Test feature generation
    df_features = generator.generate_features(sample_data)
    
    # Verify technical indicators
    expected_features = [
        'RSI',
        'SMA_20', 'EMA_20',
        'BB_upper', 'BB_middle', 'BB_lower',
        'ATR', 'volume_ma', 'PVT'
    ]
    
    # Check if expected features exist
    assert all(feature in df_features.columns for feature in expected_features)
    
    # Check for NaN values
    assert not df_features.isnull().any().any()
    
    # Basic feature validation
    assert (df_features['RSI'] >= 0).all() and (df_features['RSI'] <= 100).all()
    assert (df_features['BB_upper'] >= df_features['BB_lower']).all()
    assert (df_features['volume_ma'] >= 0).all()

def test_simple_pipeline():
    """Test complete data pipeline with a small sample"""
    # 1. Load data
    loader = DataLoader(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h"
    )
    df = loader.fetch_data('2023-01-01', '2023-01-02')
    
    # 2. Generate features
    generator = FeatureGenerator()
    df_features = generator.generate_features(df)
    
    # Basic validations
    assert not df_features.empty
    assert len(df_features.columns) > len(df.columns)
    assert not df_features.isnull().any().any()
    
    print("\nGenerated features:")
    print(list(df_features.columns))
    print("\nSample data:")
    print(df_features.head())
    
    return df_features

if __name__ == "__main__":
    print("Running basic pipeline test...")
    df = test_simple_pipeline()
    print("\nTest completed successfully!")