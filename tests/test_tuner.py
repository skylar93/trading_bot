"""Test hyperparameter tuning system"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.hyperopt import MinimalTuner as HyperParameterTuner

def generate_test_data(length: int = 1000) -> pd.DataFrame:
    """Generate test market data"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
    
    # Generate price process
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, length)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate features
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.uniform(-0.001, 0.001, length)),
        'high': price * (1 + np.random.uniform(0, 0.002, length)),
        'low': price * (1 - np.random.uniform(0, 0.002, length)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, length)
    })
    
    # Add some technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    
    return df.dropna()

def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.clip(lower=0)).rolling(window=periods).mean()
    loss = (-delta.clip(upper=0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@pytest.fixture
def test_data():
    """Create test data fixture"""
    return generate_test_data(1000)

def test_tuner_initialization(test_data):
    """Test tuner initialization"""
    tuner = HyperParameterTuner(test_data)
    assert hasattr(tuner, 'df')
    assert hasattr(tuner, 'objective')
    assert hasattr(tuner, 'run_optimization')

def test_optimization_run(test_data):
    """Test hyperparameter optimization with small search space"""
    tuner = HyperParameterTuner(test_data)
    
    # Create a minimal test config
    test_config = {
        'env': {
            'initial_balance': 10000.0,
            'trading_fee': 0.001,
            'window_size': 20
        },
        'model': {
            'fcnet_hiddens': [64, 64],
            'learning_rate': 0.001
        },
        'training': {
            'total_timesteps': 100
        }
    }
    
    # Run optimization with minimal settings for testing
    best_config = tuner.evaluate_config(test_config, episodes=1)
    
    # Check if metrics are returned
    assert isinstance(best_config, dict)
    assert 'sharpe_ratio' in best_config
    assert 'total_return' in best_config

def test_evaluate_config(test_data):
    """Test config evaluation"""
    tuner = HyperParameterTuner(test_data)
    
    test_config = {
        'env': {
            'initial_balance': 10000.0,
            'trading_fee': 0.001,
            'window_size': 20
        },
        'model': {
            'fcnet_hiddens': [64, 64],
            'learning_rate': 0.001
        },
        'training': {
            'total_timesteps': 100
        }
    }
    
    metrics = tuner.evaluate_config(test_config, episodes=1)
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert 'total_return' in metrics