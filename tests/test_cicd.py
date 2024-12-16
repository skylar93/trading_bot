"""CI/CD Pipeline Tests"""

import pytest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data.utils.data_loader import DataLoader
from training.train import train_agent
from training.backtest import Backtester
from training.evaluation import TradingMetrics

def test_data_pipeline():
    """Test data pipeline components"""
    # Initialize DataLoader
    loader = DataLoader(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h"
    )
    
    # Test data fetching
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    data = loader.fetch_data(start_date, end_date)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_backtesting():
    """Test backtesting system"""
    # Create test data
    data = pd.DataFrame({
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.abs(np.random.randn(100))
    })
    
    # Initialize backtester
    backtester = Backtester(
        data=data,
        initial_balance=10000,
        transaction_fee=0.001
    )
    
    # Create mock agent for testing
    class MockAgent:
        def get_action(self, state):
            return np.random.uniform(-1, 1)
    
    mock_agent = MockAgent()
    
    # Run backtest
    try:
        results = backtester.run(mock_agent)
        assert 'metrics' in results
        assert 'trades' in results
        assert 'portfolio_values' in results
    except Exception as e:
        pytest.fail(f"Backtesting failed with error: {str(e)}")

@pytest.mark.integration
def test_full_pipeline():
    """Test full training and backtesting pipeline"""
    # Create test data
    data = pd.DataFrame({
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.abs(np.random.randn(100))
    })
    
    # Split data
    train_data = data[:70]
    val_data = data[70:85]
    test_data = data[85:]
    
    # Train agent
    try:
        agent = train_agent(train_data, val_data)
        assert agent is not None
    except Exception as e:
        pytest.fail(f"Training failed with error: {str(e)}")
    
    # Run backtest
    backtester = Backtester(
        data=test_data,
        initial_balance=10000,
        transaction_fee=0.001
    )
    
    results = backtester.run(agent)
    assert results['metrics'] is not None
    assert len(results['trades']) > 0

@pytest.mark.performance
def test_resource_usage():
    """Test resource usage during training"""
    import psutil
    import time
    
    # Create test data
    data = pd.DataFrame({
        'open': np.random.randn(1000),
        'high': np.random.randn(1000),
        'low': np.random.randn(1000),
        'close': np.random.randn(1000),
        'volume': np.abs(np.random.randn(1000))
    })
    
    # Monitor resource usage
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Train agent
    train_data = data[:700]
    val_data = data[700:850]
    agent = train_agent(train_data, val_data)
    
    # Check resource usage
    end_time = time.time()
    final_memory = process.memory_info().rss
    
    # Log metrics
    training_time = end_time - start_time
    memory_usage = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Assert reasonable resource usage
    assert training_time < 300  # 5 minutes
    assert memory_usage < 1024  # 1 GB

if __name__ == '__main__':
    pytest.main([__file__])