"""CI/CD Pipeline Tests"""

import pytest
import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data.utils.data_loader import DataLoader
from training.train import TrainingPipeline
from training.backtest import BacktestEngine
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

def test_training_pipeline():
    """Test training pipeline"""
    # Create test data
    data = pd.DataFrame({
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.abs(np.random.randn(100))
    })
    
    # Initialize pipeline
    pipeline = TrainingPipeline("config/default_config.yaml")
    
    # Run training
    try:
        agent = pipeline.train(
            train_data=data[:70],
            val_data=data[70:85]
        )
        assert agent is not None
    except Exception as e:
        pytest.fail(f"Training failed with error: {str(e)}")

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
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_balance=10000,
        trading_fee=0.001
    )
    
    # Run backtest
    try:
        results = engine.run(data)
        assert 'portfolio_value' in results
        assert 'trades' in results
        assert 'metrics' in results
    except Exception as e:
        pytest.fail(f"Backtesting failed with error: {str(e)}")

@pytest.mark.integration
def test_full_pipeline():
    """Test full training and backtesting pipeline"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test data
        data = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.abs(np.random.randn(100))
        })
        
        # Save test data
        data_path = os.path.join(temp_dir, "test_data.csv")
        data.to_csv(data_path)
        
        # Run training
        train_script = "training/train.py"
        result = subprocess.run(
            [sys.executable, train_script, "--data", data_path],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Run backtesting
        backtest_script = "training/backtest.py"
        result = subprocess.run(
            [sys.executable, backtest_script, "--data", data_path],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Check for output files
        assert os.path.exists("mlruns")
        assert os.path.exists("training_viz")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

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
    
    # Initialize pipeline
    pipeline = TrainingPipeline("config/default_config.yaml")
    
    # Monitor resource usage
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Run training
    agent = pipeline.train(
        train_data=data[:700],
        val_data=data[700:850]
    )
    
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