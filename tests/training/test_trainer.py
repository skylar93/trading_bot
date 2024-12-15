"""Tests for Distributed Trainer"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from training.utils.trainer import DistributedTrainer, TrainingConfig

@pytest.fixture
def sample_data():
    """Create sample price data"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        end=datetime.now(),
        freq='1H'
    )
    
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum(),
        'high': np.random.randn(len(dates)).cumsum(),
        'low': np.random.randn(len(dates)).cumsum(),
        'close': np.random.randn(len(dates)).cumsum(),
        'volume': np.abs(np.random.randn(len(dates)))
    }, index=dates)
    
    return data

def test_trainer_initialization():
    """Test trainer initialization"""
    config = TrainingConfig()
    trainer = DistributedTrainer(config)
    
    assert trainer.ray_manager is not None
    assert trainer.mlflow_manager is not None

def test_environment_creation(sample_data):
    """Test trading environment creation"""
    config = TrainingConfig()
    trainer = DistributedTrainer(config)
    
    env = trainer.create_env(sample_data)
    
    assert env is not None
    assert env.df.equals(sample_data)
    assert env.initial_balance == config.initial_balance
    assert env.trading_fee == config.trading_fee

def test_agent_creation(sample_data):
    """Test PPO agent creation"""
    config = TrainingConfig()
    trainer = DistributedTrainer(config)
    
    env = trainer.create_env(sample_data)
    agent = trainer.create_agent(env)
    
    assert agent is not None

@pytest.mark.integration
def test_training_pipeline(sample_data):
    """Test full training pipeline"""
    # Reduce data size and epochs for testing
    train_data = sample_data[:48]  # 2 days
    val_data = sample_data[48:72]  # 1 day
    
    config = TrainingConfig(
        num_epochs=2,
        batch_size=16,
        num_parallel=2
    )
    
    trainer = DistributedTrainer(config)
    
    try:
        metrics = trainer.train(train_data, val_data)
        
        assert metrics is not None
        assert 'reward' in metrics
        assert 'portfolio_value' in metrics
        assert 'total_trades' in metrics
        
    finally:
        trainer.cleanup()

@pytest.mark.performance
def test_parallel_processing(sample_data):
    """Test parallel processing performance"""
    import time
    
    # Test with different numbers of parallel actors
    configs = [
        TrainingConfig(num_parallel=1),
        TrainingConfig(num_parallel=2)
    ]
    
    train_data = sample_data[:48]
    val_data = sample_data[48:72]
    
    execution_times = []
    
    for config in configs:
        trainer = DistributedTrainer(config)
        
        try:
            start_time = time.time()
            trainer.train(train_data, val_data)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
        finally:
            trainer.cleanup()
    
    # Verify that parallel execution is faster
    assert execution_times[1] < execution_times[0]

if __name__ == '__main__':
    pytest.main([__file__])