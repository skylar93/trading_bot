"""Tests for training system"""

import pytest
import numpy as np
import pandas as pd
import mlflow
from training.train import TrainingPipeline
from training.utils.mlflow_manager import MLflowManager
from envs.trading_env import TradingEnvironment
from agents.strategies.ppo_agent import PPOAgent

@pytest.fixture
def config():
    """Create training configuration"""
    return {
        'env': {
            'initial_balance': 10000.0,
            'trading_fee': 0.001,
            'window_size': 20
        },
        'model': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'batch_size': 32
        },
        'training': {
            'total_timesteps': 1000
        }
    }

@pytest.fixture
def trainer(config, mlflow_test_context):
    """Create trainer instance with MLflow integration"""
    trainer = TrainingPipeline(config)
    yield trainer

@pytest.fixture
def sample_data():
    """Create sample market data with $ prefix columns"""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
    data = pd.DataFrame({
        '$open': np.random.randn(1000).cumsum() + 100,
        '$high': np.random.randn(1000).cumsum() + 102,
        '$low': np.random.randn(1000).cumsum() + 98,
        '$close': np.random.randn(1000).cumsum() + 100,
        '$volume': np.abs(np.random.randn(1000) * 1000)
    }, index=dates)
    return data

def test_trainer_initialization(trainer):
    """Test trainer initialization"""
    assert trainer.config is not None

def test_environment_creation(sample_data, config):
    """Test environment creation"""
    env = TradingEnvironment(
        df=sample_data,
        initial_balance=config['env']['initial_balance'],
        trading_fee=config['env']['trading_fee'],
        window_size=config['env']['window_size']
    )
    
    assert isinstance(env, TradingEnvironment)
    assert env.initial_balance == 10000.0
    assert env.trading_fee == 0.001
    assert env.window_size == 20
    
    # Test environment reset
    obs, info = env.reset()
    assert obs is not None
    assert info is not None
    assert 'portfolio_value' in info

@pytest.mark.integration
def test_training_pipeline(sample_data, trainer, mlflow_test_context):
    """Test training pipeline"""
    train_size = int(len(sample_data) * 0.8)
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:]
    
    # Train agent
    agent = PPOAgent(
        observation_space=None,  # Will be set by environment
        action_space=None,      # Will be set by environment
        learning_rate=trainer.config['model']['learning_rate'],
        gamma=trainer.config['model']['gamma']
    )
    
    env = TradingEnvironment(
        df=train_data,
        initial_balance=trainer.config['env']['initial_balance'],
        trading_fee=trainer.config['env']['trading_fee'],
        window_size=trainer.config['env']['window_size']
    )
    
    # Train the agent
    training_results = agent.train(env, total_timesteps=trainer.config['training']['total_timesteps'])
    
    assert training_results is not None
    assert 'episode_rewards' in training_results
    assert 'mean_reward' in training_results
    assert 'std_reward' in training_results

if __name__ == "__main__":
    pytest.main(["-v", __file__])