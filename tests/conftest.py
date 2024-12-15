"""Test configuration and fixtures"""

import pytest
import os
import pandas as pd
import numpy as np
import tempfile
import yaml
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        end=datetime.now(),
        freq='1H'
    )
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum(),
        'high': np.random.randn(len(dates)).cumsum(),
        'low': np.random.randn(len(dates)).cumsum(),
        'close': np.random.randn(len(dates)).cumsum(),
        'volume': np.abs(np.random.randn(len(dates)))
    }).set_index('timestamp')

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

@pytest.fixture
def config_path(temp_dir):
    """Create temporary config file"""
    config = {
        'env': {
            'initial_balance': 10000,
            'trading_fee': 0.001,
            'window_size': 20
        },
        'model': {
            'hidden_size': 256,
            'num_layers': 2
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.0003,
            'num_episodes': 10  # Reduced for testing
        }
    }
    
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
def mock_ray_actor():
    """Mock Ray actor for testing"""
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    
    class MockActor:
        def process_batch(self, batch_data):
            return {
                'loss': float(np.mean(batch_data)),
                'metrics': {'batch_size': len(batch_data)}
            }
            
    return MockActor

@pytest.fixture
def mock_env():
    """Mock trading environment for testing"""
    class MockEnv:
        def __init__(self):
            self.reset()
            
        def reset(self):
            return np.zeros(10)  # State
            
        def step(self, action):
            reward = np.random.randn()
            done = np.random.random() > 0.9
            next_state = np.random.randn(10)
            info = {'portfolio_value': 10000 * (1 + reward)}
            return next_state, reward, done, info
            
    return MockEnv()

@pytest.fixture
def mock_agent():
    """Mock trading agent for testing"""
    class MockAgent:
        def __init__(self):
            pass
            
        def select_action(self, state):
            return np.random.randn()
            
        def train(self, *args, **kwargs):
            return {'loss': np.random.randn()}
            
        def save(self, path):
            pass
            
    return MockAgent()

@pytest.fixture
def mock_dataloader():
    """Mock data loader for testing"""
    class MockDataLoader:
        def fetch_data(self, start_date, end_date):
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            return pd.DataFrame({
                'open': np.random.randn(len(dates)),
                'high': np.random.randn(len(dates)),
                'low': np.random.randn(len(dates)),
                'close': np.random.randn(len(dates)),
                'volume': np.abs(np.random.randn(len(dates)))
            }, index=dates)
            
    return MockDataLoader()