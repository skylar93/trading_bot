"""Test configuration and fixtures"""

import pytest
import os
import pandas as pd
import numpy as np
import tempfile
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import mlflow
import time
from training.utils.mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Failed to cleanup temp directory: {str(e)}")
        raise

@pytest.fixture(scope="function")
def mlflow_test_context(request):
    """Create temporary MLflow test context with unique experiment name.
    
    This fixture ensures that each test gets a unique MLflow experiment name
    and properly cleans up after itself.
    """
    # Create temp directory for MLflow tracking
    temp_dir = tempfile.mkdtemp()
    
    # Create SQLite database in temp directory
    db_path = os.path.join(temp_dir, "mlflow.db")
    tracking_uri = f"sqlite:///{db_path}"
    
    # Create unique experiment name using timestamp, test name, and random suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    test_name = request.node.name.replace("[", "_").replace("]", "_")
    random_suffix = os.urandom(4).hex()
    experiment_name = f"test_experiment_{test_name}_{timestamp}_{random_suffix}"
    
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create MLflow manager
    mlflow_manager = MLflowManager(
        experiment_name=experiment_name,
        tracking_dir=temp_dir
    )
    
    yield mlflow_manager
    
    # Cleanup
    if mlflow.active_run():
        mlflow.end_run()
        time.sleep(0.1)
    
    try:
        # End any active runs
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            for run in mlflow.search_runs([experiment.experiment_id]):
                if run.info.status == "RUNNING":
                    mlflow.end_run(run_id=run.info.run_id)
            # Delete experiment
            mlflow.delete_experiment(experiment.experiment_id)
    except:
        pass
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Reset MLflow tracking URI
    mlflow.set_tracking_uri("")

@pytest.fixture
def sample_data():
    """Create sample price data for testing with $ prefix columns"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        end=datetime.now(),
        freq='1H'
    )
    
    # Generate consistent OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.01, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        '$open': prices * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
        '$high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
        '$low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
        '$close': prices,
        '$volume': np.abs(np.random.normal(1000, 100, len(dates)))
    })
    
    # Ensure high is highest and low is lowest
    df['$high'] = df[['$open', '$high', '$low', '$close']].max(axis=1)
    df['$low'] = df[['$open', '$high', '$low', '$close']].min(axis=1)
    
    return df.set_index('timestamp')

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
            obs = np.zeros((20, 5))  # (window_size, features)
            info = {'portfolio_value': 10000.0}
            return obs, info
            
        def step(self, action):
            reward = np.random.randn()
            done = np.random.random() > 0.9
            truncated = False
            next_state = np.random.randn(20, 5)  # (window_size, features)
            info = {
                'portfolio_value': 10000 * (1 + reward),
                'position': action[0],
                'current_price': 100.0
            }
            return next_state, reward, done, truncated, info
            
    return MockEnv()

@pytest.fixture
def mock_agent():
    """Mock trading agent for testing"""
    class MockAgent:
        def __init__(self):
            pass
            
        def get_action(self, state):
            return np.array([np.random.uniform(-1, 1)])
            
        def train(self, *args, **kwargs):
            return {
                'loss': np.random.randn(),
                'metrics': {
                    'sharpe_ratio': np.random.rand(),
                    'max_drawdown': -np.random.rand() * 0.1
                }
            }
            
        def save(self, path):
            pass
            
    return MockAgent()

@pytest.fixture
def mock_dataloader():
    """Mock data loader for testing"""
    class MockDataLoader:
        def fetch_data(self, start_date, end_date):
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            
            # Generate consistent OHLCV data
            base_price = 100
            returns = np.random.normal(0, 0.01, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                '$open': prices * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
                '$high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
                '$low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
                '$close': prices,
                '$volume': np.abs(np.random.normal(1000, 100, len(dates)))
            }, index=dates)
            
            # Ensure high is highest and low is lowest
            df['$high'] = df[['$open', '$high', '$low', '$close']].max(axis=1)
            df['$low'] = df[['$open', '$high', '$low', '$close']].min(axis=1)
            
            return df
            
    return MockDataLoader()