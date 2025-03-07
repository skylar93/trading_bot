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
from training.utils.unified_mlflow_manager import MLflowManager
import asyncio
import sys
import torch
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Add root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set fixed seeds for reproducible tests
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set fixed seeds for all tests to ensure reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    experiment_name = (
        f"test_experiment_{test_name}_{timestamp}_{random_suffix}"
    )

    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)

    # Create MLflow manager
    mlflow_manager = MLflowManager(
        experiment_name=experiment_name, tracking_dir=temp_dir
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
    """Generate sample price data for testing"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "$open": np.random.randn(100) * 100 + 1000,
            "$high": np.random.randn(100) * 100 + 1100,
            "$low": np.random.randn(100) * 100 + 900,
            "$close": np.random.randn(100) * 100 + 1000,
            "$volume": np.random.rand(100) * 1000,
        },
        index=dates,
    )
    return df


@pytest.fixture
def config_path(temp_dir):
    """Create temporary config file"""
    config = {
        "env": {
            "initial_balance": 10000,
            "trading_fee": 0.001,
            "window_size": 20,
        },
        "model": {"hidden_size": 256, "num_layers": 2},
        "training": {
            "batch_size": 128,
            "learning_rate": 0.0003,
            "num_episodes": 10,  # Reduced for testing
        },
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
                "loss": float(np.mean(batch_data)),
                "metrics": {"batch_size": len(batch_data)},
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
            info = {"portfolio_value": 10000.0}
            return obs, info

        def step(self, action):
            reward = np.random.randn()
            done = np.random.random() > 0.9
            truncated = False
            next_state = np.random.randn(20, 5)  # (window_size, features)
            info = {
                "portfolio_value": 10000 * (1 + reward),
                "position": action[0],
                "current_price": 100.0,
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
                "loss": np.random.randn(),
                "metrics": {
                    "sharpe_ratio": np.random.rand(),
                    "max_drawdown": -np.random.rand() * 0.1,
                },
            }

        def save(self, path):
            pass

    return MockAgent()


@pytest.fixture
def mock_dataloader():
    """Mock data loader for testing"""

    class MockDataLoader:
        def fetch_data(self, start_date, end_date):
            dates = pd.date_range(start=start_date, end=end_date, freq="1h")

            # Generate consistent OHLCV data
            base_price = 100
            returns = np.random.normal(0, 0.01, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))

            df = pd.DataFrame(
                {
                    "$open": prices
                    * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
                    "$high": prices
                    * (1 + np.random.uniform(0, 0.002, len(dates))),
                    "$low": prices
                    * (1 - np.random.uniform(0, 0.002, len(dates))),
                    "$close": prices,
                    "$volume": np.abs(np.random.normal(1000, 100, len(dates))),
                },
                index=dates,
            )

            # Ensure high is highest and low is lowest
            df["$high"] = df[["$open", "$high", "$low", "$close"]].max(axis=1)
            df["$low"] = df[["$open", "$high", "$low", "$close"]].min(axis=1)

            return df

    return MockDataLoader()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mlflow_tracking():
    """Set up and tear down MLflow tracking for all tests.
    
    This fixture:
    1. Creates a temporary directory for MLflow tracking
    2. Sets up MLflow tracking URI
    3. Creates a default experiment
    4. Cleans up after all tests
    """
    # Create temporary directory for MLflow
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    tracking_uri = f"file://{temp_dir}"
    
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create and set default experiment
    experiment_name = f"test_experiment_{int(time.time())}"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            mlflow.delete_experiment(experiment.experiment_id)
            time.sleep(0.1)  # Wait for deletion to complete
    except:
        pass
        
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    yield temp_dir
    
    # Clean up any remaining active runs
    try:
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()
            time.sleep(0.1)  # Wait for run to end
    except Exception as e:
        logger.warning(f"Error ending active run: {e}")
    
    # Delete experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            mlflow.delete_experiment(experiment.experiment_id)
    except Exception as e:
        logger.warning(f"Error deleting experiment: {e}")
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Failed to cleanup MLflow temp directory: {str(e)}")


@pytest.fixture
def mlflow_run(mlflow_tracking):
    """Create a new MLflow run for each test.
    
    This fixture ensures each test gets its own MLflow run and proper cleanup.
    """
    # End any existing runs
    try:
        if mlflow.active_run():
            mlflow.end_run()
            time.sleep(0.1)  # Wait for run to end
    except Exception as e:
        logger.warning(f"Error ending existing run: {e}")
    
    # Start a new run
    run = mlflow.start_run()
    yield run
    
    # Ensure run is ended
    try:
        if mlflow.active_run():
            mlflow.end_run()
            time.sleep(0.1)  # Wait for run to end
    except Exception as e:
        logger.warning(f"Error ending test run: {e}")


@pytest.fixture
def ray_results_dir():
    """Create and manage a temporary directory for Ray results.
    
    Returns:
        Path to temporary directory
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_price_history():
    """Generate a small price history for quick tests"""
    rows = 50
    rng = np.random.RandomState(42)
    
    # Generate price data
    close_prices = 100 + np.cumsum(rng.normal(0, 1, rows))
    
    df = pd.DataFrame({
        "$open": close_prices + rng.normal(0, 0.5, rows),
        "$high": close_prices + rng.uniform(0, 2, rows),
        "$low": close_prices - rng.uniform(0, 2, rows),
        "$close": close_prices,
        "$volume": rng.randint(100, 1000, rows)
    })
    
    df.index = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    return df


@pytest.fixture
def market_regime_data():
    """Generate synthetic data with different market regimes"""
    # Create 4 distinct regimes: trending up, trending down, ranging, volatile
    regime_length = 25  # Each regime has 25 data points
    
    # Trending up: positive drift
    trending_up = 100 + np.cumsum(np.random.normal(0.1, 0.5, regime_length))
    
    # Trending down: negative drift
    trending_down = 150 + np.cumsum(np.random.normal(-0.1, 0.5, regime_length))
    
    # Ranging: oscillating around a mean
    ranging_base = 120 + np.sin(np.linspace(0, 4*np.pi, regime_length)) * 5
    ranging = ranging_base + np.random.normal(0, 1, regime_length)
    
    # Volatile: high standard deviation
    volatile = 130 + np.cumsum(np.random.normal(0, 2.0, regime_length))
    
    # Combine the regimes
    close_prices = np.concatenate([trending_up, trending_down, ranging, volatile])
    
    # Create all price data
    rng = np.random.RandomState(42)
    rows = len(close_prices)
    
    df = pd.DataFrame({
        "$open": close_prices + rng.normal(0, 1, rows),
        "$high": close_prices + rng.uniform(0, 3, rows),
        "$low": close_prices - rng.uniform(0, 3, rows),
        "$close": close_prices,
        "$volume": rng.randint(100, 5000, rows)
    })
    
    # Add regime labels for easier analysis
    regimes = ['trending_up'] * regime_length + ['trending_down'] * regime_length + \
              ['ranging'] * regime_length + ['volatile'] * regime_length
    df['regime'] = regimes
    
    df.index = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    return df


# Helper function to create a realistic observation from price data
def create_observation(price_data: pd.DataFrame, window_size: int, step: int) -> np.ndarray:
    """
    Create a realistic observation from price data
    
    Args:
        price_data: DataFrame with OHLCV data
        window_size: Size of observation window
        step: Current step in the environment
        
    Returns:
        Observation array with shape (window_size, features)
    """
    if step < window_size:
        raise ValueError(f"Step {step} must be >= window_size {window_size}")
    
    # Extract window of price data
    window = price_data.iloc[step-window_size:step]
    
    # Extract OHLCV features
    features = window[['$open', '$high', '$low', '$close', '$volume']].values
    
    # Normalize features
    open_mean, open_std = features[:, 0].mean(), features[:, 0].std()
    high_mean, high_std = features[:, 1].mean(), features[:, 1].std()
    low_mean, low_std = features[:, 2].mean(), features[:, 2].std()
    close_mean, close_std = features[:, 3].mean(), features[:, 3].std()
    volume_mean, volume_std = features[:, 4].mean(), features[:, 4].std() + 1e-8
    
    normalized = np.zeros_like(features)
    normalized[:, 0] = (features[:, 0] - open_mean) / (open_std + 1e-8)
    normalized[:, 1] = (features[:, 1] - high_mean) / (high_std + 1e-8)
    normalized[:, 2] = (features[:, 2] - low_mean) / (low_std + 1e-8)
    normalized[:, 3] = (features[:, 3] - close_mean) / (close_std + 1e-8)
    normalized[:, 4] = (features[:, 4] - volume_mean) / (volume_std + 1e-8)
    
    return normalized
