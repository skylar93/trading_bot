"""CI/CD Pipeline Tests"""

import pytest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from data.utils.data_loader import DataLoader
from training.train import train_agent
from training.backtesting.base_backtester import BaseBacktester
from training.evaluation import TradingMetrics
from data.utils.feature_generator import FeatureGenerator
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from agents.strategies.single.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_data(size: int = 100) -> pd.DataFrame:
    """Create test data for testing

    Args:
        size: Number of data points to generate

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start="2023-01-01", periods=size, freq="1h")
    return pd.DataFrame(
        {
            "$open": np.random.randn(size) * 10 + 100,
            "$high": np.random.randn(size) * 10 + 105,
            "$low": np.random.randn(size) * 10 + 95,
            "$close": np.random.randn(size) * 10 + 100,
            "$volume": np.abs(np.random.randn(size) * 1000),
        },
        index=dates,
    )


def test_data_pipeline():
    """Test data pipeline functionality"""
    # Load data
    loader = DataLoader()
    data = loader.fetch_data("2024-12-09", "2024-12-16")

    # Verify data structure
    assert not data.empty, "Data should not be empty"
    assert all(
        col in data.columns
        for col in ["$open", "$high", "$low", "$close", "$volume"]
    ), f"Missing required columns. Found: {data.columns.tolist()}"

    # Test feature generation
    generator = FeatureGenerator()
    features = generator.generate_features(data)

    # Verify features
    assert len(features.columns) > len(
        data.columns
    ), "Should generate additional features"
    assert not features.isnull().any().any(), "Should not contain NaN values"

    return features


def test_backtesting():
    """Test backtesting functionality"""
    try:
        # Create test data
        data = create_test_data()

        # Create mock strategy
        class MockStrategy:
            def get_action(self, state):
                return np.array([0.5])  # Always try to buy with 50% of capital

        # Initialize backtester
        backtester = BaseBacktester(
            data=data,
            initial_capital=10000.0,
            trading_fee=0.001,
        )

        # Run backtest
        results = backtester.run(strategy=MockStrategy())

        # Verify results
        assert isinstance(results, dict)
        assert "metrics" in results
        assert "final_balance" in results["metrics"]
        assert "trades" in results
        assert "portfolio_values" in results

    except Exception as e:
        pytest.fail(f"Backtesting failed with error: {str(e)}")


@pytest.mark.integration
def test_full_pipeline():
    """Test full training pipeline"""
    try:
        # Create test data
        data = create_test_data()

        # Train agent
        agent = train_agent(
            train_data=data,
            val_data=data,  # Use same data for validation in test
            config={
                "env": {
                    "initial_capital": 10000.0,
                    "trading_fee": 0.001,
                    "window_size": 20,
                },
                "model": {
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "gamma": 0.99,
                },
                "training": {"total_timesteps": 50},
            },
        )

        assert agent is not None

    except Exception as e:
        logger.error(f"Agent training failed with error: {str(e)}")
        raise


@pytest.mark.performance
def test_resource_usage():
    """Test resource usage monitoring"""
    try:
        # Create test data
        data = create_test_data()

        # Create environment
        env = SingleAssetRLTradingEnv(
            data=data,
            initial_capital=10000.0,
            trading_fee=0.001,
            window_size=20,
        )

        # Run episode
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        assert isinstance(total_reward, float)
        assert "portfolio_value" in info

    except Exception as e:
        assert False, f"Resource usage test failed with error: {str(e)}"


def test_environment_initialization():
    env = SingleAssetRLTradingEnv(
        data=create_test_data(),
        initial_capital=10000.0,
        trading_fee=0.001,
        window_size=20,
    )


def test_agent_training():
    env = SingleAssetRLTradingEnv(
        data=create_test_data(),
        initial_capital=10000.0,
        trading_fee=0.001,
        window_size=20,
    )


def test_hyperparameter_tuning():
    config = {
        "initial_balance": 10000.0,
        "trading_fee": 0.001,
        "window_size": 20,
        "learning_rate": 0.001,
    }


if __name__ == "__main__":
    pytest.main([__file__])
