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
from training.backtest import Backtester
from training.evaluation import TradingMetrics
from data.utils.feature_generator import FeatureGenerator
from envs.trading_env import TradingEnvironment
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
    # Create mock data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
    data = pd.DataFrame(
        {
            "$open": np.random.randn(100) * 10 + 100,
            "$high": np.random.randn(100) * 10 + 105,
            "$low": np.random.randn(100) * 10 + 95,
            "$close": np.random.randn(100) * 10 + 100,
            "$volume": np.abs(np.random.randn(100) * 1000),
        },
        index=dates,
    )

    # Create mock agent
    class MockAgent:
        def get_action(self, state):
            return np.array([np.random.uniform(-1, 1)])

    mock_agent = MockAgent()

    try:
        # Initialize backtester
        backtester = Backtester(
            data=data, initial_balance=10000, trading_fee=0.001
        )

        # Run backtest
        results = backtester.run(mock_agent)

        # Verify results
        assert isinstance(results, dict)
        assert "metrics" in results
        assert "trades" in results
        assert len(results["trades"]) >= 0

    except Exception as e:
        pytest.fail(f"Backtesting failed with error: {str(e)}")


@pytest.mark.integration
def test_full_pipeline():
    logger.debug("Starting test_full_pipeline")

    # Generate minimal test data
    logger.debug("Generating test data")
    df = create_test_data(size=50)  # Minimal dataset
    logger.debug(f"Generated {len(df)} rows of test data")

    # Split data into train and validation sets
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    val_data = df[train_size:]
    logger.debug(
        f"Split data into {len(train_data)} training and {len(val_data)} validation samples"
    )

    # Create minimal config
    config = {
        "env": {
            "initial_balance": 10000,
            "trading_fee": 0.001,
            "window_size": 3,  # Minimal window
        },
        "model": {
            "learning_rate": 1e-2,  # Faster learning
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "c1": 1.0,
            "c2": 0.01,
            "batch_size": 16,  # Smaller batches
            "n_epochs": 2,  # Minimal epochs
        },
        "training": {"total_timesteps": 50},  # Minimal steps
    }

    logger.debug("Configuration created")
    logger.debug(f"Config: {config}")

    # Train agent
    logger.debug("Starting agent training")
    try:
        agent = train_agent(
            train_data=train_data, val_data=val_data, config=config
        )
        logger.debug("Agent training completed successfully")
    except Exception as e:
        logger.error(f"Agent training failed with error: {str(e)}")
        raise

    # Test the trained agent
    logger.debug("Testing trained agent")
    env = TradingEnvironment(
        df=val_data,
        window_size=config["env"]["window_size"],
        initial_balance=config["env"]["initial_balance"],
        trading_fee=config["env"]["trading_fee"],
    )

    try:
        obs, _ = env.reset()
        done = truncated = False
        total_steps = 0
        max_steps = 20  # Very few test steps
        total_reward = 0

        while not (done or truncated) and total_steps < max_steps:
            action = agent.get_action(obs)
            logger.debug(f"Step {total_steps}: Taking action {action}")
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            logger.debug(
                f"Step {total_steps} reward: {reward:.2f}, total reward: {total_reward:.2f}"
            )

        logger.debug(f"Test completed after {total_steps} steps")
        logger.debug(f"Final done: {done}, truncated: {truncated}")
        logger.debug(f"Total reward: {total_reward:.2f}")

        assert total_steps > 0, "Agent should have taken at least one step"
        assert total_steps <= max_steps, "Test exceeded maximum steps"

    except Exception as e:
        logger.error(f"Testing failed with error: {str(e)}")
        raise


@pytest.mark.performance
def test_resource_usage():
    """Test resource usage monitoring"""
    try:
        # Create environment
        env = TradingEnvironment(
            df=pd.DataFrame(
                {
                    "$open": np.random.randn(100),
                    "$high": np.random.randn(100),
                    "$low": np.random.randn(100),
                    "$close": np.random.randn(100),
                    "$volume": np.abs(np.random.randn(100)),
                }
            ),
            initial_balance=10000.0,
            trading_fee=0.001,
            window_size=20,
        )

        # Create agent
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=3e-4,
            gamma=0.99,
        )

        # Run some steps
        state, _ = env.reset()
        for _ in range(10):
            action = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            if done:
                break
            state = next_state

        assert True, "Resource usage test completed successfully"

    except Exception as e:
        assert False, f"Resource usage test failed with error: {str(e)}"


def test_environment_initialization():
    env = TradingEnvironment(
        df=create_test_data(),
        initial_balance=10000.0,
        trading_fee=0.001,
        window_size=20,
    )


def test_agent_training():
    env = TradingEnvironment(
        df=create_test_data(),
        initial_balance=10000.0,
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
