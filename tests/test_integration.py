import pytest
import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
import logging.config

from data.utils.data_loader import DataLoader
from envs.trading_env import TradingEnvironment
from envs.wrap_env import make_env
from training.train import load_config, create_env
from data.utils.feature_generator import FeatureGenerator


# Set up logging configuration
def setup_logging():
    """Set up logging configuration"""
    log_config_path = Path("config/logging_config.yaml")
    if log_config_path.exists():
        with open(log_config_path, "r") as f:
            config = yaml.safe_load(f)
            # Ensure log directory exists
            os.makedirs("logs", exist_ok=True)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        )


# Set up logging before tests run
setup_logging()
logger = logging.getLogger("trading_bot.tests")


def create_test_data():
    """Create sample data for testing"""
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
    return data


class TestIntegration:
    @pytest.fixture
    def config(self):
        """Load configuration"""
        config_path = Path("config/default_config.yaml")
        assert config_path.exists(), "Configuration file not found"
        return load_config(str(config_path))

    def test_data_to_env_pipeline(self, config):
        """Test data pipeline integration with environment"""
        logger.info("Starting data pipeline integration test")

        try:
            # 1. Load data
            loader = DataLoader(config["data"]["exchange"])
            logger.debug("Created DataLoader instance")

            df = loader.fetch_and_process(
                symbol=config["data"]["symbols"][0],
                timeframe=config["data"]["timeframe"],
                start_date=config["data"]["start_date"],
                limit=100,  # Use small dataset for testing
            )
            logger.debug(f"Loaded data shape: {df.shape}")

            assert not df.empty, "Failed to load data"
            required_columns = ["$open", "$high", "$low", "$close", "$volume"]
            assert all(
                col in df.columns for col in required_columns
            ), f"Missing required columns. Found: {df.columns.tolist()}"

            # Generate additional features
            feature_generator = FeatureGenerator()
            df = feature_generator.generate_features(df)
            logger.debug(f"Generated features. New shape: {df.shape}")

            # 2. Create environment
            window_size = 20  # Use smaller window size for testing
            logger.debug(
                f"Creating environment with window_size={window_size}"
            )
            env = TradingEnvironment(
                data=df,
                initial_capital=config["env"]["initial_balance"],
                trading_fee=config["env"]["trading_fee"],
                window_size=window_size,
            )
            logger.debug("Created TradingEnvironment instance")

            # 3. Apply wrappers
            logger.debug("Applying environment wrappers")
            wrapped_env = make_env(
                env,
                normalize=config["env"]["normalize"],
                stack_size=config["env"]["stack_size"],
            )
            logger.debug("Applied environment wrappers")

            # 4. Test environment functionality
            logger.debug("Testing environment reset")
            obs, info = wrapped_env.reset()
            logger.debug(f"Reset observation shape: {obs.shape}")
            logger.debug(f"Reset info: {info}")

            assert isinstance(
                obs, np.ndarray
            ), "Observation should be numpy array"
            expected_shape = (
                window_size,
                env.observation_space.shape[1],
            )
            assert (
                obs.shape == expected_shape
            ), f"Observation shape mismatch: expected {expected_shape}, got {obs.shape}"

            # 5. Test environment step
            logger.debug("Testing environment step")
            action = np.array([0.5])  # Buy position with 50% size
            obs, reward, done, truncated, info = wrapped_env.step(action)

            logger.debug(f"Step observation shape: {obs.shape}")
            logger.debug(f"Step reward: {reward}")
            logger.debug(f"Step info: {info}")

            assert isinstance(
                obs, np.ndarray
            ), "Step observation should be numpy array"
            assert isinstance(reward, float), "Reward should be float"
            assert isinstance(done, bool), "Done should be boolean"
            assert isinstance(info, dict), "Info should be dictionary"
            assert info["capital"] > 0, "Capital should be positive"

            logger.info(
                "Data pipeline integration test completed successfully"
            )

        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
            raise

    def test_create_env_function(self):
        """Test environment creation"""
        logger.info("Starting environment creation test")

        try:
            # Create test data
            data = create_test_data()

            # Create environment config
            env_config = {
                "data": data,
                "initial_capital": 10000.0,
                "trading_fee": 0.001,
                "window_size": 20,
            }

            # Create environment
            env = create_env(env_config)
            
            # Check if TradingEnvironment is in the wrapper chain
            def get_base_env(wrapped_env):
                if hasattr(wrapped_env, 'env'):
                    return get_base_env(wrapped_env.env)
                return wrapped_env
            
            base_env = get_base_env(env)
            assert isinstance(base_env, TradingEnvironment)

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            raise

    def test_full_episode(self):
        """Test running a full episode"""
        logger.info("Starting full episode test")

        try:
            # Create test data
            data = create_test_data()

            # Create environment config
            env_config = {
                "data": data,
                "initial_capital": 10000.0,
                "trading_fee": 0.001,
                "window_size": 20,
            }

            # Create environment
            env = create_env(env_config)

            # Run full episode
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
            logger.error(f"Test failed: {str(e)}")
            raise

    def test_state_transitions(self, config):
        """Test state transitions and position changes"""
        logger.info("Starting state transition test")

        try:
            # Create test environment with upward trending data
            dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
            df = pd.DataFrame(
                {
                    "$open": np.linspace(1000, 1100, 100),  # Upward trend
                    "$high": np.linspace(1010, 1110, 100),
                    "$low": np.linspace(990, 1090, 100),
                    "$close": np.linspace(1000, 1100, 100),
                    "$volume": np.random.rand(100) * 1000,
                    "RSI": np.random.uniform(0, 100, 100),
                    "MACD": np.random.normal(0, 1, 100),
                    "Signal": np.random.normal(0, 1, 100),
                },
                index=dates,
            )

            env = TradingEnvironment(
                data=df,
                initial_capital=10000.0,
                trading_fee=0.001,
                window_size=20,
            )

            # Test initial state
            obs, info = env.reset()
            assert info["position"] == 0.0, "Initial position should be zero"

            # Test buy action
            action = np.array([1.0])  # Full buy
            obs, reward, done, truncated, info = env.step(action)
            assert info["position"] > 0, "Position should be long after buy"
            initial_position = info["position"]

            # Test additional buy action
            action = np.array([1.0])  # Another buy
            obs, reward, done, truncated, info = env.step(action)
            assert info["position"] >= initial_position, "Position should increase or stay same after another buy"

            # Test sell action
            action = np.array([-1.0])  # Full sell
            obs, reward, done, truncated, info = env.step(action)
            assert info["position"] < initial_position, "Position should decrease after sell"

        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            raise

    def test_reward_calculation(self):
        """Test reward calculation"""
        logger.info("Starting reward calculation test")

        try:
            # Create environment
            env = TradingEnvironment(
                data=create_test_data(),
                initial_capital=10000.0,
                trading_fee=0.001,
                window_size=20,
            )

            # Reset environment
            obs, info = env.reset()
            assert "capital" in info
            assert info["capital"] == env.initial_capital

            # Take a buy action
            action = np.array([1.0])  # Full buy
            obs, reward, done, truncated, info = env.step(action)

            # Verify reward calculation
            assert isinstance(reward, float)
            assert "capital" in info
            assert info["capital"] > 0

        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            raise

    def _create_test_data(self):
        """Create test data with known price movements"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        return pd.DataFrame(
            {
                "$open": np.linspace(1000, 1100, 100),
                "$high": np.linspace(1010, 1110, 100),
                "$low": np.linspace(990, 1090, 100),
                "$close": np.linspace(1000, 1100, 100),
                "$volume": np.random.rand(100) * 1000,
                "RSI": np.random.uniform(0, 100, 100),
                "MACD": np.random.normal(0, 1, 100),
                "Signal": np.random.normal(0, 1, 100),
            },
            index=dates,
        )
