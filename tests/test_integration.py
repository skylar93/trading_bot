import pytest
import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
from datetime import datetime
import logging.config

from data.utils.data_loader import DataLoader
from envs.base_env import TradingEnvironment
from envs.wrap_env import make_env
from training.train import load_config, create_env


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
            from data.utils.feature_generator import FeatureGenerator

            feature_generator = FeatureGenerator()
            df = feature_generator.generate_features(df)
            logger.debug(f"Generated features. New shape: {df.shape}")

            # 2. Create environment
            window_size = 20  # Use smaller window size for testing
            logger.debug(
                f"Creating environment with window_size={window_size}"
            )

            env = TradingEnvironment(
                df=df,
                initial_balance=config["env"]["initial_balance"],
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
                env.n_features,
            )  # Updated to use actual features
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
            assert info["balance"] > 0, "Balance should be positive"

            logger.info(
                "Data pipeline integration test completed successfully"
            )

        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
            raise

    def test_create_env_function(self, config):
        """Test environment creation function used by RLlib"""
        logger.info("Starting environment creation test")

        try:
            # Create mock data
            dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
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
            logger.debug(f"Created mock data with shape: {df.shape}")

            # Create environment config
            env_config = {**config["env"], "df": df}
            logger.debug(f"Environment config: {env_config}")

            # Create environment
            env = create_env(env_config)
            logger.debug("Created environment instance")

            # Basic environment tests
            obs, info = env.reset()
            logger.debug(f"Reset observation shape: {obs.shape}")
            logger.debug(f"Reset info: {info}")

            assert isinstance(
                obs, np.ndarray
            ), "Observation should be numpy array"

            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            logger.debug(f"Step observation shape: {obs.shape}")
            logger.debug(f"Step reward: {reward}")
            logger.debug(f"Step info: {info}")

            assert isinstance(
                obs, np.ndarray
            ), "Step observation should be numpy array"
            assert isinstance(reward, float), "Reward should be float"
            assert isinstance(done, bool), "Done should be boolean"
            assert isinstance(info, dict), "Info should be dictionary"

            logger.info("Environment creation test completed successfully")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
            raise

    def test_full_episode(self, config):
        """Test running a full episode"""
        # Create mock data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
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

        # Create environment config
        env_config = {**config["env"], "df": df}

        # Create environment
        env = create_env(env_config)
        obs, info = env.reset()

        # Run episode
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 100  # Limit steps for testing

        while not done and step_count < max_steps:
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            # Validate step outputs
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

            total_reward += reward
            step_count += 1

            # Validate observation bounds if normalized
            if config["env"]["normalize"]:
                assert np.all(obs >= -1) and np.all(
                    obs <= 1
                ), "Normalized observation out of bounds"

        assert step_count > 0, "Episode should have at least one step"

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
                df=df,
                initial_balance=10000.0,
                trading_fee=0.001,
                window_size=20,
            )

            # Buy action
            obs1, info1 = env.reset()
            action1 = np.array([1.0])  # Buy
            obs1, reward1, done1, truncated1, info1 = env.step(action1)

            # Check position
            assert info1["position"].value > 0, "Position should be long"

            # Sell action
            action2 = np.array([-1.0])  # Sell
            obs2, reward2, done2, truncated2, info2 = env.step(action2)

            # Check position changed
            assert info2["position"].value < 0, "Position should be short"

        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            raise

    def test_reward_calculation(self, config):
        """Test reward calculation"""
        logger.info("Starting reward calculation test")

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
                },
                index=dates,
            )

            env = TradingEnvironment(
                df=df,
                initial_balance=10000.0,
                trading_fee=0.001,
                window_size=20,
            )

            # Initial state
            obs1, info1 = env.reset()
            initial_portfolio = info1.get(
                "portfolio_value", env.initial_balance
            )

            # Execute buy
            action = np.array([1.0])  # Full buy
            obs2, reward2, done2, truncated2, info2 = env.step(action)
            final_portfolio = info2["portfolio_value"]

            # Verify rewards
            portfolio_return = (
                final_portfolio - initial_portfolio
            ) / initial_portfolio
            assert (
                portfolio_return >= 0
            ), "Portfolio should increase in upward trend"
            assert reward2 >= 0, "Profit should give positive reward"

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
