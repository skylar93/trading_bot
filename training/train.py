import os
import logging
import yaml
import pandas as pd
import numpy as np
import torch
from data.utils.data_loader import DataLoader
from agents.strategies.single.ppo_agent import PPOAgent
from training.utils.visualization import TradingVisualizer
from training.evaluation import TradingMetrics
from envs.trading_env import TradingEnvironment
from envs.wrap_env import make_env
from training.utils.mlflow_manager import MLflowManager
from typing import Union, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import ccxt

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    """Load configuration from yaml file"""
    import os

    # Get the absolute path of the project root
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Join with the config path
    config_path = os.path.join(project_root, config_path)

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to default config
        return {
            "env": {
                "initial_balance": 10000.0,
                "trading_fee": 0.001,
                "window_size": 20,
            },
            "model": {
                "fcnet_hiddens": [64, 64],
                "lr": 0.001,
                "gamma": 0.99,
                "epsilon": 0.2,
            },
            "training": {
                "total_timesteps": 10000,
                "early_stop": 20,
                "batch_size": 64,
            },
            "paths": {
                "model_dir": os.path.join(project_root, "models"),
                "data_dir": os.path.join(project_root, "data"),
                "log_dir": os.path.join(project_root, "logs"),
            },
        }


def create_env(env_config: Dict[str, Any]) -> TradingEnvironment:
    """Create trading environment from config

    Args:
        env_config: Environment configuration

    Returns:
        Trading environment instance
    """
    # Get data from config
    data = env_config.get("df", None)
    if data is None:
        raise ValueError("No data provided in env_config")

    # Handle different data types
    if isinstance(data, (list, tuple)):
        # Convert list of dictionaries to DataFrame
        if all(isinstance(d, dict) for d in data):
            max_len = max(
                len(v) if isinstance(v, (list, np.ndarray)) else 1
                for d in data
                for v in d.values()
            )
            normalized_data = {}
            for k, v in data[0].items():
                if not isinstance(v, (list, np.ndarray)):
                    v = [v] * max_len
                normalized_data[k] = v
            data = pd.DataFrame(normalized_data)

    # Ensure column names have '$' prefix
    if isinstance(data, pd.DataFrame):
        rename_dict = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in data.columns:
                rename_dict[col] = f"${col}"
        if rename_dict:
            data = data.rename(columns=rename_dict)

    # Create environment with config parameters
    env = TradingEnvironment(
        df=data,
        initial_balance=env_config.get("initial_balance", 10000.0),
        trading_fee=env_config.get("trading_fee", 0.001),
        window_size=env_config.get("window_size", 20),
        max_position_size=env_config.get("max_position_size", 1.0),
    )

    # Apply wrappers if specified in config
    if env_config.get("normalize", True):
        env = make_env(
            env, normalize=True, stack_size=env_config.get("stack_size", 4)
        )

    return env


def train_agent(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Dict[str, Any] = None,
) -> PPOAgent:
    """Train a PPO agent

    Args:
        train_data: Training data
        val_data: Validation data
        config: Configuration dictionary

    Returns:
        Trained PPO agent
    """
    # Use default config if none provided
    if config is None:
        config = {
            "env": {
                "initial_balance": 10000.0,
                "trading_fee": 0.001,
                "window_size": 20,
            },
            "model": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "c1": 1.0,
                "c2": 0.01,
                "batch_size": 64,
                "n_epochs": 10,
            },
            "training": {"total_timesteps": 10000},
        }

    # Convert flat config to nested if needed
    if not isinstance(config.get("env"), dict):
        env_config = config.get("env", {})
        if isinstance(env_config, dict):
            # Already nested
            pass
        else:
            # Convert flat to nested
            config = {
                "env": {
                    "initial_balance": config.get("initial_balance", 10000.0),
                    "trading_fee": config.get("trading_fee", 0.001),
                    "window_size": config.get("window_size", 20),
                },
                "model": {
                    "learning_rate": config.get("learning_rate", 3e-4),
                    "gamma": config.get("gamma", 0.99),
                    "gae_lambda": config.get("gae_lambda", 0.95),
                    "clip_epsilon": config.get("clip_epsilon", 0.2),
                    "c1": config.get("c1", 1.0),
                    "c2": config.get("c2", 0.01),
                    "batch_size": config.get("batch_size", 64),
                    "n_epochs": config.get("n_epochs", 10),
                },
                "training": {
                    "total_timesteps": config.get("total_timesteps", 10000)
                },
            }

    # Ensure all required config sections exist
    for section in ["env", "model", "training"]:
        if section not in config:
            config[section] = {}

    # Set default values for required parameters
    config["env"].setdefault("initial_balance", 10000.0)
    config["env"].setdefault("trading_fee", 0.001)
    config["env"].setdefault("window_size", 20)

    config["model"].setdefault("learning_rate", 3e-4)
    config["model"].setdefault("gamma", 0.99)
    config["model"].setdefault("gae_lambda", 0.95)
    config["model"].setdefault("clip_epsilon", 0.2)
    config["model"].setdefault("c1", 1.0)
    config["model"].setdefault("c2", 0.01)
    config["model"].setdefault("batch_size", 64)
    config["model"].setdefault("n_epochs", 10)

    config["training"].setdefault("total_timesteps", 10000)

    # Create environments
    train_env = TradingEnvironment(
        df=train_data,
        initial_balance=config["env"]["initial_balance"],
        trading_fee=config["env"]["trading_fee"],
        window_size=config["env"]["window_size"],
    )

    val_env = TradingEnvironment(
        df=val_data,
        initial_balance=config["env"]["initial_balance"],
        trading_fee=config["env"]["trading_fee"],
        window_size=config["env"]["window_size"],
    )

    # Create agent
    agent = PPOAgent(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        learning_rate=config["model"]["learning_rate"],
        gamma=config["model"]["gamma"],
        gae_lambda=config["model"]["gae_lambda"],
        clip_epsilon=config["model"]["clip_epsilon"],
        c1=config["model"]["c1"],
        c2=config["model"]["c2"],
        batch_size=config["model"]["batch_size"],
        n_epochs=config["model"]["n_epochs"],
    )

    # Train agent
    train_env.reset()
    agent.train(
        train_env, total_timesteps=config["training"]["total_timesteps"]
    )

    return agent


class TrainingPipeline:
    """Training pipeline for RL agents"""

    def __init__(self, config: Optional[str | Dict] = None):
        """Initialize training pipeline

        Args:
            config: Configuration file path or dictionary
        """
        if config is None:
            self.config = {
                "env": {
                    "initial_balance": 10000.0,
                    "trading_fee": 0.001,
                    "window_size": 20,
                },
                "model": {"fcnet_hiddens": [64, 64], "learning_rate": 0.001},
                "training": {"total_timesteps": 100, "batch_size": 64},
                "paths": {"model_dir": "models", "log_dir": "logs"},
            }
        elif isinstance(config, str):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = (
                config.copy()
            )  # Make a copy to avoid modifying the original

            # Add default paths if not present
            if "paths" not in self.config:
                self.config["paths"] = {
                    "model_dir": "models",
                    "log_dir": "logs",
                }

        # Ensure required paths exist
        os.makedirs(self.config["paths"]["model_dir"], exist_ok=True)
        os.makedirs(self.config["paths"]["log_dir"], exist_ok=True)

    def train(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> PPOAgent:
        """Train an agent

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Trained agent
        """
        # Create environments
        train_env = TradingEnvironment(
            data=train_data,
            initial_balance=self.config["env"]["initial_balance"],
            trading_fee=self.config["env"]["trading_fee"],
            window_size=self.config["env"]["window_size"],
        )

        val_env = TradingEnvironment(
            data=val_data,
            initial_balance=self.config["env"]["initial_balance"],
            trading_fee=self.config["env"]["trading_fee"],
            window_size=self.config["env"]["window_size"],
        )

        # Create agent
        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]

        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config["model"]["fcnet_hiddens"],
            learning_rate=self.config["model"]["learning_rate"],
        )

        # Training loop
        best_reward = float("-inf")
        episodes_without_improvement = 0
        early_stop = self.config["training"].get("early_stop", 20)

        for episode in range(self.config["training"]["total_timesteps"]):
            # Training episode
            train_metrics = self._run_episode(
                train_env, agent, train=True, episode_idx=episode
            )

            # Validation episode
            val_metrics = self._run_episode(
                val_env, agent, train=False, episode_idx=episode
            )

            # Early stopping
            if val_metrics["total_reward"] > best_reward:
                best_reward = val_metrics["total_reward"]
                episodes_without_improvement = 0
                self.save_model(agent, "best_model.pt")
            else:
                episodes_without_improvement += 1

            if episodes_without_improvement >= early_stop:
                print(f"Early stopping at episode {episode}")
                break

            # Log metrics
            self._log_metrics(episode, train_metrics, val_metrics)

        return agent

    def _run_episode(
        self, env, agent, train: bool = False, episode_idx: int = 0
    ) -> Dict[str, float]:
        """Run a single episode

        Args:
            env: Environment to run in
            agent: Agent to use
            train: Whether to train the agent
            episode_idx: Current episode index

        Returns:
            Episode metrics
        """
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            if train:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        # Calculate metrics
        metrics = {
            "total_reward": total_reward,
            "steps": steps,
            "final_balance": env.balance,
            "total_trades": env.total_trades,
            "profitable_trades": env.profitable_trades,
        }

        if env.total_trades > 0:
            metrics["win_rate"] = env.profitable_trades / env.total_trades

        return metrics

    def _log_metrics(
        self,
        episode: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log metrics for an episode

        Args:
            episode: Episode number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        print(f"Episode {episode}:")
        print(
            f"  Train: reward={train_metrics['total_reward']:.2f}, trades={train_metrics['total_trades']}"
        )
        print(
            f"  Val: reward={val_metrics['total_reward']:.2f}, trades={val_metrics['total_trades']}"
        )

    def save_model(self, agent: PPOAgent, filename: str):
        """Save model to disk

        Args:
            agent: Agent to save
            filename: Filename to save to
        """
        save_path = os.path.join(self.config["paths"]["model_dir"], filename)
        agent.save(save_path)

    def load_model(self, filename: str) -> PPOAgent:
        """Load model from disk

        Args:
            filename: Filename to load from

        Returns:
            Loaded agent
        """
        load_path = os.path.join(self.config["paths"]["model_dir"], filename)
        return PPOAgent.load(load_path)


def fetch_training_data(
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch historical price data for training

    Args:
        exchange: Exchange name (default: binance)
        symbol: Trading pair symbol (default: BTC/USDT)
        timeframe: Candle timeframe (default: 1h)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Maximum number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange)
        exchange_instance = exchange_class(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )

        # Set up date range
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # Fetch OHLCV data
        ohlcv = exchange_instance.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=int(start_date.timestamp() * 1000),
            limit=limit,
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=[
                "timestamp",
                "$open",
                "$high",
                "$low",
                "$close",
                "$volume",
            ],
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        logger.info(f"Fetched {len(df)} candles from {exchange} for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")

        # Return dummy data for testing
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
        return pd.DataFrame(
            {
                "$open": np.random.randn(1000) * 10 + 100,
                "$high": np.random.randn(1000) * 10 + 100,
                "$low": np.random.randn(1000) * 10 + 100,
                "$close": np.random.randn(1000) * 10 + 100,
                "$volume": np.abs(np.random.randn(1000) * 1000),
            },
            index=dates,
        )


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration."""
        self.config = config
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize MLflow manager
        self.mlflow_manager = MLflowManager(
            experiment_name=config.get(
                "experiment_name", "trading_bot_training"
            ),
            tracking_dir=config.get("mlflow_tracking_dir", "./mlflow_runs"),
        )

        # Initialize other attributes
        self.window_size = config["env"]["window_size"]
        self.trading_fee = config["env"]["trading_fee"]
        self.batch_size = config["model"]["batch_size"]
        self.learning_rate = config["model"]["learning_rate"]
        self.num_episodes = config["training"]["num_episodes"]

        # Environment and agent will be set later
        self.env = None
        self.agent = None

    def train(self) -> None:
        """Train the agent with MLflow tracking."""
        if self.env is None:
            raise ValueError(
                "Environment not set. Please set env before training."
            )
        if self.agent is None:
            raise ValueError(
                "Agent not set. Please set agent before training."
            )

        with self.mlflow_manager.start_run(
            run_name=f"training_{self.start_time}"
        ):
            # Log training parameters
            self.mlflow_manager.log_params(
                {
                    "window_size": self.window_size,
                    "trading_fee": self.trading_fee,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_episodes": self.num_episodes,
                }
            )

            self.results_df = pd.DataFrame()

            for episode in range(self.num_episodes):
                # Reset environment
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                portfolio_value = self.env.initial_balance

                while not done:
                    # Get action from agent
                    action = self.agent.get_action(state)

                    # Take step in environment
                    next_state, reward, done, truncated, info = self.env.step(
                        action
                    )

                    # Update agent
                    self.agent.train_step(
                        state, action, reward, next_state, done
                    )

                    # Update state and metrics
                    state = next_state
                    episode_reward += reward
                    portfolio_value = info.get(
                        "portfolio_value", portfolio_value
                    )

                # Calculate episode metrics
                sharpe_ratio = self._calculate_sharpe_ratio(self.env)

                # Log episode metrics
                self.mlflow_manager.log_metrics(
                    {
                        "episode_reward": episode_reward,
                        "portfolio_value": portfolio_value,
                        "sharpe_ratio": sharpe_ratio,
                    },
                    step=episode,
                )

                # Store results
                self.results_df = pd.concat(
                    [
                        self.results_df,
                        pd.DataFrame(
                            {
                                "episode": [episode],
                                "reward": [episode_reward],
                                "portfolio_value": [portfolio_value],
                                "sharpe_ratio": [sharpe_ratio],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            # Log final model and results
            if hasattr(self.agent, "policy_network"):
                self.mlflow_manager.log_model(
                    self.agent.policy_network, "policy_network"
                )

            if not self.results_df.empty:
                self.mlflow_manager.log_dataframe(
                    self.results_df, "results", "training_results.parquet"
                )

    def _calculate_sharpe_ratio(self, env) -> float:
        """Calculate Sharpe ratio from environment returns."""
        if not hasattr(env, "returns") or len(env.returns) < 2:
            return 0.0

        returns = np.array(env.returns)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
