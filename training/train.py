import os
import logging
import yaml
import pandas as pd
import numpy as np
import torch
from data.utils.data_loader import DataLoader
from agents.strategies.ppo_agent import PPOAgent
from training.utils.visualization import TradingVisualizer
from training.evaluation import TradingMetrics
from envs.trading_env import TradingEnvironment
from envs.wrap_env import make_env
from training.utils.mlflow_manager import MLflowManager
from typing import Union, Dict, Any, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/default_config.yaml") -> dict:
    """Load configuration from yaml file"""
    import os
    
    # Get the absolute path of the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Join with the config path
    config_path = os.path.join(project_root, config_path)
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to default config
        return {
            'env': {
                'initial_balance': 10000.0,
                'trading_fee': 0.001,
                'window_size': 20
            },
            'model': {
                'fcnet_hiddens': [64, 64],
                'lr': 0.001,
                'gamma': 0.99,
                'epsilon': 0.2
            },
            'training': {
                'total_timesteps': 10000,
                'early_stop': 20,
                'batch_size': 64
            },
            'paths': {
                'model_dir': os.path.join(project_root, "models"),
                'data_dir': os.path.join(project_root, "data"),
                'log_dir': os.path.join(project_root, "logs")
            }
        }

def create_env(data: Union[pd.DataFrame, Dict], config: dict = None) -> TradingEnvironment:
    """Create trading environment with given data and config"""
    if config is None:
        config = load_config()
    
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        if 'df' in data:  # Handle DataFrame in config
            data = data['df']
        elif 'data' in data:  # Handle nested data structure
            data = pd.DataFrame(data['data'])
        else:
            # Ensure all values are lists of the same length
            max_len = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in data.values())
            normalized_data = {}
            for k, v in data.items():
                if not isinstance(v, (list, np.ndarray)):
                    v = [v] * max_len
                normalized_data[k] = v
            data = pd.DataFrame(normalized_data)
    
    # Ensure column names have '$' prefix
    if isinstance(data, pd.DataFrame):
        rename_dict = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                rename_dict[col] = f'${col}'
        if rename_dict:
            data = data.rename(columns=rename_dict)
    
    # Create environment with config parameters
    env = TradingEnvironment(
        df=data,
        initial_balance=config['env']['initial_balance'],
        transaction_fee=config['env']['trading_fee'],
        window_size=config['env']['window_size'],
        max_position_size=config['env'].get('max_position_size', 1.0)
    )
    
    # Apply wrappers if specified in config
    if config['env'].get('normalize', True):
        env = make_env(
            env,
            normalize=True,
            stack_size=config['env'].get('stack_size', 4)
        )
    
    return env

def train_agent(train_data: pd.DataFrame, 
              val_data: pd.DataFrame, 
              config: Union[str, Dict[str, Any]] = "config/default_config.yaml"):
    """
    Simplified interface for training an agent.
    
    Args:
        train_data: Training data
        val_data: Validation data
        config: Either a path to config file (str) or config dictionary
        
    Returns:
        Trained agent
    """
    if isinstance(config, str):
        pipeline = TrainingPipeline(config)
    else:
        # If config is a dict, load default config first then update with provided config
        default_config = load_config()
        # Ensure model config has required fields
        if 'model' not in default_config:
            default_config['model'] = {}
        if 'fcnet_hiddens' not in default_config['model']:
            default_config['model']['fcnet_hiddens'] = [64, 64]
        
        # Update with provided config
        if isinstance(config, dict):
            for key, value in config.items():
                if key in default_config:
                    if isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                else:
                    default_config[key] = value
        
        pipeline = TrainingPipeline("config/default_config.yaml")
        pipeline.config = default_config
    
    return pipeline.train(train_data, val_data)

class TrainingPipeline:
    """Training pipeline for RL agents"""
    
    def __init__(self, config: Optional[str | Dict] = None):
        """Initialize training pipeline
        
        Args:
            config: Configuration file path or dictionary
        """
        if config is None:
            self.config = {
                'env': {
                    'initial_balance': 10000.0,
                    'trading_fee': 0.001,
                    'window_size': 20
                },
                'model': {
                    'fcnet_hiddens': [64, 64],
                    'learning_rate': 0.001
                },
                'training': {
                    'total_timesteps': 100,
                    'batch_size': 64
                },
                'paths': {
                    'model_dir': 'models',
                    'log_dir': 'logs'
                }
            }
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Ensure required paths exist
        os.makedirs(self.config['paths']['model_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> PPOAgent:
        """Train an agent
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Trained agent
        """
        # Create environments
        train_env = TradingEnvironment(
            train_data,
            initial_balance=self.config['env']['initial_balance'],
            trading_fee=self.config['env']['trading_fee'],
            window_size=self.config['env']['window_size']
        )
        
        val_env = TradingEnvironment(
            val_data,
            initial_balance=self.config['env']['initial_balance'],
            trading_fee=self.config['env']['trading_fee'],
            window_size=self.config['env']['window_size']
        )
        
        # Create agent
        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['model']['fcnet_hiddens'],
            learning_rate=self.config['model']['learning_rate']
        )
        
        # Training loop
        best_reward = float('-inf')
        episodes_without_improvement = 0
        early_stop = self.config['training'].get('early_stop', 20)
        
        for episode in range(self.config['training']['total_timesteps']):
            # Training episode
            train_metrics = self._run_episode(train_env, agent, train=True, episode_idx=episode)
            
            # Validation episode
            val_metrics = self._run_episode(val_env, agent, train=False, episode_idx=episode)
            
            # Early stopping
            if val_metrics['total_reward'] > best_reward:
                best_reward = val_metrics['total_reward']
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
        
    def _run_episode(self, env, agent, train: bool = False, episode_idx: int = 0) -> Dict[str, float]:
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
            'total_reward': total_reward,
            'steps': steps,
            'final_balance': env.balance,
            'total_trades': env.total_trades,
            'profitable_trades': env.profitable_trades
        }
        
        if env.total_trades > 0:
            metrics['win_rate'] = env.profitable_trades / env.total_trades
            
        return metrics
        
    def _log_metrics(self, episode: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for an episode
        
        Args:
            episode: Episode number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        print(f"Episode {episode}:")
        print(f"  Train: reward={train_metrics['total_reward']:.2f}, trades={train_metrics['total_trades']}")
        print(f"  Val: reward={val_metrics['total_reward']:.2f}, trades={val_metrics['total_trades']}")
        
    def save_model(self, agent: PPOAgent, filename: str):
        """Save model to disk
        
        Args:
            agent: Agent to save
            filename: Filename to save to
        """
        save_path = os.path.join(self.config['paths']['model_dir'], filename)
        agent.save(save_path)
        
    def load_model(self, filename: str) -> PPOAgent:
        """Load model from disk
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded agent
        """
        load_path = os.path.join(self.config['paths']['model_dir'], filename)
        return PPOAgent.load(load_path)