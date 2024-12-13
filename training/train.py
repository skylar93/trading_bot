import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import torch
from datetime import datetime

from data.utils.data_loader import DataLoader
from data.utils.feature_generator import FeatureGenerator
from envs.trading_env import TradingEnvironment
from agents.ppo_agent import PPOAgent
from training.evaluation import TradingMetrics
from training.utils.mlflow_utils import MLflowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Training pipeline for the trading bot"""
    
    def __init__(self, config_path: str):
        """Initialize training pipeline"""
        self.config = self._load_config(config_path)
        self.experiment_dir = self._setup_experiment_dir()
        
        # Initialize MLflow
        self.mlflow_manager = MLflowManager(
            experiment_name=self.config.get('experiment_name', 'trading_bot')
        )
        
        # Initialize components
        self.data_loader = DataLoader(
            exchange_id=self.config['data']['exchange'],
            symbol=self.config['data']['symbols'][0],
            timeframe=self.config['data']['timeframe']
        )
        
        self.feature_generator = FeatureGenerator()
        self.metrics = TradingMetrics()
        self.portfolio_history = []
        self.action_history = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_experiment_dir(self) -> Path:
        """Setup experiment directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path("experiments") / timestamp
        
        # Create directories
        for subdir in ['models', 'logs', 'results']:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return exp_dir
        
    def prepare_data(self):
        """Prepare training data"""
        logger.info("Fetching and preparing data...")
        
        # Fetch data
        df = self.data_loader.fetch_data(
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        # Generate features
        df = self.feature_generator.generate_features(df)
        
        # Split data
        train_end = int(len(df) * 0.7)
        val_end = int(len(df) * 0.85)
        
        train_data = df[:train_end]
        val_data = df[train_end:val_end]
        test_data = df[val_end:]
        
        # Log data info
        self.mlflow_manager.log_params({
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'features': list(df.columns)
        })
        
        return train_data, val_data, test_data
    
    def create_env(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment"""
        return TradingEnvironment(
            df=data,
            initial_balance=self.config['env']['initial_balance'],
            transaction_fee=self.config['env']['trading_fee'],
            window_size=self.config['env']['window_size']
        )
        
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train the agent"""
        logger.info("Starting training...")
        
        # Create environments
        train_env = self.create_env(train_data)
        val_env = self.create_env(val_data)
        
        # Create agent
        agent = PPOAgent(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            hidden_dim=self.config['model']['fcnet_hiddens'][0],
            lr=self.config['model']['lr']
        )
        
        # Start MLflow run
        self.mlflow_manager.start_run()
        
        # Log configurations
        self.mlflow_manager.log_params(self.config['model'])
        self.mlflow_manager.log_params(self.config['env'])
        
        best_val_return = float('-inf')
        no_improve_count = 0
        
        for episode in range(self.config['training']['total_timesteps']):
            logger.info(f"Episode {episode+1}/{self.config['training']['total_timesteps']}")
            
            # Training episode
            train_metrics = self._run_episode(train_env, agent, train=True)
            logger.info(f"Train metrics: {train_metrics}")
            
            # Log training metrics
            self.mlflow_manager.log_metrics({
                'train_portfolio_value': train_metrics['final_balance'],
                'train_reward': train_metrics['episode_reward']
            }, step=episode)
            
            # Validation episode
            val_metrics = self._run_episode(val_env, agent, train=False)
            logger.info(f"Val metrics: {val_metrics}")
            
            # Log validation metrics
            self.mlflow_manager.log_metrics({
                'val_portfolio_value': val_metrics['final_balance'],
                'val_reward': val_metrics['episode_reward']
            }, step=episode)
            
            # Save best model
            if val_metrics['episode_reward'] > best_val_return:
                best_val_return = val_metrics['episode_reward']
                self.save_model(agent, "best_model.pt")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= self.config['training'].get('early_stop', 20):
                logger.info("Early stopping triggered")
                break
        
        # End MLflow run
        self.mlflow_manager.end_run()
        return agent
    
    def _run_episode(self, env: TradingEnvironment, agent: PPOAgent, train: bool = True) -> Dict[str, float]:
        """Run single episode"""
        state, _ = env.reset()
        done = False
        episode_reward = 0
        portfolio_values = []
        actions = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Record metrics
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            
            if train:
                agent.train(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        # Log episode metrics to MLflow
        if train:
            self.mlflow_manager.log_metrics({
                'portfolio_value': portfolio_values[-1],
                'action': actions[-1],
                'portfolio_history': portfolio_values,
                'action_history': actions
            })
        
        return {
            'episode_reward': episode_reward,
            'final_balance': portfolio_values[-1]
        }
        
    def save_model(self, agent: PPOAgent, filename: str):
        """Save model checkpoint"""
        save_path = self.experiment_dir / 'models' / filename
        agent.save(str(save_path))
        logger.info(f"Saved model to {save_path}")
        
    def evaluate(self, agent: PPOAgent, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate agent on test data"""
        logger.info("Starting evaluation...")
        
        test_env = self.create_env(test_data)
        metrics = self._run_episode(test_env, agent, train=False)
        
        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.2f}")
        
        return metrics