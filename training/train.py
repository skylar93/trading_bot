import os
import logging
import yaml
import pandas as pd
import torch
from data.utils.data_loader import DataLoader
from agents.ppo_agent import PPOAgent
from envs.trading_env import TradingEnvironment
from training.utils.mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_manager = MLflowManager()
    
    def prepare_data(self):
        """Prepare training data"""
        logger.info("Fetching and preparing data...")
        
        # Load data
        data_loader = DataLoader(
            exchange_id=self.config['data']['exchange'],
            symbol=self.config['data']['symbols'][0],
            timeframe=self.config['data']['timeframe']
        )
        
        data = data_loader.fetch_data(
            start_date=self.config['data']['start_date'],
            end_date=self.config['data'].get('end_date')
        )
        
        # Split data
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.15)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        return train_data, val_data, test_data

    def create_env(self, data):
        return TradingEnvironment(
            df=data,
            initial_balance=self.config['env']['initial_balance'],
            transaction_fee=self.config['env']['trading_fee'],
            window_size=self.config['env']['window_size']
        )
    
    def save_model(self, agent, filename):
        save_path = os.path.join(self.config['paths']['model_dir'], filename)
        agent.save(save_path)
        self.mlflow_manager.log_artifact(save_path)
    
    def _run_episode(self, env, agent, train=True):
        state, _ = env.reset()
        logger.info(f"Initial state shape: {state.shape if hasattr(state, 'shape') else 'no shape'}")
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if train:
                agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        return {
            'episode_reward': total_reward,
            'final_balance': env.balance,
            **info
        }
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, callback=None):
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
            
            # Validation episode
            val_metrics = self._run_episode(val_env, agent, train=False)
            logger.info(f"Val metrics: {val_metrics}")
            
            # Call callback if provided
            if callback:
                callback(episode, train_metrics, val_metrics)
            
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