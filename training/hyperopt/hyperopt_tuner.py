"""Hyperparameter optimization using Ray Tune"""

import os
import ray
from ray import tune
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from training.train import train_agent

class MinimalTuner:
    """Minimal hyperparameter tuner using Ray Tune"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize tuner
        
        Args:
            df: DataFrame containing market data
        """
        self.df = df
        
        # Split data into train/val
        train_size = int(len(df) * 0.8)
        self.train_data = df[:train_size]
        self.val_data = df[train_size:]
        
        # Default search space
        self.search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "hidden_size": tune.choice([32, 64, 128, 256]),
            "gamma": tune.uniform(0.9, 0.999),
            "epsilon": tune.loguniform(1e-5, 1e-3),
            "initial_balance": tune.choice([10000, 100000]),
            "transaction_fee": tune.loguniform(1e-3, 1e-2),
            "total_timesteps": tune.choice([10000, 50000])
        }
        
    def objective(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Objective function for Ray Tune
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing metrics
        """
        # Convert flat config to nested config
        config_dict = {
            'env': {
                'initial_balance': config['initial_balance'],
                'trading_fee': config['transaction_fee'],
                'window_size': 20
            },
            'model': {
                'fcnet_hiddens': [config['hidden_size'], config['hidden_size']],
                'learning_rate': config['learning_rate'],
                'gamma': config['gamma'],
                'epsilon': config['epsilon']
            },
            'training': {
                'total_timesteps': config['total_timesteps'],
                'batch_size': 64
            },
            'paths': {
                'model_dir': 'models',
                'log_dir': 'logs'
            }
        }
        
        # Train agent
        agent = train_agent(self.train_data, self.val_data, config_dict)
        
        # Evaluate on validation set
        metrics = self.evaluate_config(config_dict)
        
        # Report metrics
        tune.report(
            score=metrics['sharpe_ratio'],
            total_return=metrics['total_return'],
            max_drawdown=metrics['max_drawdown']
        )
        
        return metrics
        
    def evaluate_config(self, config: Dict[str, Any], episodes: int = 1) -> Dict[str, float]:
        """Evaluate a configuration
        
        Args:
            config: Configuration dictionary
            episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing metrics
        """
        # Train agent
        agent = train_agent(self.train_data, self.val_data, config)
        
        # Calculate metrics
        returns = []
        for _ in range(episodes):
            metrics = agent.evaluate(self.val_data)
            returns.append(metrics['total_return'])
            
        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        return {
            'total_return': mean_return,
            'return_std': std_return,
            'sharpe_ratio': sharpe_ratio
        }
        
    def optimize(self, num_samples: int = 10, max_concurrent: int = 4) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Run hyperparameter optimization
        
        Args:
            num_samples: Number of configurations to try
            max_concurrent: Maximum number of concurrent trials
            
        Returns:
            Tuple containing best configuration and metrics
        """
        # Run optimization
        analysis = tune.run(
            self.objective,
            config=self.search_space,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            resources_per_trial={'cpu': 2},
            metric='score',
            mode='max'
        )
        
        # Get best configuration
        best_config = analysis.best_config
        best_metrics = analysis.best_result
        
        return best_config, best_metrics
        
    def run_optimization(self, num_samples: int = 10) -> Dict[str, Any]:
        """Run optimization and return best configuration
        
        Args:
            num_samples: Number of configurations to try
            
        Returns:
            Best configuration dictionary
        """
        return self.optimize(num_samples=num_samples)[0]  # Return only config for backward compatibility