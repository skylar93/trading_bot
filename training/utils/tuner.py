"""
Hyperparameter tuning system using Ray Tune.
Optimizes model parameters for better trading performance.
"""

from typing import Dict, Any, Optional
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback
import logging
import mlflow
from datetime import datetime
from .mlflow_manager import MLflowManager
from ..train import TrainingPipeline

logger = logging.getLogger(__name__)

# Import MinimalTuner for backward compatibility
from training.hyperopt import MinimalTuner as HyperParameterTuner

class HyperParameterTuner:
    """Manages hyperparameter optimization using Ray Tune"""
    
    def __init__(self, 
                 data_train,
                 data_val,
                 mlflow_manager: Optional[MLflowManager] = None):
        """
        Initialize tuner
        
        Args:
            data_train: Training data
            data_val: Validation data
            mlflow_manager: Optional MLflow manager for tracking
        """
        self.data_train = data_train
        self.data_val = data_val
        self.mlflow_manager = mlflow_manager or MLflowManager()
        
    def _training_function(self, config: Dict[str, Any]):
        """Training function for Ray Tune
        
        Args:
            config: Hyperparameter configuration
        """
        # Create training pipeline with config
        pipeline = TrainingPipeline()
        
        # Update pipeline config with tune config
        pipeline.config.update({
            'env': {
                'window_size': config['window_size']
            },
            'model': {
                'hidden_size': config['hidden_size'],
                'num_layers': config['num_layers'],
                'learning_rate': config['learning_rate'],
                'gamma': config['gamma'],  # discount factor
                'lambda': config['lambda'],  # GAE parameter
                'clip_param': config['clip_param']  # PPO clip parameter
            },
            'training': {
                'batch_size': config['batch_size'],
                'num_episodes': config['num_episodes']
            }
        })
        
        # Train model
        agent = pipeline.train(
            self.data_train,
            self.data_val,
            verbose=False  # Disable verbose output during tuning
        )
        
        # Evaluate performance
        metrics = pipeline.evaluate(agent, self.data_val)
        
        # Report metrics to Ray Tune
        tune.report(
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            total_return=metrics['total_return'],
            win_rate=metrics['win_rate']
        )
    
    def run_optimization(self, 
                        num_samples: int = 10,
                        num_epochs: int = 100,
                        gpus_per_trial: float = 0.5) -> Dict:
        """Run hyperparameter optimization
        
        Args:
            num_samples: Number of configurations to try
            num_epochs: Number of epochs per trial
            gpus_per_trial: GPUs allocated per trial
            
        Returns:
            Best configuration found
        """
        # Define hyperparameter search space
        config = {
            # Environment parameters
            'window_size': tune.choice([10, 20, 30, 40, 50]),
            
            # Model architecture
            'hidden_size': tune.choice([64, 128, 256, 512]),
            'num_layers': tune.choice([1, 2, 3]),
            
            # PPO parameters
            'learning_rate': tune.loguniform(1e-5, 1e-3),
            'gamma': tune.uniform(0.9, 0.999),
            'lambda': tune.uniform(0.9, 1.0),
            'clip_param': tune.uniform(0.1, 0.3),
            
            # Training parameters
            'batch_size': tune.choice([32, 64, 128, 256]),
            'num_episodes': tune.choice([50, 100, 200])
        }
        
        # Define scheduler
        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=10,
            reduction_factor=2
        )
        
        # MLflow tracking
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=self.mlflow_manager.tracking_uri,
            experiment_name=f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Run optimization
        analysis = tune.run(
            self._training_function,
            config=config,
            metric="sharpe_ratio",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            callbacks=[mlflow_callback],
            resources_per_trial={
                "cpu": 2,
                "gpu": gpus_per_trial
            },
            local_dir="./ray_results",
            name="ppo_trading_tuning"
        )
        
        # Get best configuration
        best_config = analysis.get_best_config(metric="sharpe_ratio", mode="max")
        
        # Log best results
        best_trial = analysis.get_best_trial(metric="sharpe_ratio", mode="max")
        logger.info(f"Best trial config: {best_config}")
        logger.info(f"Best trial final metrics: {best_trial.last_result}")
        
        return best_config
    
    def apply_best_config(self, best_config: Dict[str, Any]):
        """Apply best configuration to training pipeline
        
        Args:
            best_config: Best hyperparameter configuration
        """
        pipeline = TrainingPipeline()
        pipeline.config.update({
            'env': {
                'window_size': best_config['window_size']
            },
            'model': {
                'hidden_size': best_config['hidden_size'],
                'num_layers': best_config['num_layers'],
                'learning_rate': best_config['learning_rate'],
                'gamma': best_config['gamma'],
                'lambda': best_config['lambda'],
                'clip_param': best_config['clip_param']
            },
            'training': {
                'batch_size': best_config['batch_size'],
                'num_episodes': best_config['num_episodes']
            }
        })
        
        return pipeline

# Re-export for backward compatibility
__all__ = ['HyperParameterTuner']