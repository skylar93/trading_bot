"""
Hyperparameter optimization using Ray Tune.
"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.air as air
from typing import Dict, Optional
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import mlflow
from .hyperopt.hyperopt_agent import HyperoptAgent
from .hyperopt.hyperopt_env import HyperoptEnvironment

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Ray Tune"""
    
    def __init__(self, config_path: str):
        """
        Initialize optimizer
        
        Args:
            config_path: Path to base configuration file
        """
        self.pipeline = TrainingPipeline(config_path)
        self.best_params = None
    
    def _train_with_params(self, config: Dict, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Training function for Ray Tune
        
        Args:
            config: Hyperparameter configuration to test
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with metrics
        """
        # Update pipeline config with trial parameters
        self.pipeline.config['model'].update({
            'fcnet_hiddens': [config['hidden_size']] * config['num_layers'],
            'lr': config['learning_rate']
        })
        
        # Add training parameters
        self.pipeline.config['training'].update({
            'total_timesteps': 100  # For quick testing
        })
        self.pipeline.config['training'].update({
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'gamma': config['gamma']
        })
        
        # Train model
        
        try:
            agent = self.pipeline.train(train_data, val_data)
            metrics = self.pipeline.evaluate(agent, val_data)
            
            # Report metrics to Ray Tune
            return {
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_return': metrics['total_return']
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # Report bad metrics to ensure trial is stopped
            tune.report(
                sharpe_ratio=-100,
                sortino_ratio=-100,
                max_drawdown=-1,
                total_return=-1
            )
    
    def optimize(self,
                train_data: pd.DataFrame,
                val_data: pd.DataFrame,
                num_trials: int = 50,
                cpus_per_trial: int = 2,
                gpus_per_trial: float = 0.5) -> Dict:
        """
        Run hyperparameter optimization
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_trials: Number of trials to run
            cpus_per_trial: CPUs allocated per trial
            gpus_per_trial: GPUs allocated per trial
            
        Returns:
            Dictionary with best hyperparameters
        """
        # Define search space
        search_space = {
            "hidden_size": tune.choice([64, 128, 256, 512]),
            "num_layers": tune.choice([1, 2, 3]),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "gamma": tune.uniform(0.9, 0.999)
        }
        
        # Define search algorithm
        search_algo = OptunaSearch(
            metric="sharpe_ratio",
            mode="max"
        )
        
        # Create tune config
        tune_config = tune.TuneConfig(
            metric="sharpe_ratio",
            mode="max",
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                max_t=100,
                grace_period=10,
                reduction_factor=2
            ),
            num_samples=num_trials
        )
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        # Store data in Ray object store
        train_data_ref = ray.put(train_data)
        val_data_ref = ray.put(val_data)
        
        def _trainable(config):
            """Trainable function for Ray Tune"""
            train_data = ray.get(train_data_ref)
            val_data = ray.get(val_data_ref)
            return self._train_with_params(config, train_data, val_data)
        
        # Run optimization
        analysis = tune.Tuner(
            _trainable,
            param_space=search_space,
            tune_config=tune_config,
            run_config=air.RunConfig(
                storage_path="file:///tmp/ray_results",
                name="ppo_hyperopt"
            )
        ).fit()
        
        # Get best hyperparameters
        best_result = analysis.get_best_result(metric="sharpe_ratio", mode="max")
        self.best_params = best_result.config
        
        # Log results to MLflow
        with mlflow.start_run(run_name="hyperparameter_optimization"):
            # Log best parameters
            mlflow.log_params(self.best_params)
            
            # Log best metrics
            # Log best metrics
            mlflow.log_metrics({
                "best_sharpe_ratio": best_result.metrics["sharpe_ratio"],
                "best_sortino_ratio": best_result.metrics["sortino_ratio"],
                "best_max_drawdown": best_result.metrics["max_drawdown"],
                "best_total_return": best_result.metrics["total_return"]
            })
            
            # Save optimization results
            results_df = analysis.get_dataframe()
            results_path = Path("results/hyperopt")
            results_path.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_path / "optimization_results.csv")
            mlflow.log_artifact(str(results_path / "optimization_results.csv"))
        
        return self.best_params