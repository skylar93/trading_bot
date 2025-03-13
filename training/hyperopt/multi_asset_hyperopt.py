#!/usr/bin/env python
"""
Hyperparameter Optimization for Multi-Asset Trading Environment

This module provides specialized hyperparameter optimization for multi-asset 
trading environments with integrated risk management. It extends the base
hyperopt_ray module with multi-asset specific search spaces and evaluation.

Features:
- Multi-asset specific hyperparameter search spaces
- Joint optimization of trading parameters and risk management settings
- Asset combination exploration (single vs multi-asset)
- Comparative analysis of hyperparameter importance across different assets
- Visualization of optimization results

Implementation Notes:
- Uses Ray Tune for distributed optimization
- Integrates with the unified configuration system
- Provides specialized search spaces for multi-asset trading
- Supports multiple search algorithms (random, Bayesian, BOHB)
- Handles MLflow tracking and result visualization

Recent Changes:
- Added multi-asset specific search space generator
- Implemented specialized analysis for asset correlation hyperparameters
- Enhanced visualization for multi-asset performance analysis
- Added asset combination exploration functionality
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime
import argparse
import copy

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Ray imports
import ray
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.air import session
from ray.air import CheckpointConfig, RunConfig
from ray.tune.integration.mlflow import MLflowLoggerCallback

from training.train_pipeline import train_pipeline
from training.utils.config_manager import ConfigManager
from training.utils.unified_mlflow_manager import MLflowManager
from training.hyperopt.hyperopt_ray import run_hyperparameter_optimization, create_search_algorithm, create_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"logs/multi_asset_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('multi_asset_hyperopt')

# Default paths
DEFAULT_CONFIG_PATH = "configs/risk_management.yaml"
DEFAULT_STORAGE_PATH = "./ray_results/multi_asset_opt"
DEFAULT_EXPERIMENT_NAME = "multi_asset_optimization"

def create_multi_asset_search_space(base_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create hyperparameter search space specifically for multi-asset environments.
    
    Args:
        base_config: Base configuration to build upon (optional)
        
    Returns:
        Dictionary with search space definition for Ray Tune
    """
    from ray import tune
    
    # Load base config if not provided
    if base_config is None:
        config_manager = ConfigManager()
        base_config = config_manager.load_config(DEFAULT_CONFIG_PATH)
    
    search_space = {
        # Environment configuration
        "env_config": {
            "action_type": tune.choice(["portfolio_weights", "discrete_signal"]),
            "window_size": tune.choice([10, 20, 30, 40]),
            "reward_scaling": tune.uniform(0.01, 0.1),
            "capital_allocation_mode": tune.choice(["equal", "market_cap_weighted", "inverse_volatility", "fixed"]),
            "initial_balance": 10000.0,  # Fixed value
            "trading_fee": tune.uniform(0.0005, 0.002),
            "slippage_rate": tune.uniform(0.0001, 0.001),
            "state_type": tune.choice(["raw", "normalized"]),
        },
        
        # Asset combinations to explore
        "asset_combinations": tune.choice([
            ["BTC"],  # Single crypto
            ["BTC", "ETH"],  # Crypto pair
            ["SPY"],  # Single stock
            ["BTC", "SPY"],  # Cross-asset class
            ["BTC", "ETH", "SPY"],  # Multi-asset
            ["BTC", "ETH", "SPY", "GOLD"],  # Diversified portfolio
        ]),
        
        # Risk management configuration
        "risk_config": {
            "stop_loss": {
                "use_stop_loss": tune.choice([True, False]),
                "stop_loss_threshold": tune.uniform(0.05, 0.15)
            },
            "trailing_stop": {
                "use_trailing_stop": tune.choice([True, False]),
                "trailing_stop_buffer": tune.uniform(0.03, 0.1)
            },
            "var": {
                "use_var": tune.choice([True, False]),
                "var_confidence_level": tune.choice([0.90, 0.95, 0.99]),
                "action_on_var_exceed": tune.choice(["reduce_position", "close_position"])
            },
            "correlation": {
                "use_correlation": tune.choice([True, False]),
                "correlation_threshold": tune.uniform(0.5, 0.8),
                "correlation_risk_reduction": tune.uniform(0.3, 0.7)
            },
            "portfolio_stop_loss": {
                "use_portfolio_stop_loss": tune.choice([True, False]),
                "portfolio_stop_loss_threshold": tune.uniform(0.1, 0.2)
            },
            "portfolio_trailing_stop": {
                "use_portfolio_trailing_stop": tune.choice([True, False]),
                "portfolio_trailing_stop_buffer": tune.uniform(0.05, 0.12)
            },
            "portfolio_var": {
                "use_portfolio_var": tune.choice([True, False]),
                "portfolio_var_threshold": tune.loguniform(0.01, 0.04),
                "use_parametric_var": tune.choice([True, False])
            }
        },
        
        # Agent/model configuration
        "model_config": {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "gamma": tune.uniform(0.9, 0.999),
            "gae_lambda": tune.uniform(0.9, 1.0),
            "clip_param": tune.uniform(0.1, 0.3),
            "vf_clip_param": tune.uniform(0.5, 1.0),
            "entropy_coeff": tune.loguniform(1e-5, 1e-2),
            "vf_loss_coeff": tune.uniform(0.5, 1.0),
            "hidden_dim": tune.choice([64, 128, 256, 512]),
            "hidden_layers": tune.choice([1, 2, 3]),
            "activation": tune.choice(["tanh", "relu", "swish"]),
            "use_lstm": tune.choice([True, False]),
            "lstm_cell_size": tune.choice([64, 128, 256]),
        },
        
        # Training configuration
        "training_config": {
            "batch_size": tune.choice([128, 256, 512, 1024]),
            "mini_batch_size": tune.choice([32, 64, 128]),
            "train_epochs": tune.choice([5, 10, 20]),
            "update_frequency": tune.choice([128, 256, 512]),
            "max_steps": 500000,  # Fixed value
            "eval_frequency": 10000,  # Fixed value
        },
        
        # Multi-agent configuration
        "multi_agent_config": {
            "use_multi_agent": tune.choice([True, False]),
            "agent_types": tune.choice([
                ["ppo"],  # Single agent type
                ["ppo", "momentum"],  # Two agent types
                ["ppo", "momentum", "mean_reversion"],  # Three agent types
            ]),
            "capital_manager_mode": tune.choice(["shared", "isolated"]),
            "shared_experience": tune.choice([True, False]),
            "agent_weight_allocation": tune.choice(["equal", "optimized", "dynamic"]),
        }
    }
    
    return search_space

def train_multi_asset_func(config, checkpoint_dir=None):
    """
    Training function for Ray Tune to evaluate hyperparameters.
    
    Args:
        config: Configuration from the search space
        checkpoint_dir: Directory where checkpoints are stored
        
    Returns:
        None - reports results via session.report()
    """
    import logging
    logger = logging.getLogger("train_multi_asset_func")
    
    # Extract the base configuration
    if "_full_config" in config:
        full_config = copy.deepcopy(config["_full_config"])
    else:
        full_config = copy.deepcopy(config)
    
    # Handle asset combinations (special case)
    if "asset_combinations" in config:
        assets = config["asset_combinations"]
        
        # Set assets in environment config
        if "env_config" not in full_config:
            full_config["env_config"] = {}
        full_config["env_config"]["assets"] = assets
        
        # Also update data paths if needed (assume standard data paths)
        if "paths" not in full_config:
            full_config["paths"] = {}
        if "data" not in full_config["paths"]:
            full_config["paths"]["data"] = {}
        
        # Configure data for each asset
        for asset in assets:
            full_config["paths"]["data"][asset] = f"data/{asset.lower()}_daily.csv"
    
    # Configure risk management
    if "risk_config" in config:
        risk_config = config["risk_config"]
        
        # Set risk configuration
        if "risk_management" not in full_config:
            full_config["risk_management"] = {}
        
        # Transfer risk config sections
        for section, params in risk_config.items():
            full_config["risk_management"][section] = params
    
    # Configure model parameters
    if "model_config" in config:
        model_config = config["model_config"]
        
        # Set model configuration
        if "model" not in full_config:
            full_config["model"] = {}
        
        # Transfer model config parameters
        for param, value in model_config.items():
            full_config["model"][param] = value
    
    # Configure training parameters
    if "training_config" in config:
        training_config = config["training_config"]
        
        # Set training configuration
        if "training" not in full_config:
            full_config["training"] = {}
        
        # Transfer training config parameters
        for param, value in training_config.items():
            full_config["training"][param] = value
    
    # Configure multi-agent settings
    if "multi_agent_config" in config:
        multi_agent_config = config["multi_agent_config"]
        
        # Set multi-agent configuration
        if "multi_agent" not in full_config:
            full_config["multi_agent"] = {}
        
        # Transfer multi-agent config parameters
        for param, value in multi_agent_config.items():
            full_config["multi_agent"][param] = value
    
    # Extract checkpoint path if provided
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.join("./checkpoints", f"trial_{session.get_trial_id()}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set checkpoint directory in config
    if "paths" not in full_config:
        full_config["paths"] = {}
    full_config["paths"]["checkpoints"] = checkpoint_dir
    
    try:
        # Run training with this configuration
        result = train_pipeline(
            config=full_config,
            checkpoint_path=checkpoint_path,
            trial_id=session.get_trial_id()
        )
        
        # Get evaluation metrics
        metrics = result.get("eval_metrics", {})
        
        # Create report for Ray Tune
        report_dict = {
            "episode_reward_mean": metrics.get("episode_reward_mean", -100),
            "sharpe_ratio": metrics.get("sharpe_ratio", -1),
            "sortino_ratio": metrics.get("sortino_ratio", -1),
            "max_drawdown": metrics.get("max_drawdown", 1),
            "final_portfolio_value": metrics.get("final_portfolio_value", 0),
            "win_rate": metrics.get("win_rate", 0),
            "training_iteration": result.get("training_iteration", 0),
            "total_steps": result.get("total_steps", 0),
            "time_total_s": result.get("time_total_s", 0),
            "assets": full_config["env_config"].get("assets", []),
            "risk_config": full_config.get("risk_management", {})
        }
        
        # Report metrics to Ray Tune
        session.report(report_dict)
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        # Report a very bad result on error
        session.report({
            "episode_reward_mean": -1000,
            "sharpe_ratio": -10,
            "training_iteration": 0,
            "error": str(e)
        })

def run_multi_asset_optimization(
    config_path: str = DEFAULT_CONFIG_PATH,
    storage_path: str = DEFAULT_STORAGE_PATH,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    num_samples: int = 20,
    max_concurrent_trials: int = 4,
    search_alg: str = "bayesopt",
    scheduler_type: str = "asha",
    metric: str = "sharpe_ratio",
    mode: str = "max",
    time_budget_s: Optional[int] = None,
    search_space: Optional[Dict[str, Any]] = None,
    gpus_per_trial: float = 0.25,
    cpus_per_trial: int = 2,
    experiment_id: Optional[str] = None
) -> ray.tune.ExperimentAnalysis:
    """
    Run hyperparameter optimization for multi-asset trading environment.
    
    Args:
        config_path: Path to base configuration file
        storage_path: Directory to store Ray Tune results
        experiment_name: Name of the experiment
        num_samples: Number of hyperparameter samples to try
        max_concurrent_trials: Maximum number of concurrent trials
        search_alg: Search algorithm to use
        scheduler_type: Scheduler type to use
        metric: Metric to optimize
        mode: Optimization mode ('min' or 'max')
        time_budget_s: Time budget in seconds (optional)
        search_space: Custom search space (optional)
        gpus_per_trial: GPUs per trial
        cpus_per_trial: CPUs per trial
        experiment_id: MLflow experiment ID
        
    Returns:
        Ray Tune ExperimentAnalysis with results
    """
    # Initialize ray if not initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Load base configuration
    config_manager = ConfigManager()
    base_config = config_manager.load_config(config_path)
    
    # Create search space if not provided
    if search_space is None:
        search_space = create_multi_asset_search_space(base_config)
    
    # Store the full base config in the search space
    search_space["_full_config"] = base_config
    
    # Create search algorithm
    search_algorithm = create_search_algorithm(
        algorithm=search_alg,
        search_space=search_space,
        metric=metric,
        mode=mode
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        scheduler_type=scheduler_type,
        metric=metric,
        mode=mode
    )
    
    # Set up MLflow tracking if experiment_id is provided
    callbacks = []
    if experiment_id:
        mlflow_callback = MLflowLoggerCallback(
            experiment_name=experiment_id,
            tracking_uri=MLflowManager.get_tracking_uri(),
            save_artifact=True
        )
        callbacks.append(mlflow_callback)
    
    # Configure resources per trial
    resources_per_trial = {
        "cpu": cpus_per_trial,
        "gpu": gpus_per_trial
    }
    
    # Configure checkpoint settings
    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute=metric,
        checkpoint_score_order=mode
    )
    
    # Configure run settings
    run_config = RunConfig(
        name=experiment_name,
        local_dir=storage_path,
        checkpoint_config=checkpoint_config,
        callbacks=callbacks,
        stop={
            "training_iteration": base_config.get("training", {}).get("max_steps", 500000) // 
                                base_config.get("training", {}).get("eval_frequency", 10000)
        }
    )
    
    # Run the hyperparameter optimization
    logger.info(f"Starting hyperparameter optimization with {num_samples} samples")
    tuner = tune.Tuner(
        tune.with_resources(
            train_multi_asset_func,
            resources=resources_per_trial
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_algorithm,
            max_concurrent_trials=max_concurrent_trials,
            time_budget_s=time_budget_s
        ),
        run_config=run_config,
    )
    
    # Execute the optimization
    results = tuner.fit()
    
    # Log best configuration
    best_trial = results.get_best_trial(metric=metric, mode=mode)
    best_config = best_trial.config
    best_result = best_trial.last_result
    
    logger.info(f"Best hyperparameters found: {best_config}")
    logger.info(f"Best {metric}: {best_result.get(metric)}")
    logger.info(f"Assets in best configuration: {best_result.get('assets', [])}")
    
    # Generate analysis and visualization
    generate_optimization_analysis(results, metric, mode, storage_path)
    
    return results

def generate_optimization_analysis(
    results: ray.tune.ExperimentAnalysis,
    metric: str,
    mode: str,
    output_dir: str
):
    """
    Generate analysis and visualizations from optimization results.
    
    Args:
        results: Ray Tune ExperimentAnalysis object
        metric: Metric used for optimization
        mode: Optimization mode ('min' or 'max')
        output_dir: Directory to save analysis results
    """
    # Ensure output directory exists
    output_dir = Path(output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataframe with results
    df = results.get_dataframe()
    
    # Check if we have enough data for analysis
    if len(df) <= 1:
        logger.warning("Not enough trials for meaningful analysis")
        return
    
    # Add column for asset count
    df["asset_count"] = df["assets"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # 1. Performance by asset combination
    plt.figure(figsize=(12, 8))
    asset_performance = df.groupby("assets")[metric].agg(["mean", "max", "min", "std"])
    asset_performance = asset_performance.sort_values("mean", ascending=(mode == "min"))
    
    ax = asset_performance["mean"].plot(kind="barh", xerr=asset_performance["std"], color="skyblue")
    plt.title(f"{metric.replace('_', ' ').title()} by Asset Combination")
    plt.xlabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_dir / "asset_performance.png")
    plt.close()
    
    # 2. Performance by number of assets
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="asset_count", y=metric, data=df)
    plt.title(f"{metric.replace('_', ' ').title()} by Number of Assets")
    plt.xlabel("Number of Assets")
    plt.ylabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_dir / "asset_count_performance.png")
    plt.close()
    
    # 3. Risk management impact
    risk_features = [
        "risk_config.stop_loss.use_stop_loss",
        "risk_config.trailing_stop.use_trailing_stop",
        "risk_config.var.use_var",
        "risk_config.correlation.use_correlation",
        "risk_config.portfolio_stop_loss.use_portfolio_stop_loss",
        "risk_config.portfolio_trailing_stop.use_portfolio_trailing_stop",
        "risk_config.portfolio_var.use_portfolio_var"
    ]
    
    risk_impact = pd.DataFrame()
    
    for feature in risk_features:
        if feature in df.columns:
            # Calculate mean performance with and without the feature
            with_feature = df[df[feature] == True][metric].mean(numeric_only=True)
            without_feature = df[df[feature] == False][metric].mean(numeric_only=True)
            
            # Calculate effect size (percentage difference)
            if without_feature != 0:
                effect_size = (with_feature - without_feature) / abs(without_feature)
            else:
                effect_size = 0
                
            risk_impact = risk_impact.append({
                "Risk Feature": feature.split(".")[-2],
                "With Feature": with_feature,
                "Without Feature": without_feature,
                "Effect Size": effect_size
            }, ignore_index=True)
    
    if len(risk_impact) > 0:
        plt.figure(figsize=(12, 6))
        risk_impact = risk_impact.sort_values("Effect Size", ascending=False)
        bars = plt.bar(risk_impact["Risk Feature"], risk_impact["Effect Size"], color="skyblue")
        
        # Color bars based on effect direction
        for i, bar in enumerate(bars):
            if risk_impact.iloc[i]["Effect Size"] < 0:
                bar.set_color("salmon")
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f"Impact of Risk Features on {metric.replace('_', ' ').title()}")
        plt.xlabel("Risk Feature")
        plt.ylabel(f"Effect Size (relative change in {metric})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "risk_feature_impact.png")
        plt.close()
    
    # 4. Parameter importance analysis
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        # Prepare data for parameter importance analysis
        # First, get all configuration parameters that could be important
        param_cols = [col for col in df.columns if "config" in col and col != "risk_config"]
        
        # Add risk configuration parameters
        risk_param_cols = [col for col in df.columns if col.startswith("risk_config") and ".use_" not in col]
        param_cols.extend(risk_param_cols)
        
        # Filter to only include numeric or boolean parameters
        valid_cols = []
        for col in param_cols:
            if col in df.columns:
                dtype = df[col].dtype
                if np.issubdtype(dtype, np.number) or dtype == bool:
                    valid_cols.append(col)
        
        # Ensure we have some valid columns
        if len(valid_cols) > 0:
            # Copy and prepare the data
            X = df[valid_cols].copy()
            y = df[metric].copy()
            
            # Handle categorical features
            for col in X.columns:
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)
                elif X[col].dtype == object:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
            
            # Fit a random forest model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            # Extract feature importances
            importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            })
            importances = importances.sort_values('Importance', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
            plt.title('Parameter Importance Analysis')
            plt.tight_layout()
            plt.savefig(output_dir / "parameter_importance.png")
            plt.close()
    except Exception as e:
        logger.warning(f"Parameter importance analysis failed: {e}")
    
    # 5. Generate a summary report
    with open(output_dir / "optimization_summary.txt", "w") as f:
        f.write(f"Multi-Asset Optimization Summary\n")
        f.write(f"===============================\n\n")
        f.write(f"Optimization metric: {metric} ({mode})\n")
        f.write(f"Total trials: {len(df)}\n\n")
        
        # Best trial
        best_trial = results.get_best_trial(metric=metric, mode=mode)
        best_config = best_trial.config
        best_result = best_trial.last_result
        
        f.write(f"Best Trial Results\n")
        f.write(f"-----------------\n")
        f.write(f"Trial ID: {best_trial.trial_id}\n")
        f.write(f"{metric}: {best_result.get(metric)}\n")
        f.write(f"Assets: {best_result.get('assets', [])}\n")
        if 'sharpe_ratio' in best_result:
            f.write(f"Sharpe Ratio: {best_result.get('sharpe_ratio')}\n")
        if 'sortino_ratio' in best_result:
            f.write(f"Sortino Ratio: {best_result.get('sortino_ratio')}\n")
        if 'max_drawdown' in best_result:
            f.write(f"Max Drawdown: {best_result.get('max_drawdown')}\n")
        if 'final_portfolio_value' in best_result:
            f.write(f"Final Portfolio Value: {best_result.get('final_portfolio_value')}\n")
        f.write("\n")
        
        # Asset analysis
        f.write(f"Asset Combination Performance\n")
        f.write(f"---------------------------\n")
        for idx, row in asset_performance.iterrows():
            f.write(f"{idx}: mean={row['mean']:.4f}, max={row['max']:.4f}, min={row['min']:.4f}\n")
        f.write("\n")
        
        # Risk management analysis
        if len(risk_impact) > 0:
            f.write(f"Risk Management Impact\n")
            f.write(f"---------------------\n")
            for idx, row in risk_impact.iterrows():
                f.write(f"{row['Risk Feature']}: effect={row['Effect Size']:.4f} ")
                f.write(f"(with={row['With Feature']:.4f}, without={row['Without Feature']:.4f})\n")
            f.write("\n")
        
        # Best hyperparameters
        f.write(f"Best Configuration\n")
        f.write(f"-----------------\n")
        for section, params in best_config.items():
            if section != "_full_config" and isinstance(params, dict):
                f.write(f"{section}:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
            elif section != "_full_config":
                f.write(f"{section}: {params}\n")
        
    logger.info(f"Analysis results saved to {output_dir}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for multi-asset trading")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to base configuration file")
    parser.add_argument("--storage-path", type=str, default=DEFAULT_STORAGE_PATH, help="Directory to store Ray Tune results")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="Name of the experiment")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of hyperparameter samples to try")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Maximum number of concurrent trials")
    parser.add_argument("--search-alg", type=str, default="bayesopt", help="Search algorithm to use")
    parser.add_argument("--scheduler", type=str, default="asha", help="Scheduler type to use")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric to optimize")
    parser.add_argument("--mode", type=str, default="max", help="Optimization mode ('min' or 'max')")
    parser.add_argument("--time-budget", type=int, help="Time budget in seconds (optional)")
    parser.add_argument("--gpus-per-trial", type=float, default=0.25, help="GPUs per trial")
    parser.add_argument("--cpus-per-trial", type=int, default=2, help="CPUs per trial")
    parser.add_argument("--mlflow-experiment", type=str, help="MLflow experiment name (optional)")
    args = parser.parse_args()
    
    # Create MLflow experiment if needed
    experiment_id = None
    if args.mlflow_experiment:
        mlflow_manager = MLflowManager()
        experiment_id = mlflow_manager.create_experiment(args.mlflow_experiment)
    
    # Run the optimization
    results = run_multi_asset_optimization(
        config_path=args.config,
        storage_path=args.storage_path,
        experiment_name=args.experiment_name,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent,
        search_alg=args.search_alg,
        scheduler_type=args.scheduler,
        metric=args.metric,
        mode=args.mode,
        time_budget_s=args.time_budget,
        gpus_per_trial=args.gpus_per_trial,
        cpus_per_trial=args.cpus_per_trial,
        experiment_id=experiment_id
    )
    
    # Print summary
    best_trial = results.get_best_trial(args.metric, args.mode)
    print(f"\nBest trial: {best_trial.trial_id}")
    print(f"Best {args.metric}: {best_trial.last_result.get(args.metric)}")
    print(f"Assets in best configuration: {best_trial.last_result.get('assets', [])}")
    
    # Save best configuration to file for easy reuse
    best_config = best_trial.config
    if "_full_config" in best_config:
        del best_config["_full_config"]  # Remove full config to avoid duplication
    
    best_config_path = Path(args.storage_path) / f"best_config_{args.experiment_name}.yaml"
    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"Best configuration saved to: {best_config_path}")

if __name__ == "__main__":
    main() 