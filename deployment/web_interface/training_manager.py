"""
Training Manager for Web Interface.

This module provides a bridge between the Streamlit UI and the training pipeline.
It handles training configuration, execution, and progress tracking.

Features:
- Asynchronous training execution
- Progress tracking and status updates
- Configuration validation and normalization
- Support for both single-asset and multi-asset trading
- Handling of multi-agent systems with asset assignments
- MLflow integration for experiment tracking

Implementation Notes:
- Uses the unified training pipeline from training/train_pipeline.py
- Manages communication between UI and training process
- Handles proper resource cleanup and error logging
- Supports four environment types: 
  - single_asset_rl
  - multi_asset_rl
  - multi_agent_rl
  - multi_asset_multi_agent_rl
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import yaml
import json
from datetime import datetime
import pandas as pd

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.train_pipeline import train_pipeline
from training.utils.config_manager import ConfigManager
from training.utils.unified_mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

class TrainingManager:
    """
    Manages the training process for the web interface.
    
    This class handles the interface between the Streamlit UI and the
    training pipeline, including configuration preparation, training execution,
    and progress tracking.
    
    Features:
    - Asynchronous training execution
    - Real-time progress updates
    - Configuration validation
    - Support for single/multi-asset trading
    - Support for single/multi-agent setups
    - MLflow integration
    
    Implementation Notes:
    - Converts UI parameters to training configuration format
    - Manages progress callbacks for UI updates
    - Handles proper cleanup of resources
    - Configures environment type based on asset mode and agent count
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TrainingManager with a configuration.
        
        Args:
            config: Configuration dictionary from the UI
        """
        self.config = config
        self.status = "initialized"
        self.progress = 0.0
        self.started_at = None
        self.finished_at = None
        self.current_step = 0
        self.total_steps = config.get("training", {}).get("total_timesteps", 100000)
        self.metrics = {}
        self.training_task = None
        self.mlflow_run_id = None
        
        # Prepare the configuration for the training pipeline
        self.prepared_config = self._prepare_config(config)
        
        # Set up MLflow manager for tracking
        experiment_name = f"{self.prepared_config.get('agent_type', 'ppo')}_{self.prepared_config['env']['type']}"
        self.mlflow_manager = MLflowManager(experiment_name)
    
    def _prepare_config(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert UI configuration to training pipeline format.
        
        Args:
            ui_config: Configuration from the UI
            
        Returns:
            Configuration formatted for the training pipeline
        """
        # Start with a default configuration
        config = {
            "env": {
                "type": "single_asset_rl"
            },
            "agent": {},
            "training": {},
            "paths": {
                "checkpoint_dir": "checkpoints"
            }
        }
        
        # Environment configuration
        env_config = ui_config.get("env", {})
        
        # Handle asset mode (single or multi)
        asset_mode = env_config.get("asset_mode", "single")
        config["env"]["asset_mode"] = asset_mode
        
        if asset_mode == "single":
            config["env"]["symbol"] = env_config.get("symbol", "BTC/USDT")
            config["env"]["symbols"] = [config["env"]["symbol"]]
        else:
            # Multi-asset mode
            config["env"]["symbols"] = env_config.get("symbols", ["BTC/USDT", "ETH/USDT"])
            # For backwards compatibility
            config["env"]["symbol"] = config["env"]["symbols"][0]
        
        config["env"]["timeframe"] = env_config.get("timeframe", "1h")
        config["env"]["window_size"] = env_config.get("window_size", 20)
        
        # Set data date range
        if "start_date" in env_config and "end_date" in env_config:
            config["env"]["start_date"] = env_config["start_date"]
            config["env"]["end_date"] = env_config["end_date"]
        
        # Risk management
        if env_config.get("use_stop_loss", False):
            config["env"]["stop_loss_pct"] = env_config.get("stop_loss_pct", 0.05)
        
        # Set environment type based on asset_mode and multi_agent
        is_multi_agent = env_config.get("multi_agent", False)
        
        if asset_mode == "single" and not is_multi_agent:
            config["env"]["type"] = "single_asset_rl"
        elif asset_mode == "multi" and not is_multi_agent:
            config["env"]["type"] = "multi_asset_rl"
        elif asset_mode == "single" and is_multi_agent:
            config["env"]["type"] = "multi_agent_rl"
        elif asset_mode == "multi" and is_multi_agent:
            config["env"]["type"] = "multi_asset_multi_agent_rl"
        
        # Multi-agent configuration
        if is_multi_agent:
            # Don't overwrite the env type we just set
            # config["env"]["type"] = "multi_agent_rl"
            config["env"]["multi_agent_configs"] = []
            
            # Set manager configuration if enabled
            config["env"]["use_manager"] = env_config.get("use_manager", False)
            config["env"]["ensemble_method"] = env_config.get("ensemble_method", "weighted")
            
            # Add meta-agent configuration if using meta ensemble
            if env_config.get("ensemble_method") == "meta" and "meta_config" in env_config:
                meta_config = env_config["meta_config"]
                config["env"]["meta_config"] = {
                    "id": "meta_agent",
                    "type": "meta",
                    "model": "ppo",
                    "learning_rate": meta_config.get("learning_rate", 3e-4),
                    "hidden_dim": meta_config.get("hidden_dim", 128),
                    "continuous_ensemble": meta_config.get("continuous_ensemble", True)
                }
            
            # Add shared buffer configuration if enabled
            if env_config.get("use_manager", False) and "shared_buffer" in env_config:
                shared_buffer = env_config["shared_buffer"]
                config["env"]["shared_buffer"] = {
                    "enabled": shared_buffer.get("enabled", True),
                    "min_share_reward": shared_buffer.get("min_share_reward", 0.2),
                    "max_buffer_size": shared_buffer.get("max_buffer_size", 10000)
                }
            
            # Add configuration for each agent
            agent_count = env_config.get("agent_count", 2)
            for i in range(agent_count):
                agent_config = env_config.get(f"agent_{i}", {})
                
                # Extract agent type (algorithm) and strategy separately
                agent_type = agent_config.get("type", "ppo")  # Learning algorithm
                strategy = agent_config.get("strategy", None)  # Trading strategy
                capital_pct = agent_config.get("capital_pct", 1.0 / agent_count)
                
                # Get hyperparameters
                hyperparameters = agent_config.get("hyperparameters", {})
                hidden_sizes = hyperparameters.get("hidden_sizes", agent_config.get("hidden_layers", [64, 64]))
                learning_rate = hyperparameters.get("learning_rate", agent_config.get("learning_rate", 3e-4))
                
                agent_config_dict = {
                    "id": f"agent_{i}",
                    "agent_type": agent_type,  # Learning algorithm (ppo, sac, etc.)
                    "strategy": strategy,      # Trading strategy (momentum, etc.)
                    "hyperparameters": {
                        "hidden_sizes": hidden_sizes,
                        "learning_rate": learning_rate
                    },
                    "capital_pct": capital_pct
                }
                
                # Add assigned assets for multi-asset mode
                if asset_mode == "multi" and "assigned_assets" in agent_config:
                    agent_config_dict["assigned_assets"] = agent_config.get("assigned_assets", config["env"]["symbols"])
                
                config["env"]["multi_agent_configs"].append(agent_config_dict)
            
            # Ensemble method
            config["env"]["ensemble_method"] = env_config.get("ensemble_method", "weighted")
        
        # Agent configuration
        agent_config = ui_config.get("agent", {})
        config["agent_type"] = agent_config.get("algorithm", "ppo")
        config["agent"]["hidden_sizes"] = agent_config.get("hidden_layers", [64, 64])
        config["agent"]["learning_rate"] = agent_config.get("learning_rate", 3e-4)
        
        # Use LSTM if specified
        if agent_config.get("use_lstm", False):
            config["agent"]["use_lstm"] = True
            config["agent"]["lstm_hidden_size"] = agent_config.get("lstm_hidden_size", 64)
        
        # Training configuration
        training_config = ui_config.get("training", {})
        config["training"]["total_timesteps"] = training_config.get("total_timesteps", 100000)
        config["training"]["batch_size"] = training_config.get("batch_size", 64)
        config["training"]["learning_rate"] = training_config.get("learning_rate", 3e-4)
        config["training"]["gamma"] = training_config.get("gamma", 0.99)
        config["training"]["eval_interval"] = training_config.get("eval_interval", 5000)
        config["training"]["checkpoint_interval"] = training_config.get("checkpoint_interval", 10000)
        
        # Random seed for reproducibility
        if "seed" in training_config:
            config["training"]["seed"] = training_config["seed"]
        
        return config
    
    async def run_training(self, progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Run the training process asynchronously.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with training results
        """
        self.status = "running"
        self.started_at = datetime.now()
        self.progress = 0.0
        
        # Start MLflow run
        self.mlflow_manager.start_run()
        self.mlflow_run_id = self.mlflow_manager.run_id
        
        try:
            # Log parameters to MLflow
            self.mlflow_manager.log_params(self.prepared_config)
            
            # Create a progress tracker
            def update_progress(current_step: int, total_steps: int, metrics: Dict[str, Any]):
                self.current_step = current_step
                self.progress = min(current_step / total_steps, 1.0)
                self.metrics = metrics
                
                if progress_callback:
                    progress_callback(self.progress, metrics)
            
            # Set progress update callback in config
            self.prepared_config["callbacks"] = {
                "progress_update": update_progress
            }
            
            # Run the training in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            training_result = await loop.run_in_executor(
                None, lambda: train_pipeline(self.prepared_config)
            )
            
            self.status = "completed"
            self.progress = 1.0
            self.finished_at = datetime.now()
            
            return training_result
            
        except Exception as e:
            self.status = "failed"
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
            
        finally:
            # End MLflow run
            self.mlflow_manager.end_run()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the training process.
        
        Returns:
            Dictionary with status information
        """
        duration = None
        if self.started_at:
            end_time = self.finished_at or datetime.now()
            duration = (end_time - self.started_at).total_seconds()
        
        return {
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": duration,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "metrics": self.metrics,
            "mlflow_run_id": self.mlflow_run_id
        }
    
    def get_mlflow_metrics(self) -> Dict[str, List]:
        """
        Get metrics from MLflow for the current run.
        
        Returns:
            Dictionary with metric histories
        """
        if not self.mlflow_run_id:
            return {}
        
        return self.mlflow_manager.get_metric_history() 