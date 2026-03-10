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

# Configure detailed logging
def setup_detailed_logging():
    """
    Configure detailed logging for training process
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create log directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler for DEBUG level logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"training_detailed_{timestamp}.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Define a StringIO handler to capture logs for UI display
    from io import StringIO
    log_stream = StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    return log_stream

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
        
        # Setup detailed logging and capture stream for UI
        self.log_stream = setup_detailed_logging()
        logger.info("TrainingManager initialized with configuration")
        
        # Prepare the configuration for the training pipeline
        self.prepared_config = self._prepare_config(config)
        
        # Set up MLflow manager for tracking
        experiment_name = f"{self.prepared_config.get('agent_type', 'ppo')}_{self.prepared_config['env']['type']}"
        self.mlflow_manager = MLflowManager(experiment_name)
        
        # Add estimated duration for progress calculation
        self.estimated_duration = self._estimate_training_duration()
    
    def _estimate_training_duration(self) -> float:
        """
        Estimate the training duration in seconds based on configuration.
        This is a rough estimate used for progress tracking.
        
        Returns:
            Estimated duration in seconds
        """
        total_timesteps = self.prepared_config.get("training", {}).get("total_timesteps", 100000)
        # Rough estimation: 1000 timesteps per second on average hardware
        # This will vary based on hardware, complexity, etc.
        base_rate = 1000  # timesteps per second
        
        # Adjust for complexity factors
        if self.prepared_config.get("env", {}).get("type") in ["multi_asset_rl", "multi_agent_rl"]:
            base_rate /= 1.5  # Slower for multi-asset/agent
        
        if self.prepared_config.get("env", {}).get("type") == "multi_asset_multi_agent_rl":
            base_rate /= 2.0  # Even slower for both multi-asset AND multi-agent
        
        # Adjust for window size
        window_size = self.prepared_config.get("env", {}).get("window_size", 20)
        base_rate *= (20 / max(window_size, 1))  # Slower for larger windows
        
        # Adjust for network complexity
        hidden_sizes = self.prepared_config.get("agent", {}).get("hidden_sizes", [64, 64])
        complexity_factor = sum(hidden_sizes) / 128  # Normalize by typical size
        base_rate /= max(complexity_factor, 0.5)
        
        # Calculate estimate
        estimated_seconds = total_timesteps / base_rate
        
        # Add buffer for safety
        return estimated_seconds * 1.2
    
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
            },
            "data": {
                "exchange": "binance",
                "symbols": ["BTC/USDT"],
                "timeframe": "1h",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
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
        
        # Data configuration
        config["data"]["exchange"] = env_config.get("exchange", "binance")
        config["data"]["symbols"] = config["env"]["symbols"]  # 환경에서 사용되는 심볼 그대로 사용
        config["data"]["timeframe"] = config["env"]["timeframe"]  # 환경에서 사용되는 타임프레임 그대로 사용
        
        # Set data date range
        if "start_date" in env_config and "end_date" in env_config:
            config["env"]["start_date"] = env_config["start_date"]
            config["env"]["end_date"] = env_config["end_date"]
            config["data"]["start_date"] = env_config["start_date"]
            config["data"]["end_date"] = env_config["end_date"]
        
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
        self.last_update_time = self.started_at
        self.console_output = []
        
        logger.info(f"Starting training at {self.started_at}")
        
        # Start MLflow run
        self.mlflow_manager.start_run()
        self.mlflow_run_id = self.mlflow_manager.run_id
        self.mlflow_ui_url = self.mlflow_manager.get_run_url()
        
        logger.info(f"MLflow run started: {self.mlflow_run_id}")
        
        try:
            # Log parameters to MLflow
            self.mlflow_manager.log_params(self.prepared_config)
            logger.info("Configuration parameters logged to MLflow")
            
            # Create a progress tracker
            def update_progress(current_step: int, total_steps: int, metrics: Dict[str, Any]):
                self.current_step = current_step
                
                # Calculate progress based on both steps and elapsed time
                step_progress = min(current_step / total_steps, 1.0)
                
                # Time-based progress estimation
                elapsed_seconds = (datetime.now() - self.started_at).total_seconds()
                time_progress = min(elapsed_seconds / self.estimated_duration, 1.0)
                
                # Combine both progress indicators with weights
                # Early in training, time progress is more reliable
                time_weight = max(0.8 - (current_step / total_steps), 0)
                self.progress = (step_progress * (1 - time_weight)) + (time_progress * time_weight)
                self.progress = min(self.progress, 0.99)  # Never show 100% until actually complete
                
                self.metrics = metrics
                
                # Log metrics to MLflow
                if metrics and (datetime.now() - self.last_update_time).total_seconds() >= 5:
                    try:
                        self.mlflow_manager.log_metrics(metrics)
                        self.last_update_time = datetime.now()
                        logger.debug(f"Metrics logged to MLflow: {metrics}")
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to MLflow: {str(e)}")
                
                # Capture console output for UI
                if hasattr(self, 'log_stream'):
                    log_contents = self.log_stream.getvalue()
                    if log_contents:
                        for line in log_contents.splitlines():
                            if line and line not in self.console_output:
                                self.console_output.append(line)
                
                # Update progress in UI
                if progress_callback:
                    progress_callback(self.progress, {
                        **metrics,
                        "current_step": current_step,
                        "total_steps": total_steps,
                        "elapsed_time": elapsed_seconds,
                        "estimated_total_time": self.estimated_duration,
                        "console_output": self.console_output[-50:] if hasattr(self, 'console_output') else []
                    })
            
            # Set progress update callback in config
            self.prepared_config["callbacks"] = {
                "progress_update": update_progress
            }
            
            # Load historical data
            data = None
            try:
                from data.utils.data_loader import DataLoader
                
                # Create data loader
                data_config = self.prepared_config.get("data", {})
                
                # 각 심볼에 대해 데이터를 로드하고 병합
                symbols = self.prepared_config["env"].get("symbols", ["BTC/USDT"])
                timeframe = data_config.get("timeframe", self.prepared_config["env"].get("timeframe", "1h"))
                start_date = data_config.get("start_date", self.prepared_config["env"].get("start_date", "2023-01-01"))
                end_date = data_config.get("end_date", self.prepared_config["env"].get("end_date", "2023-12-31"))
                exchange = data_config.get("exchange", "binance")
                
                # 첫 번째 심볼은 기본 데이터프레임으로 설정
                data_loader = DataLoader(exchange_id=exchange, symbol=symbols[0], timeframe=timeframe)
                data = data_loader.fetch_data(start_date=start_date, end_date=end_date)
                
                # 다중 자산 환경일 경우 여러 심볼 데이터를 처리
                if len(symbols) > 1 and self.prepared_config["env"].get("type") in ["multi_asset_rl", "multi_asset_multi_agent_rl"]:
                    combined_data = data.copy()
                    
                    # 첫 번째 자산의 컬럼 이름에 심볼 접두사 추가
                    symbol_id = symbols[0].split('/')[0]  # BTC/USDT -> BTC
                    renamed_columns = {col: f"{symbol_id}_{col}" for col in combined_data.columns}
                    combined_data = combined_data.rename(columns=renamed_columns)
                    
                    # 추가 자산 데이터 로드 및 병합
                    for symbol in symbols[1:]:
                        data_loader = DataLoader(exchange_id=exchange, symbol=symbol, timeframe=timeframe)
                        symbol_data = data_loader.fetch_data(start_date=start_date, end_date=end_date)
                        
                        # 컬럼 이름에 심볼 접두사 추가
                        symbol_id = symbol.split('/')[0]  # ETH/USDT -> ETH
                        renamed_columns = {col: f"{symbol_id}_{col}" for col in symbol_data.columns}
                        symbol_data = symbol_data.rename(columns=renamed_columns)
                        
                        # 기존 데이터프레임에 컬럼 추가
                        for col in symbol_data.columns:
                            combined_data[col] = symbol_data[col]
                    
                    data = combined_data
                
                logger.info(f"Historical data loaded for {symbols} with shape {data.shape}")
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                raise ValueError(f"Error loading data: {str(e)}")
            
            # Print training configuration summary to terminal before training starts
            print("\n" + "="*80)
            print("TRAINING CONFIGURATION SUMMARY")
            print("="*80)
            
            # Environment Settings
            env_config = self.prepared_config.get('env', {})
            asset_mode = "Multi" if env_config.get('asset_mode') == 'multi' else "Single"
            env_summary = f"\nEnvironment:\n"
            env_summary += f"  Asset Mode: {asset_mode}\n"
            env_summary += f"  Symbol(s): {', '.join(env_config.get('symbols', ['N/A']))}\n"
            env_summary += f"  Timeframe: {env_config.get('timeframe', 'N/A')}\n"
            env_summary += f"  Window Size: {env_config.get('window_size', 'N/A')}\n"
            
            print(env_summary)
            logger.info(f"Training environment: {asset_mode} asset mode, " +
                       f"Symbols: {', '.join(env_config.get('symbols', ['N/A']))}, " +
                       f"Timeframe: {env_config.get('timeframe', 'N/A')}")
            
            # Date Range
            start_date = env_config.get('start_date', 'N/A')
            end_date = env_config.get('end_date', 'N/A')
            date_summary = f"  Date Range: {start_date} to {end_date}\n"
            print(date_summary)
            logger.info(f"Training date range: {start_date} to {end_date}")
            
            if env_config.get('stop_loss_pct'):
                stop_loss = f"  Stop Loss: {env_config.get('stop_loss_pct', 0) * 100}%\n"
                print(stop_loss)
                logger.info(f"Stop loss: {env_config.get('stop_loss_pct', 0) * 100}%")
            
            # Agent Settings
            agent_config = self.prepared_config.get('agent', {})
            agent_summary = f"\nAgent Configuration:\n"
            agent_summary += f"  Algorithm: {self.prepared_config.get('agent_type', 'N/A')}\n"
            agent_summary += f"  Hidden Layers: {agent_config.get('hidden_sizes', 'N/A')}\n"
            lstm_status = 'Yes' if agent_config.get('use_lstm', False) else 'No'
            agent_summary += f"  LSTM: {lstm_status}\n"
            
            print(agent_summary)
            logger.info(f"Agent: {self.prepared_config.get('agent_type', 'N/A')}, " +
                      f"Hidden layers: {agent_config.get('hidden_sizes', 'N/A')}, LSTM: {lstm_status}")
            
            if agent_config.get('use_lstm', False):
                lstm_info = f"  LSTM Hidden Size: {agent_config.get('lstm_hidden_size', 'N/A')}\n"
                print(lstm_info)
                logger.info(f"LSTM hidden size: {agent_config.get('lstm_hidden_size', 'N/A')}")
            
            # Training Parameters
            training_config = self.prepared_config.get('training', {})
            training_summary = f"\nTraining Parameters:\n"
            training_summary += f"  Total Timesteps: {training_config.get('total_timesteps', 'N/A'):,}\n"
            training_summary += f"  Batch Size: {training_config.get('batch_size', 'N/A')}\n"
            training_summary += f"  Learning Rate: {training_config.get('learning_rate', 'N/A')}\n"
            training_summary += f"  Gamma: {training_config.get('gamma', 'N/A')}\n"
            
            print(training_summary)
            logger.info(f"Training params: {training_config.get('total_timesteps', 'N/A'):,} steps, " +
                      f"batch: {training_config.get('batch_size', 'N/A')}, " +
                      f"lr: {training_config.get('learning_rate', 'N/A')}, " +
                      f"gamma: {training_config.get('gamma', 'N/A')}")
            
            if 'seed' in training_config:
                seed_info = f"  Seed: {training_config.get('seed', 'N/A')}\n"
                print(seed_info)
                logger.info(f"Random seed: {training_config.get('seed', 'N/A')}")
                
            # Data Summary
            data_summary = f"\nData Information:\n"
            if data is not None:
                data_summary += f"  Shape: {data.shape}\n"
                data_summary += f"  Columns: {list(data.columns)[:5]}... (and {len(data.columns)-5} more)\n"
                data_summary += f"  Date Range: {data.index[0]} to {data.index[-1]}\n"
                
                print(data_summary)
                logger.info(f"Data: shape={data.shape}, date range={data.index[0]} to {data.index[-1]}")
            else:
                data_summary += "  No data loaded\n"
                print(data_summary)
                logger.warning("No data loaded for training")
            
            # MLflow Information
            mlflow_summary = f"\nMLflow Tracking:\n"
            mlflow_summary += f"  Experiment: {self.mlflow_manager.experiment_name}\n"
            mlflow_summary += f"  Run ID: {self.mlflow_run_id}\n"
            
            print(mlflow_summary)
            logger.info(f"MLflow: experiment={self.mlflow_manager.experiment_name}, run_id={self.mlflow_run_id}")
            
            print("\n" + "="*80)
            print("Starting training...")
            print("="*80 + "\n")
            logger.info("Starting training pipeline execution")
            
            # Run the training in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            training_result = await loop.run_in_executor(
                None, lambda: train_pipeline(self.prepared_config, data)
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
        
        # Get console output if available
        console_output = getattr(self, 'console_output', [])
        
        # Get MLflow URL if available
        mlflow_ui_url = getattr(self, 'mlflow_ui_url', None)
        
        return {
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": duration,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "metrics": self.metrics,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_ui_url": mlflow_ui_url,
            "console_output": console_output[-50:] if console_output else [],
            "estimated_duration": getattr(self, 'estimated_duration', None)
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