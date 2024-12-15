"""Training utilities with Ray Actor integration"""

import ray
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import mlflow

from training.utils.ray_manager import RayManager, RayConfig, BatchConfig
from agents.ppo_agent import PPOAgent
from envs.trading_env import TradingEnvironment
from training.utils.mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for distributed training"""
    
    # Environment settings
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    window_size: int = 20
    
    # Training settings
    batch_size: int = 128
    num_parallel: int = 4
    chunk_size: int = 32
    num_epochs: int = 100
    
    # Resource settings
    num_cpus: int = 4
    num_gpus: Optional[float] = None  # None means use all available
    
    # MLflow settings
    experiment_name: str = "distributed_training"

class DistributedTrainer:
    """Distributed training with Ray actors"""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.ray_config = RayConfig(
            num_cpus=config.num_cpus,
            num_gpus=config.num_gpus
        )
        self.batch_config = BatchConfig(
            batch_size=config.batch_size,
            num_parallel=config.num_parallel,
            chunk_size=config.chunk_size
        )
        
        self.ray_manager = RayManager(self.ray_config)
        self.mlflow_manager = MLflowManager()
        
    def create_env(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment
        
        Args:
            data: Training data
            
        Returns:
            Trading environment instance
        """
        return TradingEnvironment(
            df=data,
            initial_balance=self.config.initial_balance,
            trading_fee=self.config.trading_fee,
            window_size=self.config.window_size
        )
    
    def create_agent(self, env: TradingEnvironment) -> PPOAgent:
        """Create PPO agent
        
        Args:
            env: Trading environment
            
        Returns:
            PPO agent instance
        """
        return PPOAgent(env)
        
    def train(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run distributed training
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Training metrics
        """
        # Set up MLflow
        self.mlflow_manager.set_experiment(self.config.experiment_name)
        
        with mlflow.start_run() as run:
            # Log config
            self.mlflow_manager.log_params(self.config.__dict__)
            
            # Create environment and agent
            train_env = self.create_env(train_data)
            val_env = self.create_env(val_data)
            agent = self.create_agent(train_env)
            
            # Create actor pool for parallel training
            actor_pool = self.ray_manager.create_actor_pool(
                actor_class=type(agent),
                actor_config={'env': train_env},
                num_actors=self.config.num_parallel
            )
            
            best_reward = float('-inf')
            best_metrics = None
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                # Get training batches
                train_batches = self._prepare_batches(train_data)
                
                # Train in parallel
                train_results = self.ray_manager.process_in_parallel(
                    actor_pool,
                    train_batches,
                    self.batch_config
                )
                
                # Evaluate
                val_metrics = self._evaluate(agent, val_env)
                
                # Log metrics
                metrics = {
                    'epoch': epoch,
                    'train_loss': np.mean([r['loss'] for r in train_results]),
                    **val_metrics
                }
                self.mlflow_manager.log_metrics(metrics, step=epoch)
                
                # Save best model
                if val_metrics['reward'] > best_reward:
                    best_reward = val_metrics['reward']
                    best_metrics = metrics
                    self._save_model(agent, run.info.run_id)
                
                logger.info(f"Epoch {epoch}: {metrics}")
            
            return best_metrics
    
    def _prepare_batches(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data batches for training
        
        Args:
            data: Training data
            
        Returns:
            Batched data
        """
        # Convert data to numpy array and create batches
        return np.array_split(data.values, len(data) // self.config.batch_size)
    
    def _evaluate(
        self, 
        agent: PPOAgent,
        env: TradingEnvironment
    ) -> Dict[str, float]:
        """Evaluate agent performance
        
        Args:
            agent: PPO agent
            env: Trading environment
            
        Returns:
            Evaluation metrics
        """
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        return {
            'reward': total_reward,
            'portfolio_value': info['portfolio_value'],
            'total_trades': len(info.get('trades', []))
        }
    
    def _save_model(self, agent: PPOAgent, run_id: str):
        """Save model artifacts
        
        Args:
            agent: PPO agent to save
            run_id: MLflow run ID
        """
        artifacts_dir = f"models/run_{run_id}"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        model_path = os.path.join(artifacts_dir, "model.pth")
        agent.save(model_path)
        
        mlflow.log_artifact(model_path)
    
    def cleanup(self):
        """Cleanup resources"""
        self.ray_manager.shutdown()