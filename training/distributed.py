"""Distributed training with resource management"""

import ray
import logging
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import mlflow

from training.utils.resource_manager import ResourceManager
from training.utils.state_manager import StateManager
from training.utils.batch_strategy import AdaptiveBatchStrategy
from training.utils.trainer import DistributedTrainer, TrainingConfig

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    
    # Training settings
    training_config: TrainingConfig
    
    # Resource limits
    max_memory_percent: float = 90.0
    max_gpu_percent: float = 95.0
    
    # MLflow settings
    experiment_name: str = "distributed_training"
    
    # Checkpointing
    checkpoint_frequency: int = 10  # epochs
    checkpoint_dir: str = "checkpoints"

@ray.remote
class DistributedTrainingManager:
    """Manages distributed training workflow"""
    
    def __init__(self, config: DistributedConfig):
        """Initialize manager
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Initialize managers
        self.resource_manager = ResourceManager.remote(
            max_memory_percent=config.max_memory_percent,
            max_gpu_percent=config.max_gpu_percent
        )
        self.state_manager = StateManager.remote()
        
        # Initialize components
        self.trainer = DistributedTrainer(config.training_config)
        self.batch_strategy = AdaptiveBatchStrategy()
        self.recovery_manager = RecoveryManager.remote(
            checkpoint_dir=config.checkpoint_dir
        )
        
        # Start monitoring
        ray.get(self.resource_manager.start_monitoring.remote())
        ray.get(self.recovery_manager.start_monitoring.remote())
        
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
        try:
            # Set up MLflow
            mlflow.set_experiment(self.config.experiment_name)
            
            with mlflow.start_run() as run:
                # Log configuration
                mlflow.log_params(self.config.__dict__)
                
                # Initialize actors
                actor_ids = self._initialize_actors(train_data)
                
                best_metrics = None
                best_reward = float('-inf')
                
                # Training loop
                for epoch in range(self.config.training_config.num_epochs):
                    try:
                        # Get batch size from strategy
                        batch_size = self.batch_strategy.get_next_batch_size()
                        
                        # Process batches
                        metrics = self._process_epoch(
                            train_data,
                            val_data,
                            batch_size,
                            actor_ids
                        )
                        
                        # Log metrics
                        mlflow.log_metrics(metrics, step=epoch)
                        
                        # Save best model
                        if metrics['val_reward'] > best_reward:
                            best_reward = metrics['val_reward']
                            best_metrics = metrics
                            self._save_checkpoint(epoch)
                        
                        # Log progress
                        logger.info(f"Epoch {epoch}: {metrics}")
                        
                        # Periodic checkpointing
                        if epoch % self.config.checkpoint_frequency == 0:
                            self._save_checkpoint(epoch)
                            
                    except Exception as e:
                        logger.error(f"Error in epoch {epoch}: {str(e)}", exc_info=True)
                        continue
                
                return best_metrics
                
        finally:
            # Cleanup
            self.cleanup()
    
    def _initialize_actors(self, data: pd.DataFrame) -> list:
        """Initialize training actors
        
        Args:
            data: Training data
            
        Returns:
            List of actor IDs
        """
        actor_ids = []
        
        for i in range(self.config.training_config.num_parallel):
            try:
                # Request resources
                allocation = ray.get(self.resource_manager.allocate_resources.remote(
                    f"actor_{i}",
                    cpu_needed=1,
                    memory_needed=data.memory_usage().sum() / self.config.training_config.num_parallel,
                    gpu_needed=1 if self.config.training_config.num_gpus else None
                ))
                
                # Register with state manager
                ray.get(self.state_manager.register_actor.remote(f"actor_{i}"))
                
                actor_ids.append(f"actor_{i}")
                
            except Exception as e:
                logger.error(f"Error initializing actor {i}: {str(e)}", exc_info=True)
                
        return actor_ids
    
    def _process_epoch(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        batch_size: int,
        actor_ids: list
    ) -> Dict[str, float]:
        """Process one training epoch with failure recovery
        
        Args:
            train_data: Training data
            val_data: Validation data
            batch_size: Batch size to use
            actor_ids: List of active actor IDs
            
        Returns:
            Training metrics
        """
        # Split data into batches
        batches = np.array_split(train_data, len(train_data) // batch_size)
        
        # Process batches with recovery
        results = []
        unfinished_batches = list(enumerate(batches))
        
        while unfinished_batches:
            futures = {}  # batch_index -> future
            
            # Start batch processing
            for batch_index, batch in unfinished_batches:
                actor_id = actor_ids[batch_index % len(actor_ids)]
                
                # Send heartbeat and get state
                should_continue = ray.get(
                    self.recovery_manager.heartbeat.remote(
                        actor_id,
                        {'batch_index': batch_index}
                    )
                )
                
                if should_continue:
                    # Process batch
                    futures[batch_index] = self.trainer.process_batch.remote(
                        batch,
                        actor_id
                    )
                else:
                    # Load checkpoint and retry
                    checkpoint = ray.get(
                        self.recovery_manager.load_checkpoint.remote(actor_id)
                    )
                    if checkpoint:
                        # Resume from checkpoint
                        futures[batch_index] = self.trainer.resume_batch.remote(
                            batch,
                            checkpoint,
                            actor_id
                        )
                    else:
                        # Retry from scratch
                        futures[batch_index] = self.trainer.process_batch.remote(
                            batch,
                            actor_id
                        )
            
            # Wait for results with timeout
            ready_futures, pending = ray.wait(
                list(futures.values()),
                timeout=30.0  # 30 seconds timeout
            )
            
            # Process completed batches
            completed_indices = []
            for batch_index, future in futures.items():
                if future in ready_futures:
                    try:
                        result = ray.get(future)
                        results.append(result)
                        completed_indices.append(batch_index)
                        
                        # Save checkpoint
                        actor_id = actor_ids[batch_index % len(actor_ids)]
                        ray.get(self.recovery_manager.save_checkpoint.remote(
                            actor_id,
                            {'batch_result': result}
                        ))
                        
                    except Exception as e:
                        logger.error(
                            f"Error processing batch {batch_index}: {str(e)}",
                            exc_info=True
                        )
                        # Mark worker as failed
                        actor_id = actor_ids[batch_index % len(actor_ids)]
                        ray.get(self.recovery_manager.mark_worker_failed.remote(actor_id))
            
            # Remove completed batches
            unfinished_batches = [
                (i, b) for i, b in unfinished_batches
                if i not in completed_indices
            ]
        
        # Check for failed workers
        failed_workers = ray.get(self.recovery_manager.get_failed_workers.remote())
        if failed_workers:
            logger.warning(f"Workers failed during epoch: {failed_workers}")
        
        # Evaluate
        val_metrics = self.trainer.evaluate(val_data)
        
        # Combine metrics
        metrics = {
            'train_loss': np.mean([r['loss'] for r in results]),
            'batch_size': batch_size,
            'val_reward': val_metrics['reward'],
            'val_portfolio_value': val_metrics['portfolio_value'],
            'failed_workers': len(failed_workers)
        }
        
        # Update batch strategy
        self.batch_strategy.record_batch({
            'batch_size': batch_size,
            'processing_time': np.mean([r['processing_time'] for r in results]),
            'loss': metrics['train_loss'],
            'metrics': metrics
        })
        
        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint
        
        Args:
            epoch: Current epoch number
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.trainer.save(checkpoint_path)
        
        # Save batch strategy state
        strategy_state = self.batch_strategy.get_stats()
        
        # Save to MLflow
        mlflow.log_artifacts(checkpoint_path)
        mlflow.log_metrics(strategy_state, step=epoch)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop resource monitoring
            ray.get(self.resource_manager.stop_monitoring.remote())
            
            # Release resources
            actor_states = ray.get(self.state_manager.get_all_states.remote())
            for actor_id in actor_states:
                ray.get(self.resource_manager.release_resources.remote(actor_id))
            
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}", exc_info=True)