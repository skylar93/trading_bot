"""Run distributed training"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import psutil
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import ray
from data.utils.data_loader import DataLoader
from training.distributed import DistributedTrainingManager, DistributedConfig
from training.utils.trainer import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run distributed training")
    
    # Data options
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange to fetch data from"
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair symbol"
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Data timeframe"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data"
    )
    
    # Training options
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        default=128,
        help="Initial batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate"
    )
    
    # Resource options
    parser.add_argument(
        "--max-memory",
        type=float,
        default=90.0,
        help="Maximum memory usage percentage"
    )
    parser.add_argument(
        "--max-gpu",
        type=float,
        default=95.0,
        help="Maximum GPU usage percentage"
    )
    parser.add_argument(
        "--gpu-fraction",
        type=float,
        default=0.25,
        help="Fraction of GPU memory per worker"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize Ray
        ray.init(
            num_cpus=args.num_parallel,
            num_gpus=1 if torch.cuda.is_available() else None,
            object_store_memory=int(psutil.virtual_memory().total * 0.5)  # Use 50% of system memory
        )
        
        # Load data
        data_loader = DataLoader(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        data = data_loader.fetch_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Split data
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.15)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        # Create configs
        training_config = TrainingConfig(
            batch_size=args.initial_batch_size,
            num_parallel=args.num_parallel,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        distributed_config = DistributedConfig(
            training_config=training_config,
            max_memory_percent=args.max_memory,
            max_gpu_percent=args.max_gpu,
            experiment_name=f"distributed_{args.symbol}_{args.timeframe}"
        )
        
        # Create and run manager
        manager = DistributedTrainingManager.remote(distributed_config)
        
        # Run training
        metrics = ray.get(manager.train.remote(train_data, val_data))
        
        logger.info("Training completed!")
        logger.info(f"Best metrics: {metrics}")
        
        # Test final model
        test_metrics = ray.get(manager.evaluate.remote(test_data))
        logger.info(f"Test metrics: {test_metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
        
    finally:
        # Cleanup Ray
        ray.shutdown()

if __name__ == "__main__":
    main()