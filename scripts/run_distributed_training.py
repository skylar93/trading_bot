"""Run distributed training"""

import os
import sys
import logging
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from data.utils.data_loader import DataLoader
from training.utils.trainer import DistributedTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run distributed training")
    
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
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    
    args = parser.parse_args()
    
    try:
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
        
        # Configure training
        config = TrainingConfig(
            num_parallel=args.parallel,
            experiment_name=f"distributed_{args.symbol}_{args.timeframe}"
        )
        
        # Create trainer
        trainer = DistributedTrainer(config)
        
        try:
            # Run training
            metrics = trainer.train(train_data, val_data)
            
            logger.info("Training completed!")
            logger.info(f"Best metrics: {metrics}")
            
        finally:
            trainer.cleanup()
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()