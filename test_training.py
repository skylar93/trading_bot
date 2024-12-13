import yaml
import logging
from pathlib import Path
from training.train import TrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create config directory if it doesn't exist
    Path('config').mkdir(exist_ok=True)
    
    # Create minimal test configuration
    config = {
        'experiment_name': 'test_training',
        'data': {
            'exchange': 'binance',
            'symbols': ['BTC/USDT'],
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-01-07'  # 1주일 데이터로 테스트
        },
        'env': {
            'initial_balance': 10000,
            'trading_fee': 0.001,
            'window_size': 10  # 윈도우 사이즈를 줄여서 간단하게 시작
        },
        'model': {
            'fcnet_hiddens': [32],  # 더 작은 네트워크로 시작
            'lr': 0.001
        },
        'training': {
            'total_timesteps': 50,  # 매우 짧은 학습으로 시작
            'checkpoint_freq': 10,
            'early_stop': 20
        }
    }
    
    # Save test config
    config_path = 'config/test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info("Starting minimal test training pipeline...")
    
    try:
        # Create and run pipeline
        pipeline = TrainingPipeline(config_path)
        
        logger.info("Preparing data...")
        train_data, val_data, test_data = pipeline.prepare_data()
        logger.info(f"Data prepared - Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
        
        logger.info("Starting training...")
        agent = pipeline.train(train_data, val_data)
        logger.info("Training completed")
        
        logger.info("Starting evaluation...")
        metrics = pipeline.evaluate(agent, test_data)
        
        print("\nTest completed successfully!")
        print("Test metrics:", metrics)
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()