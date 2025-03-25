import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> bool:
    """데이터 유효성 검사"""
    try:
        # 필수 컬럼 확인
        required_columns = ['$open', '$high', '$low', '$close', '$volume']
        assert all(col in df.columns for col in required_columns), f"Missing columns: {[col for col in required_columns if col not in df.columns]}"
        
        # NaN 값 확인
        assert not df.isnull().any().any(), f"Found NaN values in columns: {df.columns[df.isnull().any()].tolist()}"
        
        # 데이터 타입 확인
        for col in required_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric type"
        
        # OHLC 관계 확인
        assert (df['$high'] >= df['$low']).all(), "High should be >= Low"
        assert (df['$high'] >= df['$open']).all(), "High should be >= Open"
        assert (df['$high'] >= df['$close']).all(), "High should be >= Close"
        assert (df['$low'] <= df['$open']).all(), "Low should be <= Open"
        assert (df['$low'] <= df['$close']).all(), "Low should be <= Close"
        
        # 볼륨 확인
        assert (df['$volume'] >= 0).all(), "Volume should be >= 0"
        
        logger.info("Data validation passed!")
        logger.info(f"Data shape: {df.shape}")
        logger.info("\nFirst few rows:")
        logger.info(f"\n{df.head()}")
        logger.info("\nData statistics:")
        logger.info(f"\n{df.describe()}")
        
        # 데이터 타입 정보 출력
        logger.info("\nData types:")
        logger.info(f"\n{df[required_columns].dtypes}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False

def main():
    try:
        # 설정 파일 로드
        config_path = os.path.join(os.path.dirname(__file__), "config", "default_config.yaml")
        logger.info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 설정 검증
        required_keys = ['env', 'data', 'model', 'training', 'paths']
        assert all(key in config for key in required_keys), f"Missing required keys in config: {[k for k in required_keys if k not in config]}"

        # 데이터 설정 검증
        data_config = config['data']
        assert all(k in data_config for k in ['exchange', 'symbols', 'timeframe', 'start_date']), "Missing required data config"

        logger.info("Configuration loaded successfully:")
        logger.info(f"Exchange: {data_config['exchange']}")
        logger.info(f"Symbol: {data_config['symbols'][0]}")
        logger.info(f"Timeframe: {data_config['timeframe']}")
        logger.info(f"Start date: {data_config['start_date']}")

        # end_date가 없으면 현재 날짜 사용
        end_date = data_config.get('end_date', datetime.now().strftime("%Y-%m-%d"))
        logger.info(f"End date: {end_date}")

        # 데이터 로드
        from data.utils.data_loader import DataLoader
        
        data_loader = DataLoader(
            exchange_id=config['data']['exchange'],
            symbol=config['data']['symbols'][0],
            timeframe=config['data']['timeframe']
        )

        # 데이터 가져오기
        logger.info("Fetching data...")
        data = data_loader.fetch_data(
            start_date=config['data']['start_date'],
            end_date=end_date
        )

        # 데이터 검증
        is_valid = validate_data(data)
        
        if is_valid:
            logger.info("Data pipeline validation completed successfully!")
        else:
            logger.error("Data pipeline validation failed!")

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 