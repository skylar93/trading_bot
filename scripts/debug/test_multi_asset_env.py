#!/usr/bin/env python
"""
디버깅 스크립트: MultiAssetTradingEnv 클래스의 logger 속성 문제 해결

이 스크립트는 MultiAssetTradingEnv 클래스에 logger 속성이 있는지 확인하고,
해당 속성이 올바르게 초기화되는지 테스트합니다.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from envs.multi_asset_env import MultiAssetTradingEnv
    logger.info("MultiAssetTradingEnv 클래스 성공적으로 임포트됨")
except ImportError as e:
    logger.error(f"MultiAssetTradingEnv 클래스 임포트 실패: {e}")
    sys.exit(1)

def create_synthetic_data(n_samples=200, assets=None):
    """테스트에 사용할 합성 데이터 생성"""
    if assets is None:
        assets = ["BTC", "ETH", "LTC"]
    
    data = pd.DataFrame()
    timestamps = pd.date_range(start="2023-01-01", periods=n_samples, freq="1H")
    
    for asset in assets:
        # 임의의 OHLCV 데이터 생성
        base_price = np.random.uniform(100, 1000)
        volatility = np.random.uniform(0.01, 0.05)
        
        # 난수 시드 설정
        np.random.seed(42 + assets.index(asset))
        
        # 가격 데이터 랜덤 워크 생성
        changes = np.random.normal(0, volatility, n_samples)
        close_prices = base_price * np.exp(np.cumsum(changes))
        
        # OHLCV 데이터 생성
        high_prices = close_prices * np.random.uniform(1.001, 1.02, n_samples)
        low_prices = close_prices * np.random.uniform(0.98, 0.999, n_samples)
        open_prices = low_prices + np.random.uniform(0, 1, n_samples) * (high_prices - low_prices)
        volumes = np.random.uniform(1000, 10000, n_samples) * close_prices
        
        # 데이터프레임에 추가
        data[f"{asset}_$open"] = open_prices
        data[f"{asset}_$high"] = high_prices
        data[f"{asset}_$low"] = low_prices
        data[f"{asset}_$close"] = close_prices
        data[f"{asset}_$volume"] = volumes
    
    data["timestamp"] = timestamps
    data.set_index("timestamp", inplace=True)
    
    return data

def test_multi_asset_env_logger():
    """MultiAssetTradingEnv 클래스의 logger 속성 테스트"""
    
    # 합성 데이터 생성
    logger.info("테스트 데이터 생성 중...")
    data = create_synthetic_data()
    assets = ["BTC", "ETH", "LTC"]
    
    # 환경 생성
    logger.info("환경 객체 생성 중...")
    try:
        env = MultiAssetTradingEnv(
            df=data,
            assets=assets,
            window_size=10,
            initial_balance=10000.0,
            trading_fee=0.001
        )
        logger.info("환경 객체 생성 성공!")
        
        # logger 속성 확인
        has_logger = hasattr(env, 'logger')
        logger.info(f"logger 속성 존재 여부: {has_logger}")
        
        if has_logger:
            logger.info("logger가 올바른 타입인지 확인...")
            is_logger_type = isinstance(env.logger, logging.Logger)
            logger.info(f"logger의 타입이 logging.Logger인지: {is_logger_type}")
            
            # logger 사용 테스트
            try:
                env.logger.info("이것은 환경 로거의 테스트 메시지입니다.")
                logger.info("환경 로거를 통한 로깅 성공!")
            except Exception as e:
                logger.error(f"환경 로거 사용 중 예외 발생: {e}")
        else:
            logger.error("환경 객체에 logger 속성이 없습니다.")
        
        # _convert_df_to_dfs 메서드 테스트
        logger.info("_convert_df_to_dfs 메서드 테스트 중...")
        try:
            result = env._convert_df_to_dfs(data, assets)
            logger.info(f"_convert_df_to_dfs 메서드 실행 성공! 반환된 딕셔너리 키: {list(result.keys())}")
        except Exception as e:
            logger.error(f"_convert_df_to_dfs 메서드 실행 중 예외 발생: {e}")
        
        # 환경 리셋 테스트
        logger.info("환경 리셋 테스트 중...")
        try:
            obs, info = env.reset()
            logger.info(f"환경 리셋 성공! 관측 형태: {obs.shape}, 정보: {info}")
        except Exception as e:
            logger.error(f"환경 리셋 중 예외 발생: {e}")
        
    except Exception as e:
        logger.error(f"환경 객체 생성 중 예외 발생: {e}")

if __name__ == "__main__":
    logger.info("MultiAssetTradingEnv 테스트 시작")
    test_multi_asset_env_logger()
    logger.info("MultiAssetTradingEnv 테스트 완료") 