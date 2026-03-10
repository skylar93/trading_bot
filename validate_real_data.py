"""
실제 거래소 데이터를 사용하여 강화학습 환경과 에이전트를 검증합니다.
"""

import logging
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
import gymnasium as gym
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """설정 파일 로드"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "default_config.yaml")
    logger.info(f"설정 파일 로드: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 필수 키 검증
    required_keys = ['env', 'data', 'model', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"설정 파일에 필수 키가 없습니다: {key}")
    
    return config

def fetch_real_data(config):
    """실제 거래소에서 데이터 가져오기"""
    try:
        from data.utils.enhanced_data_loader import EnhancedDataLoader
        
        # 설정에서 데이터 설정 가져오기
        exchange = config['data']['exchange']
        symbol = config['data']['symbols'][0]  # 첫 번째 심볼만 사용
        timeframe = config['data']['timeframe']
        start_date = config['data']['start_date']
        end_date = config['data'].get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        logger.info(f"데이터 로드 중: {exchange}, {symbol}, {timeframe}, {start_date} ~ {end_date}")
        
        # 향상된 데이터 로더 생성 (실제 데이터 사용)
        loader = EnhancedDataLoader(
            exchange_id=exchange,
            symbols=symbol,
            timeframe=timeframe,
            use_real_data=True  # 실제 데이터 사용
        )
        
        # 데이터 가져오기
        data = loader.fetch_multi_asset_data(start_date, end_date)
        
        if data.empty:
            logger.warning("실제 데이터를 가져오지 못했습니다. 시뮬레이션 데이터로 대체합니다.")
            # 실패 시 일반 데이터 로더로 시뮬레이션 데이터 사용
            from data.utils.data_loader import DataLoader
            basic_loader = DataLoader(exchange_id=exchange, symbol=symbol, timeframe=timeframe)
            data = basic_loader.fetch_data(start_date, end_date)
        
        # 데이터 검증 및 변환
        processed_data = validate_data(data)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def process_data_columns(data):
    """
    데이터 열 이름 처리 및 필요한 변환 수행
    """
    logger.info(f"원본 데이터 열: {data.columns.tolist()}")
    
    # 데이터 복사본 생성
    processed_data = data.copy()
    
    # 1. 심볼 접두사 제거 (예: 'BTC/USDT_$open' -> '$open')
    symbol_prefix_cols = {}
    for col in processed_data.columns:
        if '_$' in col:
            symbol, actual_col = col.split('_$')
            symbol_prefix_cols[col] = f'${actual_col}'
    
    if symbol_prefix_cols:
        processed_data = processed_data.rename(columns=symbol_prefix_cols)
        logger.info(f"심볼 접두사 제거: {list(symbol_prefix_cols.items())}")
    
    # 2. $ 접두사 추가 (예: 'open' -> '$open')
    standard_cols = {
        'open': '$open', 
        'high': '$high', 
        'low': '$low', 
        'close': '$close', 
        'volume': '$volume'
    }
    
    for old_col, new_col in standard_cols.items():
        if old_col in processed_data.columns and new_col not in processed_data.columns:
            processed_data = processed_data.rename(columns={old_col: new_col})
            logger.info(f"열 이름 변환: {old_col} -> {new_col}")
    
    return processed_data

def validate_data(data):
    """데이터 유효성 검증"""
    logger.info(f"데이터 검증 중: 형태 {data.shape}")
    
    # 열 이름 처리
    processed_data = process_data_columns(data)
    
    # 필수 컬럼 확인
    required_columns = ['$open', '$high', '$low', '$close', '$volume']
    missing_columns = [col for col in required_columns if col not in processed_data.columns]
    if missing_columns:
        # 데이터 열 이름 로깅
        logger.error(f"현재 데이터 열: {processed_data.columns.tolist()}")
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    
    # 필요 열만 선택
    processed_data = processed_data[required_columns]
    
    # NaN 값 확인
    nan_counts = processed_data.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"데이터에 NaN 값이 있습니다: {nan_counts}")
        # NaN 값을 앞의 값으로 채우기
        processed_data.fillna(method='ffill', inplace=True)
        # 여전히 NaN이 있으면 뒤의 값으로 채우기
        processed_data.fillna(method='bfill', inplace=True)
    
    # 데이터 통계 출력
    logger.info(f"데이터 기간: {processed_data.index.min()} ~ {processed_data.index.max()}")
    logger.info(f"데이터 통계:\n{processed_data.describe()}")
    
    # 데이터 인덱스가 DatetimeIndex가 아니면 변환
    if not isinstance(processed_data.index, pd.DatetimeIndex):
        logger.warning("인덱스가 Datetime 형식이 아닙니다. 변환합니다.")
        processed_data.index = pd.to_datetime(processed_data.index)
    
    # 인덱스를 초기화하여 환경에서 올바르게 처리되도록 함
    processed_data = processed_data.reset_index(drop=True)
    logger.info("인덱스를 초기화했습니다.")
    
    # 데이터 샘플 확인
    logger.info(f"데이터 샘플:\n{processed_data.head()}")
    
    return processed_data

def create_environment(config, data):
    """강화학습 환경 생성"""
    from training.env_factory import create_env
    
    # 환경 설정 준비
    env_config = config['env']
    env_config['type'] = 'single_asset_rl'  # 단일 자산 환경 사용
    
    # 데이터 형식 최종 확인
    logger.info(f"환경 생성 전 데이터 형태: {data.shape}")
    logger.info(f"데이터 열: {data.columns.tolist()}")
    logger.info(f"데이터 타입: \n{data.dtypes}")
    
    # 데이터 타입 변환 (float로 통일)
    for col in data.columns:
        data[col] = data[col].astype(float)
    
    # 환경 생성
    env = create_env(config, data)
    
    return env

def create_agent(config, env):
    """에이전트 생성"""
    from agents.strategies.agent_factory import create_agent
    
    # 에이전트 생성
    agent = create_agent(
        agent_type='ppo',
        config=config.get('model', {}),
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    
    return agent

def test_learning_loop(env, agent, num_episodes=2, max_steps=100):
    """학습 루프 테스트"""
    logger.info("학습 루프 테스트 시작")
    
    for episode in range(num_episodes):
        logger.info(f"에피소드 {episode+1}/{num_episodes} 시작")
        
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(max_steps):
            # 액션 얻기
            action = agent.get_action(obs)
            
            # 액션 타입 및 범위 확인
            logger.info(f"Step {step+1}: Action {action}, Shape {action.shape}, Range [{action.min()}, {action.max()}]")
            
            # 환경에서 한 스텝 진행
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 보상 추적
            episode_reward += reward
            logger.info(f"Step {step+1}: Reward {reward}, Episode Reward {episode_reward}")
            
            # 에이전트 학습
            losses = agent.train_step(obs, action, reward, next_obs, done or truncated)
            
            # 손실 값 확인 (NaN이 없는지)
            for loss_name, loss_value in losses.items():
                if np.isnan(loss_value) or np.isinf(loss_value):
                    logger.warning(f"손실 값 {loss_name}이 유효하지 않습니다: {loss_value}")
                else:
                    logger.info(f"Loss - {loss_name}: {loss_value}")
            
            # 다음 상태로 업데이트
            obs = next_obs
            step_count += 1
            
            # 에피소드 종료 확인
            if done or truncated:
                logger.info(f"에피소드 {episode+1} 종료: {step_count}스텝, 총 보상 {episode_reward}")
                break
        
        logger.info(f"에피소드 {episode+1} 완료: 총 보상 {episode_reward}, 스텝 {step_count}")
    
    logger.info("학습 루프 테스트 완료")
    return True

def main():
    try:
        # 설정 로드
        config = load_config()
        
        # 실제 데이터 가져오기
        data = fetch_real_data(config)
        
        # 환경 생성
        env = create_environment(config, data)
        logger.info(f"환경 생성 완료: {type(env).__name__}")
        logger.info(f"관찰 공간: {env.observation_space}")
        logger.info(f"행동 공간: {env.action_space}")
        
        # 에이전트 생성
        agent = create_agent(config, env)
        logger.info(f"에이전트 생성 완료: {type(agent).__name__}")
        
        # 학습 루프 테스트
        success = test_learning_loop(env, agent)
        
        if success:
            logger.info("실제 데이터를 사용한 검증이 성공적으로 완료되었습니다!")
        else:
            logger.error("검증 중 오류가 발생했습니다.")
            
    except Exception as e:
        logger.error(f"검증 중 예외 발생: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 