import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from datetime import datetime
import gymnasium as gym
from typing import Dict, List, Any, Tuple
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_asset_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_multi_asset_env(env, config):
    """다중 자산 환경 검증"""
    try:
        # 기본 속성 검증
        assert hasattr(env, 'observation_space'), "Environment missing observation_space"
        assert hasattr(env, 'action_space'), "Environment missing action_space"
        assert hasattr(env, 'reset'), "Environment missing reset method"
        assert hasattr(env, 'step'), "Environment missing step method"
        
        # 환경 속성 확인
        assert hasattr(env, 'assets'), "Environment missing assets attribute"
        assert len(env.assets) > 1, f"Expected multiple assets, got {len(env.assets)}"
        
        logger.info(f"Environment has {len(env.assets)} assets: {env.assets}")
        
        # 초기 상태 검증
        obs, info = env.reset()
        
        # observation 검증
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        expected_first_dim = config['env']['window_size']
        assert obs.shape[0] == expected_first_dim, f"Expected first dim {expected_first_dim}, got {obs.shape[0]}"
        
        # action space 검증
        logger.info(f"Action space: {env.action_space}")
        assert isinstance(env.action_space, gym.spaces.Box), f"Action space should be Box, got {type(env.action_space)}"
        assert env.action_space.shape[0] == len(env.assets), f"Expected action shape ({len(env.assets)},), got {env.action_space.shape}"
        
        # 스텝 실행 검증
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 결과 검증
        assert isinstance(next_obs, np.ndarray), "Next observation should be numpy array"
        assert next_obs.shape == obs.shape, f"Observation shape changed: {obs.shape} -> {next_obs.shape}"
        assert isinstance(reward, float), "Reward should be float"
        
        # done과 truncated 타입 확인 및 변환
        logger.info(f"Done type: {type(done)}")
        logger.info(f"Truncated type: {type(truncated)}")
        done = bool(done)
        truncated = bool(truncated)
        
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        
        # 포트폴리오 정보 확인 
        assert 'portfolio_value' in info, "Missing portfolio value in info"
        assert 'positions' in info, "Missing positions in info"
        assert len(info['positions']) == len(env.assets), f"Expected positions for {len(env.assets)} assets, got {len(info['positions'])}"
        
        logger.info("Multi-asset environment validation passed!")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Initial observation shape: {obs.shape}")
        logger.info(f"Sample action: {action}")
        logger.info(f"Sample reward: {reward}")
        logger.info(f"Info keys: {list(info.keys())}")
        logger.info(f"Portfolio value: {info['portfolio_value']}")
        logger.info(f"Positions: {info['positions']}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Multi-asset environment validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during multi-asset environment validation: {str(e)}", exc_info=True)
        return False

def validate_agent(agent, env):
    """다중 자산 에이전트 검증"""
    try:
        # 기본 속성 검증
        assert hasattr(agent, 'get_action'), "Agent missing get_action method"
        assert hasattr(agent, 'train_step'), "Agent missing train_step method"
        
        # 액션 생성 검증
        obs, _ = env.reset()
        action = agent.get_action(obs)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape[0] == len(env.assets), f"Expected action shape ({len(env.assets)},), got {action.shape}"
        assert env.action_space.contains(action), f"Action {action} not in action space {env.action_space}"
        
        logger.info(f"Agent action shape: {action.shape}, values: {action}")
        
        # 스텝 실행
        next_obs, reward, done, truncated, info = env.step(action)
        
        # bool로 강제 변환
        done = bool(done)
        truncated = bool(truncated)
        
        # 학습 스텝 검증
        loss_dict = agent.train_step(obs, action, reward, next_obs, done or truncated)
        
        assert isinstance(loss_dict, dict), "train_step should return dict of losses"
        assert all(isinstance(v, float) for v in loss_dict.values()), "All losses should be float"
        assert all(np.isfinite(v) for v in loss_dict.values()), "All losses should be finite"
        
        logger.info("Agent validation passed!")
        logger.info(f"Sample action: {action}")
        logger.info(f"Sample losses: {loss_dict}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Agent validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during agent validation: {str(e)}", exc_info=True)
        return False

def validate_training_step(agent, env, iterations=5):
    """여러 트레이닝 스텝 실행 검증"""
    try:
        observations, _ = env.reset()
        
        logger.info(f"Starting {iterations} training iterations...")
        
        for i in range(iterations):
            logger.info(f"Training iteration {i+1}/{iterations}")
            
            # 액션 생성
            action = agent.get_action(observations)
            logger.info(f"Action for all assets: {action}")
            
            # 환경 스텝 실행
            next_obs, reward, done, truncated, info = env.step(action)
            
            # bool로 강제 변환
            done = bool(done)
            truncated = bool(truncated)
            
            # 학습
            loss_dict = agent.train_step(observations, action, reward, next_obs, done or truncated)
            logger.info(f"Iteration {i+1} losses: {loss_dict}")
            logger.info(f"Portfolio value: {info['portfolio_value']}")
            
            # 에피소드가 끝났는지 확인
            if done or truncated:
                logger.info(f"Episode complete after {i+1} steps")
                observations, _ = env.reset()
            else:
                observations = next_obs
        
        logger.info("Training step validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Training step validation failed: {str(e)}", exc_info=True)
        return False

def load_multi_asset_config():
    """다중 자산 설정 로드 및 변경"""
    # 기본 설정 파일 로드 (일반적으로 사용)
    config_path = os.path.join(os.path.dirname(__file__), "config", "default_config.yaml")
    logger.info(f"Loading base config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 다중 자산 설정으로 수정
    config['env']['type'] = 'multi_asset_rl'
    
    # 여러 자산 추가
    config['data']['symbols'] = ["BTC/USDT", "ETH/USDT"]
    
    # 필수 키 검증
    required_keys = ['env', 'data', 'model', 'training', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")
    
    logger.info(f"Modified config to use multi-asset environment with symbols: {config['data']['symbols']}")
    return config

def prepare_multiple_asset_data(config):
    """여러 자산의 데이터 준비"""
    from data.utils.data_loader import DataLoader
    
    symbols = config['data']['symbols']
    combined_df = None
    
    for symbol in symbols:
        data_loader = DataLoader(
            exchange_id=config['data']['exchange'],
            symbol=symbol,
            timeframe=config['data']['timeframe']
        )
        
        logger.info(f"Fetching data for {symbol}...")
        data = data_loader.fetch_data(
            start_date=config['data']['start_date'],
            end_date=config['data'].get('end_date', datetime.now().strftime("%Y-%m-%d"))
        )
        
        logger.info(f"Data for {symbol} loaded with shape: {data.shape}")
        
        # 자산 심볼로 컬럼 이름 수정
        # 예: $open -> BTC_$open (멀티에셋 환경에서 기대하는 형식)
        symbol_id = symbol.split('/')[0]  # BTC/USDT -> BTC
        renamed_columns = {col: f"{symbol_id}_{col}" for col in data.columns}
        data = data.rename(columns=renamed_columns)
        
        # 합쳐진 DataFrame이 없으면 첫 번째 자산으로 초기화
        if combined_df is None:
            combined_df = data
        else:
            # 컬럼을 추가하여 기존 DataFrame에 병합
            for col in data.columns:
                combined_df[col] = data[col]
    
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    logger.info(f"Combined DataFrame columns: {combined_df.columns.tolist()}")
    
    return combined_df

def main():
    try:
        # 다중 자산 설정 로드
        config = load_multi_asset_config()
        
        # 여러 자산 데이터 준비
        asset_data = prepare_multiple_asset_data(config)
        
        # 환경 생성
        logger.info("Creating multi-asset environment...")
        from training.env_factory import create_env
        from envs.multi_asset_env import MultiAssetTradingEnv
        
        # 환경 생성
        env = create_env(config, asset_data)
        
        # 환경 타입 확인
        assert isinstance(env, MultiAssetTradingEnv), f"Expected MultiAssetTradingEnv, got {type(env)}"
        
        # 환경 검증
        if not validate_multi_asset_env(env, config):
            logger.error("Multi-asset environment validation failed!")
            return
        
        # 에이전트 생성
        logger.info("Creating agent...")
        from agents.strategies.agent_factory import create_agent
        
        agent = create_agent(
            agent_type=config.get('agent_type', 'ppo'),
            strategy=config.get('strategy', None),
            config=config.get('model', {}),
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # 에이전트 검증
        if not validate_agent(agent, env):
            logger.error("Agent validation failed!")
            return
        
        # 트레이닝 스텝 검증 (여러 스텝 실행)
        if not validate_training_step(agent, env, iterations=10):
            logger.error("Training step validation failed!")
            return
        
        logger.info("Multi-asset training validation completed successfully!")

    except Exception as e:
        logger.error(f"Error during multi-asset validation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 