import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from datetime import datetime
import gymnasium as gym

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_environment(env, config):
    """환경 검증"""
    try:
        # 기본 속성 검증
        assert hasattr(env, 'observation_space'), "Environment missing observation_space"
        assert hasattr(env, 'action_space'), "Environment missing action_space"
        assert hasattr(env, 'reset'), "Environment missing reset method"
        assert hasattr(env, 'step'), "Environment missing step method"
        
        # 초기 상태 검증
        obs, info = env.reset()
        
        # observation 검증
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape[0] == config['env']['window_size'], f"Expected first dim {config['env']['window_size']}, got {obs.shape[0]}"
        
        # action space 검증
        logger.info(f"Action space MRO: {type(env.action_space).__mro__}")
        assert isinstance(env.action_space, gym.spaces.Box), f"Action space should be Box, got {type(env.action_space)}"
        assert env.action_space.shape == (1,), f"Expected action shape (1,), got {env.action_space.shape}"
        assert env.action_space.low[0] == -1.0, f"Expected action low -1.0, got {env.action_space.low[0]}"
        assert env.action_space.high[0] == 1.0, f"Expected action high 1.0, got {env.action_space.high[0]}"
        
        # 스텝 실행 검증
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray), "Next observation should be numpy array"
        assert next_obs.shape == obs.shape, f"Observation shape changed: {obs.shape} -> {next_obs.shape}"
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        
        logger.info("Environment validation passed!")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Initial observation shape: {obs.shape}")
        logger.info(f"Sample action: {action}")
        logger.info(f"Sample reward: {reward}")
        logger.info(f"Info keys: {list(info.keys())}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Environment validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during environment validation: {str(e)}")
        return False

def validate_agent(agent, env):
    """에이전트 검증"""
    try:
        # 기본 속성 검증
        assert hasattr(agent, 'get_action'), "Agent missing get_action method"
        assert hasattr(agent, 'train_step'), "Agent missing train_step method"
        
        # 액션 생성 검증
        obs, _ = env.reset()
        action = agent.get_action(obs)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (1,), f"Expected action shape (1,), got {action.shape}"
        assert env.action_space.contains(action), f"Action {action} not in action space {env.action_space}"
        
        # 학습 스텝 검증
        next_obs, reward, done, truncated, _ = env.step(action)
        loss_dict = agent.train_step(obs, action, reward, next_obs, done or truncated)
        
        assert isinstance(loss_dict, dict), "train_step should return dict of losses"
        assert all(isinstance(v, float) for v in loss_dict.values()), "All losses should be float"
        assert all(np.isfinite(v) for v in loss_dict.values()), "All losses should be finite"
        
        logger.info("Agent validation passed!")
        logger.info(f"Sample action from agent: {action}")
        logger.info(f"Sample losses: {loss_dict}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Agent validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during agent validation: {str(e)}")
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
            end_date=config['data'].get('end_date', datetime.now().strftime("%Y-%m-%d"))
        )

        # 환경 생성
        logger.info("Creating environment...")
        from training.env_factory import create_env
        
        # env_type을 env 설정에 추가
        env_config = config['env']
        env_config['type'] = 'single_asset_rl'
        config['env'] = env_config
        
        # 기본 환경 생성 (래퍼 없이)
        env = create_env(config, data)
        
        # 환경 생성 직후 action_space 확인
        logger.info(f"Action space type: {type(env.action_space)}")
        logger.info(f"Action space: {env.action_space}")
        
        # 환경 검증
        if not validate_environment(env, config):
            logger.error("Environment validation failed!")
            return

        # 에이전트 생성
        logger.info("Creating agent...")
        from agents.strategies.agent_factory import create_agent
        
        agent = create_agent(
            agent_type='momentum_ppo',
            config=config.get('model', {}),
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # 에이전트 검증
        if not validate_agent(agent, env):
            logger.error("Agent validation failed!")
            return
            
        logger.info("Environment and agent validation completed successfully!")

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 