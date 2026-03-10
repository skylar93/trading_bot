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
        logging.FileHandler("multi_agent_multi_asset_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_multi_agent_multi_asset_env(env, config):
    """다중 에이전트, 다중 자산 환경 검증"""
    try:
        # 기본 속성 검증
        assert hasattr(env, 'reset'), "Environment missing reset method"
        assert hasattr(env, 'step'), "Environment missing step method"
        
        # 환경 속성 확인
        assert hasattr(env, 'agents'), "Environment missing agents attribute"
        assert len(env.agents) > 1, f"Expected multiple agents, got {len(env.agents)}"
        assert hasattr(env, 'assets'), "Environment missing assets attribute"
        assert len(env.assets) > 1, f"Expected multiple assets, got {len(env.assets)}"
        
        logger.info(f"Environment has {len(env.agents)} agents: {env.agents}")
        logger.info(f"Environment has {len(env.assets)} assets: {env.assets}")
        
        # 환경 스페이스 확인
        for agent_id in env.agents:
            assert agent_id in env.observation_spaces, f"Missing observation space for agent {agent_id}"
            assert agent_id in env.action_spaces, f"Missing action space for agent {agent_id}"
            
            # 각 에이전트의 액션 스페이스는 자산 수에 맞게 설정되어야 함
            action_space = env.action_spaces[agent_id]
            logger.info(f"Agent {agent_id} action space: {action_space}")
            assert isinstance(action_space, gym.spaces.Box), f"Action space should be Box, got {type(action_space)}"
            assert action_space.shape[0] == len(env.assets), f"Expected action shape ({len(env.assets)},), got {action_space.shape}"
        
        # 초기 상태 검증
        observations, info = env.reset()
        
        # observations 구조 검증
        assert isinstance(observations, dict), "Observations should be a dictionary"
        for agent_id in env.agents:
            assert agent_id in observations, f"Missing observation for agent {agent_id}"
            assert isinstance(observations[agent_id], np.ndarray), f"Observation for {agent_id} should be numpy array"
            expected_first_dim = config['env']['window_size']
            assert observations[agent_id].shape[0] == expected_first_dim, f"Expected first dim {expected_first_dim}, got {observations[agent_id].shape[0]}"
        
        # 스텝 실행 검증
        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = env.action_spaces[agent_id].sample()
        
        next_obs, rewards, dones, truncateds, infos = env.step(actions)
        
        # 결과 검증
        assert isinstance(next_obs, dict), "Next observations should be a dictionary"
        assert isinstance(rewards, dict), "Rewards should be a dictionary"
        assert isinstance(dones, dict), "Dones should be a dictionary"
        assert isinstance(truncateds, dict), "Truncateds should be a dictionary"
        assert isinstance(infos, dict), "Infos should be a dictionary"
        
        for agent_id in env.agents:
            assert agent_id in next_obs, f"Missing next observation for agent {agent_id}"
            assert agent_id in rewards, f"Missing reward for agent {agent_id}"
            assert agent_id in dones, f"Missing done for agent {agent_id}"
            assert agent_id in truncateds, f"Missing truncated for agent {agent_id}"
            assert isinstance(rewards[agent_id], float), f"Reward for {agent_id} should be float"
            
            # done과 truncated 타입 확인 및 변환
            logger.info(f"Done type for {agent_id}: {type(dones[agent_id])}")
            logger.info(f"Truncated type for {agent_id}: {type(truncateds[agent_id])}")
            
            # bool로 강제 변환
            dones[agent_id] = bool(dones[agent_id])
            truncateds[agent_id] = bool(truncateds[agent_id])
            
            # 에이전트 정보 확인
            assert 'portfolio_value' in infos[agent_id], f"Missing portfolio value for agent {agent_id}"
            assert 'positions' in infos[agent_id], f"Missing positions for agent {agent_id}"
            assert len(infos[agent_id]['positions']) == len(env.assets), f"Expected positions for {len(env.assets)} assets, got {len(infos[agent_id]['positions'])}"
        
        logger.info("Multi-agent multi-asset environment validation passed!")
        for agent_id in env.agents:
            logger.info(f"Agent {agent_id} - Observation space: {env.observation_spaces[agent_id]}")
            logger.info(f"Agent {agent_id} - Action space: {env.action_spaces[agent_id]}")
            logger.info(f"Agent {agent_id} - Initial observation shape: {observations[agent_id].shape}")
            logger.info(f"Agent {agent_id} - Sample action: {actions[agent_id]}")
            logger.info(f"Agent {agent_id} - Sample reward: {rewards[agent_id]}")
            logger.info(f"Agent {agent_id} - Portfolio value: {infos[agent_id]['portfolio_value']}")
            logger.info(f"Agent {agent_id} - Positions: {infos[agent_id]['positions']}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Multi-agent multi-asset environment validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during multi-agent multi-asset environment validation: {str(e)}", exc_info=True)
        return False

def validate_multi_agents(agents, env):
    """여러 에이전트 검증"""
    try:
        # 기본 속성 검증
        for agent_id, agent in agents.items():
            assert hasattr(agent, 'get_action'), f"Agent {agent_id} missing get_action method"
            assert hasattr(agent, 'train_step'), f"Agent {agent_id} missing train_step method"
        
        # 초기 상태 얻기
        observations, _ = env.reset()
        
        # 각 에이전트의 액션 생성 검증
        actions = {}
        for agent_id, agent in agents.items():
            action = agent.get_action(observations[agent_id])
            actions[agent_id] = action
            
            assert isinstance(action, np.ndarray), f"Action from {agent_id} should be numpy array"
            assert action.shape[0] == len(env.assets), f"Expected action shape ({len(env.assets)},), got {action.shape}"
            assert env.action_spaces[agent_id].contains(action), f"Action {action} not in action space {env.action_spaces[agent_id]}"
            
            logger.info(f"Agent {agent_id} action shape: {action.shape}, value: {action}")
        
        # 환경 스텝 실행
        next_obs, rewards, dones, truncateds, infos = env.step(actions)
        
        # bool로 강제 변환
        for agent_id in agents.keys():
            dones[agent_id] = bool(dones[agent_id])
            truncateds[agent_id] = bool(truncateds[agent_id])
        
        # 각 에이전트의 학습 스텝 검증
        for agent_id, agent in agents.items():
            try:
                loss_dict = agent.train_step(
                    observations[agent_id], 
                    actions[agent_id], 
                    rewards[agent_id], 
                    next_obs[agent_id], 
                    dones[agent_id] or truncateds[agent_id]
                )
                
                assert isinstance(loss_dict, dict), f"train_step for {agent_id} should return dict of losses"
                assert all(isinstance(v, float) for v in loss_dict.values()), f"All losses for {agent_id} should be float"
                assert all(np.isfinite(v) for v in loss_dict.values()), f"All losses for {agent_id} should be finite"
                
                logger.info(f"Agent {agent_id} train_step passed, losses: {loss_dict}")
            except Exception as e:
                logger.error(f"Error in agent {agent_id} train_step: {str(e)}", exc_info=True)
                return False
        
        logger.info("All agents validation passed!")
        return True
        
    except AssertionError as e:
        logger.error(f"Multi-agent validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during multi-agent validation: {str(e)}", exc_info=True)
        return False

def validate_training_step(agents, env, iterations=5):
    """여러 트레이닝 스텝 실행 검증"""
    try:
        observations, _ = env.reset()
        
        logger.info(f"Starting {iterations} training iterations...")
        
        for i in range(iterations):
            logger.info(f"Training iteration {i+1}/{iterations}")
            
            # 각 에이전트의 액션 생성
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.get_action(observations[agent_id])
                logger.info(f"Agent {agent_id} action: {actions[agent_id]}")
            
            # 환경 스텝 실행
            next_obs, rewards, dones, truncateds, infos = env.step(actions)
            
            # bool로 강제 변환
            for agent_id in agents.keys():
                dones[agent_id] = bool(dones[agent_id])
                truncateds[agent_id] = bool(truncateds[agent_id])
            
            # 각 에이전트 학습
            losses = {}
            for agent_id, agent in agents.items():
                loss_dict = agent.train_step(
                    observations[agent_id], 
                    actions[agent_id], 
                    rewards[agent_id], 
                    next_obs[agent_id], 
                    dones[agent_id] or truncateds[agent_id]
                )
                losses[agent_id] = loss_dict
                logger.info(f"Agent {agent_id} portfolio value: {infos[agent_id]['portfolio_value']}")
            
            logger.info(f"Iteration {i+1} losses: {losses}")
            
            # 에피소드가 끝났는지 확인
            if all(dones.values()):
                logger.info(f"Episode complete after {i+1} steps")
                observations, _ = env.reset()
            else:
                observations = next_obs
        
        logger.info("Training step validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Training step validation failed: {str(e)}", exc_info=True)
        return False

def load_multi_agent_multi_asset_config():
    """다중 에이전트, 다중 자산 설정 로드"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "multi_agent_config.yaml")
    logger.info(f"Loading multi-agent config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 타입 변경
    config['env']['type'] = 'multi_asset_multi_agent_rl'
    
    # 여러 자산 추가
    config['data']['symbols'] = ["BTC/USDT", "ETH/USDT"]
    
    # 필수 키 검증
    required_keys = ['env', 'data', 'training', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")
    
    # 에이전트 설정 확인
    env_config = config['env']
    assert 'multi_agent_configs' in env_config, "Multi-agent configs missing"
    assert isinstance(env_config['multi_agent_configs'], list), "multi_agent_configs should be a list"
    assert len(env_config['multi_agent_configs']) > 0, "multi_agent_configs is empty"
    
    logger.info(f"Modified config to use multi-agent multi-asset environment")
    logger.info(f"Assets: {config['data']['symbols']}")
    logger.info(f"Agents: {[agent['id'] for agent in config['env']['multi_agent_configs']]}")
    
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
        # 다중 에이전트, 다중 자산 설정 로드
        config = load_multi_agent_multi_asset_config()
        
        # 여러 자산 데이터 준비
        asset_data = prepare_multiple_asset_data(config)
        
        # 환경 생성
        logger.info("Creating multi-agent multi-asset environment...")
        from training.env_factory import create_env
        from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv
        
        # 환경 생성
        env = create_env(config, asset_data)
        
        # 환경 타입 확인
        assert isinstance(env, MultiAgentMultiAssetEnv), f"Expected MultiAgentMultiAssetEnv, got {type(env)}"
        
        # 환경 검증
        if not validate_multi_agent_multi_asset_env(env, config):
            logger.error("Multi-agent multi-asset environment validation failed!")
            return
        
        # 에이전트 생성
        logger.info("Creating agents...")
        from agents.strategies.agent_factory import create_agent
        
        agents = {}
        for agent_config in config['env']['multi_agent_configs']:
            agent_id = agent_config['id']
            agent_type = agent_config['agent_type']
            strategy = agent_config.get('strategy')
            
            observation_space = env.observation_spaces[agent_id]
            action_space = env.action_spaces[agent_id]
            
            logger.info(f"Creating agent {agent_id} with type {agent_type}, strategy {strategy}")
            logger.info(f"Observation space: {observation_space}")
            logger.info(f"Action space: {action_space}")
            
            agent = create_agent(
                agent_type=agent_type,
                strategy=strategy,
                config=agent_config.get('hyperparameters', {}),
                observation_space=observation_space,
                action_space=action_space
            )
            
            agents[agent_id] = agent
        
        # 에이전트 검증
        if not validate_multi_agents(agents, env):
            logger.error("Multi-agent validation failed!")
            return
        
        # 트레이닝 스텝 검증 (여러 스텝 실행)
        if not validate_training_step(agents, env, iterations=10):
            logger.error("Training step validation failed!")
            return
        
        logger.info("Multi-agent multi-asset training validation completed successfully!")

    except Exception as e:
        logger.error(f"Error during multi-agent multi-asset validation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 