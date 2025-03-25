import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from datetime import datetime
import gymnasium as gym
from typing import Dict, List, Any
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_agent_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_multi_agent_env(env, config):
    """멀티 에이전트 환경 검증"""
    try:
        # 기본 속성 검증
        assert hasattr(env, 'reset'), "Environment missing reset method"
        assert hasattr(env, 'step'), "Environment missing step method"
        
        # 환경 속성 확인
        for agent_id in env.agents:
            assert agent_id in env.observation_spaces, f"Missing observation space for agent {agent_id}"
            assert agent_id in env.action_spaces, f"Missing action space for agent {agent_id}"
        
        # 초기 상태 검증
        observations, info = env.reset()
        
        # observations 구조 검증
        assert isinstance(observations, dict), "Observations should be a dictionary"
        for agent_id in env.agents:
            assert agent_id in observations, f"Missing observation for agent {agent_id}"
            assert isinstance(observations[agent_id], np.ndarray), f"Observation for {agent_id} should be numpy array"
            expected_shape = (config['env']['window_size'], -1)  # window_size x features
            assert observations[agent_id].shape[0] == expected_shape[0], f"Expected first dim {expected_shape[0]}, got {observations[agent_id].shape[0]}"
        
        # action space 검증
        for agent_id in env.agents:
            action_space = env.action_spaces[agent_id]
            logger.info(f"Agent {agent_id} action space: {action_space}")
            assert isinstance(action_space, gym.spaces.Box), f"Action space should be Box, got {type(action_space)}"
            assert action_space.shape == (1,), f"Expected action shape (1,), got {action_space.shape}"
            assert action_space.low[0] == -1.0, f"Expected action low -1.0, got {action_space.low[0]}"
            assert action_space.high[0] == 1.0, f"Expected action high 1.0, got {action_space.high[0]}"
        
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
            
            # done과 truncated 타입을 직접 로깅하고 확인
            logger.info(f"Done type for {agent_id}: {type(dones[agent_id])}")
            logger.info(f"Truncated type for {agent_id}: {type(truncateds[agent_id])}")
            
            # bool로 강제 변환
            dones[agent_id] = bool(dones[agent_id])
            truncateds[agent_id] = bool(truncateds[agent_id])
        
        logger.info("Multi-agent environment validation passed!")
        logger.info(f"Agents: {env.agents}")
        for agent_id in env.agents:
            logger.info(f"Agent {agent_id} - Observation space: {env.observation_spaces[agent_id]}")
            logger.info(f"Agent {agent_id} - Action space: {env.action_spaces[agent_id]}")
            logger.info(f"Agent {agent_id} - Initial observation shape: {observations[agent_id].shape}")
            logger.info(f"Agent {agent_id} - Sample action: {actions[agent_id]}")
            logger.info(f"Agent {agent_id} - Sample reward: {rewards[agent_id]}")
        
        return True
        
    except AssertionError as e:
        logger.error(f"Multi-agent environment validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during multi-agent environment validation: {str(e)}", exc_info=True)
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
            assert action.shape == (1,), f"Expected action shape (1,), got {action.shape}"
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

def load_multi_agent_config():
    """멀티 에이전트 설정 로드"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "multi_agent_config.yaml")
    logger.info(f"Loading multi-agent config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 필수 키 검증
    required_keys = ['env', 'data', 'training', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")
    
    # 환경 설정 검증
    env_config = config['env']
    assert 'multi_agent_configs' in env_config, "Multi-agent configs missing"
    assert isinstance(env_config['multi_agent_configs'], list), "multi_agent_configs should be a list"
    assert len(env_config['multi_agent_configs']) > 0, "multi_agent_configs is empty"
    
    return config

def main():
    try:
        # 멀티 에이전트 설정 로드
        config = load_multi_agent_config()
        
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
        
        # 데이터 확인
        logger.info(f"Data loaded with shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data preview:\n{data.head()}")
        
        # 멀티 에이전트 환경 생성
        logger.info("Creating multi-agent environment...")
        from training.env_factory import create_env
        from envs.multi_agent_env import MultiAgentTradingEnv
        
        # env_type을 multi_agent_rl로 설정
        config['env']['type'] = 'multi_agent_rl'
        
        # 멀티 에이전트 환경 생성
        env = create_env(config, data)
        
        # 환경 타입 확인
        assert isinstance(env, MultiAgentTradingEnv), f"Expected MultiAgentTradingEnv, got {type(env)}"
        
        # 환경 검증
        if not validate_multi_agent_env(env, config):
            logger.error("Multi-agent environment validation failed!")
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
        
        logger.info("Multi-agent training validation completed successfully!")

    except Exception as e:
        logger.error(f"Error during multi-agent validation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 