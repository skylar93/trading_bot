#!/usr/bin/env python
"""
디버깅 스크립트: DummyAgent 클래스의 continuous_ensemble 속성 문제 해결

이 스크립트는 DummyAgent 클래스에 continuous_ensemble 속성이 있는지 확인하고,
multi_agent_manager에서 이 속성을 사용하는 부분을 테스트합니다.
"""

import sys
import os
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from agents.strategies.single.dummy_agent import DummyAgent
    logger.info("DummyAgent 클래스 성공적으로 임포트됨")
except ImportError as e:
    logger.error(f"DummyAgent 클래스 임포트 실패: {e}")
    sys.exit(1)

def test_dummy_agent_attributes():
    """DummyAgent 클래스의 속성 테스트"""
    
    # 간단한 observation space와 action space 생성
    observation_dim = 10
    action_dim = 1
    
    observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(observation_dim,), 
        dtype=np.float32
    )
    
    action_space = spaces.Box(
        low=-1.0, 
        high=1.0, 
        shape=(action_dim,), 
        dtype=np.float32
    )
    
    # 일반 에이전트 생성
    logger.info("일반 DummyAgent 생성 시도...")
    dummy_agent = DummyAgent(
        observation_space=observation_space,
        action_space=action_space
    )
    
    # 속성 확인
    logger.info(f"일반 DummyAgent의 속성: {dir(dummy_agent)}")
    has_continuous_ensemble = hasattr(dummy_agent, 'continuous_ensemble')
    logger.info(f"continuous_ensemble 속성 존재 여부: {has_continuous_ensemble}")
    if has_continuous_ensemble:
        logger.info(f"continuous_ensemble 값: {dummy_agent.continuous_ensemble}")
    
    # 메타 에이전트 생성
    logger.info("Meta 타입의 DummyAgent 생성 시도...")
    meta_dummy_agent = DummyAgent(
        observation_space=observation_space,
        action_space=action_space,
        agent_type="meta",
        continuous_ensemble=True
    )
    
    # 속성 확인
    logger.info(f"Meta DummyAgent의 속성: {dir(meta_dummy_agent)}")
    has_continuous_ensemble = hasattr(meta_dummy_agent, 'continuous_ensemble')
    logger.info(f"continuous_ensemble 속성 존재 여부: {has_continuous_ensemble}")
    if has_continuous_ensemble:
        logger.info(f"continuous_ensemble 값: {meta_dummy_agent.continuous_ensemble}")
    
    # kwargs를 통한 생성 테스트
    logger.info("kwargs를 통한 Meta DummyAgent 생성 시도...")
    meta_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "agent_type": "meta",
        "continuous_ensemble": True
    }
    kwargs_dummy_agent = DummyAgent(**meta_kwargs)
    
    # 속성 확인
    logger.info(f"kwargs DummyAgent의 속성: {dir(kwargs_dummy_agent)}")
    has_continuous_ensemble = hasattr(kwargs_dummy_agent, 'continuous_ensemble')
    logger.info(f"continuous_ensemble 속성 존재 여부: {has_continuous_ensemble}")
    if has_continuous_ensemble:
        logger.info(f"continuous_ensemble 값: {kwargs_dummy_agent.continuous_ensemble}")

def test_multi_agent_manager_integration():
    """MultiAgentManager와의 통합 테스트"""
    try:
        from agents.strategies.multi.multi_agent_manager import MultiAgentManager
        logger.info("MultiAgentManager 클래스 성공적으로 임포트됨")
        
        # 간단한 observation space와 action space 생성
        observation_dim = 10
        action_dim = 1
        
        observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_dim,), 
            dtype=np.float32
        )
        
        action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # 에이전트 생성
        dummy_agent1 = DummyAgent(
            observation_space=observation_space,
            action_space=action_space,
            agent_type="momentum"
        )
        dummy_agent2 = DummyAgent(
            observation_space=observation_space,
            action_space=action_space,
            agent_type="mean_reversion"
        )
        meta_agent = DummyAgent(
            observation_space=observation_space,
            action_space=action_space,
            agent_type="meta",
            continuous_ensemble=True
        )
        
        # 에이전트 딕셔너리 생성
        agents = {
            "agent1": dummy_agent1,
            "agent2": dummy_agent2,
            "meta_agent": meta_agent
        }
        
        # MultiAgentManager 생성 시도
        logger.info("MultiAgentManager 생성 시도...")
        try:
            manager = MultiAgentManager(
                agents=agents,
                meta_agent_id="meta_agent",
                ensemble_method="meta"
            )
            logger.info("MultiAgentManager 생성 성공!")
            
            # act 메서드 테스트
            logger.info("MultiAgentManager.act() 테스트 시도...")
            obs_dict = {
                "agent1": np.zeros((observation_dim,)),
                "agent2": np.zeros((observation_dim,))
            }
            try:
                actions = manager.act(obs_dict)
                logger.info(f"act() 성공, 반환된 액션: {actions}")
            except Exception as e:
                logger.error(f"act() 호출 실패: {e}")
        
        except Exception as e:
            logger.error(f"MultiAgentManager 생성 실패: {e}")
    
    except ImportError as e:
        logger.error(f"MultiAgentManager 클래스 임포트 실패: {e}")

if __name__ == "__main__":
    logger.info("DummyAgent 테스트 시작")
    test_dummy_agent_attributes()
    test_multi_agent_manager_integration()
    logger.info("DummyAgent 테스트 완료") 