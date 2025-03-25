#!/usr/bin/env python
"""
디버깅 스크립트: MetaAgent 클래스의 continuous_ensemble 매개변수 중복 문제 해결

이 스크립트는 MetaAgent 클래스 초기화 과정에서 발생하는 
'continuous_ensemble' 매개변수가 중복 제공되는 문제를 식별하고 해결합니다.
"""

import sys
import os
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from agents.strategies.meta_agent import MetaAgent
    logger.info("MetaAgent 클래스 성공적으로 임포트됨")
except ImportError as e:
    logger.error(f"MetaAgent 클래스 임포트 실패: {e}")
    sys.exit(1)

def test_meta_agent_init():
    """MetaAgent 클래스의 초기화 테스트"""
    
    # 간단한 observation space와 action space 생성
    observation_dim = 10
    action_dim = 2
    
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
    
    try:
        # 기본 파라미터로 초기화 시도
        logger.info("기본 파라미터로 MetaAgent 초기화 시도...")
        meta_agent = MetaAgent(
            observation_space=observation_space,
            action_space=action_space
        )
        logger.info("MetaAgent 초기화 성공!")
        
        # continuous_ensemble 파라미터를 한 번만 제공
        logger.info("continuous_ensemble=True로 MetaAgent 초기화 시도...")
        meta_agent = MetaAgent(
            observation_space=observation_space,
            action_space=action_space,
            continuous_ensemble=True
        )
        logger.info("continuous_ensemble=True로 MetaAgent 초기화 성공!")
        
        # kwargs를 통해 continuous_ensemble 제공
        logger.info("kwargs를 통해 continuous_ensemble 제공하여 초기화 시도...")
        meta_kwargs = {
            "observation_space": observation_space,
            "action_space": action_space,
            "continuous_ensemble": False  # 명시적 파라미터와 충돌할 수 있는 부분
        }
        try:
            meta_agent = MetaAgent(**meta_kwargs)
            logger.info("kwargs를 통해 MetaAgent 초기화 성공!")
        except TypeError as e:
            logger.error(f"kwargs 방식 초기화 실패: {e}")
            logger.info("MetaAgent 생성자에서 continuous_ensemble이 중복 정의되는 문제가 있는 것으로 판단됩니다.")
    
    except Exception as e:
        logger.error(f"MetaAgent 초기화 중 예외 발생: {e}")
        raise

if __name__ == "__main__":
    logger.info("MetaAgent 테스트 시작")
    test_meta_agent_init()
    logger.info("MetaAgent 테스트 완료") 