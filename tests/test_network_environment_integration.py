#!/usr/bin/env python
"""
네트워크와 환경의 통합 테스트.
이 테스트는 다양한 네트워크 아키텍처와 환경 설정 간의 호환성을 검증합니다.
특히 관측 공간과 액션 공간의 크기 호환성에 초점을 맞춥니다.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import pytest
import logging
from typing import Dict, List, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from envs.multi_asset_env import MultiAssetTradingEnv
from networks.multi_asset_policy import MultiAssetLSTMPolicy, MultiAssetAttentionPolicy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger('test_network_environment_integration')

# 재현성을 위한 랜덤 시드 설정
np.random.seed(42)
torch.manual_seed(42)


def create_test_data() -> pd.DataFrame:
    """테스트용 가격 데이터 생성."""
    # 날짜 범위 생성
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    
    # BTC 가격 데이터 생성
    btc_prices = 20000 + np.cumsum(np.random.normal(0, 500, len(dates)))
    btc_prices = np.maximum(btc_prices, 15000)  # 음수 가격이 없도록 함
    
    # ETH 가격 데이터 생성 (BTC와 상관관계 있음)
    eth_prices = 1500 + 0.8 * np.cumsum(np.random.normal(0, 30, len(dates))) + 0.2 * (btc_prices - 20000) / 10
    eth_prices = np.maximum(eth_prices, 1000)
    
    # SPY 가격 데이터 생성 (주식은 다른 패턴)
    spy_prices = 400 + np.cumsum(np.random.normal(0, 2, len(dates)))
    spy_prices = np.maximum(spy_prices, 380)
    
    # 거래량 생성
    btc_volumes = np.random.uniform(500, 2000, len(dates))
    eth_volumes = np.random.uniform(5000, 20000, len(dates))
    spy_volumes = np.random.uniform(1000000, 5000000, len(dates))
    
    # OHLCV 데이터 생성
    data = pd.DataFrame({
        'date': dates,
        'BTC_$open': btc_prices * 0.99,
        'BTC_$high': btc_prices * 1.02,
        'BTC_$low': btc_prices * 0.98,
        'BTC_$close': btc_prices,
        'BTC_$volume': btc_volumes,
        'ETH_$open': eth_prices * 0.99,
        'ETH_$high': eth_prices * 1.02,
        'ETH_$low': eth_prices * 0.98,
        'ETH_$close': eth_prices,
        'ETH_$volume': eth_volumes,
        'SPY_$open': spy_prices * 0.998,
        'SPY_$high': spy_prices * 1.005,
        'SPY_$low': spy_prices * 0.995,
        'SPY_$close': spy_prices,
        'SPY_$volume': spy_volumes
    })
    
    return data


class TestNetworkEnvironmentIntegration(unittest.TestCase):
    """네트워크와 환경 간의 통합 테스트 클래스."""
    
    def setUp(self):
        """테스트 환경 설정."""
        # 테스트 데이터 생성
        self.data = create_test_data()
        
        # 디바이스 설정 (가능하면 GPU, 없으면 CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 테스트 단순화 - 기본 구성 위주로 테스트
        self.env_configs = [
            # 기본 2자산, 포트폴리오 가중치 형식
            {
                'assets': ['BTC', 'ETH'],
                'action_type': 'portfolio_weights',
                'add_position_info': True,
                'format_3d': False,
                'window_size': 7
            },
            # 2자산, 이산 금액 형식
            {
                'assets': ['BTC', 'ETH'],
                'action_type': 'discrete_amount',
                'add_position_info': True,
                'format_3d': False,
                'window_size': 5
            }
        ]
    
    def test_lstm_policy_compatibility(self):
        """LSTM 정책 네트워크와 다양한 환경 설정 간의 호환성 테스트."""
        for i, config in enumerate(self.env_configs):
            with self.subTest(f"LSTM 호환성 테스트 구성 #{i+1}"):
                # 환경 생성
                env = MultiAssetTradingEnv(
                    df=self.data,
                    assets=config['assets'],
                    initial_balance=10000.0,
                    window_size=config['window_size'],
                    action_type=config['action_type'],
                    add_position_info=config['add_position_info'],
                    format_3d=config['format_3d']
                )
                
                # 관측 및 액션 공간 차원 확인
                observation_shape = env.observation_space.shape
                action_shape = env.action_space.shape
                logger.info(f"환경 구성 #{i+1}: 관측 공간 형태 {observation_shape}, "
                             f"액션 공간 형태 {action_shape}")
                
                # 환경 리셋하여 초기 관측값 받기
                obs, _ = env.reset()
                
                # 정확한 features_per_asset 계산
                # 각 자산당 특성 수는 전체 특성 수를 자산 수로 나눈 값
                total_features = observation_shape[1]  # 2D 형식: (window_size, total_features)
                true_features_per_asset = total_features // len(config['assets'])
                
                logger.info(f"자산별 특성 수: {true_features_per_asset}, "
                           f"전체 특성 수: {total_features}, "
                           f"자산 수: {len(config['assets'])}")
                
                # 네트워크 생성 - 정확한 features_per_asset 전달
                policy = MultiAssetLSTMPolicy(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    n_assets=len(config['assets']),
                    window_size=config['window_size'],
                    features_per_asset=true_features_per_asset,  # 정확한 값 사용
                    hidden_size=64,
                    lstm_layers=1
                ).to(self.device)
                
                # 텐서로 변환 (2D -> 3D 배치 추가)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 입력 텐서 형태 로깅
                logger.info(f"obs 형태: {obs.shape}, obs_tensor 형태: {obs_tensor.shape}")
                
                # 네트워크 통과 (이게 오류 없이 실행되면 입력 형태 호환 성공)
                with torch.no_grad():
                    action_tensor = policy.get_action(obs_tensor, deterministic=True)
                
                # 출력 텐서 형태 확인
                self.assertEqual(action_tensor.shape[1], env.action_space.shape[0],
                                 f"액션 출력 차원이 일치하지 않음: 예상 {env.action_space.shape[0]}, "
                                 f"실제 {action_tensor.shape[1]}")
                
                # 액션 취하기
                action = action_tensor.squeeze().cpu().numpy()
                
                # 환경 스텝 진행 (이게 오류 없이 실행되면 출력 형태 호환 성공)
                next_obs, reward, terminated, truncated, info = env.step(action)
    
    def test_attention_policy_compatibility(self):
        """Attention 정책 네트워크와 다양한 환경 설정 간의 호환성 테스트."""
        for i, config in enumerate(self.env_configs):
            with self.subTest(f"Attention 호환성 테스트 구성 #{i+1}"):
                # 환경 생성
                env = MultiAssetTradingEnv(
                    df=self.data,
                    assets=config['assets'],
                    initial_balance=10000.0,
                    window_size=config['window_size'],
                    action_type=config['action_type'],
                    add_position_info=config['add_position_info'],
                    format_3d=config['format_3d']
                )
                
                # 관측 및 액션 공간 차원 확인
                observation_shape = env.observation_space.shape
                action_shape = env.action_space.shape
                logger.info(f"환경 구성 #{i+1}: 관측 공간 형태 {observation_shape}, "
                             f"액션 공간 형태 {action_shape}")
                
                # 환경 리셋하여 초기 관측값 받기
                obs, _ = env.reset()
                
                # 정확한 features_per_asset 계산
                total_features = observation_shape[1]  # 2D 형식: (window_size, total_features)
                true_features_per_asset = total_features // len(config['assets'])
                
                logger.info(f"자산별 특성 수: {true_features_per_asset}, "
                           f"전체 특성 수: {total_features}, "
                           f"자산 수: {len(config['assets'])}")
                
                # 네트워크 생성 - 정확한 features_per_asset 전달, 파라미터 이름 수정
                policy = MultiAssetAttentionPolicy(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    n_assets=len(config['assets']),
                    window_size=config['window_size'],
                    features_per_asset=true_features_per_asset,  # 정확한 값 사용
                    hidden_size=64,
                    num_heads=4,
                    num_layers=2
                ).to(self.device)
                
                # 텐서로 변환 (2D -> 3D 배치 추가)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 입력 텐서 형태 로깅
                logger.info(f"obs 형태: {obs.shape}, obs_tensor 형태: {obs_tensor.shape}")
                
                # 네트워크 통과 (이게 오류 없이 실행되면 입력 형태 호환 성공)
                with torch.no_grad():
                    action_tensor = policy.get_action(obs_tensor, deterministic=True)
                
                # 출력 텐서 형태 확인
                self.assertEqual(action_tensor.shape[1], env.action_space.shape[0],
                                 f"액션 출력 차원이 일치하지 않음: 예상 {env.action_space.shape[0]}, "
                                 f"실제 {action_tensor.shape[1]}")
                
                # 액션 취하기
                action = action_tensor.squeeze().cpu().numpy()
                
                # 환경 스텝 진행 (이게 오류 없이 실행되면 출력 형태 호환 성공)
                next_obs, reward, terminated, truncated, info = env.step(action)
    
    def test_multi_episode_network_interaction(self):
        """여러 에피소드에 걸친 네트워크와 환경 상호작용 테스트."""
        # 환경 설정
        env = MultiAssetTradingEnv(
            df=self.data,
            assets=['BTC', 'ETH'],
            initial_balance=10000.0,
            window_size=7,
            action_type='portfolio_weights',
            add_position_info=True,
            format_3d=False
        )
        
        # 관측 공간 정보
        obs_shape = env.observation_space.shape
        features_per_asset = obs_shape[1] // 2  # BTC, ETH 두 자산
        
        # LSTM 정책 네트워크 생성
        policy = MultiAssetLSTMPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            n_assets=2,
            window_size=7,
            features_per_asset=features_per_asset,
            hidden_size=64,
            lstm_layers=1
        ).to(self.device)
        
        # 여러 에피소드 실행
        for episode in range(3):
            obs, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < 20:  # 최대 20 스텝
                # 관측을 텐서로 변환
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 네트워크에서 액션 얻기
                with torch.no_grad():
                    action_tensor = policy.get_action(obs_tensor, deterministic=False)
                
                # 텐서를 NumPy 배열로 변환
                action = action_tensor.squeeze().cpu().numpy()
                
                # 환경에서 스텝 진행
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # 상태 업데이트
                obs = next_obs
                done = terminated or truncated
                step += 1
            
            logger.info(f"에피소드 {episode+1} 완료: {step} 스텝, 최종 포트폴리오 가치: {env.portfolio_value:.2f}")
            
            # 에피소드가 올바르게 진행되었는지 확인
            self.assertGreater(step, 0, "에피소드에서 스텝이 진행되지 않음")
    
    def test_action_format_compatibility(self):
        """다양한 액션 형식과 네트워크 호환성 테스트."""
        # 액션 유형 목록
        action_types = ['portfolio_weights', 'discrete_amount', 'discrete_signal']
        
        for action_type in action_types:
            with self.subTest(f"액션 유형 {action_type} 테스트"):
                # 환경 생성
                env = MultiAssetTradingEnv(
                    df=self.data,
                    assets=['BTC', 'ETH'],
                    initial_balance=10000.0,
                    window_size=7,
                    action_type=action_type,
                    add_position_info=True,
                    format_3d=False
                )
                
                # 관측 공간 정보
                obs_shape = env.observation_space.shape
                features_per_asset = obs_shape[1] // 2  # BTC, ETH 두 자산
                
                # LSTM 정책 네트워크 생성
                policy = MultiAssetLSTMPolicy(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    n_assets=2,
                    window_size=7,
                    features_per_asset=features_per_asset,
                    hidden_size=64,
                    lstm_layers=1
                ).to(self.device)
                
                # 환경 리셋
                obs, _ = env.reset()
                
                # 관측을 텐서로 변환
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 네트워크에서 액션 얻기
                with torch.no_grad():
                    action_tensor = policy.get_action(obs_tensor, deterministic=True)
                
                # 텐서를 NumPy 배열로 변환
                action = action_tensor.squeeze().cpu().numpy()
                
                # 액션 형태 확인
                self.assertEqual(action.shape[0], env.action_space.shape[0],
                                 f"액션 차원이 일치하지 않음: 예상 {env.action_space.shape[0]}, "
                                 f"실제 {action.shape[0]}")
                
                # 환경에서 스텝 진행 (이게 오류 없이 실행되면 액션 형식 호환 성공)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # 액션 유형에 따른 검증 (네트워크 출력은 -1~1 범위, 환경이 적절히 변환)
                # 여기서는 액션 유형과 상관없이 네트워크 출력이 -1~1 범위인지만 확인
                self.assertTrue(np.all(action >= -1) and np.all(action <= 1),
                               f"네트워크 액션은 -1~1 사이여야 함, 실제: {action}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 