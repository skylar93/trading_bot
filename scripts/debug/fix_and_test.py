#!/usr/bin/env python
"""
종합 디버깅 및 수정 스크립트

이 스크립트는 다음과 같은 문제를 진단하고 수정합니다:
1. 'DummyAgent' object has no attribute 'continuous_ensemble'
2. 'MultiAssetTradingEnv' object has no attribute 'logger'
3. MetaAgent() got multiple values for keyword argument 'continuous_ensemble'
4. 평가 후 환경이 리셋되지 않는 문제

또한 기본적인 기능 테스트를 수행하여 문제가 해결되었는지 확인합니다.
"""

import sys
import os
import logging
import importlib
import inspect
import re
import shutil

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

def backup_file(file_path):
    """파일 백업"""
    backup_path = f"{file_path}.bak"
    logger.info(f"파일 백업 중: {file_path} -> {backup_path}")
    shutil.copy2(file_path, backup_path)
    return backup_path

def fix_dummy_agent():
    """DummyAgent 클래스의 continuous_ensemble 속성 문제 해결"""
    logger.info("DummyAgent 수정 시작...")
    
    try:
        from agents.strategies.dummy_agent import DummyAgent
        
        # 파일 경로 찾기
        file_path = inspect.getfile(DummyAgent)
        logger.info(f"DummyAgent 파일 위치: {file_path}")
        
        # 파일 백업
        backup_path = backup_file(file_path)
        
        # 파일 읽기
        with open(file_path, 'r') as f:
            content = f.read()
        
        # continuous_ensemble 속성이 이미 있는지 확인
        if "self.continuous_ensemble" in content:
            logger.info("continuous_ensemble 속성이 이미 존재합니다.")
            
            # 속성 설정이 올바른지 확인
            pattern = r"if agent_type == \"meta\"[^}]*?self\.continuous_ensemble\s*="
            if not re.search(pattern, content, re.DOTALL):
                logger.info("continuous_ensemble 속성이 meta 타입에 대해 올바르게 설정되지 않았습니다. 수정 중...")
                
                # __init__ 메서드에서 meta 타입 처리 추가
                init_pattern = r"(\s+)# For PPO compatibility[^}]*?self\.max_grad_norm\s*=[^}]+\n"
                meta_code = r"\1# For meta-agent compatibility\n\1if agent_type == \"meta\":\n\1    self.continuous_ensemble = kwargs.get(\"continuous_ensemble\", True)\n\1    self.observation_size = kwargs.get(\"observation_size\", 60)\n\1    self.action_dim = kwargs.get(\"action_dim\", 2)\n\1    self.hidden_dim = kwargs.get(\"hidden_dim\", 128)\n\1else:\n\1    self.continuous_ensemble = False\n\1\n"
                
                modified_content = re.sub(init_pattern, r"\g<0>" + meta_code, content)
                
                # 변경된 내용 저장
                with open(file_path, 'w') as f:
                    f.write(modified_content)
                
                logger.info("DummyAgent 클래스 수정 완료")
            else:
                logger.info("continuous_ensemble 속성이 이미 meta 타입에 대해 올바르게 설정되어 있습니다.")
        else:
            logger.info("continuous_ensemble 속성이 존재하지 않습니다. 추가 중...")
            
            # __init__ 메서드에서 meta 타입 처리 추가
            init_pattern = r"(\s+)# For PPO compatibility[^}]*?self\.max_grad_norm\s*=[^}]+\n"
            meta_code = r"\1# For meta-agent compatibility\n\1if agent_type == \"meta\":\n\1    self.continuous_ensemble = kwargs.get(\"continuous_ensemble\", True)\n\1    self.observation_size = kwargs.get(\"observation_size\", 60)\n\1    self.action_dim = kwargs.get(\"action_dim\", 2)\n\1    self.hidden_dim = kwargs.get(\"hidden_dim\", 128)\n\1else:\n\1    self.continuous_ensemble = False\n\1\n"
            
            modified_content = re.sub(init_pattern, r"\g<0>" + meta_code, content)
            
            # 변경된 내용 저장
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            logger.info("DummyAgent 클래스 수정 완료")
        
    except Exception as e:
        logger.error(f"DummyAgent 수정 중 오류 발생: {e}")

def fix_meta_agent():
    """MetaAgent 클래스의 continuous_ensemble 매개변수 중복 문제 해결"""
    logger.info("MetaAgent 수정 시작...")
    
    try:
        from agents.strategies.meta_agent import MetaAgent
        
        # 파일 경로 찾기
        file_path = inspect.getfile(MetaAgent)
        logger.info(f"MetaAgent 파일 위치: {file_path}")
        
        # 파일 백업
        backup_path = backup_file(file_path)
        
        # 파일 읽기
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 생성자에서 continuous_ensemble 매개변수 중복 확인
        constructor_pattern = r"def __init__\([^)]*continuous_ensemble[^)]*\):"
        if re.search(constructor_pattern, content):
            # 생성자에서 continuous_ensemble 매개변수가 kwargs에도 포함되는지 확인
            kwargs_pattern = r"def __init__\([^)]*continuous_ensemble[^)]*\*\*kwargs[^)]*\):"
            if re.search(kwargs_pattern, content):
                logger.info("continuous_ensemble 매개변수가 중복 정의될 수 있습니다. 수정 중...")
                
                # kwargs에서 continuous_ensemble 매개변수 제거
                constructor_code = re.search(r"def __init__\([^)]*\):", content).group()
                init_body_pattern = r"def __init__\([^)]*\):([^#]*)# "
                init_body = re.search(init_body_pattern, content, re.DOTALL).group(1)
                
                # kwargs에서 continuous_ensemble을 체크하는 코드 추가
                adjusted_init_body = init_body + "        # Check if continuous_ensemble is in kwargs and use it\n        if 'continuous_ensemble' in kwargs:\n            self.continuous_ensemble = kwargs.pop('continuous_ensemble')\n\n        # "
                
                modified_content = content.replace(init_body + "# ", adjusted_init_body)
                
                # 변경된 내용 저장
                with open(file_path, 'w') as f:
                    f.write(modified_content)
                
                logger.info("MetaAgent 클래스 수정 완료")
            else:
                logger.info("continuous_ensemble 매개변수가 중복 정의되지 않습니다.")
        else:
            logger.info("continuous_ensemble 매개변수 중복 문제가 발견되지 않았습니다.")
        
    except Exception as e:
        logger.error(f"MetaAgent 수정 중 오류 발생: {e}")

def fix_multi_asset_env():
    """MultiAssetTradingEnv 클래스의 logger 속성 문제 해결"""
    logger.info("MultiAssetTradingEnv 수정 시작...")
    
    try:
        from envs.multi_asset_env import MultiAssetTradingEnv
        
        # 파일 경로 찾기
        file_path = inspect.getfile(MultiAssetTradingEnv)
        logger.info(f"MultiAssetTradingEnv 파일 위치: {file_path}")
        
        # 파일 백업
        backup_path = backup_file(file_path)
        
        # 파일 읽기
        with open(file_path, 'r') as f:
            content = f.read()
        
        # logger 속성 초기화 확인
        logger_init_pattern = r"self\.logger\s*=\s*logging\.getLogger\("
        if re.search(logger_init_pattern, content):
            logger.info("logger 속성이 이미 초기화되어 있습니다.")
            
            # __init__ 메서드 초기에 초기화되는지 확인
            early_init_pattern = r"def __init__\([^)]*\):[^#]*super\(\)\.__init__\(\)[^#]*"
            early_logger_init = r"self\.logger\s*=\s*logging\.getLogger\("
            
            if re.search(early_init_pattern + early_logger_init, content, re.DOTALL):
                logger.info("logger 속성이 __init__ 메서드 초기에 올바르게 초기화되어 있습니다.")
            else:
                logger.info("logger 속성이 __init__ 메서드 초기에 초기화되지 않았습니다. 수정 중...")
                
                init_pattern = r"(def __init__\([^)]*\):[^#]*super\(\)\.__init__\(\))"
                logger_code = r"\1\n\n        # Initialize logger\n        self.logger = logging.getLogger(self.__class__.__name__)"
                
                modified_content = re.sub(init_pattern, logger_code, content)
                
                # 변경된 내용 저장
                with open(file_path, 'w') as f:
                    f.write(modified_content)
                
                logger.info("MultiAssetTradingEnv 클래스 수정 완료")
        else:
            logger.info("logger 속성이 초기화되지 않았습니다. 추가 중...")
            
            init_pattern = r"(def __init__\([^)]*\):[^#]*super\(\)\.__init__\(\))"
            logger_code = r"\1\n\n        # Initialize logger\n        self.logger = logging.getLogger(self.__class__.__name__)"
            
            modified_content = re.sub(init_pattern, logger_code, content)
            
            # 변경된 내용 저장
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            logger.info("MultiAssetTradingEnv 클래스 수정 완료")
        
    except Exception as e:
        logger.error(f"MultiAssetTradingEnv 수정 중 오류 발생: {e}")

def fix_train_pipeline():
    """train_pipeline.py의 환경 리셋 문제 해결"""
    logger.info("train_pipeline.py 수정 시작...")
    
    try:
        # 파일 경로 찾기
        file_path = os.path.join(root_dir, 'training', 'train_pipeline.py')
        
        if not os.path.exists(file_path):
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return
            
        logger.info(f"train_pipeline.py 파일 위치: {file_path}")
        
        # 파일 백업
        backup_path = backup_file(file_path)
        
        # 파일 읽기
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 평가 후 환경 리셋 확인
        eval_pattern = r"(# Evaluation[^#]*mean_eval_reward\s*=\s*np\.mean\(eval_rewards\)[^#]*?best_model_path\s*=\s*os\.path\.join\(checkpoint_dir,\s*\"best_agent\.pt\"\)[^#]*?logger\.info[^#]*?\"best_model_path\"[^#]*?\))"
        reset_code = r"\1\n\n            # Reset environment after evaluation\n            obs, info = env.reset()"
        
        if "# Reset environment after evaluation" in content:
            logger.info("이미 평가 후 환경 리셋 코드가 추가되어 있습니다.")
        else:
            logger.info("평가 후 환경 리셋 코드가 없습니다. 추가 중...")
            
            modified_content = re.sub(eval_pattern, reset_code, content, flags=re.DOTALL)
            
            # 변경된 내용 저장
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            logger.info("train_pipeline.py 수정 완료")
        
    except Exception as e:
        logger.error(f"train_pipeline.py 수정 중 오류 발생: {e}")

def test_fixes():
    """수정 사항 테스트"""
    logger.info("수정 사항 테스트 시작...")
    
    # DummyAgent 테스트
    try:
        from agents.strategies.dummy_agent import DummyAgent
        from gymnasium import spaces
        import numpy as np
        
        # 간단한 observation space와 action space 생성
        observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Meta 타입 DummyAgent 생성
        dummy_agent = DummyAgent(
            observation_space=observation_space,
            action_space=action_space,
            agent_type="meta",
            continuous_ensemble=True
        )
        
        # continuous_ensemble 속성 확인
        if hasattr(dummy_agent, 'continuous_ensemble'):
            logger.info(f"DummyAgent의 continuous_ensemble 속성: {dummy_agent.continuous_ensemble}")
            if dummy_agent.continuous_ensemble:
                logger.info("✅ DummyAgent 수정 테스트 통과")
            else:
                logger.warning("⚠️ DummyAgent의 continuous_ensemble 속성이 False입니다.")
        else:
            logger.error("❌ DummyAgent에 continuous_ensemble 속성이 없습니다.")
    except Exception as e:
        logger.error(f"DummyAgent 테스트 실패: {e}")
    
    # MultiAssetTradingEnv 테스트
    try:
        from envs.multi_asset_env import MultiAssetTradingEnv
        import pandas as pd
        
        # 간단한 데이터프레임 생성
        df = pd.DataFrame({
            'BTC_$open': [100, 101, 102],
            'BTC_$high': [105, 106, 107],
            'BTC_$low': [95, 96, 97],
            'BTC_$close': [103, 104, 105],
            'BTC_$volume': [1000, 1100, 1200],
        })
        
        # 환경 생성
        env = MultiAssetTradingEnv(
            df=df,
            assets=["BTC"],
            window_size=1,
            initial_balance=10000
        )
        
        # logger 속성 확인
        if hasattr(env, 'logger'):
            logger.info("✅ MultiAssetTradingEnv 수정 테스트 통과")
        else:
            logger.error("❌ MultiAssetTradingEnv에 logger 속성이 없습니다.")
    except Exception as e:
        logger.error(f"MultiAssetTradingEnv 테스트 실패: {e}")
    
    # train_pipeline.py 테스트는 실제 실행이 필요하므로 여기서는 생략
    
    logger.info("수정 사항 테스트 완료")

if __name__ == "__main__":
    logger.info("종합 디버깅 및 수정 스크립트 시작")
    
    # 수정 적용
    fix_dummy_agent()
    fix_meta_agent()
    fix_multi_asset_env()
    fix_train_pipeline()
    
    # 수정 테스트
    test_fixes()
    
    logger.info("종합 디버깅 및 수정 스크립트 완료")
    
    logger.info("\n다음 단계:")
    logger.info("1. 원본 train_pipeline.py 스크립트를 실행하여 모든 오류가 해결되었는지 확인하세요.")
    logger.info("2. 모든 수정 사항이 개발 지침에 맞게 작성되었는지 확인하세요.")
    logger.info("3. 백업된 파일은 필요하지 않다면 삭제하세요.")
    logger.info("4. 문제가 계속 발생하면 더 자세한 진단이 필요할 수 있습니다.") 