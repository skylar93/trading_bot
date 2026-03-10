#!/bin/bash

# 디버그 스크립트를 다양한 설정으로 실행하는 헬퍼 스크립트

# 기본 실행 (14일 데이터, 5 에피소드)
echo "=== 기본 실행: 14일 데이터, 5 에피소드 ==="
python debug_single_asset_env_real_data.py --episodes 5 --days 14

# 디버그 모드 (자세한 로깅)
echo "=== 상세 디버그 모드 ==="
python debug_single_asset_env_real_data.py --episodes 3 --days 14 --verbose-debug

# 다양한 rollout-steps 테스트
echo "=== 다양한 롤아웃 스텝 테스트 ==="
# 작은 롤아웃 (더 자주 업데이트)
python debug_single_asset_env_real_data.py --episodes 2 --days 14 --rollout-steps 256 

# 큰 롤아웃 (더 적게 업데이트)
python debug_single_asset_env_real_data.py --episodes 2 --days 14 --rollout-steps 2048

# 다양한 학습률 테스트
echo "=== 다양한 학습률 테스트 ==="
python debug_single_asset_env_real_data.py --episodes 2 --days 14 --learning-rate 1e-3
python debug_single_asset_env_real_data.py --episodes 2 --days 14 --learning-rate 1e-4

echo "모든 테스트 완료!" 