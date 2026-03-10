#!/bin/bash

# PPO 에이전트 디버깅 스크립트 (PolicyNetwork 디버그 메시지 없음)
# 이 스크립트는 단일 자산 환경에서 PPO 에이전트를 테스트하는데 사용됩니다.

# 기본 매개변수
EPISODES=3
DAYS=14
LEARNING_RATE=0.0003
ROLLOUT_STEPS=128

# 매개변수 파싱
while getopts ":e:d:l:r:" opt; do
  case ${opt} in
    e )
      EPISODES=$OPTARG
      ;;
    d )
      DAYS=$OPTARG
      ;;
    l )
      LEARNING_RATE=$OPTARG
      ;;
    r )
      ROLLOUT_STEPS=$OPTARG
      ;;
    \? )
      echo "사용법: $0 [-e 에피소드수] [-d 데이터일수] [-l 학습률] [-r 롤아웃스텝수]"
      exit 1
      ;;
  esac
done

echo "=== 조용한 PPO 디버그 모드 ==="
echo "에피소드 수: $EPISODES"
echo "데이터 일수: $DAYS"
echo "학습률: $LEARNING_RATE"
echo "롤아웃 스텝: $ROLLOUT_STEPS"
echo "========================="

# 스크립트 실행
python debug_single_asset_env_real_data.py \
  --episodes $EPISODES \
  --days $DAYS \
  --learning-rate $LEARNING_RATE \
  --rollout-steps $ROLLOUT_STEPS \
  --quiet-debug 