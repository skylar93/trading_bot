#!/bin/bash
#SBATCH --job-name=trading_bot_train
#SBATCH --account=stf               # 사용할 Hyak 계정
#SBATCH --partition=gpu-l40         # GPU 파티션
#SBATCH --nodes=1                   # 사용할 노드 수
#SBATCH --gpus-per-node=1           # 노드당 GPU 수
#SBATCH --mem=47G                   # 메모리 요청
#SBATCH --time=8:00:00              # 실행 시간 (8시간)
#SBATCH --cpus-per-task=7           # 태스크당 CPU 수
#SBATCH --output=logs/trading_bot_%j.out   # 표준 출력 로그
#SBATCH --error=logs/trading_bot_%j.err    # 에러 로그

# Load Python environment
module load foster/python/miniconda/3.8

# Activate virtual environment
source /gscratch/scrubbed/nlee6/trading_bot/venv/bin/activate

# Set environment variables
export PYTHONPATH=/gscratch/scrubbed/nlee6/trading_bot:$PYTHONPATH

# Check GPU availability
nvidia-smi

# Run training script
python /gscratch/scrubbed/nlee6/trading_bot/training/train.py

