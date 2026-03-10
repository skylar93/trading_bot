"""
MLflow 관련 폴더 정리 스크립트

이 스크립트는 다양한 MLflow 관련 폴더들을 정리하고, 현재 활성화된 
mlruns 폴더만 유지합니다.
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlflow_cleaner")

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="MLflow 관련 폴더 정리")
    parser.add_argument("--backup", action="store_true", help="MLflow 데이터를 삭제하기 전에 백업")
    parser.add_argument("--keep-ray-results", action="store_true", help="Ray 결과 디렉토리는 유지")
    parser.add_argument("--dry-run", action="store_true", help="실제로 파일을 삭제하지 않고 삭제될 파일만 표시")
    return parser.parse_args()

def clean_mlflow_folders(backup=False, keep_ray_results=False, dry_run=False):
    """MLflow 관련 폴더 정리 함수"""
    # 현재 작업 디렉토리
    workspace_dir = os.path.abspath(".")
    
    # 현재 활성화된 mlruns 폴더 - 유지됨
    active_mlruns = os.path.join(workspace_dir, "mlruns")
    
    # 백업 생성 (요청된 경우)
    if backup and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(workspace_dir, f"mlruns_archive_{timestamp}")
        logger.info(f"현재 MLflow 데이터를 {backup_dir}로 백업합니다")
        
        os.makedirs(backup_dir, exist_ok=True)
        if os.path.exists(active_mlruns):
            shutil.copytree(active_mlruns, os.path.join(backup_dir, "mlruns"))
    
    # 정리 대상 디렉토리 목록
    dirs_to_clean = []
    
    # 루트 디렉토리의 이전 백업/임시 mlruns 폴더 찾기
    for item in os.listdir(workspace_dir):
        full_path = os.path.join(workspace_dir, item)
        if os.path.isdir(full_path) and item.startswith("mlruns") and item != "mlruns":
            dirs_to_clean.append(full_path)
    
    # Ray 결과 디렉토리 내 mlruns 폴더 찾기 (요청된 경우)
    if not keep_ray_results:
        ray_results_dir = os.path.join(workspace_dir, "ray_results")
        if os.path.exists(ray_results_dir) and os.path.isdir(ray_results_dir):
            for root, dirs, files in os.walk(ray_results_dir):
                for dir_name in dirs:
                    if dir_name == "mlruns":
                        dirs_to_clean.append(os.path.join(root, dir_name))
    
    # 정리 실행
    total_size = 0
    for dir_path in dirs_to_clean:
        try:
            # 디렉토리 크기 계산
            dir_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(dir_path)
                for filename in filenames
            )
            total_size += dir_size
            
            # 크기를 읽기 쉬운 형식으로 변환
            size_str = f"{dir_size / (1024*1024):.2f} MB"
            
            if dry_run:
                logger.info(f"[DRY RUN] 삭제 예정: {dir_path} (크기: {size_str})")
            else:
                logger.info(f"삭제 중: {dir_path} (크기: {size_str})")
                shutil.rmtree(dir_path)
        except Exception as e:
            logger.error(f"디렉토리 {dir_path} 삭제 중 오류 발생: {str(e)}")
    
    # 총 정리된 크기
    total_size_mb = total_size / (1024*1024)
    if dry_run:
        logger.info(f"[DRY RUN] 총 정리 가능한 공간: {total_size_mb:.2f} MB")
    else:
        logger.info(f"총 정리된 공간: {total_size_mb:.2f} MB")
    
    return True

if __name__ == "__main__":
    args = parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN 모드 - 실제 삭제는 수행되지 않습니다")
    
    success = clean_mlflow_folders(
        backup=args.backup,
        keep_ray_results=args.keep_ray_results,
        dry_run=args.dry_run
    )
    
    if success:
        if args.dry_run:
            logger.info("MLflow 폴더 정리 시뮬레이션이 완료되었습니다")
        else:
            logger.info("MLflow 폴더 정리가 완료되었습니다")
            logger.info("현재 활성화된 mlruns 폴더만 유지되었습니다")
    else:
        logger.error("MLflow 폴더 정리 중 오류가 발생했습니다") 