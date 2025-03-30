"""
debug_single_asset_env_real_data.py

이 스크립트는 DataLoader를 이용해 실제 거래소 데이터를 로드한 뒤,
SingleAssetRLTradingEnv에서 PPO 학습을 수행해 보는 디버그용 예시입니다.

사용 예:
  python debug_single_asset_env_real_data.py
  python debug_single_asset_env_real_data.py --use-mlflow  # MLflow 추적 활성화
  python debug_single_asset_env_real_data.py --hyperopt  # 하이퍼파라미터 최적화 실행
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import argparse
from typing import Dict, Any, Optional

# 프로젝트 경로에 맞춰 조정 (env_factory, ppo_agent, DataLoader 위치에 따라 수정)
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from data.utils.data_loader import DataLoader
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from agents.strategies.single.ppo_agent import PPOAgent
from buffers.ppo_buffer import PPOBuffer

# MLflow 매니저 및 하이퍼파라미터 최적화 임포트
try:
    from training.utils.unified_mlflow_manager import MLflowManager
    from training.hyperopt.hyperopt_ray import run_hyperparameter_optimization, create_search_space
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow 또는 하이퍼파라미터 최적화 모듈을 가져올 수 없습니다. 해당 기능은 비활성화됩니다.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 아래 코드를 추가하여 PolicyNetwork 로거 수준을 조정하는 함수 정의
def configure_loggers(quiet_debug=False):
    """로거 설정 함수"""
    if quiet_debug:
        # PolicyNetwork 로거를 ERROR 수준으로 설정하여 DEBUG 메시지 제거
        policy_network_logger = logging.getLogger("PolicyNetwork")
        policy_network_logger.setLevel(logging.ERROR)
        logger.info("PolicyNetwork 로거가 ERROR 수준으로 설정되었습니다. 디버그 메시지가 표시되지 않습니다.")


def load_real_data(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2022-01-01",
    end_date="2022-04-10"
) -> pd.DataFrame:
    """
    DataLoader를 통해 실제 거래소(예: Binance) 데이터를 로드하여 DataFrame으로 반환.
    """
    try:
        logging.info(f"📊 Loading data for {symbol} from {start_date} to {end_date}")
        
        # 원하는 거래소, 심볼, 타임프레임 설정
        data_loader = DataLoader(
            exchange_id="binance",  # 본인의 exchange_id로 수정 가능
            symbol=symbol,
            timeframe=timeframe
        )
        
        # DataLoader에서 데이터 가져오기
        df = data_loader.fetch_data(
            start_date=start_date,
            end_date=end_date
        )
        
        logging.info(f"📈 Loaded data with shape: {df.shape}")
        logging.info("Sample data (first 3 rows):")
        logging.info(f"\n{df.head(3)}")
        
        # df가 너무 작거나, 결측치가 많거나 하면 이후 학습에 곤란할 수 있음
        return df
    
    except Exception as e:
        logging.error(f"❌ Error loading real data: {e}")
        raise

def create_env(df, config=None):
    """환경 생성 함수"""
    if config is None:
        config = {}
    
    env_config = config.get("env", {})
    
    return SingleAssetRLTradingEnv(
        data=df,
        initial_capital=env_config.get("initial_capital", 10000.0),
        trading_fee=env_config.get("trading_fee", 0.001),
        window_size=env_config.get("window_size", 20),
        max_position_size=env_config.get("max_position_size", 1.0),
        risk_adjusted_reward=env_config.get("risk_adjusted_reward", True),
        sharpe_lookback=env_config.get("sharpe_lookback", 30),
        sharpe_weight=env_config.get("sharpe_weight", 0.5),
        drawdown_penalty=env_config.get("drawdown_penalty", True),
        max_drawdown_penalty_threshold=env_config.get("max_drawdown_penalty_threshold", 0.1),
        apply_slippage=env_config.get("apply_slippage", True),
        slippage_factor=env_config.get("slippage_factor", 0.0005),
        partial_fills=env_config.get("partial_fills", True),
        min_fill_rate=env_config.get("min_fill_rate", 0.8),
        volume_slippage_factor=env_config.get("volume_slippage_factor", 0.1)
    )

def create_agent(env, config=None):
    """에이전트 생성 함수"""
    if config is None:
        config = {}
    
    agent_config = config.get("agent", {})
    
    return PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=agent_config.get("learning_rate", 3e-4),
        gamma=agent_config.get("gamma", 0.99),
        gae_lambda=agent_config.get("gae_lambda", 0.95),
        clip_epsilon=agent_config.get("clip_epsilon", 0.2),
        n_epochs=agent_config.get("n_epochs", 5),
        batch_size=agent_config.get("batch_size", 64),
        max_grad_norm=agent_config.get("max_grad_norm", 0.5),
        target_kl=agent_config.get("target_kl", 0.02),
        rollout_steps=agent_config.get("rollout_steps", 1024)
    )

def train_and_evaluate(
    agent, 
    env, 
    config=None,
    mlflow_manager: Optional[MLflowManager] = None
):
    """트레이닝 및 평가 함수"""
    if config is None:
        config = {}
    
    training_config = config.get("training", {})
    
    # 학습 파라미터 설정
    num_episodes = training_config.get("num_episodes", 10)
    update_interval = training_config.get("update_interval", 128)
    
    step_count = 0
    all_rewards = []
    
    # 추가 디버깅을 위한 통계 저장
    agent_stats = {
        "std_values": [],       # 표준편차 통계
        "ratio_stats": [],      # ratio 분포 통계
        "kl_components": [],    # KL 구성요소
        "entropy_values": [],   # 엔트로피 값
        "episode_data": []      # 에피소드별 통계
    }
    
    # 디버깅 기록 간격 설정
    debug_log_interval = 10  # 몇 개 스텝마다 표준편차 통계를 출력할지
    
    for eps in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        
        episode_stats = {
            "std_values": [],
            "actions": [],
            "rewards": []
        }
        
        while not done:
            # 행동 결정
            action = agent.get_action(obs, deterministic=False)
            
            # 표준편차 확인 (몇 스텝마다)
            if step_count % debug_log_interval == 0:
                with torch.no_grad():
                    state = obs
                    # Add explicit reshaping for 2D observations (window_size, features)
                    if isinstance(state, np.ndarray) and len(state.shape) == 2:
                        # Add batch dimension: (window_size, features) -> (1, window_size, features)
                        state = np.expand_dims(state, axis=0)
                        logger.info(f"Expanded state shape from {obs.shape} to {state.shape} for std stats calculation")
                    
                    state_tensor = torch.FloatTensor(state).to(agent.device)
                    action_mean, action_std = agent.old_network(state_tensor)
                    mean_std = action_std.mean().item()
                    min_std = action_std.min().item()
                    max_std = action_std.max().item()
                    logger.info(f"[STD STATS] step={step_count}, mean_std={mean_std:.4f}, min_std={min_std:.4f}, max_std={max_std:.4f}")
                    agent_stats["std_values"].append({"step": step_count, "mean": mean_std, "min": min_std, "max": max_std})
                    episode_stats["std_values"].append({"step": step_count, "mean": mean_std, "min": min_std, "max": max_std})
            
            # step 진행
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 에피소드 통계 업데이트
            episode_stats["actions"].append(action)
            episode_stats["rewards"].append(reward)
            
            # PPO 버퍼에 저장
            agent.train_step(obs, action, reward, next_obs, done)
            
            obs = next_obs
            step_count += 1
            ep_steps += 1
            ep_reward += reward
            
            # 일정 스텝마다 PPO 업데이트
            if step_count % update_interval == 0:
                logger.info(f"[UPDATING] step={step_count}, buffer_size={len(agent.buffer)}")
                update_metrics = agent.update_if_buffer_ready()
                
                if update_metrics:
                    # KL 발산 상세 분석
                    if "kl" in update_metrics and update_metrics["kl"] <= 0.001:  # 매우 작은 KL
                        logger.warning(
                            f"[KL WARNING] Very small KL={update_metrics['kl']:.6f} detected - "
                            f"old and new policies might be too similar"
                        )
                    
                    # 엔트로피 값이 음수인지 확인
                    if "entropy" in update_metrics and update_metrics["entropy"] < 0:
                        logger.warning(
                            f"[ENTROPY WARNING] Negative entropy={update_metrics['entropy']:.4f} detected - "
                            f"std might be too small"
                        )
                    
                    # 결과 로깅
                    logger.info(
                        f"[UPDATE] step={step_count}, "
                        f"policy_loss={update_metrics['policy_loss']:.4f}, "
                        f"value_loss={update_metrics['value_loss']:.4f}, "
                        f"entropy={update_metrics['entropy']:.4f}, "
                        f"kl={update_metrics.get('kl', 0):.4f}, "
                        f"mean_std={update_metrics.get('mean_std', 0):.4f}"
                    )
                    
                    # 디버깅 정보 저장
                    agent_stats["entropy_values"].append({"step": step_count, "value": update_metrics["entropy"]})
                    
                    if "kl" in update_metrics:
                        agent_stats["kl_components"].append({
                            "step": step_count,
                            "kl": update_metrics["kl"],
                            "policy_loss": update_metrics["policy_loss"]
                        })
                    
                    # MLflow 로깅
                    if mlflow_manager is not None:
                        mlflow_manager.log_metrics({
                            "policy_loss": float(update_metrics['policy_loss']),
                            "value_loss": float(update_metrics['value_loss']),
                            "entropy": float(update_metrics['entropy']),
                            "kl": float(update_metrics.get('kl', 0)),
                            "mean_std": float(update_metrics.get('mean_std', 0))
                        }, step=step_count)
        
        # 에피소드 종료 통계
        episode_mean_std = np.mean([stat["mean"] for stat in episode_stats["std_values"]]) if episode_stats["std_values"] else 0
        episode_action_mean = np.mean(np.abs(episode_stats["actions"]))
        
        agent_stats["episode_data"].append({
            "episode": eps,
            "reward": ep_reward,
            "steps": ep_steps,
            "mean_std": episode_mean_std,
            "mean_abs_action": episode_action_mean
        })
        
        all_rewards.append(ep_reward)
        logger.info(
            f"Episode {eps+1}/{num_episodes} ended. "
            f"Reward={ep_reward:.4f}, Steps={ep_steps}, "
            f"Mean |action|={episode_action_mean:.4f}, Mean std={episode_mean_std:.4f}"
        )
        
        # MLflow에 에피소드 보상 로깅
        if mlflow_manager is not None:
            mlflow_manager.log_metrics({
                "episode_reward": float(ep_reward),
                "avg_reward": float(np.mean(all_rewards)),
                "episode_mean_std": float(episode_mean_std),
                "episode_mean_abs_action": float(episode_action_mean)
            }, step=eps)
    
    # 남은 버퍼 처리를 위한 final update
    final_update_metrics = agent.update_if_buffer_ready()
    if final_update_metrics:
        logger.info(
            f"[FINAL UPDATE] policy_loss={final_update_metrics['policy_loss']:.4f}, "
            f"value_loss={final_update_metrics['value_loss']:.4f}, "
            f"entropy={final_update_metrics['entropy']:.4f}, "
            f"kl={final_update_metrics['kl']:.4f}, "
            f"mean_std={final_update_metrics.get('mean_std', 0):.4f}"
        )
        
        # MLflow 로깅
        if mlflow_manager is not None:
            mlflow_manager.log_metrics({
                "final_policy_loss": float(final_update_metrics['policy_loss']),
                "final_value_loss": float(final_update_metrics['value_loss']),
                "final_entropy": float(final_update_metrics['entropy']),
                "final_kl": float(final_update_metrics['kl']),
                "final_mean_std": float(final_update_metrics.get('mean_std', 0))
            })
    
    # 학습 통계 요약
    logger.info("=== Training Statistics Summary ===")
    logger.info(f"All episode rewards: {all_rewards}")
    logger.info(f"Avg reward over {num_episodes} episodes: {np.mean(all_rewards):.4f}")
    
    if agent_stats["entropy_values"]:
        entropy_values = [entry["value"] for entry in agent_stats["entropy_values"]]
        logger.info(f"Entropy stats - Min: {min(entropy_values):.4f}, Max: {max(entropy_values):.4f}, Mean: {np.mean(entropy_values):.4f}")
    
    if agent_stats["std_values"]:
        std_means = [entry["mean"] for entry in agent_stats["std_values"]]
        logger.info(f"Std stats - Min: {min(std_means):.4f}, Max: {max(std_means):.4f}, Mean: {np.mean(std_means):.4f}")
    
    if agent_stats["kl_components"]:
        kl_values = [entry["kl"] for entry in agent_stats["kl_components"]]
        logger.info(f"KL stats - Min: {min(kl_values):.6f}, Max: {max(kl_values):.6f}, Mean: {np.mean(kl_values):.6f}")
    
    # 트레이닝 히스토리
    try:
        training_history = agent.get_training_history()
        if training_history:
            logger.info("=== Agent Training History ===")
            logger.info(f"Updates: {training_history.get('update_count', 0)}")
            logger.info(f"Total steps: {training_history.get('total_steps', 0)}")
            logger.info(f"Completed episodes: {training_history.get('completed_episodes', 0)}")
            
            if 'episode_rewards' in training_history and training_history['episode_rewards']:
                logger.info(f"Final episode rewards: {training_history['episode_rewards'][-5:]}")
    except:
        logger.warning("Could not retrieve agent training history")
    
    # 모델 저장
    save_path = config.get("paths", {}).get("model_path", "ppo_agent_single_asset_100days.pt")
    agent.save(save_path)
    logger.info(f"Agent saved to {save_path}")
    
    # MLflow에 모델 아티팩트 저장
    if mlflow_manager is not None:
        mlflow_manager.log_artifact(save_path)
        mlflow_manager.log_metrics({
            "avg_reward": float(np.mean(all_rewards))
        })
    
    return {
        "avg_reward": np.mean(all_rewards),
        "episode_rewards": all_rewards,
        "model_path": save_path,
        "agent_stats": agent_stats
    }

def hyperparameter_optimization(df, config=None):
    """하이퍼파라미터 최적화 실행"""
    if not MLFLOW_AVAILABLE:
        logger.error("하이퍼파라미터 최적화를 위해서는 Ray Tune과 MLflow가 필요합니다.")
        return None
    
    if config is None:
        config = {}
    
    # 데이터를 임시 파일로 저장 (Ray Tune worker가 접근할 수 있도록)
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    temp_data_path = os.path.join(temp_dir, "temp_data.csv")
    logger.info(f"임시 데이터 파일 저장: {temp_data_path}")
    
    try:
        # DataFrame을 CSV 파일로 저장
        df.to_csv(temp_data_path, index=False)
        
        # 하이퍼파라미터 검색 공간 설정
        search_config = {
            "hyperopt": {
                "num_samples": 5,  # 검색할 샘플 수
                "parameters": {
                    "agent.learning_rate": {"distribution": "loguniform", "min": 1e-5, "max": 5e-4},
                    "agent.gamma": {"distribution": "uniform", "min": 0.9, "max": 0.999},
                    "agent.gae_lambda": {"distribution": "uniform", "min": 0.9, "max": 0.99},
                    "agent.clip_epsilon": {"distribution": "uniform", "min": 0.1, "max": 0.3},
                    "agent.n_epochs": {"distribution": "randint", "min": 3, "max": 10},
                    "env.window_size": {"distribution": "choice", "values": [3, 5]},
                    "training.update_interval": {"distribution": "choice", "values": [64, 128, 256]},
                }
            },
            "training": {
                "num_episodes": 5,  # 각 시도마다 적은 에피소드로 평가
            },
            "paths": {
                "model_path": "ppo_agent_hyperopt.pt"
            },
            # 환경 유형 및 데이터 경로 설정 추가
            "env": {
                "type": "single_asset_rl",  # 환경 유형 설정
                "initial_capital": 10000.0,
                "trading_fee": 0.001,
                "risk_adjusted_reward": True
            },
            "data": {
                "data_path": temp_data_path  # 임시 데이터 파일 경로 설정
            }
        }
        
        # 기존 설정과 검색 설정 병합
        full_config = {**config, **search_config}
        
        def train_func(trial_config):
            """Ray Tune 학습 함수"""
            # 설정 병합
            trial_full_config = {**full_config}
            
            # 하이퍼파라미터 업데이트
            for param_key, param_value in trial_config.items():
                if param_key != "_full_config":  # Ray Tune 내부 변수 무시
                    parts = param_key.split(".")
                    nested_dict = trial_full_config
                    for part in parts[:-1]:
                        if part not in nested_dict:
                            nested_dict[part] = {}
                        nested_dict = nested_dict[part]
                    nested_dict[parts[-1]] = param_value
            
            # 환경과 에이전트 생성
            env = create_env(df, trial_full_config)
            agent = create_agent(env, trial_full_config)
            
            # MLflow 매니저 설정
            mlflow_manager = MLflowManager(
                experiment_name="SingleAssetRL_Hyperopt",
                tracking_dir="./mlruns"
            )
            
            # 학습 및 평가
            with mlflow_manager.start_run() as run:
                # 하이퍼파라미터 로깅
                flat_params = {}
                for param_key, param_value in trial_config.items():
                    if param_key != "_full_config":
                        flat_params[param_key] = param_value
                mlflow_manager.log_params(flat_params)
                
                # 학습 및 평가 실행
                results = train_and_evaluate(agent, env, trial_full_config, mlflow_manager)
                
                # Ray Tune에 결과 반환
                from ray import tune
                tune.report(avg_reward=results["avg_reward"])
                
                return results
        
        # Ray Tune으로 하이퍼파라미터 최적화 실행
        logger.info("하이퍼파라미터 최적화 시작")
        search_space = create_search_space(full_config)
        best_config, optimization_results = run_hyperparameter_optimization(full_config)
        
        logger.info(f"최적화 결과: {optimization_results}")
        logger.info(f"최적 설정: {best_config}")
        
        # 최적 파라미터로 최종 학습
        logger.info("최적 파라미터로 최종 학습 실행")
        env = create_env(df, best_config)
        agent = create_agent(env, best_config)
        
        mlflow_manager = MLflowManager(
            experiment_name="SingleAssetRL_Final",
            tracking_dir="./mlruns"
        )
        
        with mlflow_manager.start_run("best_model_training") as run:
            mlflow_manager.log_params(best_config)
            final_results = train_and_evaluate(agent, env, best_config, mlflow_manager)
        
        return final_results
    
    finally:
        # 임시 디렉토리 및 파일 정리
        import shutil
        logger.info(f"임시 디렉토리 정리: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"임시 디렉토리 정리 중 오류 발생: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="단일 자산 RL 환경에서 PPO 에이전트 디버깅")
    parser.add_argument("--use-mlflow", action="store_true", help="MLflow 실험 추적 활성화")
    parser.add_argument("--hyperopt", action="store_true", help="하이퍼파라미터 최적화 실행")
    parser.add_argument("--episodes", type=int, default=5, help="학습할 에피소드 수")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="학습률")
    parser.add_argument("--days", type=int, default=14, help="데이터 로드할 일 수")
    parser.add_argument("--verbose-debug", action="store_true", help="상세 디버깅 정보 출력")
    parser.add_argument("--rollout-steps", type=int, default=1024, help="로올아웃 스텝 수 (버퍼 크기)")
    parser.add_argument("--quiet-debug", action="store_true", help="PolicyNetwork 로거를 ERROR 수준으로 설정하여 디버그 메시지 제거")
    args = parser.parse_args()
    
    # 로거 설정 함수 호출
    configure_loggers(args.quiet_debug)
    
    # 디버깅 로그 레벨 설정
    if args.verbose_debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("PPOAgent").setLevel(logging.DEBUG)
        logging.getLogger("SingleAssetRLTradingEnv").setLevel(logging.DEBUG)
    
    # MLflow 매니저 초기화
    mlflow_manager = None
    if args.use_mlflow and MLFLOW_AVAILABLE:
        try:
            mlflow_manager = MLflowManager(
                experiment_name="SingleAssetRL_Debug",
                tracking_dir="./mlruns"
            )
            logger.info("MLflow 매니저 초기화 완료")
        except Exception as e:
            logger.error(f"MLflow 매니저 초기화 실패: {str(e)}")
            mlflow_manager = None
    
    # 사용자 설정 - args.days일 정도의 데이터
    import datetime
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    logger.info(f"데이터 기간: {start_date} ~ {end_date} ({args.days}일)")
    
    # 1) 실제 데이터 로드
    df = load_real_data(symbol, timeframe, start_date, end_date)
    
    # df가 너무 작으면 학습이 잘 안 될 수 있으니, shape나 null 값 체크
    if df.shape[0] < 100:
        logger.warning("데이터가 매우 적습니다. 학습 결과가 좋지 않을 수 있습니다.")
    
    # 하이퍼파라미터 최적화 모드
    if args.hyperopt and MLFLOW_AVAILABLE:
        logger.info("하이퍼파라미터 최적화 모드로 실행합니다.")
        config = {
            "training": {
                "num_episodes": args.episodes
            },
            "agent": {
                "learning_rate": args.learning_rate
            }
        }
        hyperparameter_optimization(df, config)
        return
    
    # 2) 기본 설정 및 학습 실행
    config = {
        "env": {
            "initial_capital": 10000.0,
            "trading_fee": 0.001,
            "window_size": 20,
            "max_position_size": 1.0,
            "risk_adjusted_reward": True,
            "sharpe_lookback": 30,
            "sharpe_weight": 0.5,
            "drawdown_penalty": True,
            "max_drawdown_penalty_threshold": 0.1,
            "apply_slippage": True,
            "slippage_factor": 0.0005,
            "partial_fills": True,
            "min_fill_rate": 0.8,
            "volume_slippage_factor": 0.1
        },
        "agent": {
            "learning_rate": args.learning_rate,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "n_epochs": 5,
            "batch_size": 64,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "rollout_steps": args.rollout_steps  # 로올아웃 스텝 수 설정
        },
        "training": {
            "num_episodes": args.episodes,
            "update_interval": 128
        },
        "paths": {
            "model_path": f"ppo_agent_single_asset_{args.days}days.pt"
        }
    }
    
    # 환경과 에이전트 생성
    env = create_env(df, config)
    agent = create_agent(env, config)
    
    logger.info(f"에이전트 설정: 로올아웃 스텝={args.rollout_steps}, 학습률={args.learning_rate}")
    
    # MLflow 실험 추적
    if mlflow_manager is not None:
        with mlflow_manager.start_run("debug_training") as run:
            # 설정 로깅
            mlflow_manager.log_params({
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "data_shape": f"{df.shape}",
                "window_size": config["env"]["window_size"],
                "learning_rate": config["agent"]["learning_rate"],
                "episodes": config["training"]["num_episodes"],
                "rollout_steps": config["agent"]["rollout_steps"]
            })
            
            # 학습 및 평가 실행
            results = train_and_evaluate(agent, env, config, mlflow_manager)
            
            # 학습 URL 표시
            if hasattr(mlflow_manager, "get_run_url"):
                run_url = mlflow_manager.get_run_url()
                if run_url:
                    logger.info(f"MLflow 실험 URL: {run_url}")
    else:
        # MLflow 없이 학습
        results = train_and_evaluate(agent, env, config)
    
    # 추가 디버깅 정보 출력
    if "agent_stats" in results:
        logger.info("=== PPO 학습 디버깅 요약 ===")
        stats = results["agent_stats"]
        
        if stats["episode_data"]:
            logger.info("에피소드별 통계:")
            for ep_data in stats["episode_data"]:
                logger.info(f"에피소드 {ep_data['episode']+1}: 보상={ep_data['reward']:.4f}, 표준편차={ep_data['mean_std']:.4f}")
        
        logger.info(f"평균 보상: {results['avg_reward']:.4f}")
    
    logger.info("학습 완료!")


if __name__ == "__main__":
    main() 