# AI-Powered Trading Bot

A comprehensive trading system that combines reinforcement learning, risk management, and real-time execution.

## Core Components

### 1. Reinforcement Learning
- PPO-based trading agent
- Custom trading environments
- Multi-agent support
- Hyperparameter optimization with Ray Tune
- MLflow experiment tracking

### 2. Risk Management
- Position size control
- Portfolio VaR monitoring
- Multi-asset correlation tracking
- Dynamic risk adjustment
- Advanced backtesting

### 3. Live Trading
- Real-time execution via CCXT
- Paper trading support
- Order types: limit, stop-limit, trailing-stop
- Rate limiting and error handling
- Network resilience

### 4. Data Pipeline
- OHLCV data processing
- Technical indicators (TA-Lib)
- Market scenario simulation
- Multi-asset data handling
- Real-time data streaming

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Training an Agent**
```python
from training.agents import PPOAgent
from training.environments import TradingEnv

# Initialize and train agent
agent = PPOAgent(config)
env = TradingEnv(data)
agent.train(env)
```

2. **Backtesting**
```python
from training.utils.risk_backtest import RiskAwareBacktester
from training.utils.risk_config import RiskConfig

# Run backtest
backtester = RiskAwareBacktester(data, risk_config)
results = backtester.run(agent)
```

3. **Live Trading**
```python
from trading.live import LiveTradingEnvironment

# Start live trading
live_env = LiveTradingEnvironment(
    exchange_config=config,
    risk_config=risk_config
)
live_env.run(agent)
```


## Development
project_root/
├─ config/
│  ├─ default_config.yaml         # 기본 설정(모델 파라미터, 데이터 경로, 환경 변수)
│  ├─ hyak_config.yaml            # Hyak GPU 클러스터 전용 설정
│  ├─ env_settings.yaml           # 시장 환경 파라미터(거래 수수료, 슬리피지, 최소 거래 단위 등)
│  ├─ model_architectures.yaml    # 사용 가능한 모델 아키텍처 정의(PPO, SAC, DDPG, Transformer 기반 등)
│  └─ credentials.yaml (optional) # API Key, Secret Key 등 (Vault 연동 권장)
│
├─ data/
│  ├─ raw/
│  │  ├─ crypto/                  # ccxt로 받은 원본 크립토 시세 데이터
│  │  ├─ equity/                  # 주식 시세, 퀀들, 야후 파이낸스 등
│  │  └─ other_assets/            # 선물, 옵션 등 추가 자산
│  ├─ processed/
│  │  ├─ qlib_data/               # Qlib로 전처리한 후 저장되는 시계열 데이터
│  │  ├─ features/                # featurization 후 최종 모델 입력용
│  │  └─ cache/                   # 중간 캐싱된 데이터(Dask/Ray로 병렬 처리 시 유용)
│  └─ utils/
│     ├─ data_loader.py           # 원본데이터 로딩 유틸
│     ├─ data_cleaning.py         # 결측치, 이상치 처리
│     ├─ feature_generation.py    # 기술적 지표, NLP sentiment, on-chain 피처추출
│     └─ validation.py            # 데이터 검증(Great Expectations 등)
│
├─ envs/
│  ├─ base_env.py                 # 기본 OpenAI Gym 호환 환경 클래스 (단일 에이전트)
│  ├─ multi_agent_env.py          # 다중 에이전트 환경, PettingZoo/RLlib compatible
│  └─ wrappers.py                 # Observation, Action space 래퍼, Normalization, StackFrame 등
│
├─ agents/
│  ├─ base/
│  │  ├─ base_agent.py            # 에이전트 추상 클래스 (load/save, get_action 등 공통 인터페이스)
│  │  ├─ agent_factory.py         # agent 인스턴스화 유틸 (config 기반으로 PPO, SAC 등 만들기)
│  ├─ strategies/
│  │  ├─ market_maker.py          # 시장 메이커 전략 에이전트 구현
│  │  ├─ momentum.py              # 모멘텀 기반 전략 에이전트
│  │  ├─ mean_reversion.py        # 역추세 전략 에이전트
│  │  └─ meta_agent.py            # 메타 에이전트(상위 정책으로 다른 에이전트 조율)
│  └─ models/
│     ├─ policy_network.py        # 정책 신경망 정의 (MLP, LSTM, Transformer 등)
│     ├─ value_network.py         # 가치함수 신경망 정의
│     ├─ transformer_policy.py    # Transformer 기반 정책망 (시계열 패턴 학습)
│     └─ custom_layers.py         # Custom layer, attention module 등
│
├─ training/
│  ├─ train_local.ipynb           # 로컬 개발용 간단한 학습 notebook
│  ├─ train.py                    # 명령행으로 실행 가능한 학습 스크립트(단일 또는 멀티에이전트)
│  ├─ train_hyak.py               # Hyak GPU 클러스터용 스크립트 (Slurm job submit)
│  ├─ evaluation.py               # 백테스트 및 성능 평가 스크립트 (Sharpe, Sortino, MDD 등)
│  ├─ hyperparameter_search.py    # Optuna/Ray Tune 기반 하이퍼파라미터 최적화
│  └─ utils/
│     ├─ callbacks.py             # RL 학습용 콜백(학습률 스케줄, early stopping)
│     ├─ metrics.py               # 평가 지표 계산 함수
│     └─ logger.py                # MLflow/W&B 연동 로거
│
├─ deployment/
│  ├─ web_interface/
│  │  ├─ app.py                   # Streamlit 대시보드 진입점
│  │  ├─ pages/                   # Streamlit multi-page 구조(모델 리스트, 파라미터 튜닝, 결과 모니터링)
│  │  ├─ static/                  # CSS, JS
│  │  └─ templates/               # 추가 HTML 템플릿(필요시)
│  ├─ api/
│  │  ├─ main.py                  # FastAPI 엔드포인트 진입점
│  │  ├─ routers/
│  │  │  ├─ model_routes.py       # /models: 모델 목록 조회, 특정 모델 학습 요청
│  │  │  ├─ data_routes.py        # /data: 데이터 파이프라인 제어(API)
│  │  │  └─ training_routes.py    # /training: 학습 시작/중단, 진행상황 조회
│  │  ├─ schemas/                 # Pydantic 스키마 정의 (Request/Response)
│  │  └─ auth.py                  # 인증/인가 로직(JWT, OAuth2)
│  └─ inference/
│     ├─ run_inference.py         # 학습 완료 모델로 실시간 예측(백테스트 또는 페이퍼트레이드)
│     └─ realtime_adapter.py      # 실시간 데이터 스트림 처리(추후 확장, ZeroMQ,Kafka 연동 가능)
│
├─ scripts/
│  ├─ run_hyak_job.sh             # Slurm 배치 스크립트 예시
│  ├─ setup_hyak_env.sh           # Hyak 환경 세팅 스크립트(모듈 로드, conda env 등)
│  ├─ local_dev_setup.sh          # 로컬 개발환경 셋업(pip install 등)
│  ├─ model_export.py             # 모델 가중치 Export/Import 스크립트(MLflow 연동)
│  └─ automations/
│     ├─ daily_update.sh          # 매일 데이터 업데이트 cron job
│     ├─ weekly_retrain.sh        # 매주 재학습 파이프라인
│     └─ notify_slack.py          # Slack/Webhook 알림
│
├─ tests/
│  ├─ test_data_pipeline.py       # 데이터 로딩, 전처리, 피처링 테스트
│  ├─ test_env.py                 # 환경 reset/step 동작 테스트
│  ├─ test_models.py              # 모델 학습 step sanity check
│  ├─ test_web_interface.py       # API endpoint, Streamlit 프론트엔드 테스트(Playwright/Selenium)
│  └─ test_integration.py         # 통합테스트(데이터→모델학습→평가→UI)
│
├─ ci/
│  ├─ github_actions.yaml         # GitHub Actions CI/CD 워크플로우 정의(유닛테스트, 린트, 빌드)
│  ├─ docker/
│  │  ├─ Dockerfile               # 컨테이너 이미지 빌드
│  │  └─ docker-compose.yaml      # 로컬 서비스(MLflow, DB) 테스트용
│  └─ jenkins/                    # Jenkinsfile 혹은 Jenkins 설정(옵션)
│
└─ README.md
### Testing
```bash
python -m pytest tests/
```

### Code Quality
- Follow PEP 8
- Add docstrings
- Update CHANGELOG.md

## Documentation

- See class/method docstrings
- Check CHANGELOG.md
- Review test files

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Update documentation
5. Submit pull request

## License

MIT License