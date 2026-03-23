# Trading Bot 사용 가이드

> RTX 3060 Ti PC에서 RL 앙상블 트레이딩 봇을 실행하는 완전한 가이드입니다.

---

## 목차

1. [Quick Start (5분)](#quick-start)
2. [핵심 개념](#핵심-개념)
3. [시스템 구성요소](#시스템-구성요소)
4. [일상 운영](#일상-운영)
5. [Hyperparameter 튜닝 가이드](#hyperparameter-튜닝)
6. [고급 설정](#고급-설정)
7. [트러블슈팅](#트러블슈팅)

---

## Quick Start

### 처음 설치 (5분)

```bash
# 1. 환경 설정 (Python 3.10+, RTX 3060 Ti)
python setup_local.py --gpu 3060ti

# 2. 데이터 수집 (BTC 1년치 1시간봉 + 크로스애셋 + 대안 데이터)
python scripts/fetch_data.py --asset BTCUSDT --period 1y --interval 1h --cross-assets --alt-data

# 3. 학습 시작 (약 20시간, 하룻밤)
python -m training.train_pipeline --config config/local_3060ti.yaml

# 4. 결과 모니터링 (웹 대시보드)
streamlit run deployment/web_interface/app.py
```

### M2 MacBook에서 빠른 테스트

```bash
# 환경 설정
python setup_local.py --gpu m2

# 데이터 수집 (기본)
python scripts/fetch_data.py --asset BTCUSDT --period 3m --interval 1h

# 빠른 테스트 (100K steps, ~30분)
python -m training.train_pipeline --config config/local_m2.yaml
```

---

## 핵심 개념

### Ensemble Agent란?

이 봇은 4개의 서로 다른 AI 에이전트가 함께 결정을 내립니다.

| Agent | 성격 | 언제 강한가 |
|-------|------|-----------|
| **PPO (CVaR)** | 보수적 | 변동성 높을 때, 꼬리 위험 관리 |
| **SAC** | 균형형 | 안정적인 trending 시장 |
| **TD3** | 공격적 | 명확한 추세가 있을 때 |
| **FLAG-Trader** | 적응형 | 뉴스/이벤트가 있을 때, LLM 기반 |

Meta-Controller가 현재 시장 상황을 보고 4개 에이전트의 의견에 **가중치**를 달리 부여합니다.

### Regime Detection (시장 상태 분류)

HMM(Hidden Markov Model)이 시장을 3가지 상태로 분류합니다:

- **State 0 (Trending)**: 낮은 변동성, 추세 있음 → PPO/SAC에 높은 가중치
- **State 1 (Ranging)**: 중간 변동성, 횡보 → 균등 가중치
- **State 2 (Crisis)**: 높은 변동성, 급락/급등 → 방어 모드 (포지션 줄임)

### CVaR (꼬리 위험 관리)

CVaR = "최악의 5% 상황에서의 평균 손실"

일반 RL은 평균 수익만 최대화하려 합니다. CVaR 제약을 추가하면 **극단적 손실**도 함께 제어합니다.

- `cvar_alpha: 0.05` = 5% 최악 시나리오
- `cvar_threshold: -0.02` = 최악 상황에서도 2% 이상 손실 제한

### Walk-Forward Validation (과적합 방지)

과거 데이터를 시간 순서대로 학습/검증합니다. 미래 데이터를 학습에 쓰지 않도록 설계.

```
[학습 252일] [검증 63일] [테스트 21일]
              → [학습 252일] [검증 63일] [테스트 21일]
                            → ...  (5 folds)
```

OOS(Out-of-Sample) Sharpe가 일관되게 높아야 실제로 쓸 수 있는 모델입니다.

---

## 시스템 구성요소

```
trading_bot/
├── config/
│   ├── local_3060ti.yaml    ← PC 학습용 설정 (주로 이걸 씀)
│   ├── local_m2.yaml        ← MacBook 테스트용
│   └── training_config.yaml ← 전체 설정 레퍼런스
│
├── agents/
│   ├── sb3/                 ← SB3 기반 PPO/SAC/TD3 + CVaR
│   ├── llm_rl/              ← FLAG-Trader (LLM 기반)
│   └── ensemble/            ← Meta-Controller + 통신 프로토콜
│
├── training/
│   ├── data/                ← 특징 추출 (기술 지표, 크로스애셋, 온체인)
│   ├── regime/              ← HMM 시장 상태 감지
│   ├── monitoring/          ← Drift Detection
│   └── continual/           ← 자동 재학습 파이프라인
│
├── envs/
│   └── single_asset_rl_env.py  ← RL 거래 환경
│
├── scripts/
│   └── fetch_data.py        ← 데이터 자동 수집
│
├── setup_local.py           ← 최초 설치 스크립트
└── deployment/
    └── web_interface/       ← Streamlit 대시보드
```

---

## 일상 운영

### 데이터 갱신

매일 새 데이터를 받아서 최신 상태 유지:

```bash
# 기본 (BTCUSDT 1시간봉, 어제부터 현재까지 증분)
python scripts/fetch_data.py --asset BTCUSDT --interval 1h

# 전체 (크로스애셋 + 대안 데이터 포함)
python scripts/fetch_data.py --asset BTCUSDT --interval 1h --cross-assets --alt-data

# 자동 갱신 설정 (cron 또는 Windows Task Scheduler 설정법 출력)
python scripts/fetch_data.py --asset BTCUSDT --interval 1h --schedule daily
```

스크립트는 마지막 수집 시점을 기억해서 **증분 업데이트**만 합니다 (중복 없음).

### 모델 재학습 시점 판단

Drift Detection이 자동으로 경고를 줍니다:

```
WARNING: Concept drift detected — reward distribution shifted (ADWIN)
→ 재학습을 권장합니다.
```

이 경고가 뜨면:

```bash
# 1. 최신 데이터 수집
python scripts/fetch_data.py --asset BTCUSDT --interval 1h

# 2. 재학습 (Walk-Forward 포함)
python -m training.train_pipeline --config config/local_3060ti.yaml --retrain

# 3. 검증 결과 확인
mlflow ui --port 5000
# → http://localhost:5000 에서 OOS Sharpe 비교
```

> 재학습은 기존 모델을 덮어쓰지 않습니다. OOS 검증 통과 시에만 자동 교체됩니다.

### Paper Trading 모니터링

```bash
# Paper trading 시작 (실제 거래 없음, 시뮬레이션)
python -m deployment.paper_trading --config config/local_3060ti.yaml

# 대시보드
streamlit run deployment/web_interface/app.py
```

대시보드에서 확인할 것:
- 누적 수익률 vs Buy & Hold
- Sharpe Ratio / Max Drawdown
- 각 에이전트의 현재 가중치
- 현재 Regime (Trending/Ranging/Crisis)
- Drift 경고 여부

### 설정 변경 가이드

`config/local_3060ti.yaml` 주요 파라미터:

```yaml
training:
  total_timesteps: 500000   # 학습 steps. 늘리면 성능 ↑, 시간 ↑
  device: "cuda"            # GPU 사용. CPU로 바꾸면 매우 느림

ensemble:
  agents:
    - type: "flag_trader"
      params:
        lora_rank: 8        # 메모리 부족하면 4로 줄임

validation:
  n_folds: 5                # Walk-Forward fold 수. 늘리면 더 엄격

cvar:
  threshold: -0.02          # 허용 최대 꼬리 손실. -0.03으로 완화 가능
```

---

## Hyperparameter 튜닝

### Optuna 사용법

```bash
# 20회 자동 튜닝 (3060 Ti 기준 ~7시간)
python -m training.hyperopt.hyperopt_ray \
  --config config/local_3060ti.yaml \
  --n_trials 20

# 결과 확인
mlflow ui --port 5000
# Experiments → hyperopt → 각 trial의 OOS Sharpe 비교
```

### 주요 파라미터와 영향

| 파라미터 | 범위 | 영향 |
|---------|------|------|
| `learning_rate` | 1e-5 ~ 1e-3 | 높으면 빠르게 학습, 불안정 |
| `n_steps` | 256 ~ 2048 | 높으면 안정적, 메모리 ↑ |
| `ent_coef` | 0.001 ~ 0.1 | 높으면 탐험 ↑, 수렴 느림 |
| `gamma` | 0.95 ~ 0.999 | 높으면 장기 보상 중시 |
| `clip_range` | 0.1 ~ 0.4 | PPO 업데이트 크기 제한 |

### 추천 튜닝 순서

1. **learning_rate** — 가장 중요. 먼저 fix
2. **n_steps + batch_size** — 메모리 범위 내에서 최대화
3. **ent_coef** — 충분한 탐험 보장
4. **reward weights** — 보상 함수 균형 조정
5. **CVaR threshold** — 마지막에 리스크 조정

---

## 고급 설정

### GTrXL Feature Extractor (더 강력, 더 느림)

```yaml
# config/local_3060ti.yaml
agent:
  feature_extractor: "gtrxl"   # conv1d → gtrxl
  feature_extractor_kwargs:
    n_layers: 2        # 3 → 2 (VRAM 절약)
    d_model: 64        # 128 → 64
    memory_len: 32     # 64 → 32
```

GTrXL은 긴 시퀀스 패턴을 더 잘 학습하지만 VRAM을 2-3배 더 씁니다.

### On-Chain 데이터 활성화 (크립토 전용)

```yaml
# config/local_3060ti.yaml
cross_asset:
  enabled: true
  assets: ["SPY", "DXY", "GC=F", "ETH-USD"]
  vix_asset: "^VIX"
```

On-Chain 데이터 수집:
```bash
python scripts/fetch_data.py --asset BTCUSDT --period 1y --alt-data
```

### DT Forecaster 활성화 (예측 기반 관측 확장)

```yaml
dt_forecaster:
  enabled: true    # false → true
  hidden_size: 64
  n_layer: 2
```

활성화하면 observation에 `predicted_return_1step`, `predicted_return_5step`, `prediction_confidence` 3개 특징이 추가됩니다.

---

## 트러블슈팅

### GPU 메모리 부족 (CUDA out of memory)

```yaml
# config/local_3060ti.yaml에서:
agent:
  sb3_params:
    ppo:
      n_steps: 512          # 1024 → 512
      batch_size: 32         # 64 → 32

ensemble:
  agents:
    - type: "flag_trader"
      params:
        lora_rank: 4         # 8 → 4
        batch_size: 2        # 4 → 2
```

또는 FLAG-Trader를 임시 비활성화:
```yaml
ensemble:
  agents:
    - type: "sb3_cvar_ppo"
    - type: "sb3_sac"
    - type: "sb3_td3"
    # flag_trader 라인 제거
```

### 데이터 수집 실패

```bash
# Binance 접속 불가 시 Bybit으로 변경
python scripts/fetch_data.py --asset BTCUSDT --exchange bybit --interval 1h

# yfinance 실패 시 (방화벽 등)
# cross-assets 없이 기본 수집만
python scripts/fetch_data.py --asset BTCUSDT --interval 1h
# → cross-asset 특징은 0으로 채워짐 (graceful degradation)
```

### 학습이 수렴하지 않음 (Reward 계속 낮음)

1. **learning_rate를 낮춤**: `3e-4` → `1e-4`
2. **데이터 양 확인**: 최소 1년치 이상 권장
3. **Reward 스케일 확인**: `reward_scaling` 파라미터 조정
4. **CVaR threshold 완화**: `-0.02` → `-0.05` (제약 일시 완화)

### MLflow UI 접속 안 됨

```bash
# 포트 변경
mlflow ui --port 5001

# 또는 직접 결과 확인
python -c "
import mlflow
runs = mlflow.search_runs(experiment_names=['local_3060ti'])
print(runs[['run_id', 'metrics.sharpe_ratio', 'metrics.max_drawdown']].head(10))
"
```

### 흔한 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: ccxt` | ccxt 미설치 | `pip install ccxt` |
| `ModuleNotFoundError: yfinance` | yfinance 미설치 | `pip install yfinance` |
| `AssertionError: check_env` | Observation space 불일치 | `window_size`와 `n_features` 확인 |
| `CUDA error: device-side assert` | NaN in observation | 데이터 전처리 확인, `validate_data_pipeline.py` 실행 |
| `hmmlearn import error` | hmmlearn 미설치 | `pip install hmmlearn` (없으면 threshold fallback 사용) |

---

## Phase 3 기능 (Week 30-34)

### Monitoring 설정법 (Telegram bot)

1. **Bot 생성**: Telegram에서 `@BotFather`에게 `/newbot` 명령어를 보내어 토큰을 발급받습니다.
2. **Chat ID 확인**: Bot에 메시지를 보낸 후 `https://api.telegram.org/bot<TOKEN>/getUpdates`에서 `chat.id`를 확인합니다.
3. **환경 변수 설정**:
   ```bash
   export TELEGRAM_BOT_TOKEN="1234567890:ABCdef..."
   export TELEGRAM_CHAT_ID="-1001234567890"
   ```
4. **config 변경** (`config/local_3060ti.yaml`):
   ```yaml
   monitoring:
     alert_channels: ["console", "telegram"]
     drawdown_alert_threshold: 0.10
     daily_loss_alert: -500
   ```
5. **동작 확인**:
   ```python
   from deployment.monitoring.alerter import TradingAlerter
   alerter = TradingAlerter({"alert_channels": ["telegram"]})
   alerter.send_alert("테스트 알림", level="WARNING")
   ```

---

### Statistical Test 결과 읽는 법

`StrategyStatisticalTests`는 세 가지 주요 지표를 제공합니다.

**Bootstrap Sharpe CI** (`bootstrap_sharpe_ci`):
```
Sharpe 95% CI: [lo=0.42, point=0.87, hi=1.31]
```
- `lo`가 0보다 크면 95% 신뢰구간에서 양의 Sharpe → 통계적으로 유의한 전략
- 구간이 넓을수록 샘플 수가 부족하거나 수익률 변동성이 큼

**Permutation Test** (`permutation_test`):
```
p-value: 0.031  →  유의 (α=0.05 기준 통과)
p-value: 0.210  →  비유의, 랜덤과 구분 불가
```
- `p < 0.05` → 전략 성능이 우연이 아닐 가능성 높음
- `p > 0.05` → 과적합 의심, 하이퍼파라미터 또는 데이터 재검토

**Deflated Sharpe Ratio (DSR)**:
- 여러 번 하이퍼파라미터 탐색 시 다중검정 보정
- `n_trials=50`으로 설정하면 50회 탐색을 보정
- `DSR > 0.95` → 유의한 전략, `DSR < 0.5` → 과적합 강한 의심

---

### Regime별 성능 차이가 클 때 대응법

`regime_conditional_report`로 Bull / Bear / Sideways 성능을 분리 확인:

```python
report = st.regime_conditional_report(returns, regimes)
# {0: {sharpe: -0.3, max_drawdown: -0.15, ...},   # Bear
#  1: {sharpe: 0.5, ...},                          # Sideways
#  2: {sharpe: 1.2, ...}}                          # Bull
```

- **Bear Sharpe < -0.5**: `adjust_for_regime`의 `vol_factor`를 강화하거나 Stop-Loss 강화
- **IS/OOS 비율 > 2**: Walk-forward 경고 확인 → 학습 데이터 기간 단축 검토
- **특정 Regime에서만 수익**: 전략이 regime-specific한지 확인, MetaController 가중치 조정

---

### 일일 체크리스트

매 거래일 시작 전/후 아래 항목을 확인합니다.

**아침 (거래 시작 전)**:
- [ ] `docker-compose ps` — 모든 서비스 `healthy` 상태 확인
- [ ] MLflow에서 전일 Drift 감지 횟수 확인 (`drift_detection.n_detections`)
- [ ] 최근 7일 OOS Sharpe 추이 확인 (하락 추세면 재학습 검토)

**저녁 (거래 종료 후)**:
- [ ] `alerter.alert_history`에서 당일 Drawdown/Loss 알림 확인
- [ ] 일일 P&L이 `daily_loss_alert` 임계값 50% 초과 시 포지션 축소 검토
- [ ] Regime 분포 확인 — High-vol 비율 > 60% 시 다음날 `adjust_for_regime` 활성화 유지

---

## 문의 / 참고 자료

- 개발 가이드라인: `DEVELOPMENT_GUIDELINES.md`
- 변경 이력: `CHANGELOG.md`
- 멀티에이전트 아키텍처: `docs/MULTI_AGENT_MANAGER.md`
- MLflow 실험 관리: `docs/training_validation_guide.md`
- Phase 3 통합 테스트: `tests/test_phase3_integration.py`
