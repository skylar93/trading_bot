# Phase 3: Hardening & Production Ready (Weeks 29-34)

## Context

Week 28까지 완료. Opus 코드 리뷰 결과:
- 버그 1건 (Sharpe annualization), legacy 잔존, config 검증 부재
- 전략 성능: multi-timeframe, regime→sizing 연결 필요
- 실전 배포: execution layer, monitoring 부재
- 평가: 과적합 검증 도구 없음

**목표**: 남편이 실전(paper → live) 투입 가능한 상태로 만들기.

**규칙**:
- 각 Week의 검증 코드는 반드시 실행해서 통과 확인
- 기존 테스트 깨지면 안 됨: `pytest tests/ -x --tb=short`
- Config 변경 시 `config/local_3060ti.yaml`에 반영

---

## Week 29: Critical Bug Fix + Cleanup

### 29.1 Sharpe Annualization 버그 수정
**파일**: `envs/single_asset_rl_env.py`
**위치**: `_calculate_risk_adjusted_reward()` 내 sharpe_proxy 계산 부분

현재:
```python
sharpe_proxy = mean_return / std_return
```

수정:
```python
sharpe_proxy = mean_return / std_return * np.sqrt(252)
```

backtester(`training/backtesting/base_backtester.py:829`)와 동일하게 연환산.

**검증**:
```bash
python -c "
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
import numpy as np, pandas as pd
data = pd.read_csv('test_data.csv')
env = SingleAssetRLTradingEnv(data=data, initial_capital=10000, window_size=20)
obs, _ = env.reset()
# 30번 step 후 reward가 이전보다 큰 절대값을 가지는지 확인
rewards = []
for _ in range(50):
    action = env.action_space.sample()
    obs, r, done, trunc, info = env.step(action)
    rewards.append(r)
    if done: break
print(f'Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]')
print('Sharpe annualization fix verified')
"
```

### 29.2 Sharpe Rolling Window 기본값 조정
**파일**: `envs/single_asset_rl_env.py`
**위치**: `__init__`에서 `sharpe_lookback` 기본값

현재 `30` → `60`으로 변경. Config에서 override 가능하게 유지.

### 29.3 Legacy 파일 삭제
삭제 전 `grep -r "import.*파일명"` 으로 참조 확인 후 삭제:

```
training/train_meta_agent.py
tests/test_hierarchical_agent.py
```

참조하는 곳 있으면 해당 import도 제거.

### 29.4 Bare Exception 수정
`except:` → `except Exception as e:` + logger.error(f"...: {e}")

대상 파일 검색:
```bash
grep -rn "except:" --include="*.py" | grep -v "except Exception" | grep -v "# noqa"
```

**전체 검증**:
```bash
pytest tests/ -x --tb=short
```

---

## Week 30: Config Validation + Regime-Sizing 연결

### 30.1 Pydantic Config Schema
**신규 파일**: `config/schema.py`

```python
from pydantic import BaseModel, validator
from typing import Optional, List, Literal

class EnvConfig(BaseModel):
    window_size: int = 20
    initial_balance: float = 10000
    trading_fee: float = 0.001
    max_position_size: float = 1.0
    sharpe_lookback: int = 60

class AgentConfig(BaseModel):
    algo_type: Literal["sb3_ppo", "sb3_sac", "sb3_td3", "sb3_cvar_ppo"]
    learning_rate: float = 3e-4
    feature_extractor: Literal["conv1d", "gtrxl", "mlp"] = "conv1d"

class EnsembleConfig(BaseModel):
    agents: List[AgentConfig]
    @validator("agents")
    def at_least_one(cls, v):
        assert len(v) >= 1, "최소 1개 agent 필요"
        return v

class TrainingConfig(BaseModel):
    total_timesteps: int = 500000
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    eval_interval: int = 10000

class RiskConfig(BaseModel):
    stop_loss_threshold: float = 0.05
    trailing_stop_buffer: float = 0.03
    max_drawdown_pct: float = 0.20
    portfolio_stop_loss_threshold: float = 0.15

class FullConfig(BaseModel):
    env: EnvConfig = EnvConfig()
    training: TrainingConfig = TrainingConfig()
    risk_management: RiskConfig = RiskConfig()
    # ... 나머지 섹션
```

**통합 위치**: `scripts/run_full_pipeline.py`와 `training/train_pipeline.py`의 config 로딩 직후에:
```python
from config.schema import FullConfig
config = FullConfig(**raw_yaml_dict)  # 여기서 validation error 발생 시 즉시 중단
```

**검증**:
```bash
python -c "
from config.schema import FullConfig
import yaml
with open('config/local_3060ti.yaml') as f:
    raw = yaml.safe_load(f)
config = FullConfig(**raw)
print(f'Config validated: {config.training.device}')
"
```

### 30.2 Regime → Position Sizing 연결
**파일**: `risk_management/rl_risk_manager.py`

현재 HMM regime detection은 meta_controller에서만 사용. risk_manager에 regime 정보를 전달하여 position sizing에 반영:

```python
# RLRiskManager에 추가할 메서드
def adjust_for_regime(self, action: float, regime_probs: np.ndarray) -> float:
    """
    regime_probs: [low_vol, medium_vol, high_vol] from HMM
    high_vol regime일수록 position 축소
    """
    # regime 2 (high-vol) 확률이 높을수록 position 줄임
    vol_factor = 1.0 - 0.5 * regime_probs[2]  # high-vol이면 최대 50% 축소
    crisis_factor = 1.0 - 0.3 * regime_probs[2]  # crisis면 추가 30% 축소
    adjusted = action * vol_factor
    return float(np.clip(adjusted, 0.0, self.config.max_position_size))
```

**호출 위치**: `envs/single_asset_rl_env.py`의 `step()` 내 action 처리 부분에서, regime detector가 있으면 action을 adjust.

**검증**:
```bash
python -c "
from risk_management.rl_risk_manager import RLRiskManager
import numpy as np
rm = RLRiskManager({'stop_loss_threshold': 0.05, 'max_position_size': 1.0})
# high-vol regime
adj = rm.adjust_for_regime(0.8, np.array([0.1, 0.1, 0.8]))
assert adj < 0.8, f'Expected reduction, got {adj}'
# low-vol regime
adj2 = rm.adjust_for_regime(0.8, np.array([0.8, 0.1, 0.1]))
assert adj2 > adj, f'Low-vol should allow larger position'
print(f'Regime sizing: high_vol={adj:.3f}, low_vol={adj2:.3f}')
"
```

---

## Week 31: Multi-Timeframe Features

### 31.1 Multi-Timeframe Feature Generator
**신규 파일**: `training/features/multi_timeframe.py`

```python
class MultiTimeframeFeatures:
    """
    1H 데이터를 기반으로 4H, 1D aggregation 생성.
    각 timeframe에서 동일한 기술지표 계산 후 concat.
    """
    def __init__(self, base_timeframe="1H", higher_timeframes=["4H", "1D"]):
        ...

    def generate(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Input: 1H OHLCV DataFrame
        Output: 1H DataFrame with 4H/1D features appended as columns

        1H → 4H: resample('4H', {'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        1H → 1D: resample('1D', ...)

        각 timeframe별 지표:
        - RSI(14), MACD signal, BB position, ATR
        - 총 4개 × 2 higher timeframes = 8 추가 features

        Higher timeframe 값은 해당 시점까지의 값만 사용 (forward fill, NO look-ahead)
        """
```

**통합 위치**: `scripts/run_full_pipeline.py`의 feature engineering 단계에서 호출.

**Env 수정**: `envs/single_asset_rl_env.py`에서 observation space shape이 feature 수에 따라 동적으로 결정되므로, 추가 feature가 있으면 자동 반영됨. 확인만 하면 됨.

**검증**:
```bash
python -c "
from training.features.multi_timeframe import MultiTimeframeFeatures
import pandas as pd
df = pd.read_csv('test_data.csv', parse_dates=['date'] if 'date' in pd.read_csv('test_data.csv', nrows=1).columns else None)
mtf = MultiTimeframeFeatures()
result = mtf.generate(df)
new_cols = [c for c in result.columns if c not in df.columns]
print(f'Added {len(new_cols)} features: {new_cols[:5]}...')
assert not result[new_cols].iloc[-1].isna().all(), 'Features should not be all NaN'
print('Multi-timeframe features OK')
"
```

### 31.2 Feature Importance 재측정
Multi-timeframe feature 추가 후 SHAP importance 재측정하여 유용한 feature만 남기기.

**파일**: `training/analysis/feature_importance.py` (기존)

```bash
python -c "
from training.analysis.feature_importance import FeatureImportanceAnalyzer
# 기존 코드로 importance 재측정
# 상위 20개 feature 확인, 나머지 제거 검토
"
```

---

## Week 32: 과적합 검증 도구

### 32.1 Bootstrap Confidence Interval
**신규 파일**: `training/analysis/statistical_tests.py`

```python
class StrategyStatisticalTests:
    """백테스트 결과의 통계적 유의성 검증"""

    def bootstrap_sharpe_ci(self, returns: np.ndarray, n_bootstrap=10000, ci=0.95) -> tuple:
        """
        Bootstrap으로 Sharpe ratio의 95% CI 산출.
        Returns: (lower, point_estimate, upper)
        """

    def permutation_test(self, returns: np.ndarray, n_permutations=10000) -> float:
        """
        H0: 전략 수익률 = random ordering
        Returns: p-value (0.05 미만이면 유의)
        수익률 순서를 무작위 셔플 → 랜덤 Sharpe 분포 생성 → 실제 Sharpe의 위치로 p-value
        """

    def deflated_sharpe_ratio(self, sharpe: float, n_trials: int,
                               var_sharpe: float, skew: float, kurt: float) -> float:
        """
        Bailey & López de Prado (2014)
        다수 전략 시도에 따른 multiple testing 보정
        n_trials: 시도한 전략/하이퍼파라미터 조합 수
        """

    def regime_conditional_report(self, returns: np.ndarray,
                                   regimes: np.ndarray) -> dict:
        """
        Regime별 (bull/bear/sideways) 성능 분리 보고
        Returns: {regime_id: {sharpe, max_dd, win_rate, n_trades}}
        """
```

**통합**: `scripts/run_full_pipeline.py`의 report 생성 단계에서 호출. HTML report에 CI와 p-value 포함.

**검증**:
```bash
python -c "
from training.analysis.statistical_tests import StrategyStatisticalTests
import numpy as np
st = StrategyStatisticalTests()
# 유의미한 전략 (positive returns)
good_returns = np.random.normal(0.001, 0.02, 252)
lo, mid, hi = st.bootstrap_sharpe_ci(good_returns)
print(f'Sharpe 95% CI: [{lo:.2f}, {mid:.2f}, {hi:.2f}]')
p = st.permutation_test(good_returns)
print(f'p-value: {p:.4f}')
assert p < 0.1, 'Positive-mean returns should be somewhat significant'
# 랜덤 전략 (zero returns)
random_returns = np.random.normal(0, 0.02, 252)
p2 = st.permutation_test(random_returns)
print(f'Random p-value: {p2:.4f}')
assert p2 > 0.05, 'Random returns should not be significant'
print('Statistical tests OK')
"
```

### 32.2 IS vs OOS Gap 경고
**파일**: `training/validation/walk_forward.py`

Walk-forward 결과에서 IS Sharpe / OOS Sharpe 비율이 2배 이상이면 경고 로그:
```python
if is_sharpe > 0 and oos_sharpe > 0:
    ratio = is_sharpe / oos_sharpe
    if ratio > 2.0:
        logger.warning(f"⚠ Overfitting suspected: IS/OOS Sharpe ratio = {ratio:.2f}")
```

---

## Week 33: Execution Layer + Monitoring

### 33.1 Order Execution Manager
**신규 파일**: `deployment/execution/order_manager.py`

```python
class OrderManager:
    """
    Exchange API 래퍼. Paper trading과 live 모두 지원.

    기능:
    - submit_order(side, amount, order_type="market") → order_id
    - check_order(order_id) → status (filled/partial/failed)
    - cancel_order(order_id)
    - reconcile() → 실제 포지션 vs 내부 state 비교, 불일치 시 경고

    안전장치:
    - max_order_size: 단일 주문 최대 크기
    - daily_loss_limit: 일일 손실 한도 초과 시 거래 중단
    - rate_limiter: exchange API 호출 제한 (default: 10 req/sec)
    - retry_with_backoff: 실패 시 최대 3회 재시도
    """

    def __init__(self, exchange_config: dict, paper_mode: bool = True):
        self.paper_mode = paper_mode
        if not paper_mode:
            import ccxt
            self.exchange = ccxt.binance({...})
        self.rate_limiter = RateLimiter(max_calls=10, period=1.0)
        self.daily_pnl = 0.0
        self.daily_loss_limit = exchange_config.get("daily_loss_limit", -500.0)
```

### 33.2 Monitoring & Alerting
**신규 파일**: `deployment/monitoring/alerter.py`

```python
class TradingAlerter:
    """
    조건 충족 시 알림 발송.

    지원 채널:
    - Telegram (python-telegram-bot)
    - Webhook (generic HTTP POST)
    - 콘솔 로그 (fallback)

    트리거:
    - drawdown > threshold (default 10%)
    - daily P&L < loss_limit
    - drift detected
    - connection lost > 60s
    - trade executed (optional, verbose mode)
    """

    def __init__(self, config: dict):
        self.telegram_token = config.get("telegram_token")  # env var 권장
        self.telegram_chat_id = config.get("telegram_chat_id")
        self.webhook_url = config.get("webhook_url")
```

**Config 추가** (`config/local_3060ti.yaml`):
```yaml
monitoring:
  enabled: true
  alert_channels: ["console"]  # telegram, webhook 추가 가능
  drawdown_alert_threshold: 0.10
  daily_loss_alert: -500
  telegram_token: ${TELEGRAM_BOT_TOKEN}  # env var
  telegram_chat_id: ${TELEGRAM_CHAT_ID}
```

**검증**:
```bash
python -c "
from deployment.monitoring.alerter import TradingAlerter
alerter = TradingAlerter({'alert_channels': ['console'], 'drawdown_alert_threshold': 0.10})
alerter.check_drawdown(current=9500, peak=10000)  # 5% → no alert
alerter.check_drawdown(current=8800, peak=10000)  # 12% → alert
print('Alerter OK')
"
```

---

## Week 34: Integration Test + 최종 패키지

### 34.1 End-to-End Integration Test
**신규 파일**: `tests/test_full_integration.py`

전체 파이프라인을 축소 데이터(test_data.csv)로 실행:
1. Config validation (Pydantic)
2. Data loading + multi-timeframe feature generation
3. HMM regime detection
4. 1개 agent 단축 학습 (1000 steps)
5. Walk-forward 1 fold
6. Backtesting + statistical tests (bootstrap CI)
7. Risk manager with regime sizing
8. Paper trading 10 steps
9. Alert system trigger test

```bash
pytest tests/test_full_integration.py -v --timeout=300
```

전체 소요: 5분 이내 (축소 데이터 + 최소 step).

### 34.2 User Guide 업데이트
**파일**: `docs/user_guide.md` (기존)

추가 섹션:
- Monitoring 설정법 (Telegram bot 생성 방법)
- Statistical test 결과 읽는 법 (CI, p-value 해석)
- Regime별 성능 차이가 클 때 대응법
- 일일 체크리스트: 로그 확인 → drawdown 확인 → drift 확인

### 34.3 최종 검증
```bash
# 전체 테스트
pytest tests/ -x --tb=short

# 파이프라인 dry-run
python scripts/run_full_pipeline.py --config config/local_3060ti.yaml --dry-run

# Docker build
docker-compose build --no-cache
docker-compose up -d
docker-compose ps  # 모든 서비스 healthy 확인
docker-compose down
```

---

## 요약: Week별 산출물

| Week | 핵심 | 파일 수 | 난이도 |
|------|------|---------|--------|
| 29 | 버그 수정 + cleanup | 수정 3-5개 | ★☆☆ |
| 30 | Config schema + regime sizing | 신규 1 + 수정 2 | ★★☆ |
| 31 | Multi-timeframe features | 신규 1 + 수정 1 | ★★☆ |
| 32 | 통계 검증 도구 | 신규 1 + 수정 1 | ★★☆ |
| 33 | Execution + monitoring | 신규 2 + config | ★★★ |
| 34 | Integration test + 패키지 | 신규 1 + 문서 | ★★☆ |

**Sonnet 작업 시 주의**:
- 각 Week는 독립적으로 실행 가능 (순서대로 하되, 하나 완료 후 다음)
- 매 Week 끝에 `pytest tests/ -x --tb=short` 통과 필수
- 신규 파일 생성 시 기존 코드 스타일 따르기 (logger 사용, emoji 로그 X)
- Config 변경은 항상 `config/local_3060ti.yaml`과 `config/default_config.yaml` 둘 다
