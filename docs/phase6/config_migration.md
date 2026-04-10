# Config Migration — Week 63 (S39)

**작성일**: 2026-04-10  
**전**: 22개 YAML 파일 → **후**: 10개 (55% 감소)

---

## 파일 매핑 (삭제된 파일 → 새 위치)

| 삭제된 파일 | 내용이 들어간 곳 | 비고 |
|------------|----------------|------|
| `config/auto_iterate.yaml` | `config/base.yaml#auto_iterate` | auto_iterate 섹션 |
| `config/default_config.yaml` | `config/base.yaml` | env/agent/training/data 기본값 |
| `config/ensemble.yaml` | `config/base.yaml#ensemble` | agents 인라인, config_path 참조 제거 |
| `config/env_settings.yaml` | `config/trading.yaml#trading` | observation/action space |
| `config/flag_trader.yaml` | `config/base.yaml#flag_trader` + `config/base.yaml#ensemble.agents[3]` | 앙상블 내 params로 통합 |
| `config/local_3060ti.yaml` | `config/env/local_3060ti.yaml` | diff만 (공통값은 base) |
| `config/local_m2.yaml` | `config/env/local_m2.yaml` | diff만 |
| `config/multi_agent.yaml` | `config/trading.yaml#multi_agent` | multi_agent.enabled: false로 기본 |
| `config/multi_agent_config.yaml` | `config/trading.yaml#multi_agent` | multi_agent.yaml과 중복 → 통합 |
| `config/multi_agent_multi_asset.yaml` | `config/trading.yaml#multi_agent` | multi_agent.assets[]로 표현 |
| `config/multi_asset.yaml` | `config/trading.yaml#multi_agent` | 위와 동일 |
| `config/paper_trading.yaml` | `config/deployment.yaml#paper_trading` | secrets도 deployment로 |
| `config/regime.yaml` | `config/base.yaml#regime` | HMM 파라미터 |
| `config/risk_management.yaml` | `config/risk.yaml` | 전체 흡수 |
| `config/risk_reward_trading_env.yaml` | `config/risk.yaml#risk_management` + `config/trading.yaml#reward` | 리워드/마찰 분리 |
| `config/sac.yaml` | `config/base.yaml#agent.sb3_params.sac` + `config/base.yaml#ensemble.agents[1]` | 인라인 |
| `config/single_agent.yaml` | `config/base.yaml` | env/agent defaults로 |
| `config/td3.yaml` | `config/base.yaml#agent.sb3_params.td3` + `config/base.yaml#ensemble.agents[2]` | 인라인 |
| `config/test_config.yaml` | `config/env/test.yaml` | CI 전용 오버라이드로 |
| `config/training_config.yaml` | `config/base.yaml` (핵심만) | 384줄 중 research-only 설정 제거 |

---

## 신규 파일

| 파일 | 역할 |
|------|------|
| `config/base.yaml` | 공통 defaults 전체 |
| `config/trading.yaml` | 거래 환경 & 실행 |
| `config/risk.yaml` | 리스크 관리 전체 |
| `config/monitoring.yaml` | 모니터링 & 알림 (기존 파일 재작성) |
| `config/deployment.yaml` | 배포, persistence, secrets |
| `config/env/local_3060ti.yaml` | RTX 3060 Ti diff |
| `config/env/local_m2.yaml` | M2 Mac diff |
| `config/env/test.yaml` | CI/테스트 diff |
| `config/env/uw_gpu.yaml` | UW HPC GPU diff (신규) |
| `config/logging_config.yaml` | Python logging (유지) |

---

## 코드 변경 필요 사항

### 업데이트된 파일
| 파일 | 변경 내용 |
|------|---------|
| `scripts/run_full_pipeline.py` | `--env` 플래그 추가, `config.loader.load()` 사용 |
| `scripts/test_training.py` | `load("test")` 사용 |

### 여전히 레거시 로딩 사용 (점진적 마이그레이션 대상)
| 파일 | 현재 방식 | 권장 교체 |
|------|---------|---------|
| `training/train_ensemble.py` | `yaml.safe_load(config_path)` | `load_raw(config_path)` 또는 `load(env)` |
| `training/train_pipeline.py` | `FullConfig(**raw)` 직접 호출 | `load(env)` (내부에서 validate) |
| `scripts/generate_report.py` | `yaml.safe_load` | `load_raw(path)` |
| `scripts/validate_training.py` | `configs/` 디렉터리 참조 (오타) | `config/` + `load(env)` |

### 삭제된 ensemble config_path 참조
기존 `ensemble.yaml`은 각 에이전트에 `config_path:` 를 사용했음:
```yaml
agents:
  - type: ppo
    config_path: config/single_agent.yaml  # ← 삭제됨
```
새 방식은 `base.yaml#ensemble.agents[].params`에 직접 인라인.
`train_ensemble.py`에서 `config_path` 로드 로직이 있다면 제거 필요.

---

## Loader 사용법

```python
# 새 방식 (권장)
from config.loader import load
cfg = load("local_3060ti")   # config/env/local_3060ti.yaml 오버라이드 적용
cfg = load("test")           # CI용
cfg = load("uw_gpu")         # UW HPC

# 환경변수로도 제어 가능
# TRADING_BOT_ENV=local_3060ti python scripts/run_full_pipeline.py --env local_3060ti

# 레거시 단일 YAML (하위호환)
from config.loader import load_raw
cfg = load_raw("path/to/any.yaml")   # 기존 yaml.safe_load + FullConfig validate
```

---

## 충돌 해소 결정

| 키 | 충돌 값들 | 최종 결정 | 근거 |
|----|---------|---------|------|
| `env.initial_balance` | 10000 (default_config) vs 100000 (local_3060ti) | **100000** | 실거래 규모 |
| `risk_management.max_drawdown_pct` | 0.15 (risk_management.yaml) vs 0.20 (local_3060ti.yaml) | **0.20** | local_3060ti가 실운영 설정 |
| `risk_management.stop_loss_threshold` | 0.10 (risk_management.yaml) vs 0.05 (local_3060ti.yaml) | **0.05** | 보수적 (실거래 기준) |
| `training.total_timesteps` | 1M (single_agent) vs 500K (local_3060ti) vs 10K (test) | **1M (base)**, env 오버라이드로 감소 | base는 최대값, env에서 감소 |
