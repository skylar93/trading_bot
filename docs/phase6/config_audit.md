# Config Audit — Week 63 (S36)

**작성일**: 2026-04-10  
**기준**: `config/*.yaml` (root-level 22개 파일)

---

## 파일 매트릭스

| 파일 | 역할 | 주요 top-level 키 | 중복/충돌 | 처리 |
|------|------|-------------------|-----------|------|
| `auto_iterate.yaml` | 자동 반복 학습 파라미터 | auto_iterate, seed_strategy, evaluation, output | training과 중복 | → `base.yaml` 흡수 |
| `default_config.yaml` | 가장 기본 defaults | env, data, model, training, paths, risk_management | local_3060ti.yaml과 충돌 (initial_balance 10000 vs 100000) | → `base.yaml` 흡수 |
| `ensemble.yaml` | 4-agent ensemble 정의 | agents (config_path 참조), meta_controller, ood_detector, walk_forward | agents가 sac/td3/flag_trader.yaml 포인팅 | → `base.yaml` 흡수 (inline) |
| `env_settings.yaml` | 거래 환경 세부 설정 | trading, observation_space, action_space | local_3060ti.yaml env 섹션과 중복 | → `trading.yaml` 흡수 |
| `flag_trader.yaml` | FLAG-Trader (LLM) 설정 | flag_trader, ensemble, mlflow | local_3060ti.yaml flag_trader 섹션과 중복 | → `base.yaml` 흡수 |
| `local_3060ti.yaml` | RTX 3060 Ti 환경 설정 | env, agent, ensemble, training, data, monitoring, ... (20+ 섹션) | **가장 포괄적인 config** — 많은 파일과 중복 | → `config/env/local_3060ti.yaml` (diff만) |
| `local_m2.yaml` | M2 Mac 환경 설정 | env, agent, ensemble, training, data, validation, hyperopt, ... | local_3060ti.yaml과 많은 공통 키, device/size만 다름 | → `config/env/local_m2.yaml` (diff만) |
| `logging_config.yaml` | Python logging 설정 | version, formatters, handlers, loggers, root | Python logging dict 형식 — YAML이지만 특수 | **유지** (별도 처리) |
| `monitoring.yaml` | 모니터링 설정 | monitoring (max_history, prometheus, dashboard, drift) | local_3060ti.yaml monitoring 섹션과 중복 | → `monitoring.yaml` 재작성 |
| `multi_agent.yaml` | 다중 에이전트 환경 | env, agent_type, agent, training, paths, data | multi_agent_config.yaml과 90% 중복 | → 삭제 (multi_agent_config.yaml도 삭제) |
| `multi_agent_config.yaml` | 다중 에이전트 대안 | env, data, training, shared_experience, paths, mlflow | multi_agent.yaml 중복 | → 삭제 |
| `multi_agent_multi_asset.yaml` | 멀티에이전트+멀티자산 | env, agent_type, agent, training, paths, data | multi_asset.yaml과 구조 동일, type만 다름 | → 삭제 (base + trading.yaml에서 커버) |
| `multi_asset.yaml` | 멀티 자산 | env, agent_type, agent, training, paths, data | multi_agent_multi_asset.yaml과 중복 | → 삭제 |
| `paper_trading.yaml` | Paper trading 설정 | paper_trading, agent, mlflow, llm_review | local_3060ti.yaml paper_trading 섹션과 중복 | → `deployment.yaml` 흡수 |
| `regime.yaml` | 레짐 감지 설정 | regime_detector, labels | local_3060ti.yaml regime 섹션과 중복 | → `base.yaml` 흡수 |
| `risk_management.yaml` | 리스크 관리 세부 | stop_loss, trailing_stop, var, cvar, drawdown, correlation, portfolio_* | local_3060ti.yaml risk_management + schema.py RiskConfig와 일부 중복 | → `risk.yaml` 흡수 |
| `risk_reward_trading_env.yaml` | 보상 리스크 환경 | data, env, training, logging, visualization | env 섹션 중복, logging 특수 | → `risk.yaml` + `trading.yaml` 흡수 |
| `sac.yaml` | SAC 에이전트 설정 | agent, ensemble | ensemble.yaml 내부 sac 설정과 중복 | → `base.yaml` 흡수 (inline) |
| `single_agent.yaml` | 단일 에이전트 환경 | env, agent_type, agent, training, paths, data | default_config.yaml + local_3060ti.yaml env와 중복 | → `base.yaml` 흡수 |
| `td3.yaml` | TD3 에이전트 설정 | agent, ensemble | ensemble.yaml 내부 td3 설정과 중복 | → `base.yaml` 흡수 (inline) |
| `test_config.yaml` | 테스트 최소 설정 | data, env, agent, training, paths | total_timesteps: 10000 (CI용) | → `config/env/test.yaml` |
| `training_config.yaml` | 가장 포괄적인 학습 설정 | 384줄, 25+ 섹션 (decision_transformer, sentiment, onchain, ...) | 대부분 연구용 / dead code. 실제 사용은 local_3060ti.yaml | → `base.yaml`에 핵심만 흡수, 나머지 삭제 |

---

## 중복/충돌 식별

### 충돌 (다른 값)
| 키 | 파일 A | 값 A | 파일 B | 값 B |
|----|--------|------|--------|------|
| `env.initial_balance` | default_config.yaml | 10000 | local_3060ti.yaml | 100000 |
| `training.total_timesteps` | single_agent.yaml | 1,000,000 | local_3060ti.yaml | 500,000 | test_config.yaml | 10,000 |
| `risk_management.max_drawdown_pct` | risk_management.yaml | 0.15 | local_3060ti.yaml | 0.20 |
| `risk_management.stop_loss_threshold` | risk_management.yaml (stop_loss.stop_loss_threshold) | 0.10 | local_3060ti.yaml | 0.05 |
| `ensemble.enabled` | local_3060ti.yaml | true | local_m2.yaml | false |
| `training.device` | single_agent.yaml | "auto" | local_3060ti.yaml | "cuda:0" | local_m2.yaml | "mps" |

### Dead Config (사용되지 않음)
- `training_config.yaml`: `onchain`, `calendar`, `prediction_market`, `torchrl`, `continual_learning` 등 — 구현 없거나 실험적 기능
- `multi_agent_config.yaml`: `multi_agent.yaml`과 거의 동일, 실제 진입점 불명확
- `auto_iterate.yaml`: `scripts/auto_iterate.py` 존재하지만 실거래 경로 외부

---

## 새 레이아웃 결정

| 신규 파일 | 흡수하는 기존 파일 | 역할 |
|-----------|------------------|------|
| `config/base.yaml` | default_config, training_config (핵심), single_agent, ensemble, sac, td3, flag_trader, regime, auto_iterate | 공통 defaults 전체 |
| `config/trading.yaml` | env_settings, multi_agent, multi_agent_config, multi_agent_multi_asset, multi_asset, risk_reward_trading_env (env 부분) | 거래 환경 설정 |
| `config/risk.yaml` | risk_management, risk_reward_trading_env (risk 부분) | 리스크 관리 전체 |
| `config/monitoring.yaml` | monitoring (재작성) | 모니터링 + 알림 |
| `config/deployment.yaml` | paper_trading | 배포/운영 설정 |
| `config/env/local_3060ti.yaml` | local_3060ti (diff) | 3060 Ti 오버라이드 |
| `config/env/local_m2.yaml` | local_m2 (diff) | M2 Mac 오버라이드 |
| `config/env/test.yaml` | test_config | CI/테스트 오버라이드 |
| `config/env/uw_gpu.yaml` | (신규) | UW HPC GPU 오버라이드 |
| `config/logging_config.yaml` | (유지) | Python logging (특수 형식) |

**결과: 22 → 10 파일 (55% 감소)**

---

## Loader 동작 방식

```
load("local_3060ti")
 └─ base.yaml           (공통 defaults)
 └─ trading.yaml        (거래 환경)
 └─ risk.yaml           (리스크 설정)
 └─ monitoring.yaml     (모니터링)
 └─ deployment.yaml     (배포 설정)
     ↓ deep_merge
 └─ env/local_3060ti.yaml  (환경별 override)
     ↓ FullConfig validation (Pydantic v2)
 → dict
```
