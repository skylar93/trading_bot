# Week 80 — Scheduled Retraining & Web Interface 정리 (H11-H14)

**날짜**: 2026-04-21  
**브랜치**: claude/suspicious-lalande-c7663d  
**전제**: Week 79 (H6-H10) 완료 — MLflow, DVC, pandera, feature registry, walkforward 모두 active

---

## H11 — Airflow vs Prefect 결정

**결정**: **Prefect 2 로컬 워커** 채택 (plan 권장대로)

- Airflow: 스케줄러 + 워커 + 메타DB 분리 필요. 솔로 dev 기준 과도.
- Prefect 2: `prefect serve` 한 줄로 로컬 워커 실행. UI도 내장.
- `requirements.txt`에 `prefect>=2.14.0` 추가.

---

## H12 — Retrain Flow

**파일**: [`training/pipelines/retrain_flow.py`](../../training/pipelines/retrain_flow.py)

### 5단계 파이프라인

| Task | 역할 |
|------|------|
| `fetch_latest_data` | CSV / CCXT / yfinance 소스에서 OHLCV 로드. retries=2. |
| `compute_features` | feature_engineering → fallback minimal (RSI/SMA/vol_ratio). FeatureRegistry 자동 등록. |
| `train_model` | env_factory + agent_factory → `agent.learn()` → checkpoint 저장. timeout 1시간. |
| `walkforward_eval` | `evaluate_for_promotion()` (H10 purged K-fold). report JSON 저장. |
| `register_staging` | gate pass 시에만 ModelRegistry 등록 + `candidate → staging` 자동 promote. |

### 자동 승급 금지

`staging → canary`는 `scripts/promote_model.py` 수동 실행 필요 (G5 원칙).

### RetrainingTrigger 연동

```python
from training.pipelines.retrain_flow import make_retrain_callback
from deployment.monitoring.retraining_trigger import RetrainingTrigger

trigger = RetrainingTrigger(
    config={"drawdown_trigger_pct": 0.15},
    on_trigger=make_retrain_callback(config=pipeline_config),
)
# trading loop 내 trigger.check() → drawdown/drift 초과 시 background thread로 retrain_flow 실행
```

중복 실행 방지: `threading.Lock()` + 실행 중 새 트리거 → drop + 경고 로그.

---

## H13 — Web Interface 제거 (Option A 채택)

**결정**: 완전 제거. Grafana로 대체.

**삭제 항목**:
- `deployment/web_interface/` (36 files)
- `tests/test_week11.py` (108 tests)
- `tests/test_backtest_integration.py`

**대체**:
- 운영 모니터링 → `deployment/monitoring/grafana_dashboard.json` (Week 78, H2)
- 알림 → `deployment/monitoring/alerter.py` (Discord/webhook)
- 실시간 메트릭 → Prometheus + Grafana (H1)

**이유**: realtime_trading_manager.py:106, utils/data_stream.py:116 가 TODO stub. Grafana가 PnL attribution, latency, drawdown, drift alarm, kill switch를 이미 커버. 별도 Streamlit UI의 추가 가치 없음.

→ 상세 기록: [`docs/phase7/deleted_tests.md`](deleted_tests.md)

---

## H14 — Phase 7 최종 검증

**스모크 테스트**: [`scripts/phase7_smoke.py`](../../scripts/phase7_smoke.py)

| Track | 검증 항목 | 결과 |
|-------|----------|------|
| **E** | UnifiedRiskManager (check_drawdown / compute_var / check_trailing_stop) | ✅ |
| **E/G** | PreTradeComplianceChecker (G6-G10: position limits, self-trade, notional cap, wash trade) | ✅ |
| **F** | RetrainingTrigger (drawdown + drift conditions, callback invocation) | ✅ |
| **G** | ModelRegistry promotion state machine (5 stage transitions, invalid block, history) | ✅ |
| **H1** | MetricsExporter.update + snapshot + history | ✅ |
| **H2** | grafana_dashboard.json 존재 | ✅ |
| **H3** | TradingAlerter (drawdown alert, notify_error, notify_kill_switch) | ✅ |
| **H6** | pandera OHLCV schema (NaN/negative 감지) | ✅ |
| **H7** | MLflowRegistryBridge import | ✅ |
| **H8** | dvc.yaml 존재 | ✅ |
| **H9** | FeatureRegistry (register, get, drift_report) | ✅ |
| **H10** | WalkForwardReport (staging gate, save/load) | ✅ |
| **H11** | retrain_flow + make_retrain_callback importable, RetrainingTrigger → callback 연동 | ✅ |
| **H12** | 5개 Prefect task 모두 callable, Prefect 미설치 시 graceful fallback | ✅ |
| **H13** | deployment/web_interface/ 삭제 확인 | ✅ |

**결과**: 67/67 checks passed

### 전체 pytest 결과

```
2332 passed, 41 skipped, 0 failed (177.48s)
```

- pytest.ini ignore: 2개 (test_multi_asset_data.py, test_multi_asset_env.py) — 플랜 목표 ≤5 ✅
- flaky 0 ✅

---

## Phase 7 완료 상태

| Track | 주차 | 완료 |
|-------|------|------|
| E. Hardening Debt | 69-71 | ✅ |
| F. Real Connectivity | 72-74 | ✅ |
| G. Governance & Go-Live Gate | 75-77 | ✅ |
| H. Integration Layer | 78-80 | ✅ |

**Phase 7 전체 완료** — 2332 passed, 0 failed, phase7_smoke.py 67/67.
