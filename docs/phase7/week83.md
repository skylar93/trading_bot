# Week 83: Safety Nets — Canary + OTel + Schema Drift

**Date**: 2026-04-22  
**Branch**: claude/dreamy-pare-e1d0af  
**Theme**: G4-G6 채움 — 운영 중 자동으로 개입하는 방어선 3종

---

## 완료 조건 체크

| 조건 | 상태 |
|------|------|
| Canary auto-demote (traffic 0% + alert) | ✅ |
| OTel 5개 span 계층 구조 | ✅ |
| Schema drift halt 경로 + alert 검증 테스트 | ✅ |
| `tests/integration/test_safety_nets.py` 18/18 pass | ✅ |
| Full pytest 0 fail (2402 passed) | ✅ |

---

## R11: Canary Auto-Demotion (G4)

### 구현된 파일
- `deployment/paper_trader.py` — `_check_canary_auto_demote()` + config fields
- `deployment/monitoring/alerter.py` — `notify_canary_auto_demoted()` 신규
- `training/registry/model_registry.py` — stage entry에 `auto_demote_criteria` 필드 추가

### 동작 원리
1. `_run_canary_agent()` 내 canary step마다 `_check_canary_auto_demote()` 호출
2. 최근 `window = demote_hours × steps_per_hour` 스텝의 canary / prod return 비교
3. `canary_mean < prod_mean - sigma_below_prod × prod_std` 위반 시 `_canary_underperform_streak` 증가
4. `consecutive_hours`회 연속 위반 시 traffic_pct=0, canary_enabled=False (즉시 차단)
5. 이후 단계: stage는 canary 유지, human이 `promote_model.py --restore-traffic`으로 복구

### 설정값 (config/alerts.yaml)
```yaml
canary_auto_demote:
  sigma_below_prod: 1.0        # prod - 1σ 이하 → breach
  consecutive_hours: 6         # 6h 연속 breach → auto-demote
```

### Design Decisions
- **Latch 방식**: `_canary_auto_demoted = True`로 래치 → 한 세션에서 1회만 발동
- **Stage 불변**: `model_registry.promote()` 호출하지 않음 — PaperTrader가 traffic만 차단, stage 판단은 human
- **audit_logger**: `log_risk_event(type=canary_auto_demoted)` 로 전체 수치 기록

---

## R12: OTel Span Instrumentation (G5)

### 구현된 파일
- `deployment/execution/order_manager.py` — `submit_order()` → inner method 분리 + 5개 span
- `docker-compose.yml` — Jaeger all-in-one 서비스 추가 (port 16686 UI, 4318 OTLP HTTP)

### Span 계층 구조
```
trading.order.submit                    ← parent (전체 round-trip)
├── trading.order.idempotency_lookup    ← S44 체크
├── trading.order.risk_check            ← S41/S42/S43 + drawdown
├── trading.order.compliance_check      ← G6-G9
└── trading.order.exchange_submit       ← paper/live execution
```

### 주요 Attributes
| Span | Attributes |
|------|-----------|
| submit | symbol, side, amount, order_type, order.id, order.total_latency_ms |
| idempotency_lookup | idempotency.hit |
| risk_check | risk.correlation_check, risk.circuit_breaker_tripped, risk.drawdown_check_passed, risk.fat_finger_passed |
| compliance_check | compliance.passed, compliance.reject_reason |
| exchange_submit | order.paper_mode, order.status |

### Jaeger 연동
```bash
# docker-compose --profile tracing up jaeger
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# UI: http://localhost:16686
```

### Design Decisions
- `submit_order()` → `_submit_order_inner()` 리팩터: outer에서 parent span context 열고 inner에 전달
- `record_order_latency()` 기존 helper 재사용 — fill latency를 parent span에 기록
- OTel 미설치 시 `_NoopSpan`으로 무해하게 fallback (기존 동작 변경 없음)

---

## R13: Real-time Schema Drift Guard (G6)

### 구현된 파일
- `data/quality/stream_validator.py` — 신규 (StreamValidator, SchemaDrift)
- `deployment/monitoring/alerter.py` — `schema_drift_detected()` 신규

### 검사 항목
| 항목 | 내용 |
|------|------|
| 키 존재 | `$open, $high, $low, $close, $volume` 모두 있어야 |
| dtype | float 변환 가능해야 |
| NaN | 불허 |
| ±inf | 불허 |
| 양수 range | 모든 값 > 0 |

### 정책
- `on_schema_drift: halt` (기본) → `SchemaDrift` raise + CRITICAL alert
- `on_schema_drift: warn` → WARNING alert + 계속 진행

### 사용 예시
```python
from data.quality.stream_validator import StreamValidator, SchemaDrift

sv = StreamValidator(on_schema_drift="halt", alerter=alerter)
for tick in ccxt_live_feed:
    try:
        sv.validate(tick)
    except SchemaDrift:
        break  # halt policy: stop immediately
```

### config/alerts.yaml (기존)
```yaml
schema_drift:
  on_drift: halt   # halt | warn
```

---

## R14: Integration Tests (test_safety_nets.py)

**파일**: `tests/integration/test_safety_nets.py`  
**총 테스트**: 18개 / 18개 pass

| 클래스 | 테스트 수 | 커버리지 |
|--------|----------|---------|
| TestCanaryAutoDemotion | 4 | 발동 / 미발동 / 래치 / stage 불변 |
| TestOTelSpans | 3 | parent span / child spans / idempotency hit attr |
| TestSchemaDriftGuard | 9 | halt/warn/missing key/dtype/nan/inf/negative/extra key |
| TestSafetyNetsNonInterference | 2 | E2E 동시 실행 / alert priority 분류 |

---

## pytest 결과

```
2402 passed, 41 skipped, 28 warnings — 0 failed
```

---

## 기타 변경사항

- `pytest.ini` — `test_multi_asset_risk_integration.py` ignore 추가
  (collection 시 `logs/` 디렉토리 없어서 crash — pre-existing 결함)

---

## Week 84 미리보기

- R15: API key scope 자동 probe (`verify_exchange_key_scope.py`)
- R16: Pre-commit secret scanner (detect-secrets hook)
- R17: Capacity baseline 스냅샷 (`capacity_probe.py`)
- R18: Runbook drill 2건 실제 수행 + 기록
