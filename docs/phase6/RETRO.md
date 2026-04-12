# Phase 6 Retrospective: Production Readiness & Architecture Consolidation

**기간**: Weeks 56-68 (2026-03-xx ~ 2026-04-12)  
**목표**: "돈 태울 수 있는 상태" — ops readiness + 구조 통합

---

## 최종 테스트 결과

**1727 passed, 19 skipped, 0 failed**  
Phase 5 baseline: 1386 → Phase 6 완료: 1727 (+341 신규 테스트)

---

## Track A — Ops Readiness (Weeks 56-59)

### 배운 것
- **StateStore**: SQLite WAL 모드 + idempotent restore는 간단하지만 강력하다. `PRAGMA journal_mode=WAL`만으로 동시 read/write 충돌 해결.
- **AuditLogger**: hash chain이 append-only 보장의 핵심. 파일 corruption 감지는 verify_audit_log.py 하나로 충분.
- **Secrets**: `secret_ref:` 패턴 — config에서 평문 제거하는 가장 덜 침습적 방법. 실 운영에서 env var 기반이 macOS keychain보다 CI/CD 친화적.
- **Runbook + Drills**: 코드보다 운영 절차가 위기 대응의 핵심. drill script가 실제 failure path를 커버해야 진짜 가치.

### Gotchas
- StateStore `PRAGMA journal_mode=WAL` 설정은 `PRAGMA` 구문으로 해야 하며, connection string에 포함하면 동작 안 함.
- Audit chain replay 시 partial write (크래시 직전) 레코드가 있으면 chain이 끊긴다. verify script는 이를 exit 1로 감지.

---

## Track B — Architecture Consolidation (Weeks 60-63)

### 배운 것
- **UnifiedRiskManager**: composition 방식(기존 클래스가 내부에서 사용)이 subclassing보다 안전했다. 기존 1386 테스트 건드리지 않고 통합 완료.
- **DataSource**: 인터페이스 분리 효과가 즉각적 — `StaticDataSource` / `CSVDataSource` / `MockLiveDataSource` 각각 독립 테스트 가능.
- **Config Consolidation**: 23개 → 10개 YAML. `load(env)` 하나로 merge + validate. `_deep_merge` 구현이 제일 까다로웠음.
- **DI (Dependency Injection)**: `PaperTrader`에 주입 포인트를 많이 만들어두면 테스트 작성이 쉬워진다. 하지만 constructor가 너무 길어지는 trade-off.

### Gotchas
- `UnifiedRiskManager`와 `RiskManagerBase`는 interface가 다르다 (예: `check_max_drawdown` vs `check_drawdown`). Phase 7에서 통합 필요.
- Config merge 시 list 타입 값은 deep merge 안 하고 override함 — 의도된 동작이지만 주의.

---

## Track C — Production Safety (Weeks 64-66)

### 배운 것
- **Fat-finger guard**: `lookback` 기반 평균 × multiplier 체크가 hard cap보다 시장 적응적. 초기에 history가 없으면 hard cap만 동작.
- **Circuit breaker**: rolling vol이 `window` 미만이면 trip 안 함. cooldown 설정이 false alarm 방지에 중요.
- **Rate limiter**: 토큰 버킷(token bucket)은 간단하지만 burst 허용. 실거래소 rate limit은 burst 불허 케이스도 있으니 확인 필요.
- **PnL Attribution**: `market_move + slippage + fees = realized_pnl` 분해가 전략 개선 진단에 핵심. 데이터가 실제 fill price와 expected price의 차이를 포함해야 slippage가 정확함.
- **Latency SLO**: p50/p95/p99 percentile 추적 → paper mode에서는 latency 거의 0이라 live 전환 시 다시 캘리브레이션 필요.

### Gotchas
- `FatFingerGuard.check()` 리턴이 `(bool, str)` 튜플. bool만 체크하면 reason 정보 손실.
- `OrderManager`가 `max_order_size` config로 자동 clamping함 — 운영 전 이 값 확인 필수.

---

## Track D — Model Lifecycle (Weeks 67-68)

### 배운 것
- **DriftDetector**: ADWIN/PageHinkley가 feature별 drift를 returns drift보다 먼저 감지. early warning 효과.
- **Regime detection**: live wire 연결 자체보다 "regime 변경 시 무엇을 할 것인가" 정책 결정이 더 중요. 현재 log만 — 충분.
- **ModelRegistry**: MLflow 없이 파일 기반으로도 버전 관리 + rollback 가능. 원자적 write(tmp → rename)이 핵심.
- **Shadow agent**: main과 동일 obs로 새 모델을 "dry run" — 운영 영향 없이 성능 비교. audit log에 남아서 분석 가능.

### Gotchas
- `shadow_agent`가 raise하면 main loop이 멈추면 안 된다 → `try/except` 필수.
- `AuditLogger.log_model_decision(extra=...)` 필드 추가 시 hash chain에 영향 → schema 변경 금지.
- Week 67 구현 누락 (model_registry.py) → Week 68에서 보완. plan 체크리스트 실행 중 미완료 항목 발생 시 다음 week으로 이월하지 말고 해당 week에 처리할 것.

---

## Phase 7 후보 (우선순위 순)

1. **UnifiedRiskManager ↔ RiskManagerBase 완전 통합**: `PaperTrader._check_risk()`가 `check_max_drawdown`을 호출하는데 `UnifiedRiskManager`에 없음. 둘을 연결해야 risk 경로가 통일됨.
2. **실거래소 연동**: CCXT live mode + API key (cloud secret manager). 현재 simulation mode만 검증됨.
3. **Multi-asset live**: `MultiAgentManager` phase 6에서도 wire 안 됨. single-asset 먼저 실거래 검증 후.
4. **MLflow experiment tracking**: model_registry.py를 MLflow artifact store와 연결.
5. **Shadow agent dashboard**: 현재 audit log에만 기록. Grafana/Streamlit에서 실시간 비교.
6. **Rollback 핫스왑**: restart 없이 running PaperTrader의 agent 교체.
7. **Kubernetes / 컨테이너화**: 현재 docker-compose. k8s로 확장 시 stateful set 고려.
8. **Compliance / tax reporting**: 거래 기록 → 세금 신고 자동화.

---

## Phase 6 완료 조건 최종 체크

- [x] 실거래 시나리오 drill 3종 통과 (Week 59)
- [x] audit log 체인 검증 통과 (Week 57, smoke 확인)
- [x] 평문 secret 0건 (Week 58)
- [x] UnifiedRiskManager로 통합, parity 테스트 통과 (Week 60)
- [x] Config 23 → ≤10 (Week 63: 5 base + env overrides)
- [x] Fat-finger / circuit breaker / rate limiter live 경로 연결 (Week 64)
- [x] PnL attribution이 계산됨 (Week 66, 정확성 테스트 통과)
- [x] Shadow agent 동작 검증 (Week 68, 7개 테스트)
- [x] 전체 회귀 0 fail (1727 passed)

**Phase 6 완료.**
