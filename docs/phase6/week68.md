# Week 68: Rollback / Shadow Deploy & Final Validation (S61-S65)

**날짜**: 2026-04-12  
**브랜치**: `claude/elated-ptolemy`  
**목표**: Phase 6 마지막 week — shadow agent, model rollback, 전체 smoke, 최종 회귀

---

## What (무엇을 했나)

### S61 — Shadow Agent (PaperTrader)
- `PaperTrader.__init__`에 `shadow_agent=None` 파라미터 추가
- 매 step 관찰 후 shadow agent의 `predict()` 호출 → **주문 없음**
- 결정 비교 (`main_action` vs `shadow_action`) → `AuditLogger.log_model_decision(source="shadow")`
- `_run_shadow_agent()` 예외 처리: shadow 실패해도 main loop 계속
- `AuditLogger.log_model_decision`에 `extra: dict` 파라미터 추가 (하위호환)
- 테스트: `tests/deployment/test_shadow_mode.py` (7개)

### S62 — Model Registry + Rollback
- `training/registry/model_registry.py` 생성 (Week 67에서 누락)
  - 파일 기반 로컬 레지스트리 (MLflow 불필요)
  - `register()`, `set_active()`, `get_active()`, `get_version()`, `list_versions()`, `rollback()`, `delete_version()`
  - 원자적 index 쓰기 (`registry.json.tmp` → rename)
  - thread-safe (`threading.Lock`)
- `scripts/rollback_model.py` CLI
  - `<version>` positional arg, `--list`, `--show`, `--active-model-path`, `--registry-dir`
  - 성공 시 exit 0, 버전 없으면 exit 1
- 테스트: `tests/training/test_model_registry.py` (20개)

### S63 — Phase 6 Smoke Test
- `scripts/phase6_smoke.py` — 12개 섹션, 33개 체크
  - Track A: StateStore (save/load/clear), AuditLogger (chain verify)
  - Track B: UnifiedRiskManager (position limit), StaticDataSource (window/staleness), config loader
  - Track C: FatFingerGuard, VolatilityCircuitBreaker, RateLimiter, PnLAttributor
  - Track D: Shadow agent audit, ModelRegistry + rollback CLI
  - Full E2E: 모든 컴포넌트 동시 실행 (PaperTrader + StateStore + AuditLogger + OrderManager + shadow)
- 33/33 PASS, exit 0

### S64 — 최종 회귀
- **1727 passed, 19 skipped, 0 failed** (baseline 1386 + Phase 6 신규 341)
- pytest.ini ignore 목록 변동 없음

### S65 — 문서
- 이 문서 + `docs/phase6/RETRO.md`

---

## Why (왜 이렇게 했나)

**Shadow agent**: 실거래 전 새 모델을 "live trial" 없이 검증하는 가장 안전한 방법. 기존 main agent의 주문에 전혀 영향 없이 결정 비교 가능. audit log에 남기면 사후 분석도 된다.

**ModelRegistry 파일 기반**: MLflow 없어도 동작해야 함 (부록 B: MLflow 도입은 Phase 7). `registry.json.tmp → rename` 패턴으로 partial write 방지.

**Rollback CLI**: PaperTrader restart 전제 — 운영 중 핫스왑 지원 안 하는 것이 맞다. registry pointer만 바꾸고 별도 `--active-model-path` 옵션으로 파일도 교체 가능.

**Smoke test 설계**: 각 컴포넌트가 실제로 `동작하는지`를 별도 단위로 검증 (unit test) + 마지막에 전부 묶어서 통합(E2E). 하나라도 실패하면 exit 1.

---

## Gotchas (주의사항)

1. **UnifiedRiskManager ↔ PaperTrader**: `PaperTrader._check_risk()`는 `risk_manager.check_max_drawdown()`을 호출하지만 `UnifiedRiskManager`엔 이 메서드가 없다. 둘은 아직 완전히 연결 안 됨 — `check_drawdown`이 있고 signature도 다름. Phase 7에서 `RiskManagerBase`를 `UnifiedRiskManager`로 교체 시 주의.

2. **`AuditLogger.log_model_decision` extra 필드**: hash chain은 `payload` 전체를 포함하므로 `extra` 키가 바뀌면 chain이 이어지지 않는다. schema 바꾸지 말 것.

3. **VolatilityCircuitBreaker threshold**: 실제 운영 설정에서 `vol_threshold`를 너무 낮게 잡으면 정상 시장에서도 trip 발생. 초기 calibration 필요.

4. **shadow_agent가 main_agent와 동일한 obs를 받음**: obs는 같지만 shadow가 main과 다른 내부 state를 갖는 SB3 모델이라면 predict() 결과가 달라질 수 있다. 의도된 동작.

---

## 미완료 / Phase 7 후보

- UnifiedRiskManager ↔ PaperTrader 완전 연결 (check_max_drawdown 구현 or PaperTrader 수정)
- Shadow agent live dashboard (현재 audit log에만 기록)
- Model registry → MLflow 연동 (Phase 7)
- Rollback 중 running PaperTrader 핫스왑 (현재 restart 필요)
