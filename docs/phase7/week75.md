# Week 75: Model Promotion & Canary (G1-G5)

**날짜**: 2026-04-19  
**Track**: G — Governance & Go-Live Gate  
**PR**: TBD

---

## 완료 항목

### G1. Promotion 상태 기계 (`training/registry/model_registry.py`)
- Stage: `candidate → staging → canary → prod → retired`
- `VALID_TRANSITIONS` dict로 허용 전이 명시
- `PROMOTION_CRITERIA` dict로 각 전이 조건 문서화
- 신규 메서드: `promote()`, `get_stage()`, `get_promotion_history()`, `list_by_stage()`, `check_promotion_conditions()`
- 각 전이 이벤트: timestamp / actor / reason 기록 (registry.json `stages` 섹션)
- `register()` 시 자동으로 `candidate` 스테이지 초기화
- `force=True` 옵션: 테스트/긴급 상황용 전이 검사 우회

### G2. Canary 라우팅 (`deployment/paper_trader.py`)
- `shadow_agent` → `canary_agent`로 개명 (backward-compat alias 유지)
- `canary.enabled: true/false` (기본 off), `canary.traffic_pct: 0.10` config 지원
- `_run_canary_agent()`: traffic routing + observe-only mode 분리
- 성과 추적: `_canary_returns` / `_prod_returns` 누적
- 자동 promotion 제안: 168-step 윈도우에서 canary mean ≥ prod - 0.5σ 시 audit log에 경고 (자동 승급 X)
- audit log source: `"canary_observe"` (관찰만) / `"canary_active"` (실 실행)

### G3. Promotion 기준 문서 (`docs/phase7/promotion_criteria.md`)
- 전이별 정량 기준 (Sharpe, drawdown, walkforward, ruin prob)
- CLI 사용법
- 강등(demote) 절차

### G4. 수동 승급 CLI (`scripts/promote_model.py`)
- `--check` dry-run 모드
- `--from / --to / --version / --actor / --reason` 인자
- `canary → prod` 시 `--actor`, `--reason` 필수 강제
- `--json` 출력 모드 (scripting용)
- 조건 미달 시 비-0 exit code

### G5. Rollback 핫스왑 (`deployment/paper_trader.py`)
- `PaperTrader.replace_agent(new_agent, actor, reason)` 메서드
- `PositionTracker._lock` 안에서 원자적 교체
- 교체 이벤트 audit log 기록 (`type: "agent_hotswap"`)
- 내부 상태(포지션/잔고/히스토리/step) 유지

---

## 테스트 결과

| 테스트 클래스 | 개수 | 결과 |
|--------------|------|------|
| `TestPromotionStateMachine` | 18 | ✅ |
| `TestCanaryRouting` | 7 | ✅ |
| `TestPromoteModelCLI` | 7 | ✅ |
| `TestAgentHotSwap` | 5 | ✅ |
| **신규 합계** | **37** | ✅ |
| 전체 pytest | 2297 passed / 0 failed / 42 skipped | ✅ |

완료 조건 검증:
- [x] canary → prod 전이 시뮬레이션 1회 완료 (`test_full_pipeline_candidate_to_prod`)
- [x] 핫스왑 테스트 pass (`test_replace_agent_50_steps_then_swap`)

---

## 변경 파일

- `training/registry/model_registry.py` — G1 promotion 상태 기계
- `deployment/paper_trader.py` — G2 canary routing + G5 replace_agent
- `docs/phase7/promotion_criteria.md` — G3 기준 문서
- `scripts/promote_model.py` — G4 CLI (신규)
- `tests/deployment/test_governance.py` — 37개 신규 테스트 (신규)
- `tests/deployment/test_shadow_mode.py` — G2 rename에 맞춰 필드명 업데이트
