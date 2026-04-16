# Week 70 — UnifiedRiskManager 실질 통합 (E7-E12)

**완료일**: 2026-04-16  
**브랜치**: claude/dazzling-wright

---

## 완료 조건 검증

| 조건 | 상태 |
|------|------|
| `rg "check_max_drawdown"` → 0건 (alias 제외) | ✅ |
| Parity 100회 전부 동일 | ✅ (`TestRandomScenarioParity`) |
| 신규 테스트 실패 0 | ✅ |

---

## E7 — RiskManagerBase 인터페이스 통일

`risk_management/risk_manager_base.py`:

| 구 이름 (abstract) | 신 이름 (abstract) | 구 이름 (shim) |
|---|---|---|
| `check_max_drawdown` | `check_drawdown` | `check_max_drawdown` → DeprecationWarning |
| `calculate_var` | `compute_var` | `calculate_var` → DeprecationWarning |
| `check_stop_loss` | `check_trailing_stop` | `check_stop_loss` → DeprecationWarning |

---

## E8 — Caller 사이트 일괄 수정

| 파일 | 변경 |
|------|------|
| `deployment/paper_trader.py:701` | `check_max_drawdown` → `check_drawdown` |
| `deployment/execution/order_manager.py:279` | `check_max_drawdown` → `check_drawdown` |
| `envs/single_asset_rl_env.py:794` | `check_max_drawdown` → `check_drawdown` |
| `deployment/execution/order_manager.py:108` | 문서 업데이트 |
| Tests (7개 파일) | `check_max_drawdown` → `check_drawdown` 전체 교체 |

---

## E9 — 구체 구현 정리

**BacktestingRiskManager**:
- `check_max_drawdown` → `check_drawdown` (impl), 구 이름은 deprecated shim
- `calculate_var` → `compute_var` (impl), 구 이름은 deprecated shim
- `check_stop_loss` → `check_trailing_stop` (impl), 구 이름은 deprecated shim
- `calculate_cvar` 내부 호출 → `self.compute_var` 업데이트

**RLRiskManager**:
- `check_max_drawdown` → `check_drawdown` (impl), 구 이름은 deprecated shim
- `calculate_var` → `compute_var` (impl), 구 이름은 deprecated shim
- `check_trailing_stop` 이미 존재 → abstract 충족
- `check_stop_loss` — entry_price 기반 실제 로직, 구현 유지 (base shim과 다른 semantics)
- `check_var_exceed` 내부 → `self.compute_var` 업데이트

---

## E10 — Parity 100회 시나리오

`tests/test_parity.py`에 `TestRandomScenarioParity` 추가:
- `test_drawdown_parity_100_random` — peak/current 100개 무작위 → BRM, RL, Unified 일치 확인
- `test_var_parity_100_random` — n=15~100 무작위 수익률 100개 → BRM, RL, Unified VaR 일치 확인

---

## E11 — 기존 관리자 클래스 제거 경로

Week 60에서 `BacktestingRiskManager.__init__`과 `RLRiskManager.__init__`에 `DeprecationWarning` 추가됨.  
Week 70에서 메서드 수준 deprecated shim 완성.  
**Phase 8 예정**: alias-only로 축소 (class body 제거, `BacktestingRiskManager = ...`).

---

## E12 — End-to-End Risk Path 테스트

`tests/integration/test_risk_path.py` 신규 작성:
- `PaperTrader → OrderManager → UnifiedRiskManager` 단일 경로 확인 (mock 없이)
- 시나리오: drawdown 초과 시 order reject 확인
