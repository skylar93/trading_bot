# Week 60 Retrospective — RiskManager 통합 (S20-S25)

**Date:** 2026-04-09
**Branch:** claude/clever-grothendieck
**Track:** B — Architecture Consolidation

---

## What

`UnifiedRiskManager` 신규 클래스를 생성하고, 기존 `BacktestingRiskManager`와 `RLRiskManager`가 이를 composition으로 활용하도록 수정했다.

| 항목 | 내용 |
|---|---|
| 신규 파일 | `risk_management/unified_risk_manager.py` |
| 수정 파일 | `risk_management/rl_risk_manager.py`, `risk_management/backtesting_risk_manager.py` |
| 신규 테스트 | `tests/test_parity.py` (34개) |
| 신규 문서 | `docs/architecture/risk_manager.md`, `docs/phase6/week60.md` |

### 위임(delegate)된 메서드

| UnifiedRiskManager 메서드 | BRM 내 사용처 | RLRM 내 사용처 |
|---|---|---|
| `check_drawdown` | `check_max_drawdown` | `check_max_drawdown` (patterns 1, 2, 3) |
| `compute_var` | `calculate_var` | `calculate_var` |
| `check_correlation` | `check_correlation_limits` | `_check_correlation` |
| `check_trailing_stop` | 직접 사용 없음 (BRM 자체 상태 기반) | 직접 사용 없음 |
| `check_position_limit` | 새로운 공용 API | 새로운 공용 API |

---

## Why

`BacktestingRiskManager.calculate_var`와 `RLRiskManager.calculate_var`가 동일한 수식을 독립적으로 구현하고 있었다. 한쪽에서 버그를 수정하면 다른 쪽에도 동일하게 적용해야 했다. 이 Week의 목표는 그 중복을 제거하는 것.

---

## Key Decisions

1. **Composition, not inheritance**: `UnifiedRiskManager`는 `RiskManagerBase`를 상속하지 않는다. 상태 모델이 두 기존 클래스 사이에서 너무 달라 (BRM은 `StopLossConfig` 기반, RLRM은 `deque` 기반) 단순 상속으로는 정리가 어렵다.

2. **`threading.RLock` 선택**: 기존 `RLRiskManager`는 `threading.Lock`을 사용한다. `UnifiedRiskManager`는 composing class의 lock 내에서 호출되는 경우를 대비해 `RLock`(reentrant)을 사용했다.

3. **`var_method`는 `mode`와 독립**: 계획(S20)대로 VaR 계산 방식은 backtest/live 모드와 무관하게 설정 가능하다.

4. **DeprecationWarning 타이밍**: 기존 클래스 `__init__`에 즉시 경고를 추가했다. 실제 제거는 다음 Phase에서 진행.

---

## Gotchas

1. **경계값 테스트**: `peak=1000, current=850` → `drawdown = 0.15`. `>` (strict greater)이므로 False. 처음에 `True`로 예상했다가 수정했다.

2. **BRM `calculate_var` 최솟값 조건**: BRM은 `len < 2`면 0.0을 반환하지만, `UnifiedRiskManager.compute_var`는 `len < 10`이면 `None`을 반환한다. BRM에서 `result if result is not None else 0.0` 패턴으로 차이를 유지했다. (BRM caller들은 0.0을 기대하고 있음)

3. **Correlation semantics inversion**: BRM의 `check_correlation_limits`는 True = "안전" 의미. UnifiedRiskManager의 `check_correlation`은 True = "위험 초과". BRM 내에서 `not self._unified.check_correlation(...)` 로 처리.

---

## Results

- Parity tests: 34/34 passed
- Full regression: **1489 passed, 0 failed**, 19 skipped
- Phase 5 baseline (1386) 대비 103개 증가 (Weeks 56-60 누적 신규 테스트)

---

## Phase 7 후보

- `BacktestingRiskManager`와 `RLRiskManager` 실제 제거 (stub 또는 alias로 대체)
- `UnifiedRiskManager`에 state 관리(peak_values, returns_history 등) 통합 — 현재는 composing class에 분산됨
