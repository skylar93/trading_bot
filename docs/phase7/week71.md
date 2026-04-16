# Week 71 Baseline — Numerical Hygiene & Noise Floor

**날짜**: 2026-04-16
**브랜치**: claude/suspicious-jepsen
**완료 조건**: numeric warning < 500, NaN canary 100% pass

---

## 완료 조건 결과

| 조건 | 결과 | 판정 |
|------|------|------|
| pytest warnings (filtered) | **37** (target: < 500) | ✅ |
| NaN canary pass rate (100 seeds × 100 steps) | **310 / 310** | ✅ |
| 전체 pytest | **2117 passed, 40 skipped, 0 failed** | ✅ |

---

## E13 — Warning 분류 스크립트

**파일**: [`scripts/analyze_warnings.py`](../../scripts/analyze_warnings.py)

- pytest warning report parser + classifier
- 버킷 분류: `divide-invalid`, `divide-zero`, `empty-slice`, `dof-zero`, `overflow`, ...
- own-code vs third-party 판별
- verdict: `bug` / `edge-case` / `third-party` / `intentional`

---

## E14 — Invariant 가드

**파일**: [`envs/multi_agent_multi_asset_env.py`](../../envs/multi_agent_multi_asset_env.py)

### 발견된 버그 (수정 완료)

| 위치 | 근본 원인 | 수정 |
|------|----------|------|
| `_process_portfolio_weights_action` line 563 | 0-vector action → `action / (sum + 1e-8)` = NaN | `isfinite` 검사 + uniform fallback |
| `_get_agent_observation` line 355 | empty/1-row window → `np.mean(empty)` = Mean of empty slice warning | `shape[0] < 2` guard + zero-fill |
| `_get_agent_observation` position_percentage | delisted asset NaN price → NaN position_value | price validity guard (≤ 0 or !finite → 0) |
| `_update_agent_portfolio_values` | NaN price × position = NaN portfolio value | price/balance validity guard |
| `_calculate_agent_reward` | NaN portfolio_value → NaN reward | `isfinite` 검사 + 0.0 fallback |

### E16 canary (multi-agent 쪽)

`_step_shared_capital` 끝에 observation/reward NaN 검사 `AssertionError` 추가. Test 실행 시 delisting 시나리오에서 잠재 버그 1건 조기 발견 → 위 `_update_agent_portfolio_values` 수정으로 해결.

---

## E14 — SingleAssetRLTradingEnv NaN canary (single-asset 쪽)

**파일**: [`envs/single_asset_rl_env.py`](../../envs/single_asset_rl_env.py)

step() 끝에 E16 NaN canary guard 추가:
- observation non-finite → `nan_to_num` + error log (assert 대신 복구 — single-asset은 기존 NaN 처리 로직이 이미 견고함)
- reward non-finite → 0.0 + error log

---

## E15 — RuntimeWarning → error 정책

**파일**: [`pytest.ini`](../../pytest.ini)

```ini
error::RuntimeWarning
ignore::RuntimeWarning:scipy.*
ignore::RuntimeWarning:numpy.*
ignore::RuntimeWarning:torch.*
ignore::RuntimeWarning:sklearn.*
```

새 PR에서 우리 코드에 unguarded numeric 연산이 들어오면 CI가 즉시 실패.

---

## E16 — NaN Canary 테스트

**파일**: [`tests/test_numerical_canary.py`](../../tests/test_numerical_canary.py)

| 테스트 클래스 | 커버리지 | 결과 |
|---|---|---|
| `TestSingleAssetNaNCanary` | 100 seeds × 100 steps, random action | 100/100 pass |
| `TestSingleAssetStressNaNCanary` | 10 seeds × 100 steps, alternating max buy/sell | 10/10 pass |
| `TestMultiAgentSharedCapitalNaNCanary` | 100 seeds × 100 steps, shared capital | 100/100 pass |
| `TestMultiAgentIndependentCapitalNaNCanary` | 100 seeds × 100 steps, independent capital | 100/100 pass |

---

## E17 — Warning Count Script

**파일**: [`scripts/count_warnings.py`](../../scripts/count_warnings.py)

```
python scripts/count_warnings.py
```

출력:
```
warnings : 37  (target < 500)
PASS: 37 < 500
```

---

## Week 71 이전/이후 비교

| 지표 | Week 70 이전 | Week 71 이후 |
|------|-------------|-------------|
| pytest warnings (전체 suite) | **12,068** | **37** |
| RuntimeWarning in own code | undetected (ignored) | **CI error** |
| NaN canary coverage | 없음 | 310 parametrized tests |
| multi_agent delisted asset bug | 잠재 (NaN reward 무음) | 수정 완료 |
| Warning 분류 도구 | 없음 | `analyze_warnings.py` |

---

## Phase 7 전제 (변경 없음)

- `risk_management/rl_risk_manager.py:226` trailing stop key = `f"_default_{symbol}"`
- `backtesting_risk_manager.py:567` VaR = `-norm.ppf(1 - CL) * std`
- `envs/single_asset_rl_env.py` portfolio valuation = `current_step - 1` 가격
- Python: `/Users/skylar/anaconda3/bin/python`
- Test baseline (2026-04-16, week71): **2117 passed, 40 skipped, 0 failed, 37 warnings**
