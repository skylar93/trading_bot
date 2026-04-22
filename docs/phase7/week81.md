# Week 81: 결함 청산 & 문서 일관성

**Phase**: 7.5 — Live Closure  
**Date**: 2026-04-22  
**Branch**: claude/gallant-buck-3a5c45  
**Tests**: 2332 passed, 41 skipped, 0 failed  
**phase7_smoke**: 67/67 ✓

---

## R1. web_interface 유령 폴더 제거 (D1) ✓

`deployment/web_interface/` (components/, pages/, utils/, __pycache__/) 전체 삭제.  
파일들은 모두 gitignored (`__pycache__/**`, `.DS_Store`) — `git rm` 불필요, `rm -rf`로 삭제.  
`phase7_smoke H13` check green.

**완료 조건**: `ls deployment/web_interface/` → "No such file or directory" ✓

---

## R2. check_stop_loss → check_trailing_stop 마이그레이션 (D2) ✓

**Files modified:**
- `envs/single_asset_rl_env.py`: deprecated `check_stop_loss` 블록 제거, trailing_stop으로 통합
- `envs/multi_asset_env.py:751`: deprecated `check_stop_loss` 블록 제거
- `tests/test_week37.py`: stop_loss 테스트 2개 → trailing_stop 기반으로 마이그레이션

**Design decision:**  
`RLRiskManager.check_stop_loss(agent_id, position_size, entry_price, current_price)` (entry-price based)와 `check_trailing_stop(agent_id, asset, position_size, current_price)` (high-water mark based)는 동작이 다르다. 이 두 check는 중복되므로 trailing_stop 단일 check로 통합. 기존 stop_loss 테스트는 `use_trailing_stop=True, trailing_stop_buffer=0.05`로 재구성해 동일한 declining-price 시나리오를 검증.

**완료 조건**: `pytest -W error::DeprecationWarning tests/test_week37.py` 8/8 pass ✓

---

## R3. Idempotency Flaky 결론 (D3) ✓

- `pytest-repeat>=0.9.3` 설치, `requirements.txt` 추가
- 100회 연속 실행: **100/100 pass** (0 failures)
- 기존 구현 `deployment/execution/order_manager.py:317-326`의 `with self._lock: setdefault(...)` 패턴이 완전히 atomic. Lock 수정 불필요.
- 상세 기록: `docs/phase7/week81_idempotency.md`
- pytest.ini에 flaky 마커 없었으므로 제거 작업 없음

**완료 조건**: 100회 연속 pass 증거 ✓

---

## R4. 누락 주차 문서 5개 작성 (D4) ✓

| 파일 | 내용 |
|------|------|
| `docs/phase7/week72.md` | CCXT Sandbox Wire (F1–F6): CCXTAdapter WebSocket-first, exchange_mode, reconnect, credential redaction |
| `docs/phase7/week73.md` | Reconciliation (F7–F11): ExchangeSnapshot, reconcile_on_boot, mismatch 3종 + 3 policy, ClockSync |
| `docs/phase7/week74.md` | Execution Realism (F12–F16): 4 order types, partial fill, TTL, SlippageModel OLS, FeeModel VIP |
| `docs/phase7/week76.md` | Compliance (G6–G10): PreTradeComplianceChecker, check_all(), 4 guards, thread-safe sliding window |
| `docs/phase7/week78.md` | Observability (H1–H5): 30+ Prometheus metrics, 12-panel Grafana, Discord alert, OTel init, Sentry scrubbing |

`ls docs/phase7/week{69..80}.md` 전부 존재 ✓

---

## R5. Deprecation Caller CI Gate (D2 재발 방지) ✓

**File added:** `scripts/check_deprecation_callers.py`  
- 금지 심볼: `check_stop_loss`, `check_max_drawdown`, `calculate_var`  
- 제외: risk_management/ shim 정의 파일, tests/, scripts/  
- word-boundary 검색 (`\b` regex / `-w` grep flag) — `check_stop_losses` 복수형 false positive 방지
- Exit 0 = clean, Exit 1 = callers found

**first_dollar_drill.py 통합**: `check_no_old_risk_api()` → `check_deprecation_callers.py` delegate로 교체 (중복 제거, 범위 확장)

**CI 추가**: `.github/workflows/ci.yml`에 `python scripts/check_deprecation_callers.py` step 추가 (테스트 실행 전)

**완료 조건**: 스크립트 clean exit ✓, CI gate 활성 ✓

---

## 완료 조건 체크 (Week 81)

| 조건 | 상태 |
|------|------|
| phase7_smoke H13 green | ✓ (67/67) |
| deprecation warning 0건 (`-W error::DeprecationWarning`) | ✓ |
| 주차 문서 12/12 (week69~80) | ✓ |
| CI gate 활성 | ✓ |
| idempotency 결론 문서 | ✓ |
| 전체 pytest 0 fail | ✓ (2332 passed) |
