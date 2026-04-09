# Week 56 — State Persistence (S1-S6)

**Branch**: `claude/interesting-shtern`
**Goal**: PaperTrader는 크래시 후 재시작 시 position / cash / portfolio_history /
trades 를 정확히 복원할 수 있어야 한다.

## What

- `deployment/persistence/state_store.py` — SQLite (`PRAGMA journal_mode=WAL`)
  기반 `StateStore`. 테이블 3개 (`positions`, `orders`, `account_state`).
  `account_state.full_state_json` 컬럼에 `TradingState.to_dict()` 결과를
  통째로 보관해서 `portfolio_history` 와 `trades` 까지 손실 없이 복원.
- `TradingState.to_dict()` / `from_dict()` 추가 (`deployment/paper_trader.py`).
  - datetime → ISO string, NaN/inf 는 저장 단계 (`StateStore._reject_nonfinite`)
    에서 거부.
- `PaperTrader.__init__(state_store=...)` 옵션 + config `persistence:` 블록
  (`enabled / db_path / checkpoint_every_n_steps`). 미설정 시 None (하위호환).
- `PaperTrader._checkpoint()` — `PositionTracker._lock` 안에서 snapshot 떠서
  `state_store.save_snapshot()` 호출. run loop 내에서 `_log_step_metrics()`
  직후 매 step (또는 N step 마다) 실행.
- `PaperTrader.restore(state_store, agent, config, **kwargs)` 클래스메서드.
  복원 후 첫 step 에서 `restored: ...` 로그 기록.
- `config/local_3060ti.yaml` 에 `persistence:` 블록 추가.

## Why

Phase 5 까지 PaperTrader 는 in-memory 상태만 들고 있었다 (JSON
`save_checkpoint` 가 있지만 수동 호출). 실거래 직전에 SIGKILL 한 번이면
포지션·체결 이력·peak 가 모두 사라지는 상태였다. Phase 6 Track A 의
출발점으로, 매 step 자동 체크포인트 + 정확히 같은 결과로 재개되는 경로를
확보했다.

## Tests

`tests/deployment/test_state_persistence.py` 6 케이스:
1. `StateStore` round-trip / `clear()`
2. NaN/inf 거부
3. `TradingState.to_dict / from_dict` round-trip
4. **Idempotent restart** — 100 step 풀런 vs 50 + restore + 50 의
   `cash / position / peak / portfolio_value / num_trades / shutdown` 가 정확
   일치.
5. persistence 블록 없을 때 `state_store is None`
6. `persistence.enabled=true` 설정만으로 자동 StateStore 생성 + DB 파일 생성
   확인

## Regression

`/Users/skylar/anaconda3/bin/python -m pytest -q`
→ **1392 passed, 19 skipped, 0 failed** (Phase 5 baseline 1386 + 6 신규).

## Gotchas / 결정

- `PaperTrader.run()` 의 local `step` 변수를 `self.state.step` 으로 시드해서
  복원 후에도 누적 step 이 이어지도록 했다. 이게 없으면 split run 의
  최종 `state.step` 이 50 으로 덮어써져 baseline (100) 과 어긋난다.
- 복원 시 `_price_history` 는 마지막 가격으로 `window_size` 만큼 시드했다.
  `_build_observation()` 가 윈도우를 채우길 요구하기 때문이고, 실 거래에서는
  복원 후 들어오는 새 feed 가 점진적으로 윈도우를 다시 채우게 된다.
- `_slippage_records` 는 직렬화하지 않았다 (관측용 통계, 복원 필수 아님).
  Week 57 audit log 가 들어오면 그쪽에서 더 정확하게 본다.
- 현재 `orders` 테이블은 상시 비어 있다 (PaperTrader 가 OrderManager 미사용
  경로에서는 open order 를 들고 있지 않음). Week 57 / 64 에서 OrderManager
  연동 시 채워질 예정.

## 다음 (Week 57)

Immutable Audit Log — `deployment/audit/audit_logger.py` + chain hash 검증
스크립트.
