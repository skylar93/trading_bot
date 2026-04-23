# I1 완료 보고서 — 문서-코드 정합성 복구

**Date**: 2026-04-23  
**Plan ref**: Phase 7.6 I1 (B1-B3 해결)  
**Branch**: claude/condescending-napier-51268d  

---

## 완료 항목

### I1-a: `scripts/run_paper_trader.py` 신규 작성

**파일**: `scripts/run_paper_trader.py`

`deployment/paper_trader.PaperTrader` 를 감싸는 thin wrapper.

| Flag | 기본값 | 설명 |
|------|--------|------|
| `--config` | (required) | YAML config 경로 |
| `--exchange-mode` | `paper` | paper / sandbox / live |
| `--duration-hours` | None (무제한) | 실행 시간 (시간 단위) |
| `--log-dir` | `logs/` | 로그 + 최종 리포트 저장 경로 |
| `--pid-file` | `state/paper_trader.pid` | PID 파일 경로 |

동작:
- 시작 시 PID 파일 작성, SIGTERM/SIGINT → `_trigger_shutdown()` graceful halt
- 종료 시 PID 파일 제거 + `logs/paper_trader_{start_ts}.json` 리포트 저장

### I1-b: `deployment/paper_trader.py` CLI 확장

**파일**: `deployment/paper_trader.py` — `_main()` 함수

- `--duration-hours` 추가 (내부에서 초 변환)
- `--exchange-mode {paper,sandbox,live}` 추가 → `config.execution.exchange_mode` / `config.paper_trading.exchange_mode` override
- 기존 `--duration` (초) 는 `DeprecationWarning` 유지

### I1-c: `scripts/first_dollar_drill.py --live` mode

**파일**: `scripts/first_dollar_drill.py`

새 함수: `run_live_drill(capital, symbol, *, _exchange=None, _order_manager=None)`

| 가드 | 조건 |
|------|------|
| capital 한도 | `capital > 100` → 즉시 FAIL |
| 심볼 확인 | `symbol != BTC/USDT` → interactive confirm |
| 10분 timeout | 초과 시 `cancel_all_orders()` + abort |

실행 흐름:
1. `SecretProvider` 로 `EXCHANGE_BINANCE_KEY` / `EXCHANGE_BINANCE_SECRET` 로드 → 없으면 exit 1
2. `verify_exchange_key_scope.run_probes()` 직접 호출 → Withdraw 감지 시 exit 1
3. ccxt `fetch_ticker()` → mid price
4. `OrderManager(paper_mode=False)` → limit buy $50 @ mid × 0.98
5. 3분 폴링 (30s 간격) → open / filled / partial 분기 처리
6. `verify_audit_log.py` 자동 실행
7. `docs/phase7.6/live_drill_{ts}.md` 리포트 자동 생성

테스트 주입: `_exchange` / `_order_manager` 파라미터로 mock 주입 가능 (실 HTTP 호출 없음).

### I1-d: `docs/phase7/week85_first_dollar.md` 교정

Phase B Commands 블록을 `--live --capital 100` 명령으로 교체. "NOTE: simulation-only" 주석 제거.

---

## 신규 테스트

| 파일 | 커버리지 |
|------|---------|
| `tests/scripts/test_run_paper_trader_cli.py` | `_write_pid`, `_remove_pid`, `--help` flags, exchange-mode choices, config injection |
| `tests/scripts/test_first_dollar_drill_live.py` | capital guard, unfilled→cancel, filled→sell, partial→cancel+sell, audit, missing creds exit |

---

## 완료 조건 체크

- [x] `scripts/run_paper_trader.py` 존재, `--exchange-mode` + `--duration-hours` 동작
- [x] `deployment/paper_trader.py` `_main()` 동일 플래그 지원
- [x] `scripts/first_dollar_drill.py --live --capital 100` — creds 없을 때 `exit 1` + 명확한 에러
- [x] `docs/phase7/week85_first_dollar.md` Phase B 명령이 복붙 즉시 실행 가능
- [x] `docs/phase7/week85_72h.md` 명령 유효 (이미 올바른 상태, 변경 없음)
- [x] 신규 테스트 파일 2개 (`tests/scripts/`) 작성
