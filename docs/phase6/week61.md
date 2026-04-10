# Week 61: Dependency Injection Refactor (S26-S30)

**Date**: 2026-04-10
**Branch**: claude/keen-buck
**Tests**: 1514 passed, 0 failed (baseline 1386 + 25 new DI tests + other pre-existing additions)

---

## What

### S26 — PaperTrader `risk_manager` type annotation
- `deployment/paper_trader.py`: `risk_manager=None` → `risk_manager: Optional[RiskManagerBase] = None`
- Added `from risk_management.risk_manager_base import RiskManagerBase` import
- 하위호환 유지: None이면 기존 동작 그대로

### S27 — DataSource abstraction skeleton + Env injection
- 신규 파일: `data/sources/base.py` — `DataSource` ABC + `StaticDataSource` 구현
  - `get_window(start, end)`, `latest()`, `__len__()`, `is_live()` 인터페이스
  - `StaticDataSource`: in-memory DataFrame wrapper, index 자동 reset
- `envs/single_asset_rl_env.py` 수정:
  - `data_source: Optional[DataSource] = None` 파라미터 추가
  - `data_source=` 가 있으면 그것을 사용 (priority)
  - 기존 `data=` 인자는 내부에서 `StaticDataSource(data)`로 자동 래핑
  - `self.data_source` 속성으로 저장 → 외부에서 접근 가능

### S28 — `training/factories/build_system.py`
- 신규 디렉토리: `training/factories/`
- `build_system(config, *, data, data_source, risk_manager, agent, trader)` 함수
  - 조립 순서: data_source → env → risk_manager → agent → trader
  - 각 단계에서 `override=` 인자로 외부 주입 허용
  - 어떤 단계가 실패해도 warning 로그 + None 반환 (부분 조립 허용)
  - 반환: `SystemComponents` dataclass
- `training/factories/__init__.py`: `build_system`, `SystemComponents` export

### S29 — `scripts/run_full_pipeline.py` 수정
- Step 9 (Paper Trading) 에서 `build_system` 호출
- `paper_trading.enabled=true`일 때 전체 시스템 조립 시도 후 로그

### S30 — `tests/integration/test_build_system.py`
- 25개 테스트, 전부 통과
- 커버리지: StaticDataSource, Env DI, PaperTrader DI, build_system 조립

---

## Why

Week 60 (RiskManager 통합) 이후 의존성이 여러 곳에 하드코딩되어 있었음.
build_system 단일 엔트리포인트를 만들어:
1. 테스트에서 mock 주입이 가능해짐
2. 새 컴포넌트(DataSource, UnifiedRiskManager 등)를 교체해도 호출처 변경 불필요
3. Week 62 DataSource 확장, Week 63 Config 통합의 기반이 됨

---

## Gotchas

- `create_env`가 요구하는 env type은 `"single_asset_rl"` (not `"single_asset"`) — 테스트에서 처음엔 틀린 값으로 실패했음
- `StaticDataSource`는 index를 reset 해야 `get_window(start, end)` iloc이 올바르게 동작함
- `data_source=`가 있어도 `StaticDataSource`가 아닌 경우 `data_source.df` 접근이 불가 → `_build_env`에서 명시적 분기 처리

---

## Week 62 Preview

- `data/sources/base.py`에 `CSVDataSource`, `MockLiveDataSource` 추가
- `SingleAssetRLTradingEnv` 내부에서 `data_source.get_window()` 직접 사용 (현재는 `self.data` DataFrame 직접 접근)
- Data quality gate (`data/quality/gate.py`)
