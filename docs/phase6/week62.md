# Week 62 Retrospective — DataSource Abstraction (S31-S35)

**날짜**: 2026-04-10  
**브랜치**: claude/determined-keller  
**PR**: Week 62 DataSource Abstraction (S31-S35)

---

## What

DataSource 인터페이스를 완전히 확장하고, `SingleAssetRLTradingEnv`가 내부에서 `df.iloc[step]`을 직접 쓰지 않도록 리팩터링했다.

### 변경 파일 요약

| 파일 | 역할 |
|------|------|
| `data/sources/base.py` | DataSource ABC + StaticDataSource (Week 61에서 존재, 확인) |
| `data/sources/csv_source.py` | CSVDataSource — lazy load, 컬럼명 자동 정규화 |
| `data/sources/mock_live_source.py` | MockLiveDataSource — tick 기반 bar-by-bar 시뮬레이션 |
| `data/quality/gate.py` | DataQualityGate + DataIssue — NaN/inf, 음수 가격, zero volume, 시간 gap 체크 |
| `data/sources/__init__.py` | 신규 exports 추가 |
| `data/quality/__init__.py` | 패키지 초기화 |
| `envs/single_asset_rl_env.py` | `_ds_len()`, `_row_at()`, `_window_at()` 헬퍼 추가, 모든 내부 `self.data.iloc[...]` / `len(self.data)` 제거 |
| `tests/data/test_data_sources.py` | 44개 신규 테스트 |

---

## Why

- 기존 Env는 `self.data.iloc[step]`으로 DataFrame을 직접 접근했다. `StaticDataSource`가 아닌 DataSource(`MockLiveDataSource` 등)를 주입하면 `self.data`가 `None`이어서 즉시 crash.
- Live 모드 테스트, CSV 로딩, 미래 데이터 차단 등을 위해 DataSource 인터페이스를 완전히 활용할 수 있어야 한다.
- DataSource를 거치기 전 데이터 품질을 검증하는 gate가 없으면 NaN이 환경에 흘러들어 보상이 오염된다.

---

## Implementation Notes

### S34 접근 방식

헬퍼 세 개를 추가해 모든 내부 접근을 간접화:

```python
def _ds_len(self) -> int: ...
def _row_at(self, step: int) -> pd.Series: ...
def _window_at(self, start: int, end: int) -> pd.DataFrame: ...
```

- `self.data` 속성은 **그대로 유지** (외부 테스트 및 기존 코드가 직접 읽음). backward-compat 유지.
- MockLiveDataSource를 주입하면 `self.data = None`이지만, `_ds_len()` / `_row_at()` 는 `self.data_source`를 경유하므로 문제없음.

### MockLiveDataSource tick 모델

- `tick()` 호출 전까지 미래 bar는 보이지 않음 (`get_window`가 visible_end로 clamp).
- `__len__()` = 전체 데이터셋 길이. 환경이 에피소드 종료 시점을 판단하는 데 사용.
- `current_tick` property로 현재 몇 번째 bar까지 보였는지 추적.

### DataQualityGate time gap 체크

- DataFrame 인덱스가 DatetimeIndex이거나 `timestamp` / `time` / `date` 컬럼이 있으면 활성화.
- median bar duration 기준으로 `max_gap_bars + 1` 배 초과 시 issue 발생.
- 컬럼/인덱스 없으면 gap 체크 skipped (조용히 무시, 에러 아님).

---

## Gotchas

1. **sentiment_data 길이 체크**: 기존엔 `len(self.data) != len(sentiment_data)` 비교했는데, `self.data`가 None인 경우를 방어해야 해서 `_ds_len()` 사용으로 변경.
2. **`_window_at(0, end_idx)` 에서 end_idx가 0인 경우**: `get_window(0, 0)`은 빈 DataFrame을 반환하도록 MockLiveDataSource가 처리함.
3. **CSVDataSource 열 rename 순서**: `$open`이 이미 있는데 `open`도 있으면 충돌할 수 있음. rename_map은 plain name만 타겟팅해 실제론 문제없음.

---

## Test Results

```
새 테스트:  44 passed (tests/data/test_data_sources.py)
전체 회귀:  1558 passed, 0 failed, 19 skipped
```

기존 1386 baseline 그대로 유지.

---

## Phase 6 완료 조건 진행 상황

- [x] UnifiedRiskManager (Week 60)
- [x] DI Refactor (Week 61)
- [x] DataSource Abstraction (Week 62) ← 이번 주
- [ ] Config Consolidation (Week 63)
