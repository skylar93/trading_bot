# pytest.ini Ignore Audit — Week 69

**작성일**: 2026-04-15  
**감사 기준**: Phase 6 완료 직후 main 상태 (1780 passed, 19 skipped, 1 flaky)  
**목표**: 17개 ignore → ≤ 5

---

## 결론 요약

| 파일 | 실패 원인 | 처리 방침 |
|------|----------|----------|
| `test_basic.py` | `import ccxt` (not installed) | **DELETE** |
| `test_advanced_networks.py` | `from gym import spaces` (old API, `gymnasium`으로 교체됨) | **DELETE** |
| `test_action_space.py` | `data.utils.multi_asset_data_loader` non-existent | **DELETE** |
| `test_live_trading.py` | `envs/live_trading_env.py` → `data.utils.websocket_loader` + `ccxt` | **SKIP** `HAS_CCXT` |
| `test_live_trading_advanced.py` | `import ccxt` (not installed) | **SKIP** `HAS_CCXT` |
| `test_multi_agent_integration.py` | `agents.strategies.single.dummy_agent.DummyAgent` non-existent | **DELETE** |
| `test_multi_agent_multi_asset_integration.py` | `import gym` (old API) | **DELETE** |
| `test_multi_asset_data.py` | `data.utils.multi_asset_data_loader` + `data_synchronization` non-existent | **JUSTIFY+TAG** (Week 72) |
| `test_multi_asset_env.py` | `data.utils.multi_asset_data_loader` non-existent | **JUSTIFY+TAG** (Week 72) |
| `test_paper_trading.py` | `envs/paper_trading_env.py` → `data.utils.websocket_loader` | **FIX** (import conditional) |
| `test_ppo_advantage_update.py` | 파일 자체 없음 (already deleted) | **REMOVE FROM INI** |
| `test_realtime_data.py` | `data.utils.realtime_data.RealtimeDataManager` non-existent | **DELETE** |
| `tests/training/hyperopt/test_ray_hyperopt.py` | `import ray` (not installed) | **SKIP** `HAS_RAY` |
| `tests/training/utils/test_ray_manager.py` | `import ray` (not installed) | **SKIP** `HAS_RAY` |
| `test_data_fetcher.py` | `data.utils.data_loader.DataLoader` non-existent | **DELETE** |
| `test_data_pipeline.py` | `data.utils.data_loader.DataLoader` non-existent | **DELETE** |
| `test_enhanced_backtester.py` | `training/backtesting/enhanced_backtester.py` → `data.utils.enhanced_data_loader` | **DELETE** |

---

## 상세 분석

### 1. `tests/test_basic.py`
- **실패 원인**: `import ccxt` — ccxt 패키지 미설치
- **내용**: ccxt exchanges 목록 출력, pandas 동작 확인. 단순 smoke test.
- **처리**: **DELETE** — Track F (Week 72)에서 ccxt 설치 시 별도 smoke 포함 예정. 이 파일은 Phase 6+ 기능을 검증하지 않음.

### 2. `tests/test_advanced_networks.py`
- **실패 원인**: `from gym import spaces` — `gym` 패키지 없음 (현재 codebase는 `gymnasium` 사용)
- **내용**: ConvLSTMPolicy, TabNetPolicy 등 아키텍처 unit tests
- **처리**: **DELETE** — `gymnasium`으로 교체된 이후 방치된 파일. 현재 아키텍처 테스트는 다른 파일에서 커버됨.

### 3. `tests/test_action_space.py`
- **실패 원인**: `from data.utils.multi_asset_data_loader import MultiAssetDataLoader` — `data.utils` 모듈 없음
- **내용**: MultiAssetTradingEnv의 action space 테스트. Script-style (if __name__ == "__main__" 구조).
- **처리**: **DELETE** — `data.utils.multi_asset_data_loader`는 현재 codebase에 존재하지 않음. 데이터 계층이 `data/sources/`로 재구성됨. 새 테스트 작성 없이 복구 불가.

### 4. `tests/test_live_trading.py`
- **실패 원인**: `envs/live_trading_env.py:9` → `data.utils.websocket_loader` + `ccxt.async_support`
- **내용**: LiveTradingEnvironment 단위 테스트. Exchange/WebSocket을 AsyncMock으로 주입.
- **처리**: **SKIP (HAS_CCXT)** — Track F (Week 72)에서 ccxt 설치 후 자동 활성화. `envs/live_trading_env.py` 조건부 import 처리 + 파일 상단에 skipif 추가. ignore에서 제거.

### 5. `tests/test_live_trading_advanced.py`
- **실패 원인**: `import ccxt` at module level
- **내용**: 고급 LiveTradingEnvironment 시나리오. ccxt 직접 사용.
- **처리**: **SKIP (HAS_CCXT)** — 상단에 `HAS_CCXT` guard + skipif 데코레이터 추가. ignore에서 제거.

### 6. `tests/test_multi_agent_integration.py`
- **실패 원인**: `agents.strategies.test_agent_factory` → `agents.strategies.single.dummy_agent.DummyAgent` non-existent
- **내용**: MultiAgentTradingEnv 통합 테스트.
- **처리**: **DELETE** — `DummyAgent`가 현재 codebase에 없음. `agents/strategies/single/`에는 `ppo_agent.py`, `sac_agent.py`, `td3_agent.py`만 존재. 복구 시 새 모듈 작성 필요 → E2 범위 초과.

### 7. `tests/test_multi_agent_multi_asset_integration.py`
- **실패 원인**: `import gym` — `gym` 패키지 없음
- **내용**: Multi-agent Multi-asset 장기 시뮬레이션 통합 테스트.
- **처리**: **DELETE** — `gym` → `gymnasium` 마이그레이션 후 방치. 현재 `test_multi_agent_multi_asset_*.py` 파일들 (unit/risk/scenarios/mock)이 해당 env를 커버.

### 8. `tests/test_multi_asset_data.py`
- **실패 원인**: `data.utils.multi_asset_data_loader` + `data.utils.data_synchronization` non-existent
- **내용**: MultiAssetDataLoader, 타임스탬프 정렬, outlier 감지 등
- **처리**: **JUSTIFY+TAG (Week 72)** — `MultiAssetDataLoader`는 Track F에서 `CCXTLiveDataSource` 구현 시 함께 도입 예정. 현재 `data/sources/` 계층에 해당 클래스 없음. 이 파일은 ignore에 유지하되, Week 72에서 `data/sources/multi_asset.py` 추가 시 복구.

### 9. `tests/test_multi_asset_env.py`
- **실패 원인**: `data.utils.multi_asset_data_loader` non-existent
- **내용**: MultiAssetTradingEnv 환경 테스트 (obs space, portfolio 계산 등). `envs/multi_asset_env.py`는 동작 중.
- **처리**: **JUSTIFY+TAG (Week 72)** — 환경 자체(`envs/multi_asset_env.py`)는 정상이지만 테스트가 `MultiAssetDataLoader`로 데이터를 준비함. Week 72에서 DataLoader 교체 시 복구.

### 10. `tests/test_paper_trading.py`
- **실패 원인**: `envs/paper_trading_env.py:10` → `data.utils.websocket_loader`
- **내용**: PaperTradingEnvironment 비동기 테스트. `test_mode=True`로 websocket 실제 연결 없음.
- **처리**: **FIX** — `envs/paper_trading_env.py`의 `WebSocketLoader` import를 `try/except`로 조건부 처리. `test_mode=True`일 때 `WebSocketLoader` 미사용 → 수정 후 테스트 통과 예상. ignore에서 제거.

### 11. `tests/test_ppo_advantage_update.py`
- **실패 원인**: 파일 없음 (pytest.ini에만 참조 남음)
- **처리**: pytest.ini에서 참조 **제거**.

### 12. `tests/test_realtime_data.py`
- **실패 원인**: `data.utils.realtime_data.RealtimeDataManager, TradingDataStream` non-existent
- **내용**: 실시간 데이터 스트림 테스트
- **처리**: **DELETE** — 해당 클래스가 현재 codebase에 없음. `data/sources/mock_live_source.py`가 MockLiveDataSource로 대체.

### 13. `tests/training/hyperopt/test_ray_hyperopt.py`
- **실패 원인**: `import ray` — ray 패키지 미설치
- **내용**: Ray Tune 기반 hyperparameter optimization 테스트.
- **처리**: **SKIP (HAS_RAY)** — 모듈 상단에 `HAS_RAY = False; try: import ray; HAS_RAY = True` + 각 test에 `@pytest.mark.skipif(not HAS_RAY, reason="ray not installed")`. ignore에서 제거.

### 14. `tests/training/utils/test_ray_manager.py`
- **실패 원인**: `import ray` — ray 패키지 미설치
- **내용**: RayManager, RayActor 테스트.
- **처리**: **SKIP (HAS_RAY)** — 동일 처리. 파일 상단 `@ray.remote` decorator도 조건부 처리.

### 15. `tests/test_data_fetcher.py`
- **실패 원인**: `data.utils.data_loader.DataLoader` non-existent
- **내용**: DataLoader fetch_data, 캐싱, 타임존 처리 등
- **처리**: **DELETE** — DataLoader가 `data.utils`에 없음. 현재 `data/sources/csv_source.py` 등으로 대체됨.

### 16. `tests/test_data_pipeline.py`
- **실패 원인**: `data.utils.data_loader.DataLoader` + `data.utils.feature_generator.FeatureGenerator` non-existent
- **내용**: 구 DataLoader + FeatureGenerator 파이프라인 테스트.
- **처리**: **DELETE** — Week 65 Data Pipeline Safety는 `tests/deployment/test_data_pipeline_safety.py`에서 커버됨. 이 파일은 OLD 파이프라인 테스트 (non-existent classes).

### 17. `tests/test_enhanced_backtester.py`
- **실패 원인**: `training/backtesting/enhanced_backtester.py:16` → `data.utils.enhanced_data_loader.EnhancedDataLoader` non-existent
- **내용**: EnhancedBacktester 테스트 (regime detection, ensemble backtest 등)
- **처리**: **DELETE** — import chain에서 non-existent 모듈. `training/backtesting/enhanced_backtester.py` 자체도 broken.

---

## 처리 결과 (Week 69 완료 후)

| 처리 방침 | 파일 수 | ignore 남음 |
|----------|--------|-----------|
| DELETE | 9 | 0 |
| FIX (import) | 1 (test_paper_trading) | 0 |
| SKIP HAS_CCXT | 2 (test_live_trading, test_live_trading_advanced) | 0 |
| SKIP HAS_RAY | 2 (test_ray_hyperopt, test_ray_manager) | 0 |
| JUSTIFY+TAG | 2 (test_multi_asset_env, test_multi_asset_data) | **2** |
| REMOVE FROM INI | 1 (test_ppo_advantage_update, 파일 없음) | 0 |
| **합계** | **17** | **2 ≤ 5 ✅** |

---

## JUSTIFY+TAG 상세

`test_multi_asset_env.py` 와 `test_multi_asset_data.py` 는 다음 이유로 ignore 유지:
- `MultiAssetDataLoader` 클래스가 현재 codebase에 없음
- Track F (Week 72) `CCXTLiveDataSource` 구현 시 multi-asset data layer 도입 예정
- 테스트 파일 자체 로직은 유효함 (MultiAssetTradingEnv는 동작 중)
- 삭제 시 Week 72에서 재작성 필요 → 보존이 더 효율적
- Tag: `# WEEK72: Activate when MultiAssetDataLoader available in data/sources/`
