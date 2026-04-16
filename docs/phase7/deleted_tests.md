# Deleted Test Files — Week 69

**작성일**: 2026-04-15  
**이유**: pytest.ini ignore 17개 → ≤5 목표 달성을 위해 복구 불가 deprecated 테스트 삭제  
**원칙**: "Phase 6 기능을 이미 검증하고 있던 테스트만 복구" — 이하 파일들은 Phase 6+ 기능을 검증하지 않음

---

## 삭제 파일 목록 (9개)

### 1. `tests/test_basic.py`
- **삭제 이유**: `import ccxt` 의존. ccxt 미설치. Phase 1 시절 작성된 trivial smoke test (exchanges 목록 출력). Phase 6+ 기능 비검증.
- **대체**: Track F (Week 72) sandbox smoke 테스트 (`scripts/sandbox_smoke.py`)에서 ccxt 설치 후 커버.
- **커버리지 공백**: 없음.

### 2. `tests/test_advanced_networks.py`
- **삭제 이유**: `from gym import spaces` — 구버전 `gym` 사용. 현재 codebase는 `gymnasium` 사용. ConvLSTMPolicy, TabNetPolicy 등 아키텍처 테스트.
- **대체**: 아키텍처별 테스트는 `test_networks.py` 등에서 부분 커버. Phase 6에서 실사용 아키텍처는 PPO/SAC/TD3.
- **커버리지 공백**: ConvLSTM, TabNet, TCN 직접 unit test 없음. Phase 8에서 재작성 필요 시 `gymnasium`으로 작성.

### 3. `tests/test_action_space.py`
- **삭제 이유**: `from data.utils.multi_asset_data_loader import MultiAssetDataLoader` — `data.utils` 모듈 없음 (현재 `data/sources/`). Script-style 파일 (if __name__ == "__main__").
- **대체**: `MultiAssetTradingEnv` action space는 `test_multi_agent_multi_asset_*.py` 에서 간접 커버.
- **커버리지 공백**: MultiAssetTradingEnv action space 직접 unit test 없음 → Week 72 DataLoader 도입 시 복구.

### 4. `tests/test_multi_agent_integration.py`
- **삭제 이유**: `agents.strategies.test_agent_factory` → `agents.strategies.single.dummy_agent.DummyAgent` — 해당 모듈 없음 (`agents/strategies/single/`에 ppo/sac/td3만 존재).
- **대체**: Multi-agent 통합 테스트는 `tests/strategies/test_multi_agent_strategies.py`에서 커버.
- **커버리지 공백**: DummyAgent 기반 multi-agent 통합 시나리오.

### 5. `tests/test_multi_agent_multi_asset_integration.py`
- **삭제 이유**: `import gym` — 구버전 API. 현재 `gymnasium`. 장기 시뮬레이션/메모리 사용량 테스트.
- **대체**: `test_multi_agent_multi_asset_unit.py`, `test_multi_agent_multi_asset_scenarios.py` 등이 동일 env 커버.
- **커버리지 공백**: 장기(extended) 실행 + psutil 메모리 모니터링 시나리오 없음.

### 6. `tests/test_realtime_data.py`
- **삭제 이유**: `from data.utils.realtime_data import RealtimeDataManager, TradingDataStream` — 해당 클래스 없음. `data/sources/mock_live_source.py`로 대체됨.
- **대체**: `tests/data/test_data_sources.py`에서 MockLiveDataSource 커버.
- **커버리지 공백**: RealtimeDataManager 구현 시 복구 필요. Track F (Week 72)에서 CCXTLiveDataSource 추가 시 함께.

### 7. `tests/test_data_fetcher.py`
- **삭제 이유**: `from data.utils.data_loader import DataLoader` — 해당 클래스 없음. 구 DataLoader (ccxt 기반 fetch) 테스트.
- **대체**: `data/sources/csv_source.py` (CSVDataSource)가 현재 데이터 로딩 담당.
- **커버리지 공백**: exchange 기반 OHLCV fetch 단위 테스트 없음 → Track F에서 CCXTLiveDataSource 구현 시.

### 8. `tests/test_data_pipeline.py`
- **삭제 이유**: `from data.utils.data_loader import DataLoader` + `FeatureGenerator` — 구 파이프라인 클래스 없음.
- **오해 주의**: plan에서 "Week 65 Data Pipeline Safety와 직접 관련"으로 표기되었으나, 실제 Week 65 테스트는 `tests/deployment/test_data_pipeline_safety.py` (active, running). 이 파일은 구 데이터 파이프라인 테스트.
- **대체**: `tests/deployment/test_data_pipeline_safety.py` (완전 커버).
- **커버리지 공백**: 없음.

### 9. `tests/test_enhanced_backtester.py`
- **삭제 이유**: `training/backtesting/enhanced_backtester.py:16` → `data.utils.enhanced_data_loader.EnhancedDataLoader` — import chain에서 non-existent 모듈. enhanced_backtester.py 자체가 broken.
- **대체**: `tests/test_base_backtester.py`, `tests/test_backtest_execution.py`, `tests/test_backtest_integration.py`에서 backtesting 커버.
- **커버리지 공백**: regime-aware, ensemble backtest 시나리오.

---

## 커버리지 공백 요약 (추후 작업)

| 영역 | 공백 | 예정 시기 |
|------|------|---------|
| ConvLSTM/TabNet/TCN unit test | gymnasium 기반 재작성 필요 | Phase 8 |
| MultiAssetDataLoader 기반 action space test | Track F MultiAssetDataLoader 도입 시 | Week 72 |
| Exchange OHLCV fetch 단위 테스트 | CCXTLiveDataSource 구현 시 | Week 72 |
| Enhanced/Ensemble backtest | 별도 신규 테스트 (신규 추가 금지로 연기) | Phase 8 |

---

## 파일 위치 (git history)

삭제 전 마지막 커밋에서 참조 가능.  
`git log --all -- tests/test_basic.py` 등으로 조회.
