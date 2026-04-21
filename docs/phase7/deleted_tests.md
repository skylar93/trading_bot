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

---

## Week 80 추가 삭제 (H13 — web_interface 제거)

**결정**: Option A 채택 (Grafana로 대체). `deployment/web_interface/` 전체 제거.

### 10. `deployment/web_interface/` (전체 디렉토리)
- **삭제 이유**: H13 결정 — 36개 파일, 핵심 realtime 경로 2건이 TODO stub (`realtime_trading_manager.py:106`, `utils/data_stream.py:116`). Grafana + MetricsExporter (H1-H3)가 operational monitoring을 대체.
- **대체**: `deployment/monitoring/grafana_dashboard.json` (Week 78). 향후 간단한 Streamlit 스크립트가 필요하면 `scripts/` 에 단일 파일로 추가.
- **커버리지 공백**: 없음 (realtime 경로는 stub이었고, backtest UI는 `scripts/run_full_pipeline.py` + MLflow UI로 대체 가능).

### 11. `tests/test_week11.py`
- **삭제 이유**: `deployment/web_interface.*` 전적으로 의존. 108개 테스트 전부 web_interface 페이지·컴포넌트 단위 테스트.
- **대체**: Grafana dashboard validation은 smoke test 수준이면 충분.
- **커버리지 공백**: web_interface 페이지 단위 테스트 없음 — 의도된 것.

### 12. `tests/test_backtest_integration.py`
- **삭제 이유**: `deployment.web_interface.utils.backtest.BacktestManager`만 import. BacktestManager는 web_interface 내부 helper이며 다른 경로에서 사용되지 않음.
- **대체**: 실제 backtesting 커버리지는 `tests/test_base_backtester.py`, `tests/test_backtest_execution.py` 에서 유지.
- **커버리지 공백**: BacktestManager 기반 agent-backtest integration 시나리오. 실사용 없는 path.

---

## 파일 위치 (git history)

삭제 전 마지막 커밋에서 참조 가능.  
`git log --all -- tests/test_basic.py` 등으로 조회.  
`git log --all -- deployment/web_interface/app.py` 등으로 web_interface 조회.
