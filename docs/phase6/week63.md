# Week 63: Config Consolidation (S36-S40) — 회고

**날짜**: 2026-04-10  
**브랜치**: claude/nice-snyder  
**결과**: 1584 passed, 0 failed

---

## What

22개의 분산된 YAML 설정 파일을 10개로 통합 (55% 감소).

### 변경 파일

| 생성/수정 | 파일 |
|----------|------|
| 생성 | `config/base.yaml` |
| 생성 | `config/trading.yaml` |
| 생성 | `config/risk.yaml` |
| 재작성 | `config/monitoring.yaml` |
| 생성 | `config/deployment.yaml` |
| 생성 | `config/env/local_3060ti.yaml` |
| 생성 | `config/env/local_m2.yaml` |
| 생성 | `config/env/test.yaml` |
| 생성 | `config/env/uw_gpu.yaml` |
| 생성 | `config/loader.py` |
| 생성 | `docs/phase6/config_audit.md` |
| 생성 | `docs/phase6/config_migration.md` |
| 생성 | `tests/config/test_config_loader.py` (26 tests) |
| 수정 | `scripts/run_full_pipeline.py` (`--env` 플래그 추가) |
| 수정 | `scripts/test_training.py` |
| 수정 | `tests/test_deployment.py` |
| 수정 | `tests/deployment/test_secrets.py` |
| 수정 | `tests/test_auto_iterate.py` |
| 수정 | `tests/test_week17.py` |
| 수정 | `tests/config/__init__.py` (symlink → real dir) |
| 삭제 | 22개 → 20개 config 파일 (루트 레벨) |

---

## Why

Phase 6 플랜 S36-S40 명시 요구사항:
- Config 수 50% 이상 감소
- 단일 loader로 base + env 머지 + schema validate
- 모든 스크립트가 새 loader 사용
- 환경별 오버라이드는 diff만 포함 (DRY 원칙)

---

## Gotchas

### 1. tests/config symlink
`tests/config` 디렉터리가 자기 자신을 가리키는 circular symlink였음 (→ `config`). 실제 디렉터리로 교체.

### 2. flag_trader lora_rank 레이어
`flag_trader.lora_rank`가 두 곳에 있었음:
- `config/base.yaml#flag_trader.lora_rank: 16` (top-level 설정, FLAGTrader가 읽음)
- `config/env/local_3060ti.yaml#ensemble.agents[3].params.lora_rank: 8` (앙상블 내부)

env 오버라이드에 `flag_trader.lora_rank: 8`도 추가해야 FLAGTrader가 올바른 값을 읽음.

### 3. auto_iterate seed_strategy 누락
기존 `auto_iterate.yaml`의 `seed_strategy`, `evaluation`, `output` 섹션이 `base.yaml`에 없었음. 테스트 실패 후 추가.

### 4. 충돌 해소
- `initial_balance` 충돌 (10000 vs 100000) → 100000으로 통일
- `max_drawdown_pct` 충돌 (0.15 vs 0.20) → 0.20으로 통일 (실거래 기준)

### 5. load_raw() 하위호환
기존 코드(`yaml.safe_load(open(path))` 패턴)를 한 번에 전부 교체하면 위험. `load_raw(path)` 함수로 점진 마이그레이션 지원.

---

## 테스트 결과

```
26 new config loader tests: 26 passed
Full regression: 1584 passed, 0 failed (baseline 1386 대비 +198)
```

---

## Phase 7 후보 / 미완료

- `training/train_ensemble.py`, `training/train_pipeline.py`, `scripts/validate_training.py` 등 아직 `yaml.safe_load` 직접 사용 → `load_raw()` 또는 `load(env)` 점진 교체 필요
- `config/env/uw_gpu.yaml` — 실제 UW 클러스터 설정과 대조 후 fine-tuning 필요
- `scripts/validate_training.py` — `configs/` (오타, 실제 없음) 디렉터리 참조; 별도 이슈
