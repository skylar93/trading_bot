# Model Promotion Criteria (Week 75 G3)

**작성일**: 2026-04-19  
**적용 버전**: Phase 7+  
**원칙**: 자동 승급 금지. 모든 `canary → prod` 전이는 반드시 사람이 승인해야 한다.

---

## Stage 정의

| Stage | 의미 |
|-------|------|
| `candidate` | 신규 등록된 모델. 아직 평가 미완. |
| `staging` | 오프라인 백테스트 통과. 카나리 진입 자격 있음. |
| `canary` | 실제 트래픽 일부(기본 10%)로 shadow-execute 중. |
| `prod` | 전체 트래픽 담당 중인 운영 모델. |
| `retired` | 더 이상 사용하지 않음. 아카이브 보존. |

---

## 허용 전이

```
candidate → staging
staging   → canary | retired
canary    → prod | staging | retired
prod      → retired
```

강등(demote): `canary → staging` 허용 (성과 기준 미달 시).

---

## 전이별 필수 조건

### candidate → staging

오프라인 평가 기준 (모두 충족 필요):

| 지표 | 기준 |
|------|------|
| Sharpe ratio (in-sample) | ≥ 0.5 |
| Max drawdown | ≤ 30% |
| Parity with baseline | return diff ≤ 5%, DD diff ≤ 3%, slippage diff ≤ 0.2% |
| Walk-forward CV | purged K-fold — 모든 fold에서 Sharpe ≥ 0 |

절차:
1. `python scripts/validate_training.py --checkpoint <path>` 실행
2. Walk-forward 리포트 확인
3. `scripts/promote_model.py --from candidate --to staging --version N --actor <name> --reason "<reason>"`

---

### staging → canary

| 요건 | 내용 |
|------|------|
| staging 조건 | 모두 통과 (위 참조) |
| Walk-forward 리포트 | 자동 생성 필수 — 미생성 시 거부 |
| 사람 승인 | `promote_model.py` 실행 전 리포트 직접 검토 |
| 배포 범위 | 기본 traffic_pct=10%. config에서 조정 가능. |

---

### canary → prod

엄격한 기준. 다음 전부 충족 필요:

| 요건 | 기준 |
|------|------|
| 카나리 운영 기간 | **≥ 7일** 연속 |
| 트래픽 분배 | ≥ 10% 실 트래픽으로 shadow-execute |
| 수익률 비교 | canary mean return ≥ prod mean return − 0.5σ |
| 파산 확률 | 부트스트랩 CI 95% 기준 < 1% |
| 자동 제안 | `PaperTrader._canary_promotion_suggested == True` (단, 자동 승급 X) |
| 사람 승인 | **필수** — `promote_model.py` 실행자가 본인 이름으로 `--actor` 기록 |
| audit log | 모든 승급 이벤트 기록 확인 |

파산 확률 계산:
```python
# bootstrap CI (1000회 resampling)
returns = canary_returns[-168:]  # 7일 * 24시간
ruin_prob = sum(1 for _ in range(1000) if cumreturn(resample(returns)) < -0.5) / 1000
assert ruin_prob < 0.01
```

---

### canary → staging (강등)

| 조건 | 내용 |
|------|------|
| 강등 트리거 | canary mean return < prod − 0.5σ (7일 기준) |
| 강등 절차 | `promote_model.py --from canary --to staging --reason "underperformance"` |
| 재진입 조건 | 코드 수정 또는 재학습 후 `candidate` 재등록부터 |

---

### prod → retired

| 조건 | 내용 |
|------|------|
| 트리거 | 새 모델이 prod 승급, 또는 명시적 deprecation |
| 절차 | 신규 모델 승급 직전 기존 prod를 `retired`로 전이 |
| 보존 | 파일 삭제 금지. 아카이브 유지. |

---

## 승급 CLI 사용법

```bash
# 조건 충족 여부 사전 확인 (dry-run)
python scripts/promote_model.py --check --from candidate --to staging --version 3

# 실제 승급
python scripts/promote_model.py \
  --from staging --to canary \
  --version 3 \
  --actor "skylar" \
  --reason "walkforward Sharpe=0.82 all folds positive"

# canary → prod (7일 이상 운영 후)
python scripts/promote_model.py \
  --from canary --to prod \
  --version 3 \
  --actor "skylar" \
  --reason "7d canary passed, ruin_prob=0.003 < 1%, human approved"
```

---

## 완료 조건 (Week 75)

- [ ] canary → prod 전이 시뮬레이션 1회 완료 (테스트에서 검증)
- [ ] 핫스왑 테스트 pass (50-step 도중 agent 교체)
- [ ] promote_model.py --check 조건 미달 시 거부 확인
- [ ] 모든 승급 이벤트 audit log 기록 확인
