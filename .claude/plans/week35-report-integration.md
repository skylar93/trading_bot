# Week 35: Statistical Tests → HTML Report 통합

## Context

Phase 3 (Week 29-34) 전부 완료됨. 남은 이슈 1건:
- `training/analysis/statistical_tests.py`가 구현되어 있고 테스트도 통과하지만,
  `scripts/generate_report.py`의 HTML 리포트에 통합되지 않음
- 남편이 파이프라인 돌리면 HTML report에 bootstrap CI, p-value, DSR이 안 나옴

**목표**: Report에 "Section 6: Statistical Significance" 추가. 파이프라인 연결 완료.

**규칙**:
- 기존 테스트 깨지면 안 됨: `pytest tests/ -x --tb=short`
- 기존 report 섹션(1-5) 건드리지 않기
- plotly 차트 추가 불필요 — 텍스트 테이블로 충분

---

## 35.1 generate_report.py 수정

**파일**: `scripts/generate_report.py`

### Step 1: import 추가 (파일 상단 import 영역)

```python
from training.analysis.statistical_tests import StrategyStatisticalTests
```

### Step 2: generate() 메서드에 stat test 계산 추가

`generate()` 메서드 내, summary dict 구성 직후 (`summary = { ... }` 블록 뒤)에 추가:

```python
# --- Statistical significance tests ---
stat_tester = StrategyStatisticalTests()
all_returns = []
for f in walk_forward_results:
    eq = np.array(f.get("equity_curve", [1.0]))
    if len(eq) > 1:
        ret = np.diff(eq) / np.array(eq[:-1])
        all_returns.extend(ret.tolist())

stat_results = {}
if len(all_returns) >= 30:
    returns_arr = np.array(all_returns)
    lo, mid, hi = stat_tester.bootstrap_sharpe_ci(returns_arr, n_bootstrap=5000)
    p_val = stat_tester.permutation_test(returns_arr, n_permutations=5000)
    n_folds = len(walk_forward_results)
    sharpe_var = float(np.var([f.get("oos_sharpe", 0.0) for f in walk_forward_results]))
    dsr = stat_tester.deflated_sharpe_ratio(
        sharpe=mid, n_trials=max(n_folds, 1),
        var_sharpe=max(sharpe_var, 1e-6), skew=0.0, kurt=0.0,
    )
    stat_results = {
        "sharpe_ci_lower": lo,
        "sharpe_ci_point": mid,
        "sharpe_ci_upper": hi,
        "permutation_p_value": p_val,
        "deflated_sharpe_ratio": dsr,
    }
```

### Step 3: _build_html() 시그니처 확장

현재 시그니처:
```python
def _build_html(self, summary, equity_fig_json, sharpe_fig_json,
                feature_fig_json, drawdown_fig_json, fold_table_rows, config_json):
```

변경 (마지막에 파라미터 1개 추가):
```python
def _build_html(self, summary, equity_fig_json, sharpe_fig_json,
                feature_fig_json, drawdown_fig_json, fold_table_rows, config_json,
                stat_results=None):
```

### Step 4: HTML 본문에 Section 6 추가

`_build_html()` 내부, Section 5 (Configuration) 닫는 `</div>` 뒤, `</div></body></html>` 앞에 추가:

```python
# Section 6: Statistical Significance
stat_html = ""
if stat_results:
    ci_lo = stat_results.get("sharpe_ci_lower", 0)
    ci_pt = stat_results.get("sharpe_ci_point", 0)
    ci_hi = stat_results.get("sharpe_ci_upper", 0)
    p_val = stat_results.get("permutation_p_value", 1)
    dsr = stat_results.get("deflated_sharpe_ratio", 0)

    p_color = "#4CAF50" if p_val < 0.05 else ("#FF9800" if p_val < 0.10 else "#F44336")
    ci_color = "#4CAF50" if ci_lo > 0 else "#F44336"
    dsr_color = "#4CAF50" if dsr > 0.95 else ("#FF9800" if dsr > 0.5 else "#F44336")

    stat_html = f"""
  <h2>6. Statistical Significance</h2>
  <table style="width:100%; border-collapse:collapse; margin:1em 0;">
    <tr style="border-bottom:1px solid #333;">
      <th style="text-align:left; padding:8px;">Metric</th>
      <th style="text-align:left; padding:8px;">Value</th>
      <th style="text-align:left; padding:8px;">Interpretation</th>
    </tr>
    <tr>
      <td style="padding:8px;">Bootstrap Sharpe 95% CI</td>
      <td style="padding:8px; color:{ci_color};">[{ci_lo:.3f}, {ci_pt:.3f}, {ci_hi:.3f}]</td>
      <td style="padding:8px;">{"✅ CI lower > 0 → 유의" if ci_lo > 0 else "⚠ CI에 0 포함 → 비유의"}</td>
    </tr>
    <tr>
      <td style="padding:8px;">Permutation p-value</td>
      <td style="padding:8px; color:{p_color};">{p_val:.4f}</td>
      <td style="padding:8px;">{"✅ p < 0.05 → 우연 아님" if p_val < 0.05 else "⚠ p ≥ 0.05 → 과적합 의심"}</td>
    </tr>
    <tr>
      <td style="padding:8px;">Deflated Sharpe Ratio</td>
      <td style="padding:8px; color:{dsr_color};">{dsr:.4f}</td>
      <td style="padding:8px;">{"✅ DSR > 0.95 → 다중검정 통과" if dsr > 0.95 else "⚠ DSR ≤ 0.95 → 탐색 횟수 대비 유의성 부족"}</td>
    </tr>
  </table>
"""
```

이 `stat_html`을 HTML 문자열 결합 시 Section 5 뒤에 삽입.

### Step 5: generate()에서 _build_html() 호출 시 stat_results 전달

현재:
```python
html = self._build_html(
    summary=summary,
    equity_fig_json=equity_fig_json,
    sharpe_fig_json=sharpe_fig_json,
    feature_fig_json=feature_fig_json,
    drawdown_fig_json=drawdown_fig_json,
    fold_table_rows=fold_table_rows,
    config_json=config_json,
)
```

변경:
```python
html = self._build_html(
    summary=summary,
    equity_fig_json=equity_fig_json,
    sharpe_fig_json=sharpe_fig_json,
    feature_fig_json=feature_fig_json,
    drawdown_fig_json=drawdown_fig_json,
    fold_table_rows=fold_table_rows,
    config_json=config_json,
    stat_results=stat_results,
)
```

---

## 35.2 Pipeline 연결 확인

**파일**: `scripts/run_full_pipeline.py`

변경 불필요. 이미 `step_generate_report()`가 `ReportGenerator.generate()`를 호출하고, walk_forward_results를 전달함. generate() 내부에서 stat test를 자동 계산하므로 파이프라인 수정 없음.

---

## 35.3 테스트 추가

**파일**: `tests/test_phase3_integration.py`

기존 `TestBacktestingAndStatisticalTests` 클래스 끝에 테스트 1개 추가:

```python
def test_report_includes_stat_results(self, tmp_path):
    """HTML report에 statistical significance 섹션이 포함되는지 확인."""
    from scripts.generate_report import ReportGenerator

    rg = ReportGenerator(output_dir=tmp_path)
    report_path = rg.generate(output_path=tmp_path / "test_report.html")

    html = report_path.read_text(encoding="utf-8")
    assert "Statistical Significance" in html, "Report must contain stat section"
    assert "Bootstrap Sharpe" in html
    assert "Permutation p-value" in html
    assert "Deflated Sharpe" in html
```

---

## 검증

```bash
# 1. 단위 테스트
pytest tests/test_phase3_integration.py -v -k "stat" --tb=short

# 2. 전체 테스트 (기존 테스트 안 깨지는지)
pytest tests/ -x --tb=short

# 3. Report 생성 확인
python scripts/generate_report.py --dry-run
# 출력에 "Report saved" 또는 "DRY RUN" 메시지 나오면 OK

# 4. (선택) 실제 HTML 열어서 Section 6 확인
python scripts/generate_report.py --output results/test_report.html
open results/test_report.html
```

---

## 요약

| 항목 | 파일 | 작업 |
|------|------|------|
| Report 수정 | `scripts/generate_report.py` | import 추가, generate()에 stat 계산, _build_html()에 Section 6 |
| Pipeline | `scripts/run_full_pipeline.py` | 변경 없음 (자동 연결) |
| 테스트 | `tests/test_phase3_integration.py` | 테스트 1개 추가 |

난이도: ★☆☆ (파일 2개 수정, 테스트 1개 추가)
예상 소요: 15-20분
