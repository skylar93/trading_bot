"""
Results Report page — Streamlit UI (Week 28)

학습 결과 리포트를 웹에서 열람합니다.
- 기존 HTML 리포트 파일 로드 및 표시
- 새 리포트 on-demand 생성
- Walk-forward Sharpe 요약 차트
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _list_reports() -> list[Path]:
    """results/ 디렉터리에서 HTML 리포트 목록을 반환합니다."""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("report_*.html"), reverse=True)


def _load_feature_importance() -> dict:
    fi_path = PROJECT_ROOT / "results" / "feature_importance.json"
    if fi_path.exists():
        try:
            with open(fi_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_wf_results() -> list[dict]:
    wf_dir = PROJECT_ROOT / "results" / "walk_forward"
    if not wf_dir.exists():
        return []
    results = []
    for p in sorted(wf_dir.glob("fold_*.json")):
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


def _render_summary_metrics(wf_results: list[dict]) -> None:
    if not wf_results:
        st.info("Walk-forward 결과 없음 (학습 완료 후 리포트 생성 버튼을 누르세요)")
        return

    oos_sharpes = [f.get("oos_sharpe", 0.0) for f in wf_results]
    is_sharpes = [f.get("is_sharpe", 0.0) for f in wf_results]
    all_equity = []
    for f in wf_results:
        all_equity.extend(f.get("equity_curve", []))

    oos_mean = float(np.mean(oos_sharpes))
    is_mean = float(np.mean(is_sharpes))
    if all_equity:
        eq = np.array(all_equity)
        peak = np.maximum.accumulate(eq)
        max_dd = float(((eq - peak) / (peak + 1e-8)).min()) * 100
    else:
        max_dd = 0.0

    stability = 0.0
    if len(oos_sharpes) > 1:
        stability = float(1.0 - np.array(oos_sharpes).std() / (abs(np.mean(oos_sharpes)) + 1e-8))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OOS Sharpe (평균)", f"{oos_mean:.3f}",
                delta=f"{oos_mean - is_mean:+.3f} vs IS")
    col2.metric("Max Drawdown", f"{max_dd:.1f}%",
                delta=None,
                delta_color="inverse" if max_dd < -10 else "normal")
    col3.metric("Stability Ratio", f"{stability:.3f}")
    col4.metric("Folds", str(len(wf_results)))


def _render_sharpe_chart(wf_results: list[dict]) -> None:
    if not wf_results:
        return

    try:
        import plotly.graph_objects as go  # noqa: PLC0415

        labels = [f.get("label", f"Fold {i+1}") for i, f in enumerate(wf_results)]
        is_sharpe = [f.get("is_sharpe", 0.0) for f in wf_results]
        oos_sharpe = [f.get("oos_sharpe", 0.0) for f in wf_results]

        fig = go.Figure(data=[
            go.Bar(name="IS Sharpe", x=labels, y=is_sharpe, marker_color="#2196F3"),
            go.Bar(name="OOS Sharpe", x=labels, y=oos_sharpe, marker_color="#4CAF50"),
        ])
        fig.update_layout(
            barmode="group",
            title="Walk-Forward: IS vs OOS Sharpe",
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback: simple table
        df = pd.DataFrame({
            "Fold": [f.get("label", f"Fold {i+1}") for i, f in enumerate(wf_results)],
            "IS Sharpe": [f.get("is_sharpe", 0.0) for f in wf_results],
            "OOS Sharpe": [f.get("oos_sharpe", 0.0) for f in wf_results],
        })
        st.dataframe(df, use_container_width=True)


def _render_feature_importance(importance: dict) -> None:
    if not importance:
        st.info("Feature importance 데이터 없음")
        return

    try:
        import plotly.graph_objects as go  # noqa: PLC0415

        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        features = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in values]

        fig = go.Figure(go.Bar(
            x=values[::-1], y=features[::-1],
            orientation="h",
            marker_color=colors[::-1],
        ))
        fig.update_layout(
            title="Feature Importance (Top 20)",
            template="plotly_dark",
            height=500,
            margin=dict(l=160, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        top = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        df = pd.DataFrame(top, columns=["Feature", "Importance"])
        st.dataframe(df, use_container_width=True)


def _generate_report_button() -> None:
    st.markdown("---")
    st.subheader("리포트 생성")
    col1, col2 = st.columns([2, 1])

    with col1:
        dry_run = st.checkbox("Dry-run (더미 데이터로 리포트 생성)", value=False)

    with col2:
        if st.button("HTML 리포트 생성", type="primary", use_container_width=True):
            with st.spinner("리포트 생성 중 ..."):
                try:
                    from scripts.generate_report import ReportGenerator  # noqa: PLC0415
                    rg = ReportGenerator(output_dir=PROJECT_ROOT / "results")
                    report_path = rg.generate(dry_run=dry_run)
                    if dry_run:
                        st.info(f"[Dry-run] 리포트 경로: `{report_path}`")
                    else:
                        st.success(f"리포트 생성 완료: `{report_path}`")
                        st.balloons()
                except Exception as exc:
                    st.error(f"리포트 생성 실패: {exc}")


def render_results_report() -> None:
    """Streamlit page: Results Report."""
    st.header("Results Report")
    st.caption("학습 결과 분석 — Walk-Forward, Feature Importance, Risk")

    # ── Tabs ──
    tab_summary, tab_report, tab_generate = st.tabs(
        ["Summary", "HTML Report", "Generate"]
    )

    with tab_summary:
        st.subheader("Executive Summary")
        wf_results = _load_wf_results()
        _render_summary_metrics(wf_results)

        st.subheader("Walk-Forward Sharpe")
        _render_sharpe_chart(wf_results)

        st.subheader("Feature Importance")
        importance = _load_feature_importance()
        _render_feature_importance(importance)

    with tab_report:
        st.subheader("HTML 리포트 열람")
        reports = _list_reports()

        if not reports:
            st.info("생성된 리포트가 없습니다. 'Generate' 탭에서 리포트를 생성하세요.")
        else:
            report_names = [p.name for p in reports]
            selected_name = st.selectbox("리포트 선택", report_names, index=0)
            selected_path = PROJECT_ROOT / "results" / selected_name

            if selected_path.exists():
                html_content = selected_path.read_text(encoding="utf-8")
                st.markdown(
                    f"**파일:** `{selected_path}` &nbsp;|&nbsp; "
                    f"**크기:** {len(html_content):,} bytes &nbsp;|&nbsp; "
                    f"**수정:** {datetime.fromtimestamp(selected_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}"
                )
                with st.expander("HTML 미리보기 (iframe)"):
                    # Streamlit components iframe embed
                    import streamlit.components.v1 as components  # noqa: PLC0415
                    components.html(html_content, height=800, scrolling=True)

                # Download button
                st.download_button(
                    label="다운로드",
                    data=html_content.encode("utf-8"),
                    file_name=selected_name,
                    mime="text/html",
                )

    with tab_generate:
        _generate_report_button()

        st.markdown("---")
        st.subheader("Pipeline Runner")
        st.markdown("""
전체 파이프라인을 터미널에서 실행하세요:

```bash
# Full pipeline (데이터 수집 → 학습 → 검증 → 리포트)
python scripts/run_full_pipeline.py --config config/local_3060ti.yaml

# 데이터 수집 건너뛰기
python scripts/run_full_pipeline.py --config config/local_3060ti.yaml --skip-data

# Dry-run (구조 확인)
python scripts/run_full_pipeline.py --dry-run
```
""")
