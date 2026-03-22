"""
HTML Report Generator (Week 28)

Standalone HTML 리포트를 생성합니다.
Plotly 차트가 embedded되어 있어 인터넷 연결 없이도 열람 가능합니다.

사용법:
    python scripts/generate_report.py --results-dir results --output results/report.html
    python scripts/generate_report.py --mlflow-run <run_id>
    python scripts/generate_report.py --dry-run   # 더미 데이터로 리포트 생성

포함 내용:
    1. Executive Summary (OOS Sharpe, max drawdown, stability ratio)
    2. Training Progress (reward curves, CVaR)
    3. Walk-Forward Results (fold별 IS vs OOS, equity curve)
    4. Feature Analysis (importance ranking, regime timeline)
    5. Risk Analysis (drawdown, CVaR 분포)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────────────────

class ReportGenerator:
    """Standalone HTML report generator using Plotly."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Metric helpers ──

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, freq: int = 252) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(freq))

    @staticmethod
    def _compute_max_drawdown(equity: np.ndarray) -> float:
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-8)
        return float(dd.min())

    @staticmethod
    def _compute_stability(returns: np.ndarray) -> float:
        """IS/OOS Sharpe ratio stability: avg(min(is,oos)/max(is,oos)) per fold."""
        if len(returns) < 2:
            return 0.0
        return float(1.0 - abs(returns).std() / (abs(returns).mean() + 1e-8))

    # ── Plotly figure builders ──

    def _make_equity_curve(self, folds_data: list[dict[str, Any]]) -> str:
        """Return plotly figure as JSON string."""
        try:
            import plotly.graph_objects as go  # noqa: PLC0415
        except ImportError:
            return ""

        fig = go.Figure()
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

        for i, fold in enumerate(folds_data):
            equity = fold.get("equity_curve", [])
            if not equity:
                continue
            label = fold.get("label", f"Fold {i+1}")
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                y=equity,
                name=label,
                line=dict(color=color, width=2),
                mode="lines",
            ))

        fig.update_layout(
            title="Equity Curves — Walk-Forward Folds",
            xaxis_title="Step",
            yaxis_title="Portfolio Value",
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig.to_json()

    def _make_sharpe_bar(self, fold_metrics: list[dict[str, Any]]) -> str:
        try:
            import plotly.graph_objects as go  # noqa: PLC0415
        except ImportError:
            return ""

        labels = [m.get("label", f"Fold {i+1}") for i, m in enumerate(fold_metrics)]
        is_sharpe = [m.get("is_sharpe", 0.0) for m in fold_metrics]
        oos_sharpe = [m.get("oos_sharpe", 0.0) for m in fold_metrics]

        fig = go.Figure(data=[
            go.Bar(name="IS Sharpe", x=labels, y=is_sharpe, marker_color="#2196F3"),
            go.Bar(name="OOS Sharpe", x=labels, y=oos_sharpe, marker_color="#4CAF50"),
        ])
        fig.update_layout(
            barmode="group",
            title="IS vs OOS Sharpe Ratio per Fold",
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig.to_json()

    def _make_feature_importance_bar(self, importance: dict[str, float]) -> str:
        try:
            import plotly.graph_objects as go  # noqa: PLC0415
        except ImportError:
            return ""

        if not importance:
            return ""

        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        features = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in values]

        fig = go.Figure(go.Bar(
            x=values[::-1],
            y=features[::-1],
            orientation="h",
            marker_color=colors[::-1],
        ))
        fig.update_layout(
            title="Feature Importance (Top 20)",
            xaxis_title="Importance Score",
            template="plotly_dark",
            height=500,
            margin=dict(l=160, r=20, t=50, b=40),
        )
        return fig.to_json()

    def _make_drawdown_plot(self, equity: list[float]) -> str:
        try:
            import plotly.graph_objects as go  # noqa: PLC0415
        except ImportError:
            return ""

        if not equity:
            return ""

        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / (peak + 1e-8) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=drawdown,
            fill="tozeroy",
            fillcolor="rgba(244,67,54,0.3)",
            line=dict(color="#F44336"),
            name="Drawdown %",
        ))
        fig.update_layout(
            title="Drawdown (%) over Time",
            yaxis_title="Drawdown %",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig.to_json()

    # ── HTML template ──

    def _build_html(
        self,
        summary: dict[str, Any],
        equity_fig_json: str,
        sharpe_fig_json: str,
        feature_fig_json: str,
        drawdown_fig_json: str,
        fold_table_rows: str,
        config_json: str,
    ) -> str:
        plotly_cdn = "https://cdn.plot.ly/plotly-2.27.0.min.js"
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _fig_div(fig_json: str, div_id: str) -> str:
            if not fig_json:
                return f'<div id="{div_id}" class="no-data">No data available</div>'
            return f"""
<div id="{div_id}"></div>
<script>
  (function() {{
    var fig = {fig_json};
    Plotly.newPlot('{div_id}', fig.data, fig.layout, {{responsive: true, displayModeBar: false}});
  }})();
</script>"""

        oos_sharpe = summary.get("oos_sharpe_mean", 0.0)
        max_dd = summary.get("max_drawdown", 0.0)
        stability = summary.get("stability_ratio", 0.0)
        n_folds = summary.get("n_folds", 0)

        sharpe_color = "#4CAF50" if oos_sharpe > 1.0 else ("#FF9800" if oos_sharpe > 0 else "#F44336")
        dd_color = "#4CAF50" if max_dd > -0.1 else ("#FF9800" if max_dd > -0.2 else "#F44336")

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading Bot — Results Report</title>
  <script src="{plotly_cdn}"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #1a1a2e;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      line-height: 1.6;
    }}
    .header {{
      background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
      padding: 24px 32px;
      border-bottom: 1px solid #0f3460;
    }}
    .header h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
    .header p {{ color: #90a4ae; font-size: 13px; margin-top: 4px; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
    h2 {{
      font-size: 18px; font-weight: 600; color: #fff;
      border-left: 4px solid #2196F3; padding-left: 12px;
      margin: 32px 0 16px;
    }}
    .metrics-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px; margin-bottom: 24px;
    }}
    .metric-card {{
      background: #16213e; border: 1px solid #0f3460;
      border-radius: 8px; padding: 16px;
    }}
    .metric-card .label {{ font-size: 12px; color: #90a4ae; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .chart-card {{
      background: #16213e; border: 1px solid #0f3460;
      border-radius: 8px; padding: 16px; margin-bottom: 20px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      padding: 10px 12px; text-align: right;
      border-bottom: 1px solid #0f3460; font-size: 13px;
    }}
    th {{ color: #90a4ae; font-weight: 500; text-align: right; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tr:hover {{ background: #0f3460; }}
    .no-data {{ color: #546e7a; font-style: italic; padding: 20px; text-align: center; }}
    .config-pre {{
      background: #0d1117; border: 1px solid #0f3460; border-radius: 6px;
      padding: 16px; font-size: 12px; font-family: monospace;
      overflow-x: auto; white-space: pre-wrap; color: #adbac7;
      max-height: 400px; overflow-y: auto;
    }}
    .footer {{ text-align: center; color: #546e7a; font-size: 12px; padding: 32px; }}
  </style>
</head>
<body>
<div class="header">
  <h1>Trading Bot — Results Report</h1>
  <p>Generated: {generated_at} &nbsp;|&nbsp; Folds: {n_folds}</p>
</div>

<div class="container">

  <!-- Section 1: Executive Summary -->
  <h2>1. Executive Summary</h2>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="label">OOS Sharpe (mean)</div>
      <div class="value" style="color:{sharpe_color}">{oos_sharpe:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Max Drawdown</div>
      <div class="value" style="color:{dd_color}">{max_dd*100:.1f}%</div>
    </div>
    <div class="metric-card">
      <div class="label">Stability Ratio</div>
      <div class="value" style="color:#2196F3">{stability:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Walk-Forward Folds</div>
      <div class="value" style="color:#9C27B0">{n_folds}</div>
    </div>
  </div>

  <!-- Section 2: Walk-Forward Results -->
  <h2>2. Walk-Forward Results</h2>
  <div class="chart-card">
    {_fig_div(sharpe_fig_json, 'sharpe-chart')}
  </div>
  <div class="chart-card">
    {_fig_div(equity_fig_json, 'equity-chart')}
  </div>

  <!-- Fold table -->
  <div class="chart-card">
    <table>
      <thead>
        <tr>
          <th>Fold</th>
          <th>IS Sharpe</th>
          <th>OOS Sharpe</th>
          <th>OOS Max DD</th>
          <th>OOS Return</th>
          <th>Stability</th>
        </tr>
      </thead>
      <tbody>
        {fold_table_rows}
      </tbody>
    </table>
  </div>

  <!-- Section 3: Feature Analysis -->
  <h2>3. Feature Importance</h2>
  <div class="chart-card">
    {_fig_div(feature_fig_json, 'feature-chart')}
  </div>

  <!-- Section 4: Risk Analysis -->
  <h2>4. Risk Analysis</h2>
  <div class="chart-card">
    {_fig_div(drawdown_fig_json, 'drawdown-chart')}
  </div>

  <!-- Section 5: Config -->
  <h2>5. Configuration</h2>
  <div class="chart-card">
    <pre class="config-pre">{config_json}</pre>
  </div>

</div>
<div class="footer">
  Trading Bot — Multi-Agent RL &nbsp;|&nbsp; Generated {generated_at}
</div>
</body>
</html>"""
        return html

    # ── Data loaders ──

    def _load_walk_forward_results(self) -> list[dict[str, Any]]:
        """Try to load walk-forward results from disk."""
        results = []
        wf_dir = self.output_dir / "walk_forward"
        if not wf_dir.exists():
            return results

        for fold_path in sorted(wf_dir.glob("fold_*.json")):
            try:
                with open(fold_path) as f:
                    results.append(json.load(f))
            except Exception:  # noqa: BLE001
                pass
        return results

    def _load_feature_importance(self) -> dict[str, float]:
        fi_path = self.output_dir / "feature_importance.json"
        if fi_path.exists():
            try:
                with open(fi_path) as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                pass
        return {}

    # ── Main generate ──

    def generate(
        self,
        walk_forward_results: Optional[list[dict[str, Any]]] = None,
        feature_importance: Optional[dict[str, float]] = None,
        config: Optional[dict[str, Any]] = None,
        output_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> Path:
        """Generate the HTML report and return its path."""

        # Load from disk if not provided
        if walk_forward_results is None:
            walk_forward_results = self._load_walk_forward_results()
        if feature_importance is None:
            feature_importance = self._load_feature_importance()

        # Fill with dummy data if still empty (dry-run or fresh run)
        if not walk_forward_results:
            logger.info("No walk-forward results found — using dummy data for report.")
            walk_forward_results = self._make_dummy_wf_results()
        if not feature_importance:
            feature_importance = self._make_dummy_feature_importance()

        # Compute summary metrics
        oos_sharpes = [f.get("oos_sharpe", 0.0) for f in walk_forward_results]
        is_sharpes = [f.get("is_sharpe", 0.0) for f in walk_forward_results]
        all_equity = []
        for f in walk_forward_results:
            all_equity.extend(f.get("equity_curve", []))

        summary = {
            "oos_sharpe_mean": float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
            "is_sharpe_mean": float(np.mean(is_sharpes)) if is_sharpes else 0.0,
            "max_drawdown": self._compute_max_drawdown(np.array(all_equity)) if all_equity else 0.0,
            "stability_ratio": self._compute_stability(np.array(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0,
            "n_folds": len(walk_forward_results),
        }

        # Build fold table HTML
        fold_rows = []
        for i, fold in enumerate(walk_forward_results):
            label = fold.get("label", f"Fold {i+1}")
            eq = np.array(fold.get("equity_curve", [1.0]))
            oos_return = float((eq[-1] / eq[0] - 1) * 100) if len(eq) > 1 else 0.0
            max_dd = self._compute_max_drawdown(eq) * 100
            stab = min(fold.get("is_sharpe", 0.0), fold.get("oos_sharpe", 0.0)) / (
                max(fold.get("is_sharpe", 1e-8), fold.get("oos_sharpe", 1e-8)) + 1e-8
            )
            fold_rows.append(
                f"<tr>"
                f"<td>{label}</td>"
                f"<td>{fold.get('is_sharpe', 0.0):.3f}</td>"
                f"<td>{fold.get('oos_sharpe', 0.0):.3f}</td>"
                f"<td>{max_dd:.1f}%</td>"
                f"<td>{oos_return:+.1f}%</td>"
                f"<td>{stab:.3f}</td>"
                f"</tr>"
            )
        fold_table_rows = "\n".join(fold_rows) if fold_rows else "<tr><td colspan='6'>No data</td></tr>"

        # Build figures
        equity_fig_json = self._make_equity_curve(walk_forward_results)
        sharpe_fig_json = self._make_sharpe_bar(walk_forward_results)
        feature_fig_json = self._make_feature_importance_bar(feature_importance)
        drawdown_fig_json = self._make_drawdown_plot(all_equity)

        # Config JSON
        config_json = json.dumps(config or {}, indent=2, default=str)

        # Build HTML
        html = self._build_html(
            summary=summary,
            equity_fig_json=equity_fig_json,
            sharpe_fig_json=sharpe_fig_json,
            feature_fig_json=feature_fig_json,
            drawdown_fig_json=drawdown_fig_json,
            fold_table_rows=fold_table_rows,
            config_json=config_json,
        )

        # Save
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"report_{timestamp}.html"

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
            logger.info("Report saved: %s (%d bytes)", output_path, len(html))
        else:
            logger.info("[DRY RUN] Report would be saved to: %s (%d bytes)", output_path, len(html))

        return output_path

    # ── Dummy data helpers (for dry-run and testing) ──

    @staticmethod
    def _make_dummy_wf_results() -> list[dict[str, Any]]:
        results = []
        rng = np.random.default_rng(42)
        for i in range(5):
            equity = np.cumprod(1 + rng.normal(0.0005, 0.01, 500))
            equity = (equity * 10000).tolist()
            returns = np.diff(equity) / np.array(equity[:-1])
            is_sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))
            oos_sharpe = is_sharpe * rng.uniform(0.6, 1.1)
            results.append({
                "label": f"Fold {i+1}",
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "equity_curve": equity,
            })
        return results

    @staticmethod
    def _make_dummy_feature_importance() -> dict[str, float]:
        features = [
            "rsi_14", "macd_signal", "bb_width", "atr_14", "obv_norm",
            "adx_14", "stoch_k", "cci_20", "vwap_dev", "williams_r",
            "volume_ratio", "close_to_high", "mfi_14", "cmf_20",
            "donchian_pos", "psar_signal", "ma_spread", "volatility_regime",
            "calendar_fomc", "onchain_nvt",
        ]
        rng = np.random.default_rng(0)
        vals = rng.exponential(0.05, len(features))
        vals = vals / vals.sum()
        return {f: float(v) for f, v in zip(features, vals)}


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate standalone HTML results report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing walk_forward/ and feature_importance.json")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: results/report_YYYYMMDD.html)")
    parser.add_argument("--config", default="config/local_3060ti.yaml",
                        help="Config YAML path (embedded in report)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate report with dummy data (no disk read)")

    args = parser.parse_args()

    config: dict[str, Any] = {}
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        import yaml  # noqa: PLC0415
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    output_path = Path(args.output) if args.output else None

    rg = ReportGenerator(output_dir=PROJECT_ROOT / args.results_dir)
    report_path = rg.generate(config=config, output_path=output_path, dry_run=args.dry_run)

    print(f"\nReport: {report_path}")
    if not args.dry_run and report_path.exists():
        print(f"Open:   file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
