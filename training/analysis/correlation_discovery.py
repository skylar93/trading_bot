"""
Correlation Discovery Engine — automatically finds meaningful relationships
between features and target returns.

Three analysis modes:
    1. Lagged Cross-Correlation  — finds leading indicators (e.g. VIX leads BTC by 2 days)
    2. Granger Causality Test    — statistical "does X predict Y?" (p < 0.05)
    3. Mutual Information        — captures nonlinear dependencies

Outputs:
    - significant_pairs.json     — Granger-causal pairs with lag and direction
    - correlation_report.html    — interactive Plotly heatmap + lag analysis
    - MLflow artifact (optional)

Can also be run as a script::

    python -m training.analysis.correlation_discovery \\
        --data data/BTCUSDT_1h.csv \\
        --cross_assets SPY,DXY,VIX,ETH \\
        --max_lag 20
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.feature_selection import mutual_info_regression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for the correlation discovery engine."""
    max_lag: int = 20                        # maximum lag to test
    granger_max_lag: int = 10                # max lag for Granger test (fewer for speed)
    significance_level: float = 0.05        # p-value threshold for Granger
    top_k_pairs: int = 20                   # report only top-K significant pairs
    min_correlation: float = 0.1            # minimum abs correlation to report
    output_dir: str = "reports/correlation"  # where to save outputs
    log_to_mlflow: bool = False             # log artifacts to MLflow
    # Feature columns to treat as "target" (usually log returns of price)
    target_col: str = "log_return"
    # Rolling window for correlation stability analysis
    stability_window: int = 60
    # Whether to include mutual information analysis (slower)
    use_mutual_info: bool = True


@dataclass
class SignificantPair:
    """Represents a statistically significant feature-target relationship."""
    feature: str
    target: str
    lag: int                 # feature leads target by this many steps
    correlation: float       # Pearson correlation at best lag
    p_value: float           # Granger p-value (if computed)
    mi_score: float          # mutual information score
    direction: str           # "positive" | "negative" | "nonlinear"
    stability_score: float   # rolling correlation stability [0, 1]

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature,
            "target": self.target,
            "lag": self.lag,
            "correlation": round(self.correlation, 4),
            "p_value": round(self.p_value, 6),
            "mi_score": round(self.mi_score, 4),
            "direction": self.direction,
            "stability_score": round(self.stability_score, 4),
        }


class CorrelationDiscoveryEngine:
    """
    Analyzes feature-return relationships in historical price + feature data.

    Usage::

        engine = CorrelationDiscoveryEngine(config)
        results = engine.analyze(df, feature_cols=['rsi', 'macd', 'vix', ...])
        engine.save_report(results)
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.cfg = config or CorrelationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ) -> List[SignificantPair]:
        """
        Run full correlation analysis.

        Parameters
        ----------
        df : DataFrame with datetime index; must contain price/feature columns.
        feature_cols : columns to test as predictors (all numeric cols if None).
        target_col : target column name; if absent, compute log_return from '$close'.

        Returns
        -------
        List of SignificantPair, sorted by |correlation| descending.
        """
        df = df.copy()
        target_col = target_col or self.cfg.target_col

        # Build target series
        if target_col not in df.columns:
            if "$close" in df.columns:
                df[target_col] = np.log(df["$close"] / df["$close"].shift(1))
            else:
                raise ValueError(
                    f"Target column '{target_col}' not found and no '$close' column."
                )

        # Select feature columns
        if feature_cols is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric if c != target_col]

        df = df[[target_col] + feature_cols].dropna(how="all")

        logger.info(
            "CorrelationDiscovery: %d features × %d rows, max_lag=%d",
            len(feature_cols), len(df), self.cfg.max_lag,
        )

        # 1. Lagged cross-correlation analysis
        lag_results = self._lagged_correlations(df, feature_cols, target_col)

        # 2. Granger causality
        granger_results: Dict[str, Tuple[int, float]] = {}
        if _STATSMODELS_AVAILABLE:
            granger_results = self._granger_causality(df, feature_cols, target_col)
        else:
            logger.warning("statsmodels not installed — skipping Granger causality tests.")

        # 3. Mutual information
        mi_scores: Dict[str, float] = {}
        if self.cfg.use_mutual_info and _SKLEARN_AVAILABLE:
            mi_scores = self._mutual_information(df, feature_cols, target_col)
        elif self.cfg.use_mutual_info:
            logger.warning("sklearn not installed — skipping mutual information.")

        # 4. Correlation stability
        stability: Dict[str, float] = self._correlation_stability(
            df, feature_cols, target_col
        )

        # 5. Assemble results
        pairs = self._assemble_pairs(
            lag_results, granger_results, mi_scores, stability, target_col
        )

        return pairs

    def save_report(
        self,
        pairs: List[SignificantPair],
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Save analysis outputs:
        - significant_pairs.json
        - correlation_report.html (if plotly available)
        - MLflow artifacts (if configured)

        Returns dict of {artifact_name: file_path}.
        """
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        outputs: Dict[str, str] = {}

        # JSON report
        json_path = os.path.join(self.cfg.output_dir, "significant_pairs.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "max_lag": self.cfg.max_lag,
                        "significance_level": self.cfg.significance_level,
                        "top_k": self.cfg.top_k_pairs,
                    },
                    "n_significant": len(pairs),
                    "pairs": [p.to_dict() for p in pairs],
                },
                f,
                indent=2,
            )
        outputs["significant_pairs"] = json_path
        logger.info("Saved significant pairs → %s", json_path)

        # HTML report
        if _PLOTLY_AVAILABLE and df is not None and feature_cols is not None:
            html_path = os.path.join(self.cfg.output_dir, "correlation_report.html")
            self._build_html_report(df, pairs, feature_cols, html_path)
            outputs["correlation_report"] = html_path

        # MLflow logging
        if self.cfg.log_to_mlflow and _MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(nested=True):
                    for name, path in outputs.items():
                        mlflow.log_artifact(path, artifact_path="correlation")
                    mlflow.log_metric("n_significant_pairs", len(pairs))
            except Exception as exc:
                logger.warning("MLflow logging failed: %s", exc)

        return outputs

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _lagged_correlations(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Dict[str, Tuple[int, float]]:
        """
        For each feature, find the lag [0, max_lag] with highest |correlation|.

        Returns dict: feature → (best_lag, best_correlation).
        """
        target = df[target_col].values
        results: Dict[str, Tuple[int, float]] = {}

        for col in feature_cols:
            series = df[col].values
            best_lag, best_corr = 0, 0.0
            for lag in range(self.cfg.max_lag + 1):
                if lag == 0:
                    x, y = series, target
                else:
                    # feature leads target by `lag` steps
                    x = series[:-lag]
                    y = target[lag:]
                # Remove NaN pairs
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 30:
                    continue
                try:
                    r, _ = stats.pearsonr(x[mask], y[mask])
                    if abs(r) > abs(best_corr):
                        best_corr = r
                        best_lag = lag
                except Exception:
                    pass
            results[col] = (best_lag, best_corr)

        return results

    def _granger_causality(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Dict[str, Tuple[int, float]]:
        """
        Granger causality test: does feature X help predict target Y?

        Returns dict: feature → (best_lag, min_p_value).
        Only features with p < significance_level are returned.
        """
        results: Dict[str, Tuple[int, float]] = {}
        target = df[target_col].dropna()

        for col in feature_cols:
            try:
                series = df[col].dropna()
                combined = pd.concat([target, series], axis=1).dropna()
                if len(combined) < self.cfg.granger_max_lag * 4 + 10:
                    continue

                test_result = grangercausalitytests(
                    combined, maxlag=self.cfg.granger_max_lag, verbose=False
                )

                # Find minimum p-value across lags
                best_lag, best_p = 1, 1.0
                for lag, res in test_result.items():
                    # F-test p-value (first test result)
                    p = res[0]["ssr_ftest"][1]
                    if p < best_p:
                        best_p = p
                        best_lag = lag

                if best_p < self.cfg.significance_level:
                    results[col] = (best_lag, best_p)

            except Exception as exc:
                logger.debug("Granger test failed for %s: %s", col, exc)

        return results

    def _mutual_information(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Dict[str, float]:
        """
        Compute mutual information between each feature and the target.
        Captures nonlinear relationships that Pearson correlation misses.
        """
        results: Dict[str, float] = {}
        target = df[target_col].values

        for col in feature_cols:
            try:
                series = df[col].values
                mask = np.isfinite(series) & np.isfinite(target)
                if mask.sum() < 30:
                    continue
                X = series[mask].reshape(-1, 1)
                y = target[mask]
                mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)[0]
                results[col] = float(mi)
            except Exception as exc:
                logger.debug("MI failed for %s: %s", col, exc)

        return results

    def _correlation_stability(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Dict[str, float]:
        """
        Rolling correlation stability: measures how consistent the correlation
        between each feature and target is over time.

        Returns a score in [0, 1]:  1 = very stable, 0 = completely unstable.
        """
        results: Dict[str, float] = {}
        w = self.cfg.stability_window

        for col in feature_cols:
            try:
                combined = df[[col, target_col]].dropna()
                if len(combined) < w * 2:
                    results[col] = 0.5
                    continue
                rolling_corr = combined[col].rolling(w).corr(combined[target_col]).dropna()
                if len(rolling_corr) == 0:
                    results[col] = 0.5
                    continue
                # Stability = 1 - coefficient of variation of rolling correlation
                mean_abs = rolling_corr.abs().mean()
                std_abs = rolling_corr.abs().std()
                if mean_abs < 1e-9:
                    results[col] = 0.0
                else:
                    cv = std_abs / (mean_abs + 1e-9)
                    results[col] = float(np.clip(1.0 - cv, 0.0, 1.0))
            except Exception:
                results[col] = 0.5

        return results

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_pairs(
        self,
        lag_results: Dict[str, Tuple[int, float]],
        granger_results: Dict[str, Tuple[int, float]],
        mi_scores: Dict[str, float],
        stability: Dict[str, float],
        target_col: str,
    ) -> List[SignificantPair]:
        """Combine all analysis results into SignificantPair objects."""
        pairs: List[SignificantPair] = []

        for feature, (best_lag, best_corr) in lag_results.items():
            if abs(best_corr) < self.cfg.min_correlation:
                continue

            granger_lag, p_val = granger_results.get(feature, (best_lag, 1.0))
            mi = mi_scores.get(feature, 0.0)
            stab = stability.get(feature, 0.5)

            if best_corr > 0:
                direction = "positive"
            elif best_corr < 0:
                direction = "negative"
            else:
                direction = "nonlinear"

            # If MI is high but correlation is low → nonlinear
            if mi > 0.05 and abs(best_corr) < 0.15:
                direction = "nonlinear"

            pairs.append(SignificantPair(
                feature=feature,
                target=target_col,
                lag=best_lag,
                correlation=float(best_corr),
                p_value=float(p_val),
                mi_score=float(mi),
                direction=direction,
                stability_score=float(stab),
            ))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
        return pairs[: self.cfg.top_k_pairs]

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _build_html_report(
        self,
        df: pd.DataFrame,
        pairs: List[SignificantPair],
        feature_cols: List[str],
        output_path: str,
    ) -> None:
        """Build an interactive Plotly HTML report with heatmaps and lag plots."""
        if not _PLOTLY_AVAILABLE:
            return

        figs = []

        # 1. Correlation heatmap (features × features)
        numeric_cols = [c for c in feature_cols if c in df.columns]
        if numeric_cols:
            corr_matrix = df[numeric_cols].corr()
            heatmap = go.Figure(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale="RdBu",
                    zmin=-1, zmax=1,
                    colorbar=dict(title="Pearson r"),
                )
            )
            heatmap.update_layout(
                title="Feature Cross-Correlation Heatmap",
                height=max(400, len(numeric_cols) * 20),
                width=max(600, len(numeric_cols) * 20),
            )
            figs.append(heatmap)

        # 2. Top significant pairs bar chart
        if pairs:
            features = [p.feature for p in pairs[:15]]
            corrs = [p.correlation for p in pairs[:15]]
            colors = ["green" if c > 0 else "red" for c in corrs]
            bar = go.Figure(go.Bar(
                x=features,
                y=corrs,
                marker_color=colors,
                text=[f"lag={p.lag}" for p in pairs[:15]],
                textposition="outside",
            ))
            bar.update_layout(
                title="Top Significant Feature-Return Correlations (best lag)",
                xaxis_title="Feature",
                yaxis_title="Pearson r",
                height=450,
            )
            figs.append(bar)

        # 3. Stability scatter: |corr| vs stability score
        if pairs:
            scatter = go.Figure(go.Scatter(
                x=[p.stability_score for p in pairs],
                y=[abs(p.correlation) for p in pairs],
                mode="markers+text",
                text=[p.feature for p in pairs],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=[p.mi_score for p in pairs],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="MI score"),
                ),
            ))
            scatter.update_layout(
                title="Feature Stability vs Predictive Power",
                xaxis_title="Stability Score [0=unstable, 1=stable]",
                yaxis_title="|Pearson r| at best lag",
                height=450,
            )
            figs.append(scatter)

        # Combine into a single HTML file
        html_parts = ["<html><head><meta charset='utf-8'/></head><body>"]
        html_parts.append("<h1>Correlation Discovery Report</h1>")
        html_parts.append(f"<p>Analyzed {len(feature_cols)} features | "
                          f"{len(pairs)} significant pairs</p>")

        for fig in figs:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            html_parts.append("<hr/>")

        # Summary table
        html_parts.append("<h2>Significant Pairs Summary</h2>")
        html_parts.append("<table border='1' cellpadding='4' cellspacing='0'>")
        html_parts.append(
            "<tr><th>Feature</th><th>Best Lag</th><th>r</th>"
            "<th>p-value</th><th>MI</th><th>Direction</th><th>Stability</th></tr>"
        )
        for p in pairs:
            html_parts.append(
                f"<tr><td>{p.feature}</td><td>{p.lag}</td>"
                f"<td>{p.correlation:.3f}</td><td>{p.p_value:.4f}</td>"
                f"<td>{p.mi_score:.3f}</td><td>{p.direction}</td>"
                f"<td>{p.stability_score:.3f}</td></tr>"
            )
        html_parts.append("</table></body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        logger.info("Saved correlation report → %s", output_path)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Correlation Discovery Engine")
    parser.add_argument("--data", required=True, help="Path to price CSV")
    parser.add_argument(
        "--cross_assets",
        default="",
        help="Comma-separated cross-asset tickers to include (e.g. SPY,DXY,VIX)",
    )
    parser.add_argument("--max_lag", type=int, default=20)
    parser.add_argument("--output_dir", default="reports/correlation")
    parser.add_argument("--significance", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    logger.info("Loaded %d rows from %s", len(df), args.data)

    # Add cross-asset data if available
    feature_cols = [c for c in df.columns if c not in {"$open", "$high", "$low", "$close", "$volume"}]

    cfg = CorrelationConfig(
        max_lag=args.max_lag,
        output_dir=args.output_dir,
        significance_level=args.significance,
    )
    engine = CorrelationDiscoveryEngine(cfg)
    pairs = engine.analyze(df, feature_cols=feature_cols or None)

    logger.info("Found %d significant pairs", len(pairs))
    outputs = engine.save_report(pairs, df=df, feature_cols=feature_cols or list(df.columns))
    for name, path in outputs.items():
        logger.info("  %s → %s", name, path)


if __name__ == "__main__":
    _main()
