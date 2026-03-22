"""
Full Pipeline Runner (Week 28)

자동 실행 순서:
    1. 데이터 갱신 (fetch_data)
    2. Feature engineering (technical + cross-asset + on-chain)
    3. HMM regime detection (fit on training data)
    4. 4-agent ensemble 학습 (sequential on 3060 Ti)
    5. Walk-forward validation (5 folds)
    6. Feature importance analysis
    7. Correlation discovery
    8. 결과 리포트 생성 (HTML)
    9. Paper trading 시작 (optional)

사용법:
    python scripts/run_full_pipeline.py --config config/local_3060ti.yaml
    python scripts/run_full_pipeline.py --config config/local_3060ti.yaml --skip-data
    python scripts/run_full_pipeline.py --config config/local_3060ti.yaml --dry-run

예상 시간 (3060 Ti):
    Step 1:   ~5분  (2년치 데이터)
    Step 2-3: ~10분
    Step 4:   ~1.5시간 (1M timesteps)
    Step 5:   ~8시간  (5 folds)
    Step 6-8: ~30분
    Total:    ~10시간
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────
# Pipeline State — for checkpoint/resume
# ──────────────────────────────────────────────────────────

class PipelineState:
    """Persist pipeline progress so it can be resumed after interruption."""

    STATE_FILE = PROJECT_ROOT / "logs" / "pipeline_state.json"

    def __init__(self) -> None:
        self.steps_done: list[str] = []
        self.start_time: float = time.time()
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE) as f:
                    data = json.load(f)
                self.steps_done = data.get("steps_done", [])
                self.start_time = data.get("start_time", self.start_time)
                self.metadata = data.get("metadata", {})
                logger.info("Resumed pipeline state: %d steps done.", len(self.steps_done))
            except Exception:  # noqa: BLE001
                pass

    def save(self) -> None:
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE_FILE, "w") as f:
            json.dump({
                "steps_done": self.steps_done,
                "start_time": self.start_time,
                "metadata": self.metadata,
            }, f, indent=2)

    def mark_done(self, step: str, **meta) -> None:  # type: ignore[type-arg]
        if step not in self.steps_done:
            self.steps_done.append(step)
        self.metadata[step] = {"done_at": datetime.utcnow().isoformat(), **meta}
        self.save()

    def is_done(self, step: str) -> bool:
        return step in self.steps_done

    def clear(self) -> None:
        self.steps_done = []
        self.metadata = {}
        self.start_time = time.time()
        self.save()


# ──────────────────────────────────────────────────────────
# Step helpers
# ──────────────────────────────────────────────────────────

def _header(step_num: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'=' * 60}")


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ──────────────────────────────────────────────────────────
# Individual steps
# ──────────────────────────────────────────────────────────

def step_fetch_data(cfg: dict[str, Any], state: PipelineState, dry_run: bool) -> Optional[str]:
    if state.is_done("fetch_data"):
        logger.info("Step 1 already done — skipping.")
        return state.metadata.get("fetch_data", {}).get("output_path")

    data_cfg = cfg.get("data", {})
    asset = data_cfg.get("symbols", ["BTC/USDT"])[0].replace("/", "")
    interval = data_cfg.get("timeframe", "1h")
    output_path = str(PROJECT_ROOT / data_cfg.get("data_path", f"data/{asset}_{interval}.csv"))

    from scripts.fetch_data import fetch_data  # noqa: PLC0415
    fetch_data(
        asset=asset,
        period="2y",
        interval=interval,
        output=output_path,
        dry_run=dry_run,
    )

    state.mark_done("fetch_data", output_path=output_path)
    logger.info("Step 1 done: %s", output_path)
    return output_path


def step_load_data(cfg: dict[str, Any], data_path: str) -> Any:
    import pandas as pd  # noqa: PLC0415

    p = Path(data_path)
    # Prefer parquet for speed
    parquet = p.with_suffix(".parquet")
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        df = pd.read_csv(p, index_col=0, parse_dates=True)

    if not df.index.tzinfo:
        df.index = df.index.tz_localize("UTC")

    logger.info("Loaded %d rows from %s", len(df), p.name)
    return df


def step_feature_engineering(
    cfg: dict[str, Any],
    df: Any,
    state: PipelineState,
    dry_run: bool,
) -> Any:
    if state.is_done("feature_engineering"):
        logger.info("Step 2 already done — skipping.")
        return df

    try:
        from training.data.feature_engineering import FeatureEngineer  # noqa: PLC0415
        fe = FeatureEngineer(cfg.get("feature_engineering", {}))
        df = fe.transform(df)
        logger.info("Feature engineering done: %d columns", len(df.columns))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature engineering failed (%s) — using raw OHLCV.", exc)

    if not dry_run:
        state.mark_done("feature_engineering", n_features=len(df.columns))
    return df


def step_regime_detection(
    cfg: dict[str, Any],
    df: Any,
    state: PipelineState,
    dry_run: bool,
) -> None:
    if state.is_done("regime_detection"):
        logger.info("Step 3 already done — skipping.")
        return

    try:
        from training.signals.regime_detector import RegimeDetector  # noqa: PLC0415
        regime_cfg = cfg.get("regime", {})
        rd = RegimeDetector(
            method=regime_cfg.get("method", "hmm"),
            n_regimes=regime_cfg.get("n_regimes", 3),
        )
        import numpy as np  # noqa: PLC0415
        close = df["$close"].values if hasattr(df, "__getitem__") else df
        log_returns = np.diff(np.log(close + 1e-8))
        if not dry_run:
            rd.fit(log_returns)
            ckpt_path = PROJECT_ROOT / "checkpoints" / "regime_detector.pkl"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            import pickle  # noqa: PLC0415
            with open(ckpt_path, "wb") as f:
                pickle.dump(rd, f)
            logger.info("Regime detector saved: %s", ckpt_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Regime detection failed (%s) — skipping.", exc)

    state.mark_done("regime_detection")


def step_train_agents(
    cfg: dict[str, Any],
    df: Any,
    state: PipelineState,
    dry_run: bool,
) -> dict[str, Any]:
    if state.is_done("train_agents"):
        logger.info("Step 4 already done — skipping.")
        return {}

    results: dict[str, Any] = {}

    if dry_run:
        logger.info("[DRY RUN] Skipping agent training.")
        state.mark_done("train_agents")
        return results

    try:
        from training.train_pipeline import train_pipeline  # noqa: PLC0415
        metrics = train_pipeline(config=cfg, data=df)
        results["train_metrics"] = metrics
        logger.info("Training done: %s", metrics)
    except Exception as exc:  # noqa: BLE001
        logger.error("Training failed: %s", exc, exc_info=True)
        results["train_error"] = str(exc)

    state.mark_done("train_agents", **{k: str(v) for k, v in results.items()})
    return results


def step_walk_forward(
    cfg: dict[str, Any],
    df: Any,
    state: PipelineState,
    dry_run: bool,
) -> list[Any]:
    if state.is_done("walk_forward"):
        logger.info("Step 5 already done — skipping.")
        return []

    if dry_run:
        logger.info("[DRY RUN] Skipping walk-forward validation.")
        state.mark_done("walk_forward")
        return []

    results = []
    try:
        from training.validation.walk_forward import WalkForwardValidator  # noqa: PLC0415
        wf_cfg = cfg.get("walk_forward", {})
        validator = WalkForwardValidator(
            n_folds=wf_cfg.get("n_folds", 5),
            train_window=wf_cfg.get("train_window", 8000),
            val_window=wf_cfg.get("val_window", 2000),
        )
        results = validator.run(cfg=cfg, data=df)
        logger.info("Walk-forward done: %d folds", len(results))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Walk-forward failed (%s) — skipping.", exc)

    state.mark_done("walk_forward", n_folds=len(results))
    return results


def step_feature_importance(
    cfg: dict[str, Any],
    df: Any,
    state: PipelineState,
    dry_run: bool,
) -> dict[str, Any]:
    if state.is_done("feature_importance"):
        logger.info("Step 6 already done — skipping.")
        return {}

    if dry_run:
        logger.info("[DRY RUN] Skipping feature importance.")
        state.mark_done("feature_importance")
        return {}

    results: dict[str, Any] = {}
    try:
        from training.analysis.shap_analysis import SHAPAnalyzer  # noqa: PLC0415
        analyzer = SHAPAnalyzer()
        importance = analyzer.compute(df)
        results["feature_importance"] = importance
        out_path = PROJECT_ROOT / "results" / "feature_importance.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({k: float(v) for k, v in importance.items()}, f, indent=2)
        logger.info("Feature importance saved: %s", out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature importance failed (%s) — skipping.", exc)

    state.mark_done("feature_importance")
    return results


def step_correlation_discovery(
    cfg: dict[str, Any],
    df: Any,
    state: PipelineState,
    dry_run: bool,
) -> dict[str, Any]:
    if state.is_done("correlation_discovery"):
        logger.info("Step 7 already done — skipping.")
        return {}

    if dry_run:
        logger.info("[DRY RUN] Skipping correlation discovery.")
        state.mark_done("correlation_discovery")
        return {}

    results: dict[str, Any] = {}
    try:
        from training.analysis.correlation_discovery import CorrelationDiscovery  # noqa: PLC0415
        cd_cfg = cfg.get("correlation_discovery", {})
        cd = CorrelationDiscovery(
            max_lag=cd_cfg.get("max_lag", 20),
            significance_level=cd_cfg.get("significance_level", 0.05),
        )
        report = cd.analyze(df)
        results = report
        out_dir = PROJECT_ROOT / cd_cfg.get("output_dir", "reports/correlation")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "correlation_report.json", "w") as f:
            json.dump({k: str(v) for k, v in report.items()}, f, indent=2)
        logger.info("Correlation discovery saved: %s", out_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Correlation discovery failed (%s) — skipping.", exc)

    state.mark_done("correlation_discovery")
    return results


def step_generate_report(
    cfg: dict[str, Any],
    walk_forward_results: list[Any],
    feature_importance: dict[str, Any],
    state: PipelineState,
    dry_run: bool,
) -> Optional[Path]:
    if state.is_done("generate_report"):
        logger.info("Step 8 already done — skipping.")
        return None

    out_dir = PROJECT_ROOT / cfg.get("report", {}).get("output_dir", "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"report_{timestamp}.html"

    from scripts.generate_report import ReportGenerator  # noqa: PLC0415
    rg = ReportGenerator(output_dir=out_dir)
    rg.generate(
        walk_forward_results=walk_forward_results,
        feature_importance=feature_importance,
        config=cfg,
        dry_run=dry_run,
    )

    state.mark_done("generate_report", path=str(report_path))
    logger.info("Report generated: %s", report_path)
    return report_path


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def run_pipeline(
    config_path: str,
    skip_data: bool = False,
    resume: bool = True,
    dry_run: bool = False,
    paper_trading: bool = False,
) -> None:
    t0 = time.time()

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Config loaded: %s", config_path)

    # Pipeline state
    state = PipelineState()
    if not resume:
        state.clear()
        logger.info("Pipeline state cleared (fresh run).")

    # ── Step 1: Data ──
    _header(1, "Data Update")
    data_path: Optional[str] = None
    if skip_data:
        data_cfg = cfg.get("data", {})
        asset = data_cfg.get("symbols", ["BTC/USDT"])[0].replace("/", "")
        interval = data_cfg.get("timeframe", "1h")
        data_path = str(PROJECT_ROOT / data_cfg.get("data_path", f"data/{asset}_{interval}.csv"))
        logger.info("--skip-data: using existing %s", data_path)
    else:
        data_path = step_fetch_data(cfg, state, dry_run)

    # ── Load DataFrame ──
    if data_path and Path(data_path).exists():
        df = step_load_data(cfg, data_path)
    else:
        logger.warning("Data file not found (%s). Creating synthetic data for dry-run.", data_path)
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        dates = pd.date_range("2023-01-01", periods=5000, freq="h", tz="UTC")
        df = pd.DataFrame({
            "$open": np.random.uniform(30000, 50000, 5000),
            "$high": np.random.uniform(30000, 50000, 5000),
            "$low": np.random.uniform(30000, 50000, 5000),
            "$close": np.random.uniform(30000, 50000, 5000),
            "$volume": np.random.uniform(100, 10000, 5000),
        }, index=dates)

    # ── Step 2: Feature Engineering ──
    _header(2, "Feature Engineering")
    df = step_feature_engineering(cfg, df, state, dry_run)

    # ── Step 3: Regime Detection ──
    _header(3, "HMM Regime Detection")
    step_regime_detection(cfg, df, state, dry_run)

    # ── Step 4: Agent Training ──
    _header(4, "Agent Training (4-agent ensemble)")
    train_results = step_train_agents(cfg, df, state, dry_run)

    # ── Step 5: Walk-Forward Validation ──
    _header(5, "Walk-Forward Validation (5 folds)")
    wf_results = step_walk_forward(cfg, df, state, dry_run)

    # ── Step 6: Feature Importance ──
    _header(6, "Feature Importance Analysis")
    fi_results = step_feature_importance(cfg, df, state, dry_run)

    # ── Step 7: Correlation Discovery ──
    _header(7, "Correlation Discovery")
    step_correlation_discovery(cfg, df, state, dry_run)

    # ── Step 8: Report Generation ──
    _header(8, "Report Generation")
    report_path = step_generate_report(cfg, wf_results, fi_results, state, dry_run)

    # ── Step 9: Paper Trading ──
    if paper_trading:
        _header(9, "Paper Trading Start")
        pt_cfg = cfg.get("paper_trading", {})
        if pt_cfg.get("enabled", False):
            logger.info("Paper trading configured — start manually or via web UI.")
        else:
            logger.info("paper_trading.enabled=false in config. Set to true to auto-start.")

    # ── Final Summary ──
    elapsed = _elapsed(t0)
    print(f"\n{'=' * 60}")
    print(f"  Pipeline Complete — Elapsed: {elapsed}")
    print(f"  Steps done: {', '.join(state.steps_done)}")
    if report_path:
        print(f"  Report: {report_path}")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Pipeline Runner — fetch → train → validate → report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/local_3060ti.yaml",
                        help="Config YAML path")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data fetch (use existing CSV)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Clear pipeline state and start fresh")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test pipeline structure without heavy computation")
    parser.add_argument("--paper-trading", action="store_true",
                        help="Start paper trading after training (if configured)")

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        skip_data=args.skip_data,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        paper_trading=args.paper_trading,
    )


if __name__ == "__main__":
    main()
