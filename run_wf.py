"""Walk-forward driver for end-to-end pipeline runs.

Phase 8-Alpha (2026-05-04): added argparse so --config / --n_splits /
--total_timesteps actually take effect. Before this fix the script ignored
sys.argv and always loaded config/base.yaml with hardcoded n_splits=12 and
total_timesteps=1_000_000, which silently nullified the futures_maker
override during the Phase 8-Alpha 1M re-train and the funding ablation.
"""
import argparse
import sys

sys.path.insert(0, ".")

import pandas as pd
import yaml

from config.loader import load_raw, _deep_merge
from training.train_pipeline import train_pipeline


def _load_config(override_path: str | None) -> dict:
    cfg = load_raw("config/base.yaml")
    if override_path:
        with open(override_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
        print(f"Config: base.yaml + override from {override_path}")
    else:
        print("Config: base.yaml (no override)")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file deep-merged on top of base.yaml.",
    )
    parser.add_argument("--n_splits", type=int, default=12)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--data",
        default="data/BTCUSDT_1h.csv",
        help="CSV path with OHLCV bars (index=timestamp).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg["walk_forward"]["enabled"] = True
    cfg["walk_forward"]["n_splits"] = args.n_splits
    cfg.setdefault("training", {})["total_timesteps"] = args.total_timesteps

    env_cfg = cfg.get("env", {})
    print(
        f"Effective env: cost_model={env_cfg.get('cost_model', 'spot_taker')}, "
        f"trading_fee={env_cfg.get('trading_fee', 0.001)}, "
        f"apply_slippage={env_cfg.get('apply_slippage', True)}, "
        f"slippage_factor={env_cfg.get('slippage_factor', 0.0005)}, "
        f"funding_rate_per_8h={env_cfg.get('funding_rate_per_8h', 0.0001)}"
    )
    print(
        f"Effective walk-forward: n_splits={args.n_splits}, "
        f"total_timesteps={args.total_timesteps}"
    )

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    print(f"Data: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

    # Phase 8-Gamma G1: fit HMM detector on full data if regime gate is enabled.
    # Diagnostic-stage simplification: full-data fit (leakage caveat — see plan §2 arch).
    regime_cfg = cfg.get("env", {}).get("regime_gate", {}) or {}
    if regime_cfg.get("enabled", False):
        from training.signals.regime_detector import RegimeDetector
        detector_kwargs = regime_cfg.get("detector", {}) or {}
        detector = RegimeDetector(**detector_kwargs)
        detector.fit(df)
        print(
            f"RegimeDetector fitted on full data ({len(df)} rows). "
            f"Per-regime mean returns: "
            f"{detector._model.means_[detector._regime_order, 0].round(6).tolist()}"
        )
        cfg["env"]["_fitted_detector"] = detector

    result = train_pipeline(config=cfg, data=df)
    print("=== RESULT ===")
    print(result)


if __name__ == "__main__":
    main()
