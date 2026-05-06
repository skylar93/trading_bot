"""Walk-forward driver for end-to-end pipeline runs.

Phase 8-Alpha (2026-05-04): added argparse so --config / --n_splits /
--total_timesteps actually take effect. Before this fix the script ignored
sys.argv and always loaded config/base.yaml with hardcoded n_splits=12 and
total_timesteps=1_000_000, which silently nullified the futures_maker
override during the Phase 8-Alpha 1M re-train and the funding ablation.
"""
import argparse
import json
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


def _write_summary_json(path: str, result, cfg: dict, args) -> None:
    """Write a structured JSON summary of a completed walk-forward run."""
    import dataclasses
    import math

    def _safe_float(v) -> float:
        """Convert numpy/python float; map nan/inf to None for JSON safety."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if (math.isnan(f) or math.isinf(f)) else f

    folds_json = []
    for fold in result.folds:
        d = {}
        for f in dataclasses.fields(fold):
            if not f.repr:  # skip oos_returns (repr=False, large array)
                continue
            val = getattr(fold, f.name)
            if isinstance(val, dict):
                d[f.name] = val
            else:
                d[f.name] = _safe_float(val) if isinstance(val, float) else val
        folds_json.append(d)

    summary = {k: _safe_float(v) for k, v in result.summary().items()}

    payload = {
        "command": sys.argv,
        "config": {
            "env": cfg.get("env", {}),
            "walk_forward": cfg.get("walk_forward", {}),
            "training": cfg.get("training", {}),
        },
        "aggregates": summary,
        "folds": folds_json,
    }

    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Summary JSON written to {path}")


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
    parser.add_argument(
        "--summary-json",
        default=None,
        metavar="PATH",
        help="If set, write a JSON summary of the run to this path in addition to stdout.",
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

    result = train_pipeline(config=cfg, data=df)
    print("=== RESULT ===")
    print(result)

    if args.summary_json:
        _write_summary_json(args.summary_json, result, cfg, args)


if __name__ == "__main__":
    main()
