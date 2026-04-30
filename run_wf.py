"""Quick walk-forward driver for end-to-end pipeline verification.

Reduced scale (3 folds × 50k timesteps × 5000 rows) to finish in ~15-30 min
on Windows CPU/GPU. Goal is to confirm the pipeline runs end-to-end, not to
produce a real evidence pack — bump scale back up for real A0 training.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from config.loader import load_raw
from training.train_pipeline import train_pipeline

cfg = load_raw("config/base.yaml")
cfg["walk_forward"]["enabled"] = True
cfg["walk_forward"]["n_splits"] = 12
cfg.setdefault("training", {})["total_timesteps"] = 1000000

df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
df = df
print(f"Data: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

result = train_pipeline(config=cfg, data=df)
print("=== RESULT ===")
print(result)
