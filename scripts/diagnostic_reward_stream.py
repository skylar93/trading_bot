"""Empirical reward-stream confirmation for Phase 8-Beta plan §0.

Three policies on a 250-bar BTC slice with futures_maker-style env:
  P1: hold-only (action=0 every step)
  P2: buy-and-hold (action=+1 step 0, then 0)
  P3: buy-then-let-drawdown (action=+1 step 0, hold; show drawdown_penalty kicking in)
"""

import os
import sys
import numpy as np
import pandas as pd

from envs.single_asset_rl_env import SingleAssetRLTradingEnv

# Prefer full ~8760-row data if available; fall back to 200-row test sample.
_CANDIDATE_PATHS = [
    "/Users/skylar/Desktop/trading_bot/.claude/worktrees/laughing-benz-3e8884/data/raw/BTCUSDT_1h.csv",
    "/Users/skylar/Desktop/trading_bot/data/BTCUSDT_1h.csv",
]
DATA_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[-1])


def make_env(slice_start: int, slice_len: int = 250):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.iloc[slice_start : slice_start + slice_len].reset_index(drop=True)
    return SingleAssetRLTradingEnv(
        data=df,
        initial_capital=100000.0,
        trading_fee=0.00018,
        window_size=20,
        max_position_size=1.0,
        sharpe_lookback=60,
        sharpe_weight=0.1,
        risk_adjusted_reward=True,
        drawdown_penalty=True,
        max_drawdown_penalty_threshold=0.1,
        apply_slippage=False,
        slippage_factor=0.0,
        partial_fills=False,
        cost_model="futures_maker",
        funding_rate_per_8h=0.0001,
        data_frequency="hourly",
        min_episode_steps=30,
    )


def run_policy(name: str, slice_start: int, action_fn, n_steps: int = 200):
    env = make_env(slice_start)
    obs, _ = env.reset()
    total = 0.0
    components = {"basic": 0.0, "sharpe": 0.0, "drawdown_penalty": 0.0}
    n_trades = 0
    last_position = 0.0
    dd_hits = 0
    rewards = []
    for t in range(n_steps):
        a = action_fn(t)
        obs, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        rewards.append(r)
        total += r
        rd = info.get("reward_debug", {})
        components["basic"] += rd.get("basic_reward", 0.0)
        components["sharpe"] += rd.get("sharpe_component", 0.0)
        dd = rd.get("drawdown_penalty", 0.0)
        components["drawdown_penalty"] += dd
        if dd < -1e-9:
            dd_hits += 1
        pos = float(env.current_position)
        if abs(pos - last_position) > 1e-6:
            n_trades += 1
            last_position = pos
        if term or trunc:
            break
    final_pv = float(env.portfolio_value)
    print(f"\n=== {name} (slice_start={slice_start}, steps={t+1}) ===")
    print(f"  Final portfolio: ${final_pv:,.2f}  (return {(final_pv/100000-1)*100:+.2f}%)")
    print(f"  Trades:          {n_trades}")
    print(f"  Total reward:    {total:+.6f}")
    print(f"  Components (sum): basic={components['basic']:+.6f}  "
          f"sharpe_proxy={components['sharpe']:+.4f}  "
          f"drawdown_penalty={components['drawdown_penalty']:+.4f}")
    print(f"  Drawdown-penalty steps: {dd_hits}/{t+1}")
    print(f"  Reward stats: mean={np.mean(rewards):+.6f}  "
          f"std={np.std(rewards):.6f}  "
          f"min={np.min(rewards):+.6f}  max={np.max(rewards):+.6f}")
    return total, components, n_trades, dd_hits


# Pick 3 slices: bull, bear, choppy. From BTCUSDT_1h.csv (2024-05 onward, 17000+ rows).
# 2024-05 -> ~ +50% bull rally over ~2k bars
# 2025-Q3 area should be sideways/bear

# Bull slice: rows 0-250 (2024-05 early)
# Drawdown slice: pick a slice with a known dump — try several offsets

print("Phase 8-Beta §0 empirical reward dump")
print("=" * 60)

# P1: hold-only on bull slice
run_policy("P1 hold-only (bull slice)", slice_start=100, action_fn=lambda t: 0.0)

# P2: buy-and-hold on bull slice
run_policy(
    "P2 buy-and-hold (bull slice)",
    slice_start=100,
    action_fn=lambda t: 1.0 if t == 0 else 0.0,
)

# P3: buy-and-hold on a drawdown slice — find one
# Scan for a 200-bar window with big peak-to-trough drop
df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
closes = df_full["$close"].values
window = 200
worst_dd = 0.0
worst_idx = 0
for i in range(0, len(closes) - window, 50):
    seg = closes[i : i + window]
    peak = np.maximum.accumulate(seg)
    dd = ((peak - seg) / peak).max()
    if dd > worst_dd:
        worst_dd = dd
        worst_idx = i
print(f"\n[Worst 200-bar drawdown found: {worst_dd*100:.1f}% at row {worst_idx}]")

run_policy(
    "P3 buy-and-hold (worst drawdown slice)",
    slice_start=worst_idx,
    action_fn=lambda t: 1.0 if t == 0 else 0.0,
)

# P4: alternating buy/sell — emulate active trader to see fee impact
run_policy(
    "P4 active trader (flip every 10 bars, bull slice)",
    slice_start=100,
    action_fn=lambda t: 1.0 if (t // 10) % 2 == 0 else -1.0,
)
