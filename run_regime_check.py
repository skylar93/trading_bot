"""Regime + trade-frequency check.

Hypothesis (from prior diagnostic, 2026-04-29 bear slice):
  no_fee variant near-breakeven on bear, baseline -10.89%.
  → Is no_fee positive across BOTH bull and bear regimes?
  → How many trades per episode? (Overtrading vs honest signal)

Setting: 12-fold expanding walk-forward × 50k timesteps × 20 eval episodes.
Same splits as A0 overnight, but per-fold timestep budget reduced 20× to keep
runtime under 30 min.

Outputs:
  - Per-fold OOS return (fixed + random start)
  - Per-fold trade count (mean per eval episode)
  - Per-fold BTC buy-hold over OOS slice (regime label)
  - Verdict: regime-robust (Case A) / regime-dependent (Case B) / overtrading (Case C)
"""
import sys, logging
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from config.loader import load_raw
from agents.strategies.agent_factory import create_agent
from training.env_factory import create_env
from training.validation.walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("regime_check")
logger.setLevel(logging.INFO)

logging.getLogger("SingleAssetRLTradingEnv").setLevel(logging.WARNING)

cfg = load_raw("config/base.yaml")
cfg["walk_forward"]["enabled"] = True
cfg["walk_forward"]["n_splits"] = 12
cfg["walk_forward"]["train_ratio"] = 0.5
cfg["walk_forward"]["gap_days"] = 5
cfg.setdefault("training", {})["total_timesteps"] = 50_000

# no_fee variant: fee + slippage off
cfg.setdefault("env", {}).update({
    "trading_fee": 0.0,
    "apply_slippage": False,
})

EVAL_EPISODES = 20

df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
logger.info("Data: %d rows, %s → %s", len(df), df.index[0], df.index[-1])

wf_cfg = cfg["walk_forward"]
validator = WalkForwardValidator(
    n_splits=wf_cfg["n_splits"],
    train_ratio=wf_cfg["train_ratio"],
    gap_days=wf_cfg["gap_days"],
    mode=wf_cfg.get("mode", "expanding"),
)

agent_type = cfg.get("agent", {}).get("type", "ppo")
agent_cfg = cfg.get("agent", {})

def agent_factory():
    env = create_env(cfg, df.iloc[:100], validate=False)
    return create_agent(
        agent_type=agent_type, config=agent_cfg,
        observation_space=env.observation_space, action_space=env.action_space,
    )

def env_factory(fold_df):
    return create_env(cfg, fold_df)

result = validator.validate(
    agent_factory=agent_factory, env_factory=env_factory, data=df,
    total_timesteps=cfg["training"]["total_timesteps"],
    eval_episodes=EVAL_EPISODES, random_start_eval=True,
)

# ── BTC buy-hold per OOS slice (regime label) ────────────────────────────────
splits = WalkForwardValidator(
    n_splits=wf_cfg["n_splits"],
    train_ratio=wf_cfg["train_ratio"],
    gap_days=wf_cfg["gap_days"],
).split(df)

close_col = "$close" if "$close" in splits[0][1].columns else "close"
bh_returns = []
for _, test_df in splits:
    bh = (test_df[close_col].iloc[-1] / test_df[close_col].iloc[0]) - 1.0
    bh_returns.append(bh)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("REGIME + TRADE FREQUENCY CHECK — no_fee variant, 12-fold × 50k × 20 eval")
print("=" * 110)
print(f"{'Fold':>4} {'Period start':>12} {'BTC b&h':>10} {'OOS ret (fix)':>15} {'OOS ret (rnd)':>15} {'Trades/ep (fix)':>16} {'Trades/ep (rnd)':>16}")
print("-" * 110)

bull_returns_fix = []
bear_returns_fix = []
bull_returns_rnd = []
bear_returns_rnd = []

for i, fold in enumerate(result.folds):
    bh = bh_returns[i] if i < len(bh_returns) else float("nan")
    test_start = splits[i][1].index[0] if i < len(splits) else "?"
    is_bull = bh > 0
    label = "bull" if is_bull else "bear"
    print(f"{i:>4} {str(test_start)[:12]:>12} {bh*100:>9.1f}% {fold.oos_total_return*100:>14.1f}% {fold.oos_total_return_random*100:>14.1f}% {fold.oos_trade_count_mean:>16.1f} {fold.oos_trade_count_random_mean:>16.1f}  ({label})")

    if is_bull:
        bull_returns_fix.append(fold.oos_total_return)
        bull_returns_rnd.append(fold.oos_total_return_random)
    else:
        bear_returns_fix.append(fold.oos_total_return)
        bear_returns_rnd.append(fold.oos_total_return_random)

print("-" * 110)
mean_fix = np.mean([f.oos_total_return for f in result.folds])
mean_rnd = np.mean([f.oos_total_return_random for f in result.folds])
mean_trades_fix = np.mean([f.oos_trade_count_mean for f in result.folds])
mean_trades_rnd = np.mean([f.oos_trade_count_random_mean for f in result.folds])
print(f"  ALL {'':>12} {'':>10} {mean_fix*100:>14.1f}% {mean_rnd*100:>14.1f}% {mean_trades_fix:>16.1f} {mean_trades_rnd:>16.1f}")
if bull_returns_fix:
    print(f" BULL {'':>12} {'':>10} {np.mean(bull_returns_fix)*100:>14.1f}% {np.mean(bull_returns_rnd)*100:>14.1f}%  ({len(bull_returns_fix)} folds)")
if bear_returns_fix:
    print(f" BEAR {'':>12} {'':>10} {np.mean(bear_returns_fix)*100:>14.1f}% {np.mean(bear_returns_rnd)*100:>14.1f}%  ({len(bear_returns_fix)} folds)")

print("=" * 110)

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\nVERDICT:")
bull_pos = bool(bull_returns_rnd) and np.mean(bull_returns_rnd) > -0.01
bear_pos = bool(bear_returns_rnd) and np.mean(bear_returns_rnd) > -0.01
high_freq = mean_trades_rnd > 200

if bull_pos and bear_pos and high_freq:
    print("  CASE A1: regime-robust + overtrading.")
    print("    → no_fee works in both bull and bear, but agent trades very heavily.")
    print("    → Phase 8-Alpha: turnover penalty reward redesign + 1M re-train.")
elif bull_pos and bear_pos and not high_freq:
    print("  CASE A2: regime-robust + low-frequency.")
    print("    → Real signal exists, fee rate (0.1% + 0.05% slippage) is structurally too high.")
    print("    → Phase 8-Alpha: exchange/cost model audit + maker-only execution + 1M re-train.")
elif bear_pos and not bull_pos:
    print("  CASE B: bear-only signal.")
    print("    → Agent is essentially shorting in down moves; no real bull-market alpha.")
    print("    → Phase 8-Alpha: feature engineering (regime detection / funding / on-chain) BEFORE re-train.")
elif not bear_pos and bull_pos:
    print("  CASE B': bull-only signal (rare).")
    print("    → Agent rides trend but fails in chop / bear; needs regime-conditional strategy.")
else:
    print("  CASE D: no_fee not regime-robust either.")
    print("    → Earlier diagnostic was bear-slice luck. Strategy genuinely lacks signal.")
    print("    → Phase 8-Alpha: full feature + reward redesign required, no quick fix.")

print(f"\nKey numbers for Phase 8-Alpha plan:")
print(f"  Mean trades/episode (random): {mean_trades_rnd:.1f}")
if bull_returns_rnd:
    print(f"  Bull mean OOS return (random): {np.mean(bull_returns_rnd)*100:.2f}%")
else:
    print(f"  (no bull folds)")
if bear_returns_rnd:
    print(f"  Bear mean OOS return (random): {np.mean(bear_returns_rnd)*100:.2f}%")
else:
    print(f"  (no bear folds)")
print(f"  Folds positive (random): {sum(1 for f in result.folds if f.oos_total_return_random > 0)}/{len(result.folds)}")
print()
