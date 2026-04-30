"""Random-start sanity experiment.

Trains 1 OOS fold (50k steps) then evaluates with:
  - fixed start  (original behaviour)
  - random start (episode start randomised across OOS window)

Judgement (pre-committed):
  oos_total_return_random > -0.05  → metric bias suspected; full re-run with fix warranted
  -0.05 to -0.15                   → ambiguous; increase eval_episodes to 50 next
  < -0.15                          → NO-GO confirmed; strategy is genuinely failing
"""
import sys
sys.path.insert(0, ".")

import logging
import pandas as pd

from config.loader import load_raw
from agents.strategies.agent_factory import create_agent
from training.env_factory import create_env
from training.validation.walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_sanity")

# ── config ───────────────────────────────────────────────────────────────────
cfg = load_raw("config/base.yaml")
cfg["walk_forward"]["enabled"] = True
cfg["walk_forward"]["n_splits"] = 2      # 1 real OOS fold
cfg.setdefault("training", {})["total_timesteps"] = 50_000

EVAL_EPISODES = 20   # enough to get a stable mean from random starts

# ── data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
logger.info("Data: %d rows, %s → %s", len(df), df.index[0], df.index[-1])

# ── validator setup ───────────────────────────────────────────────────────────
wf_cfg = cfg.get("walk_forward", {})
validator = WalkForwardValidator(
    n_splits=wf_cfg.get("n_splits", 2),
    train_ratio=wf_cfg.get("train_ratio", 0.5),
    gap_days=wf_cfg.get("gap_days", 5),
    mode=wf_cfg.get("mode", "expanding"),
)

_agent_type = cfg.get("agent", {}).get("type", "ppo")
agent_cfg = cfg.get("agent", {})

def agent_factory():
    env = create_env(cfg, df.iloc[:100], validate=False)
    return create_agent(
        agent_type=_agent_type,
        config=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

def env_factory(fold_df: pd.DataFrame):
    return create_env(cfg, fold_df)

total_timesteps = cfg.get("training", {}).get("total_timesteps", 50_000)

# ── run ───────────────────────────────────────────────────────────────────────
result = validator.validate(
    agent_factory=agent_factory,
    env_factory=env_factory,
    data=df,
    total_timesteps=total_timesteps,
    eval_episodes=EVAL_EPISODES,
    random_start_eval=True,
)

# ── report ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY EXPERIMENT RESULTS")
print("=" * 60)
for fold in result.folds:
    print(f"\nFold {fold.fold_idx}:")
    print(f"  IS  Sharpe             : {fold.is_sharpe:.4f}")
    print(f"  OOS Sharpe  (fixed)    : {fold.oos_sharpe:.4f}")
    print(f"  OOS return  (fixed)    : {fold.oos_total_return:.4f}  [{fold.oos_total_return*100:.1f}%]")
    print(f"  OOS return  (random)   : {fold.oos_total_return_random:.4f}  [{fold.oos_total_return_random*100:.1f}%]")
    print(f"  OOS Sharpe  (random)   : {fold.oos_sharpe_random:.4f}")
    print(f"  OOS Max DD  (fixed)    : {fold.oos_max_drawdown:.4f}  [{fold.oos_max_drawdown*100:.1f}%]")

print("\n" + "-" * 60)
print(f"Mean OOS return (fixed)  : {result.oos_total_return_mean:.4f}  [{result.oos_total_return_mean*100:.1f}%]")
print(f"Mean OOS return (random) : {result.oos_total_return_random_mean:.4f}  [{result.oos_total_return_random_mean*100:.1f}%]")
print("-" * 60)

# ── judgement ─────────────────────────────────────────────────────────────────
r = result.oos_total_return_random_mean
print("\nJUDGEMENT:")
if r > -0.05:
    print("  >> metric bias suspected — random-start OOS return is near-zero or positive.")
    print("     Consider full re-run with random_start_eval=True + more timesteps.")
elif r > -0.15:
    print("  >> AMBIGUOUS — increase eval_episodes to 50 or run a second fold.")
else:
    print("  >> NO-GO CONFIRMED — strategy is genuinely failing regardless of start-point bias.")
    print("     Proceed with A0 NO-GO branch (plan phase8-restructured.md, section A0 NO-GO).")
print("=" * 60 + "\n")
