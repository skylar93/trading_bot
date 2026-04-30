"""Diagnostic experiment: which design lever is the dominant failure cause?

Compares 5 settings on the same 50k × 1 fold setup:
  baseline   : default config (matches A0 NO-GO regime)
  no_fee     : trading_fee=0.0, slippage off
  no_sharpe  : sharpe_weight=0, risk_adjusted_reward=False
  long_only  : action_space [0, 1]
  all_off    : all three above combined

Hypothesis: if all_off shows positive return, design is the problem.
"""
import sys
import copy
import logging

sys.path.insert(0, ".")
import pandas as pd

from config.loader import load_raw
from agents.strategies.agent_factory import create_agent
from training.env_factory import create_env
from training.validation.walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_diagnostic")
logger.setLevel(logging.INFO)

BASE_CFG = load_raw("config/base.yaml")
BASE_CFG["walk_forward"]["enabled"] = True
BASE_CFG["walk_forward"]["n_splits"] = 2  # 1 OOS fold
BASE_CFG.setdefault("training", {})["total_timesteps"] = 50_000

VARIANTS = {
    "baseline":  {},
    "no_fee":    {"env": {"trading_fee": 0.0, "apply_slippage": False}},
    "no_sharpe": {"env": {"sharpe_weight": 0.0, "risk_adjusted_reward": False}},
    "long_only": {"env": {"long_only": True}},
    "all_off":   {"env": {"trading_fee": 0.0, "apply_slippage": False,
                          "sharpe_weight": 0.0, "risk_adjusted_reward": False,
                          "long_only": True}},
}

df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
EVAL_EPISODES = 20

results = {}
for name, override in VARIANTS.items():
    logger.info("=" * 60)
    logger.info("VARIANT: %s — overrides: %s", name, override)
    logger.info("=" * 60)

    cfg = copy.deepcopy(BASE_CFG)
    cfg.setdefault("env", {}).update(override.get("env", {}))

    wf_cfg = cfg.get("walk_forward", {})
    validator = WalkForwardValidator(
        n_splits=wf_cfg.get("n_splits", 2),
        train_ratio=wf_cfg.get("train_ratio", 0.5),
        gap_days=wf_cfg.get("gap_days", 5),
        mode=wf_cfg.get("mode", "expanding"),
    )

    agent_type = cfg.get("agent", {}).get("type", "ppo")
    agent_cfg = cfg.get("agent", {})

    def agent_factory(cfg=cfg, agent_type=agent_type, agent_cfg=agent_cfg):
        env = create_env(cfg, df.iloc[:100], validate=False)
        return create_agent(
            agent_type=agent_type, config=agent_cfg,
            observation_space=env.observation_space, action_space=env.action_space,
        )

    def env_factory(fold_df, cfg=cfg):
        return create_env(cfg, fold_df)

    result = validator.validate(
        agent_factory=agent_factory, env_factory=env_factory, data=df,
        total_timesteps=cfg["training"]["total_timesteps"],
        eval_episodes=EVAL_EPISODES, random_start_eval=True,
    )
    results[name] = result

# Report
print("\n" + "=" * 80)
print("DIAGNOSTIC RESULTS — 50k × 1 fold × 20 eval episodes")
print("=" * 80)
print(f"{'Variant':<12} {'OOS ret (fixed)':>18} {'OOS ret (random)':>18} {'OOS DD (fixed)':>18}")
for name, result in results.items():
    f = result.folds[-1]  # last fold = real OOS
    print(f"{name:<12} {f.oos_total_return:>17.4f}  {f.oos_total_return_random:>17.4f}  {f.oos_max_drawdown:>17.4f}")

# Buy-and-hold baseline on same OOS slice
close_col = "close" if "close" in df.columns else "$close"
splits = WalkForwardValidator(n_splits=2, train_ratio=0.5, gap_days=5).split(df)
oos_df = splits[-1][1]
bh_return = (oos_df[close_col].iloc[-1] / oos_df[close_col].iloc[0]) - 1.0
print(f"\nBaseline BTC buy-hold over same OOS slice: {bh_return:>17.4f}  ({bh_return * 100:.1f}%)")
