"""Quick smoke test for the env fixes — runs a short PPO training and reports
reward distribution, capital health, and which agent class was loaded.
"""
import argparse
import logging
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from agents.strategies.agent_factory import create_agent
from envs.single_asset_rl_env import SingleAssetRLTradingEnv

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="Optional YAML config file to override env params")
args = parser.parse_args()

env_kwargs: dict = {}
if args.config:
    import yaml
    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ec = raw.get("env", {})
    for key in ("trading_fee", "apply_slippage", "slippage_factor", "cost_model",
                "funding_rate_per_8h", "window_size", "max_position_size",
                "sharpe_weight", "inactivity_penalty", "sharpe_clip_value",
                "reward_function"):
        if key in ec:
            env_kwargs[key] = ec[key]
    print(f"Config loaded from {args.config}: {env_kwargs}")
else:
    raw = {}

df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
df = df.iloc[:2000].copy()
print(f"Data: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

# Phase 8-Gamma G1: fit detector and build regime_track if gate is enabled in config
regime_cfg = (raw.get("env", {}) or {}).get("regime_gate", {}) or {}
if regime_cfg.get("enabled", False):
    from training.signals.regime_detector import RegimeDetector
    from training.env_factory import _compute_regime_track
    det = RegimeDetector(**(regime_cfg.get("detector", {}) or {}))
    det.fit(df)
    print(f"RegimeDetector fitted on {len(df)} rows.")
    env_kwargs["regime_track"] = _compute_regime_track(det, df)
    env_kwargs["regime_gate_enabled"] = True
    env_kwargs["regime_gate_mode"] = regime_cfg.get("mode", "close")
    env_kwargs["regime_gate_bear_threshold"] = regime_cfg.get("bear_threshold", 0.5)

env = SingleAssetRLTradingEnv(
    data=df,
    initial_capital=100000.0,
    window_size=env_kwargs.pop("window_size", 20),
    max_position_size=env_kwargs.pop("max_position_size", 1.0),
    **env_kwargs,
)
print(
    f"Effective env (incl. defaults): "
    f"cost_model={getattr(env, 'cost_model', 'spot_taker')}, "
    f"trading_fee={env.trading_fee}, "
    f"apply_slippage={getattr(env, 'apply_slippage', False)}, "
    f"slippage_factor={getattr(env, 'slippage_factor', 0.0)}, "
    f"funding_rate_per_8h={getattr(env, 'funding_rate_per_8h', 0.0)}, "
    f"sharpe_weight={getattr(env, 'sharpe_weight', 0.1)}, "
    f"inactivity_penalty={getattr(env, 'inactivity_penalty', 0.0)}, "
    f"sharpe_clip_value={getattr(env, 'sharpe_clip_value', 10.0)}, "
    f"regime_gate_enabled={getattr(env, 'regime_gate_enabled', False)}, "
    f"regime_gate_mode={getattr(env, 'regime_gate_mode', 'close')}, "
    f"regime_gate_bear_threshold={getattr(env, 'regime_gate_bear_threshold', 0.5)}, "
    f"reward_function={getattr(env, 'reward_function', 'sharpe_ratio')}"
)

agent = create_agent(
    agent_type="sb3_cvar_ppo",
    config={"learning_rate": 3e-4, "n_steps": 256, "batch_size": 64},
    observation_space=env.observation_space,
    action_space=env.action_space,
)
print(f"Agent class: {type(agent).__name__}")
inner = getattr(agent, "model", None) or getattr(agent, "agent", None)
print(f"Inner model class: {type(inner).__name__ if inner is not None else 'n/a'}")

obs, _ = env.reset()
rewards = []
capitals = []
neg_capital = 0
ep_ends = 0
done = False
for step in range(1500):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    rewards.append(float(r))
    capitals.append(float(env.current_capital))
    if env.current_capital < 0:
        neg_capital += 1
    if term or trunc:
        ep_ends += 1
        obs, _ = env.reset()

rewards = np.array(rewards)
capitals = np.array(capitals)

print()
print("=" * 60)
print(f"steps run:           {len(rewards)}")
print(f"episode resets:      {ep_ends}")
print(f"reward min/mean/max: {rewards.min():.4f} / {rewards.mean():.4f} / {rewards.max():.4f}")
print(f"reward std:          {rewards.std():.4f}")
print(f"reward unique-ish:   {len(np.unique(np.round(rewards, 3)))} distinct (rounded 3dp)")
print(f"reward at -5 floor:  {(rewards <= -4.99).sum()} / {len(rewards)} steps")
print(f"reward at +5 ceil:   {(rewards >= 4.99).sum()} / {len(rewards)} steps")
print(f"capital min/mean:    {capitals.min():.2f} / {capitals.mean():.2f}")
print(f"NEGATIVE capital:    {neg_capital} steps {'(BUG STILL PRESENT!)' if neg_capital else '(OK)'}")
if getattr(env, 'regime_gate_enabled', False):
    print(f"regime_gate_fires:   {getattr(env, '_gate_fires', 0)} (last episode)")
if getattr(env, 'reward_function', '') == 'realized_pnl':
    n_realized = sum(1 for r in rewards if abs(r) > 1e-9)
    print(f"realized_pnl trades: {n_realized} / {len(rewards)} steps")
print("=" * 60)
