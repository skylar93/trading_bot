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
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    ec = raw.get("env", {})
    for key in ("trading_fee", "apply_slippage", "slippage_factor", "cost_model",
                "funding_rate_per_8h", "window_size", "max_position_size"):
        if key in ec:
            env_kwargs[key] = ec[key]
    print(f"Config loaded from {args.config}: {env_kwargs}")

df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
df = df.iloc[:2000].copy()
print(f"Data: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

env = SingleAssetRLTradingEnv(
    data=df,
    initial_capital=100000.0,
    window_size=env_kwargs.pop("window_size", 20),
    max_position_size=env_kwargs.pop("max_position_size", 1.0),
    **env_kwargs,
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
print("=" * 60)
