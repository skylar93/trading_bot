#!/usr/bin/env python
"""
Smoke test: train PPO briefly, run paper trading steps, verify no crash.

Usage:
    python scripts/smoke_test.py [--steps 100] [--timesteps 512]
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Trading bot smoke test")
    parser.add_argument("--steps", type=int, default=100, help="Paper trading steps")
    parser.add_argument("--timesteps", type=int, default=512, help="Training timesteps")
    args = parser.parse_args()

    from envs.single_asset_rl_env import SingleAssetRLTradingEnv

    # Generate synthetic data
    rng = np.random.default_rng(42)
    n = 200
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "$open": price, "$high": price * 1.01,
        "$low": price * 0.99, "$close": price,
        "$volume": rng.uniform(1e5, 1e6, n),
    })

    env = SingleAssetRLTradingEnv(data=df, window_size=20)

    # Try SB3 PPO first, fall back to stub
    try:
        from stable_baselines3 import PPO
        agent = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
        agent.learn(total_timesteps=args.timesteps, progress_bar=False)
        predict_fn = lambda obs: agent.predict(obs, deterministic=True)
    except ImportError:
        from agents.strategies.agent_factory import create_agent
        agent = create_agent("PPO", config={}, observation_space=env.observation_space,
                             action_space=env.action_space)
        predict_fn = lambda obs: agent.predict(obs, deterministic=True)

    # Paper trading loop
    obs, _ = env.reset()
    for step in range(args.steps):
        action, _ = predict_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Smoke test passed: {args.steps} paper-trading steps OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
