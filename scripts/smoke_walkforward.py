"""Quick smoke test: walk-forward training with CVaRPPO end-to-end.

Verifies the SB3 / custom-wrapper agent dispatch in WalkForwardValidator
without spending hours of GPU time. Uses 2 folds, a tiny timestep budget,
and a small slice of BTC data.
"""
import logging
import sys
import warnings

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from agents.strategies.agent_factory import create_agent
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from training.validation.walk_forward import WalkForwardValidator

df = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True).iloc[:1500].copy()
print(f"Data: {len(df)} rows")


def env_factory(d):
    return SingleAssetRLTradingEnv(
        data=d, initial_capital=100000.0, window_size=20, max_position_size=1.0,
    )


# Build one env once to derive obs/action spaces for the agent
proto_env = env_factory(df.iloc[:200])


def agent_factory():
    return create_agent(
        agent_type="sb3_cvar_ppo",
        config={"learning_rate": 3e-4, "n_steps": 128, "batch_size": 32},
        observation_space=proto_env.observation_space,
        action_space=proto_env.action_space,
    )


validator = WalkForwardValidator(n_splits=2, train_ratio=0.5, gap_days=2, min_test_size=100)
print("Running walk-forward (2 folds, 512 timesteps each)...")
result = validator.validate(
    agent_factory=agent_factory,
    env_factory=env_factory,
    data=df,
    total_timesteps=512,
    eval_episodes=5,
)

print()
print("=" * 60)
print(f"folds completed:    {len(result.folds)}")
print(f"OOS Sharpe (mean):  {result.oos_sharpe:.4f}")
print(f"OOS Sharpe (std):   {result.oos_sharpe_std:.4f}")
print(f"Stability ratio:    {result.stability_ratio:.4f}")
for i, f in enumerate(result.folds):
    print(f"  fold {i}: IS Sharpe={f.is_sharpe:.3f}, OOS Sharpe={f.oos_sharpe:.3f}, OOS DD={f.oos_max_drawdown:.3f}")
print("=" * 60)
