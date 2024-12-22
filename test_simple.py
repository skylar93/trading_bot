import pandas as pd
import numpy as np
from envs.trading_env import TradingEnvironment
from agents.strategies.ppo_agent import PPOAgent
import ccxt


def quick_test():
    print("Starting quick test...")

    # 1. Fetch some test data
    print("\nFetching test data...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=500)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(f"Data shape: {df.shape}")

    # 2. Create environment
    print("\nCreating environment...")
    env = TradingEnvironment(df, initial_balance=10000, window_size=20)

    # 3. Create agent
    print("\nCreating agent...")
    agent = PPOAgent(env.observation_space, env.action_space)

    # 4. Run a few test episodes
    print("\nRunning test episodes...")
    n_episodes = 5
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.get_action(state, deterministic=False)
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1

        print(f"\nEpisode {episode + 1}:")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Portfolio Value: {info['portfolio_value']:.2f}")


if __name__ == "__main__":
    quick_test()
