#!/usr/bin/env python
"""
debug_obs_shapes.py

This script tests observation shapes from SingleAssetRLTradingEnv to debug 
dimension mismatches.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import argparse
from typing import Dict, Any, Optional

# Project path setup
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from data.utils.data_loader import DataLoader
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from agents.strategies.single.ppo_agent import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(days=14):
    """Load real trading data"""
    import datetime
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    
    data_loader = DataLoader(
        exchange_id="binance",
        symbol=symbol,
        timeframe=timeframe
    )
    
    df = data_loader.fetch_data(
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"Loaded data with shape: {df.shape}")
    return df

def test_observation_shapes(days=14, window_size=20):
    """Test observation shapes from environment"""
    # Load data
    df = load_data(days)
    
    # Create environment
    env = SingleAssetRLTradingEnv(
        data=df,
        window_size=window_size,
        initial_capital=10000.0,
        trading_fee=0.001
    )
    
    # Test reset
    logger.info("Testing env.reset()")
    obs, info = env.reset()
    logger.info(f"Observation shape after reset: {obs.shape}")
    
    # Test several steps
    for i in range(10):
        # Random action between -1 and 1
        action = np.array([np.random.uniform(-1, 1)])
        
        logger.info(f"Step {i+1}: Taking action {action}")
        obs, reward, done, truncated, info = env.step(action)
        
        logger.info(f"Observation shape: {obs.shape}")
        logger.info(f"Current step: {env.current_step}, Window size: {env.window_size}")
        logger.info(f"Reward: {reward}, Done: {done}")
        
        if done:
            logger.info("Episode ended")
            break
    
    # Test with policy network
    logger.info("\nTesting with PolicyNetwork")
    
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rollout_steps=32  # Small for testing
    )
    
    # Reset environment
    obs, info = env.reset()
    
    for i in range(10):
        # Get action from agent
        logger.info(f"Agent step {i+1}")
        # Test shape explicitly before sending to agent
        logger.info(f"Observation shape before agent: {obs.shape}")
        
        # Convert to tensor to check shape handling
        obs_tensor = torch.FloatTensor(obs)
        logger.info(f"Observation tensor shape: {obs_tensor.shape}")
        
        # Use forward directly to test shape handling
        with torch.no_grad():
            action_mean, action_std = agent.network(obs_tensor)
            logger.info(f"Action distribution - Mean shape: {action_mean.shape}, Std shape: {action_std.shape}")
        
        # Get action through agent
        action = agent.get_action(obs)
        
        # Take step in environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store experience and get metrics
        metrics = agent.train_step(obs, action, reward, next_obs, done)
        
        # Log metrics if available
        if metrics:
            logger.info(f"Update metrics: {metrics}")
            
        # Update observation
        obs = next_obs
        
        if done:
            logger.info("Episode ended")
            break
    
    logger.info("Test completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test observation shapes")
    parser.add_argument("--days", type=int, default=14, help="Number of days of data to load")
    parser.add_argument("--window", type=int, default=20, help="Window size for observations")
    
    args = parser.parse_args()
    
    test_observation_shapes(days=args.days, window_size=args.window) 