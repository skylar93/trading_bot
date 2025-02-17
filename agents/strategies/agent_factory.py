"""
Agent Factory module for creating different types of trading agents.

This module provides a centralized factory for instantiating various trading agents
based on their name/type. It supports both single and multi-agent strategies.
"""

from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np

# Single Agents
from agents.strategies.single.dummy_agent import DummyAgent
from agents.strategies.single.ppo_agent import PPOAgent

# Multi Agents
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
from agents.strategies.multi.multi_agent_manager import MultiAgentManager

# Default dummy spaces for testing
# For PPO agents, observation space must be 2D (window_size, features)
# Assuming OHLCV data format: open, high, low, close, volume
WINDOW_SIZE = 20
N_FEATURES = 5  # OHLCV format
dummy_obs_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(WINDOW_SIZE, N_FEATURES),
    dtype=np.float32
)
dummy_act_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(1,),
    dtype=np.float32
)

def create_agent(agent_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Creates and returns an instance of the specified trading agent.
    
    Args:
        agent_name: Name/type of the agent to create (e.g., "Dummy", "PPO", "MeanReversion")
        config: Optional configuration dictionary for agent initialization
        
    Returns:
        An instance of the specified agent
        
    Raises:
        ValueError: If the specified agent type is not supported
    """
    if config is None:
        config = {}
    
    # Add dummy spaces for PPO-based agents if needed
    if agent_name in ("PPO", "MeanReversion", "Momentum"):
        if "env" not in config and ("observation_space" not in config or "action_space" not in config):
            if agent_name == "PPO":
                config["observation_space"] = dummy_obs_space
                config["action_space"] = dummy_act_space
            else:
                # For MeanReversion and Momentum agents, observation_space is a direct field
                config = {
                    "observation_space": dummy_obs_space,
                    "action_space": dummy_act_space,
                    **config  # Keep any other config values
                }
    
    # Single Agents
    if agent_name == "Dummy":
        return DummyAgent(**config)
    elif agent_name == "PPO":
        return PPOAgent(**config)
        
    # Multi Agents
    elif agent_name == "MeanReversion":
        # MeanReversionPPOAgent expects a single config argument
        return MeanReversionPPOAgent(config)
    elif agent_name == "Momentum":
        # MomentumPPOAgent expects a single config argument
        return MomentumPPOAgent(config)
    elif agent_name == "MultiAgent":
        # Ensure agent_configs exists for MultiAgentManager
        if "agent_configs" not in config:
            config["agent_configs"] = []
        return MultiAgentManager(**config)
        
    raise ValueError(f"Unsupported agent type: {agent_name}")

def list_available_agents() -> Dict[str, str]:
    """
    Returns a dictionary of available agent types and their descriptions.
    
    Returns:
        Dict mapping agent names to their descriptions
    """
    return {
        "Dummy": "Simple dummy agent for testing",
        "PPO": "Single PPO agent for general trading",
        "MeanReversion": "Mean reversion strategy using PPO",
        "Momentum": "Momentum-based strategy using PPO",
        "MultiAgent": "Manager for multiple trading agents"
    } 