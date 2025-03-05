"""
Agent Factory module for creating different types of trading agents.

This module provides a centralized factory for instantiating various trading agents
based on their name/type. It supports both single and multi-agent strategies.
"""

from typing import Optional, Dict, Any, Union, List

import gymnasium as gym
import numpy as np
import logging

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

logger = logging.getLogger(__name__)

def create_agent(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
    observation_space: Optional[gym.spaces.Box] = None,
    action_space: Optional[gym.spaces.Box] = None,
) -> Union[DummyAgent, PPOAgent, MeanReversionPPOAgent, MomentumPPOAgent, MultiAgentManager]:
    """Create an agent based on type and configuration.
    
    Args:
        agent_type: Type of agent to create (e.g. "Dummy", "PPO", "MeanReversion")
        config: Agent configuration dictionary (optional)
        observation_space: Observation space (optional)
        action_space: Action space (optional)
        
    Returns:
        Created agent instance
        
    Raises:
        ValueError: If agent_type is not supported or if required config is missing
    """
    # Initialize empty config if None
    if config is None:
        config = {}
    else:
        # Create a copy to avoid modifying the original
        config = config.copy()
        
    # Remove observation_space and action_space from config if they exist
    # to avoid passing them twice
    config.pop("observation_space", None)
    config.pop("action_space", None)
    
    # Use provided spaces or defaults
    obs_space = observation_space or dummy_obs_space
    act_space = action_space or dummy_act_space
    
    # Create agent based on type
    agent_type = agent_type.lower()
    
    try:
        if agent_type == "dummy":
            return DummyAgent(
                observation_space=obs_space,
                action_space=act_space,
                **config
            )
        elif agent_type == "ppo":
            return PPOAgent(
                observation_space=obs_space,
                action_space=act_space,
                **config
            )
        elif agent_type == "meanreversion":
            return MeanReversionPPOAgent(
                observation_space=obs_space,
                action_space=act_space,
                **config
            )
        elif agent_type == "momentum":
            return MomentumPPOAgent(
                observation_space=obs_space,
                action_space=act_space,
                **config
            )
        elif agent_type == "multiagent":
            # MultiAgentManager requires a list of agent configs
            if "agents" not in config:
                config["agents"] = []  # Provide empty list as default
            return MultiAgentManager(agent_configs=config["agents"])
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
            
    except TypeError as e:
        logger.error(f"Failed to create agent of type {agent_type}: {str(e)}")
        raise

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