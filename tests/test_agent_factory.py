"""
Tests for the agent factory module.
"""

import pytest
from typing import Dict, Any
import numpy as np
import gymnasium as gym

from agents.strategies.agent_factory import create_agent, list_available_agents
from agents.strategies.single.dummy_agent import DummyAgent
from agents.strategies.single.ppo_agent import PPOAgent
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
from agents.strategies.multi.multi_agent_manager import MultiAgentManager

def test_create_dummy_agent():
    """Test creation of DummyAgent."""
    agent = create_agent("Dummy")
    assert isinstance(agent, DummyAgent)

def test_create_ppo_agent():
    """Test creation of PPOAgent."""
    agent = create_agent("PPO")
    assert isinstance(agent, PPOAgent)

def test_create_mean_reversion_agent():
    """Test creation of MeanReversionPPOAgent."""
    agent = create_agent("MeanReversion")
    assert isinstance(agent, MeanReversionPPOAgent)

def test_create_momentum_agent():
    """Test creation of MomentumPPOAgent."""
    agent = create_agent("Momentum")
    assert isinstance(agent, MomentumPPOAgent)

def test_create_multi_agent():
    """Test creation of MultiAgentManager."""
    agent = create_agent("MultiAgent")
    assert isinstance(agent, MultiAgentManager)

def test_create_agent_with_config():
    """Test agent creation with configuration."""
    config = {
        "learning_rate": 0.001,
        "batch_size": 64
    }
    agent = create_agent("PPO", config=config)
    assert isinstance(agent, PPOAgent)
    # Additional config verification could be added here if the agents expose their config

def test_create_invalid_agent():
    """Test error handling for invalid agent type."""
    with pytest.raises(ValueError) as exc_info:
        create_agent("InvalidAgent")
    assert "Unsupported agent type: InvalidAgent" in str(exc_info.value)

def test_list_available_agents():
    """Test listing available agents."""
    agents = list_available_agents()
    
    # Check that we have all expected agent types
    expected_agents = {"Dummy", "PPO", "MeanReversion", "Momentum", "MultiAgent"}
    assert set(agents.keys()) == expected_agents
    
    # Check that all values are non-empty strings
    for description in agents.values():
        assert isinstance(description, str)
        assert len(description) > 0

def test_create_agent_empty_config():
    """Test agent creation with empty config."""
    agent = create_agent("Dummy", config={})
    assert isinstance(agent, DummyAgent)

def test_create_agent_none_config():
    """Test agent creation with None config."""
    agent = create_agent("Dummy", config=None)
    assert isinstance(agent, DummyAgent)

def test_agent_identity_persistence():
    """Test that created agents maintain their identity and don't default to DummyAgent."""
    # Create different types of agents
    agents = {
        "Dummy": create_agent("Dummy"),
        "PPO": create_agent("PPO"),
        "MeanReversion": create_agent("MeanReversion"),
        "Momentum": create_agent("Momentum")
    }
    
    # Verify each agent is of the correct type
    assert isinstance(agents["Dummy"], DummyAgent)
    assert isinstance(agents["PPO"], PPOAgent)
    assert isinstance(agents["MeanReversion"], MeanReversionPPOAgent)
    assert isinstance(agents["Momentum"], MomentumPPOAgent)
    
    # Verify agents are different instances
    assert agents["MeanReversion"] != agents["Momentum"]
    assert agents["PPO"] != agents["Dummy"]

def test_agent_config_persistence():
    """Test that agent configurations are properly maintained."""
    test_config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "observation_space": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, 5), dtype=np.float32
        ),
        "action_space": gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32
        )
    }
    
    # Create agents with config
    ppo_agent = create_agent("PPO", config=test_config)
    mean_rev_agent = create_agent("MeanReversion", config=test_config)
    
    # Verify config persistence (assuming agents expose these properties)
    assert hasattr(ppo_agent, "learning_rate")
    assert hasattr(ppo_agent, "batch_size")
    assert ppo_agent.learning_rate == test_config["learning_rate"]
    assert ppo_agent.batch_size == test_config["batch_size"]

def test_agent_behavior_differentiation():
    """Test that different agents produce different actions for the same input."""
    # Create test observation
    obs = np.random.random((20, 5)).astype(np.float32)
    
    # Create different agents
    agents = {
        "Dummy": create_agent("Dummy"),
        "MeanReversion": create_agent("MeanReversion"),
        "Momentum": create_agent("Momentum")
    }
    
    # Get actions from each agent
    actions = {name: agent.predict(obs) for name, agent in agents.items()}
    
    # Verify actions are different
    assert not np.array_equal(actions["Dummy"], actions["MeanReversion"])
    assert not np.array_equal(actions["MeanReversion"], actions["Momentum"])

def test_multi_agent_configuration():
    """Test proper configuration of MultiAgentManager."""
    agent_configs = [
        {
            "id": "mean_rev_1",
            "strategy": "mean_reversion",
            "weight": 0.5,
            "observation_space": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(20, 5), dtype=np.float32
            ),
            "action_space": gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(1,), dtype=np.float32
            )
        },
        {
            "id": "momentum_1",
            "strategy": "momentum",
            "weight": 0.5,
            "observation_space": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(20, 5), dtype=np.float32
            ),
            "action_space": gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(1,), dtype=np.float32
            )
        }
    ]
    
    config = {
        "agent_configs": agent_configs
    }
    
    # Create multi-agent
    multi_agent = create_agent("MultiAgent", config=config)
    
    # Verify agent composition
    assert len(multi_agent.agents) == 2
    assert isinstance(multi_agent.agents["mean_rev_1"], MeanReversionPPOAgent)
    assert isinstance(multi_agent.agents["momentum_1"], MomentumPPOAgent)
    assert all(cfg["weight"] == 0.5 for cfg in agent_configs)

def test_agent_state_initialization():
    """Test that agents are properly initialized with different random states."""
    # Create multiple instances of the same agent type
    agent1 = create_agent("PPO")
    agent2 = create_agent("PPO")
    
    # Create test observation
    obs = np.random.random((20, 5)).astype(np.float32)
    
    # Get actions from both agents
    action1 = agent1.predict(obs)
    action2 = agent2.predict(obs)
    
    # Verify actions are different (agents should have different random states)
    assert not np.array_equal(action1, action2)

def test_agent_type_specific_config():
    """Test that agent-specific configurations are properly handled."""
    # Test MeanReversion specific config
    mean_rev_config = {
        "rsi_window": 50,
        "bb_window": 20,
        "bb_std": 2.0,
        "observation_space": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, 5), dtype=np.float32
        ),
        "action_space": gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32
        )
    }
    
    mean_rev_agent = create_agent("MeanReversion", config=mean_rev_config)
    assert hasattr(mean_rev_agent, "rsi_window")
    assert mean_rev_agent.rsi_window == 50
    assert hasattr(mean_rev_agent, "bb_window")
    assert mean_rev_agent.bb_window == 20
    
    # Test Momentum specific config
    momentum_config = {
        "momentum_window": 20,
        "observation_space": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, 5), dtype=np.float32
        ),
        "action_space": gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32
        )
    }
    
    momentum_agent = create_agent("Momentum", config=momentum_config)
    assert hasattr(momentum_agent, "momentum_window")
    assert momentum_agent.momentum_window == 20 