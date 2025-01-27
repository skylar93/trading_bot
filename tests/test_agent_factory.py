"""
Tests for the agent factory module.
"""

import pytest
from typing import Dict, Any

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