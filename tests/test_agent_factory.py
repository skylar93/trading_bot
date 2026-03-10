"""Tests for the agent factory module (SB3-based)."""

import pytest
import numpy as np
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)

from agents.strategies.agent_factory import create_agent, list_available_agents
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
from agents.sb3.sb3_agent_wrapper import SB3AgentWrapper

WINDOW_SIZE = 20
N_FEATURES = 5

obs_space_1d = gym.spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
obs_space_2d = gym.spaces.Box(low=-10, high=10, shape=(WINDOW_SIZE, N_FEATURES), dtype=np.float32)
act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


# ---------------------------------------------------------------------------
# SB3 agent types
# ---------------------------------------------------------------------------

def test_create_sb3_ppo():
    agent = create_agent("sb3_ppo", observation_space=obs_space_1d, action_space=act_space)
    assert isinstance(agent, SB3AgentWrapper)
    assert agent.algo_type in ("ppo", "sb3_ppo")


def test_create_sb3_sac():
    agent = create_agent("sb3_sac", observation_space=obs_space_1d, action_space=act_space)
    assert isinstance(agent, SB3AgentWrapper)


def test_create_sb3_td3():
    agent = create_agent("sb3_td3", observation_space=obs_space_1d, action_space=act_space)
    assert isinstance(agent, SB3AgentWrapper)


def test_create_sb3_a2c():
    agent = create_agent("sb3_a2c", observation_space=obs_space_1d, action_space=act_space)
    assert isinstance(agent, SB3AgentWrapper)


def test_create_ppo_alias():
    """Plain 'ppo' should map to SB3AgentWrapper."""
    agent = create_agent("ppo", observation_space=obs_space_1d, action_space=act_space)
    assert isinstance(agent, SB3AgentWrapper)


# ---------------------------------------------------------------------------
# Strategy agents (momentum / meanreversion)
# ---------------------------------------------------------------------------

def test_create_momentum_agent():
    agent = create_agent("momentum", observation_space=obs_space_2d)
    assert isinstance(agent, MomentumPPOAgent)


def test_create_mean_reversion_agent():
    agent = create_agent("meanreversion", observation_space=obs_space_2d)
    assert isinstance(agent, MeanReversionPPOAgent)


# ---------------------------------------------------------------------------
# Default spaces (no explicit spaces passed)
# ---------------------------------------------------------------------------

def test_create_sb3_ppo_default_spaces():
    agent = create_agent("sb3_ppo")
    assert agent is not None
    assert isinstance(agent, SB3AgentWrapper)


# ---------------------------------------------------------------------------
# SB3 agent: predict returns correct shape
# ---------------------------------------------------------------------------

def test_sb3_ppo_predict():
    agent = create_agent("sb3_ppo", observation_space=obs_space_1d, action_space=act_space)
    obs = obs_space_1d.sample()
    action = agent.get_action(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == act_space.shape


def test_sb3_ppo_predict_2d_obs():
    agent = create_agent("sb3_ppo", observation_space=obs_space_2d, action_space=act_space)
    obs = obs_space_2d.sample()
    action = agent.get_action(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == act_space.shape


# ---------------------------------------------------------------------------
# list_available_agents
# ---------------------------------------------------------------------------

def test_list_available_agents():
    types = list_available_agents()
    assert isinstance(types, dict)
    assert len(types) > 0
    for k, v in types.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_list_available_agents_includes_sb3():
    types = list_available_agents()
    assert any("sb3" in k.lower() or "ppo" in k.lower() for k in types)


# ---------------------------------------------------------------------------
# Config passing
# ---------------------------------------------------------------------------

def test_create_with_sb3_params():
    config = {
        "sb3_params": {"learning_rate": 1e-4, "n_steps": 512},
        "verbose": 0,
    }
    agent = create_agent("sb3_ppo", config=config,
                         observation_space=obs_space_1d, action_space=act_space)
    assert isinstance(agent, SB3AgentWrapper)


# ---------------------------------------------------------------------------
# Multi-agent via factory
# ---------------------------------------------------------------------------

def test_create_multiagent():
    agent_configs = [
        {"id": "a1", "type": "momentum"},
        {"id": "a2", "type": "meanreversion"},
    ]
    obs_space = gym.spaces.Box(low=-10, high=10, shape=(WINDOW_SIZE, N_FEATURES), dtype=np.float32)
    manager = create_agent("multiagent", config={"agent_configs": agent_configs},
                           observation_space=obs_space)
    assert hasattr(manager, "agents")
    assert len(manager.agents) == 2
    assert "a1" in manager.agents
    assert "a2" in manager.agents


# ---------------------------------------------------------------------------
# SB3 agent: save / load
# ---------------------------------------------------------------------------

def test_sb3_save_load(tmp_path):
    agent = create_agent("sb3_ppo", observation_space=obs_space_1d, action_space=act_space)
    save_path = str(tmp_path / "test_model")
    agent.save(save_path)

    loaded = SB3AgentWrapper.load(
        save_path,
        observation_space=obs_space_1d,
        action_space=act_space,
    )
    obs = obs_space_1d.sample()
    action = loaded.get_action(obs)
    assert action.shape == act_space.shape
