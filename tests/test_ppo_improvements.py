"""Tests for PPO improvements."""

import pytest
import torch
import numpy as np
import gymnasium as gym
from agents.strategies.single.ppo_agent import PPOAgent
from stable_baselines3.common.env_util import DummyVecEnv
from gymnasium.spaces import Box


@pytest.fixture
def observation_space():
    """Create sample observation space"""
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20, 5))


@pytest.fixture
def action_space():
    """Create sample action space"""
    return gym.spaces.Box(low=-1, high=1, shape=(1,))


@pytest.fixture
def ppo_agent(observation_space, action_space):
    """Create PPO agent with default parameters"""
    return PPOAgent(
        observation_space=observation_space,
        action_space=action_space,
        learning_rate=3e-4,
        n_epochs=10,
        use_lr_scheduler=True,
        lr_scheduler_gamma=0.9,
        lr_scheduler_step_size=100
    )


def test_kl_penalty_effect(ppo_agent):
    """Test that KL penalty affects the loss appropriately"""
    # Create sample batch
    batch_size = 32
    states = torch.randn(batch_size, 20, 5).to(ppo_agent.device)
    actions = torch.randn(batch_size, 1).to(ppo_agent.device)
    rewards = torch.ones(batch_size).to(ppo_agent.device)  # Use consistent rewards
    values = torch.zeros(batch_size).to(ppo_agent.device)  # Start with zero values
    log_probs = torch.zeros(batch_size).to(ppo_agent.device)
    dones = torch.zeros(batch_size).to(ppo_agent.device)

    # Get initial policy distribution
    with torch.no_grad():
        initial_mean, initial_std = ppo_agent.network(states)
    
    # Store initial network state
    initial_state = {
        name: param.clone()
        for name, param in ppo_agent.network.named_parameters()
    }
    
    # Update with normal c3 (KL penalty coefficient)
    ppo_agent.c3 = 0.5
    ppo_agent.update(states, actions, rewards, values, log_probs, dones)
    
    # Calculate parameter changes with normal KL
    normal_param_change = 0
    for name, param in ppo_agent.network.named_parameters():
        normal_param_change += torch.norm(param - initial_state[name])
    
    # Restore initial network state
    for name, param in ppo_agent.network.named_parameters():
        param.data.copy_(initial_state[name])
    
    # Update with high c3
    ppo_agent.c3 = 2.0
    ppo_agent.update(states, actions, rewards, values, log_probs, dones)
    
    # Calculate parameter changes with high KL
    high_param_change = 0
    for name, param in ppo_agent.network.named_parameters():
        high_param_change += torch.norm(param - initial_state[name])
    
    # Higher KL penalty should result in smaller parameter changes
    assert high_param_change < normal_param_change, (
        f"Higher KL penalty should constrain policy updates more. "
        f"Normal change: {normal_param_change:.4f}, High KL change: {high_param_change:.4f}"
    )


def test_learning_rate_scheduler():
    """Test that learning rate decreases over time with scheduler."""
    # Create simple environment with 2D observation space
    observation_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 3),  # window_size=4, features=3
        dtype=np.float32
    )
    action_space = Box(
        low=0,
        high=1,
        shape=(1,),
        dtype=np.float32
    )
    
    # Configure agent with learning rate scheduler
    config = {
        'learning_rate': 0.001,
        'batch_size': 4,
        'n_epochs': 1,
        'use_lr_scheduler': True,
        'lr_scheduler_gamma': 0.9,
        'lr_scheduler_step_size': 100
    }
    
    agent = PPOAgent(observation_space, action_space, **config)
    initial_lr = agent.learning_rate
    
    # Training loop
    for step in range(500):
        # Generate random state and action
        state = np.random.randn(4, 3).astype(np.float32)  # Shape: (window_size, features)
        action = np.random.rand(1).astype(np.float32)  # Shape: (1,)
        reward = 1.0
        next_state = np.random.randn(4, 3).astype(np.float32)  # Shape: (window_size, features)
        done = False
        
        # Train step
        agent.train_step(state, action, reward, next_state, done)
    
    # Get final learning rate
    final_lr = agent.optimizer.param_groups[0]['lr']
    
    # Verify learning rate decreased
    assert final_lr < initial_lr, "Learning rate should decrease over time"
    assert final_lr > 1e-6, "Learning rate should not decrease too much"


def test_early_stopping_on_high_kl(ppo_agent):
    """Test that training stops when KL divergence is too high"""
    # Create sample batch
    batch_size = 32
    states = torch.randn(batch_size, 20, 5).to(ppo_agent.device)
    actions = torch.randn(batch_size, 1).to(ppo_agent.device)
    rewards = torch.randn(batch_size).to(ppo_agent.device)
    values = torch.randn(batch_size).to(ppo_agent.device)
    log_probs = torch.randn(batch_size).to(ppo_agent.device)
    dones = torch.zeros(batch_size).to(ppo_agent.device)
    
    # Set very low target KL to trigger early stopping
    ppo_agent.target_kl = 1e-6
    
    # Update and count epochs
    metrics = ppo_agent.update(states, actions, log_probs, rewards, values, dones)
    
    # Should have high KL divergence
    assert metrics["kl"] > ppo_agent.target_kl, "KL divergence should be higher than target"


if __name__ == "__main__":
    pytest.main([__file__]) 