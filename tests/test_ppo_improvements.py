import pytest
import torch
import numpy as np
import gymnasium as gym
from agents.strategies.single.ppo_agent import PPOAgent


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
        n_epochs=10
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


def test_learning_rate_scheduler(ppo_agent):
    """Test that learning rate scheduler works correctly"""
    initial_lr = ppo_agent.optimizer.param_groups[0]["lr"]
    
    # Create sample batch
    batch_size = 32
    states = torch.randn(batch_size, 20, 5).to(ppo_agent.device)
    actions = torch.randn(batch_size, 1).to(ppo_agent.device)
    rewards = torch.randn(batch_size).to(ppo_agent.device)
    values = torch.randn(batch_size).to(ppo_agent.device)
    log_probs = torch.randn(batch_size).to(ppo_agent.device)
    dones = torch.zeros(batch_size).to(ppo_agent.device)
    
    # Update multiple times
    for _ in range(5):
        ppo_agent.update(states, actions, rewards, values, log_probs, dones)
    
    final_lr = ppo_agent.optimizer.param_groups[0]["lr"]
    
    # Learning rate should decrease
    assert final_lr < initial_lr, "Learning rate should decrease over time"
    assert final_lr >= initial_lr * 0.1, "Learning rate should not go below min_lr"


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
    ppo_agent.update(states, actions, rewards, values, log_probs, dones)
    
    # Should have stopped before n_epochs
    assert ppo_agent.scheduler.last_epoch < ppo_agent.n_epochs - 1, "Should stop early due to high KL"


if __name__ == "__main__":
    pytest.main([__file__]) 