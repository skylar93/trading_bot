import pytest
import torch
import numpy as np
import gymnasium as gym
from agents.models.policy_network import PolicyNetwork
from agents.models.value_network import ValueNetwork

@pytest.fixture
def observation_space():
    """Create sample observation space"""
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20, 5))

@pytest.fixture
def action_space():
    """Create sample action space"""
    return gym.spaces.Box(low=-1, high=1, shape=(1,))

def test_policy_network_initialization(observation_space, action_space):
    """Test policy network initialization"""
    network = PolicyNetwork(observation_space, action_space)
    
    # Check network structure
    assert isinstance(network, torch.nn.Module)
    assert hasattr(network, 'shared')
    assert hasattr(network, 'mean')
    assert hasattr(network, 'log_std')
    
    # Check input/output dimensions
    input_size = int(np.prod(observation_space.shape))
    output_size = int(np.prod(action_space.shape))
    assert network.shared[0].in_features == input_size
    assert network.mean.out_features == output_size
    assert network.log_std.shape == torch.Size([output_size])

def test_policy_network_forward(observation_space, action_space):
    """Test policy network forward pass"""
    network = PolicyNetwork(observation_space, action_space)
    batch_size = 32
    
    # Create sample input
    obs = torch.randn(batch_size, *observation_space.shape)
    
    # Forward pass
    mean, std = network(obs)
    
    # Check output shapes and values
    assert mean.shape == (batch_size, *action_space.shape)
    assert std.shape == mean.shape  # Standard deviation should match mean shape
    assert torch.all(std > 0)  # Standard deviation should be positive

def test_value_network_initialization(observation_space):
    """Test value network initialization"""
    network = ValueNetwork(observation_space)
    
    # Check network structure
    assert isinstance(network, torch.nn.Module)
    assert hasattr(network, 'network')
    
    # Check input/output dimensions
    input_size = int(np.prod(observation_space.shape))
    assert network.network[0].in_features == input_size
    assert network.network[-1].out_features == 1  # Value network outputs a single value

def test_value_network_forward(observation_space):
    """Test value network forward pass"""
    network = ValueNetwork(observation_space)
    batch_size = 32
    
    # Create sample input
    obs = torch.randn(batch_size, *observation_space.shape)
    
    # Forward pass
    value = network(obs)
    
    # Check output shape
    assert value.shape == (batch_size, 1)

def test_network_gradient_flow(observation_space, action_space):
    """Test gradient flow through networks"""
    policy_net = PolicyNetwork(observation_space, action_space)
    value_net = ValueNetwork(observation_space)
    
    # Sample batch
    batch_size = 16
    obs = torch.randn(batch_size, *observation_space.shape, requires_grad=True)
    
    # Policy loss
    policy_net.zero_grad()
    mean, std = policy_net(obs)
    policy_loss = -(mean * std).mean()  # Use both outputs in loss
    policy_loss.backward(retain_graph=True)
    
    # Check policy gradients
    for param in policy_net.parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)
    
    # Value loss
    value_net.zero_grad()
    value = value_net(obs)
    value_loss = value.mean()
    value_loss.backward()
    
    # Check value gradients
    for param in value_net.parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)

def test_action_bounds(observation_space, action_space):
    """Test if policy network respects action bounds"""
    network = PolicyNetwork(observation_space, action_space)
    batch_size = 100
    
    # Generate multiple observations
    obs = torch.randn(batch_size, *observation_space.shape)
    
    # Get actions
    mean, std = network(obs)
    
    # Convert action space bounds to tensors
    low = torch.tensor(action_space.low).float()
    high = torch.tensor(action_space.high).float()
    
    # Check if mean actions are within bounds
    assert torch.all(mean >= low)
    assert torch.all(mean <= high)

if __name__ == "__main__":
    pytest.main([__file__]) 