import pytest
import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
# Legacy MLP / ValueMLP removed in Week 19 (replaced by SB3 policy networks).
pytest.importorskip(
    "agents.models.architectures.mlp",
    reason="Legacy MLP architectures removed in Week 19 (replaced by SB3 policy).",
)
from agents.models.architectures.mlp import PolicyNetwork
from agents.models.architectures.value_mlp import ValueNetwork
from agents.models.architectures.base import BaseNetwork


@pytest.fixture
def observation_space():
    """Create observation space for testing."""
    return Box(
        low=-np.inf,
        high=np.inf,
        shape=(10, 5),  # (window_size, features)
        dtype=np.float32
    )


@pytest.fixture
def action_space():
    """Create action space for testing."""
    return Box(
        low=0,
        high=1,
        shape=(1,),
        dtype=np.float32
    )


def test_policy_network_initialization(observation_space, action_space):
    """Test policy network initialization."""
    network = PolicyNetwork(observation_space, action_space)
    
    # Verify network components
    assert isinstance(network, PolicyNetwork)
    assert hasattr(network, "shared")
    assert hasattr(network, "mean_head")
    assert hasattr(network, "std_head")
    
    # Verify network can process input
    batch_size = 1
    obs = torch.randn(batch_size, *observation_space.shape)
    mean, std = network(obs)
    
    # Now we can check mean and std properties
    assert hasattr(network, "mean")
    assert hasattr(network, "std")
    assert torch.allclose(network.mean, mean)
    assert torch.allclose(network.std, std)
    
    # Verify output shapes
    assert mean.shape == (batch_size, action_space.shape[0])
    assert std.shape == (batch_size, action_space.shape[0])


def test_policy_network_forward(observation_space, action_space):
    """Test policy network forward pass."""
    network = PolicyNetwork(observation_space, action_space)
    
    # Single observation
    obs = torch.randn(1, *observation_space.shape)
    mean, std = network(obs)
    
    assert mean.shape == (1, action_space.shape[0])
    assert std.shape == (1, action_space.shape[0])
    assert torch.all(mean >= 0) and torch.all(mean <= 1)
    assert torch.all(std >= 0) and torch.all(std <= 1)
    
    # Batch of observations
    batch_size = 32
    obs_batch = torch.randn(batch_size, *observation_space.shape)
    mean_batch, std_batch = network(obs_batch)
    
    assert mean_batch.shape == (batch_size, action_space.shape[0])
    assert std_batch.shape == (batch_size, action_space.shape[0])
    assert torch.all(mean_batch >= 0) and torch.all(mean_batch <= 1)
    assert torch.all(std_batch >= 0) and torch.all(std_batch <= 1)


def test_value_network_initialization(observation_space):
    """Test value network initialization."""
    network = ValueNetwork(observation_space)
    
    assert isinstance(network, ValueNetwork)
    assert hasattr(network, "network")
    
    # Verify network can process input
    batch_size = 1
    obs = torch.randn(batch_size, *observation_space.shape)
    value = network(obs)
    
    assert value.shape == (batch_size, 1)


def test_value_network_forward(observation_space):
    """Test value network forward pass."""
    network = ValueNetwork(observation_space)
    
    # Single observation
    obs = torch.randn(1, *observation_space.shape)
    value = network(obs)
    
    assert value.shape == (1, 1)
    assert not torch.isnan(value).any()
    
    # Batch of observations
    batch_size = 32
    obs_batch = torch.randn(batch_size, *observation_space.shape)
    value_batch = network(obs_batch)
    
    assert value_batch.shape == (batch_size, 1)
    assert not torch.isnan(value_batch).any()


def test_network_gradient_flow():
    """Test that gradients flow properly through networks."""
    obs_space = Box(low=-np.inf, high=np.inf, shape=(10, 5), dtype=np.float32)
    act_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    policy_net = PolicyNetwork(obs_space, act_space)
    value_net = ValueNetwork(obs_space)
    
    # Test policy network gradients
    obs = torch.randn(1, 10, 5, requires_grad=True)
    mean, std = policy_net(obs)
    loss = mean.mean() + std.mean()
    loss.backward()
    
    assert obs.grad is not None, "No gradients flowing to policy network input"
    
    # Test value network gradients
    obs = torch.randn(1, 10, 5, requires_grad=True)
    value = value_net(obs)
    value.mean().backward()
    
    assert obs.grad is not None, "No gradients flowing to value network input"


def test_action_bounds(observation_space, action_space):
    """Test that actions are properly bounded."""
    network = PolicyNetwork(observation_space, action_space)
    
    # Test with various inputs
    for _ in range(100):
        obs = torch.randn(1, *observation_space.shape)
        mean, std = network(obs)
        
        assert torch.all(mean >= 0) and torch.all(mean <= 1), "Action mean should be in [0, 1]"
        assert torch.all(std >= 0) and torch.all(std <= 1), "Action std should be in [0, 1]"


def test_save_load(observation_space, action_space, tmp_path):
    """Test network save and load functionality."""
    policy_net = PolicyNetwork(observation_space, action_space)
    value_net = ValueNetwork(observation_space)
    
    # Generate random input
    obs = torch.randn(1, *observation_space.shape)
    
    # Get outputs before saving
    mean1, std1 = policy_net(obs)
    value1 = value_net(obs)
    
    # Save networks
    policy_path = tmp_path / "policy.pt"
    value_path = tmp_path / "value.pt"
    policy_net.save(policy_path)
    value_net.save(value_path)
    
    # Create new networks and load saved weights
    policy_net2 = PolicyNetwork(observation_space, action_space)
    value_net2 = ValueNetwork(observation_space)
    policy_net2.load(policy_path)
    value_net2.load(value_path)
    
    # Get outputs after loading
    mean2, std2 = policy_net2(obs)
    value2 = value_net2(obs)
    
    # Compare outputs
    assert torch.allclose(mean1, mean2), "Policy mean changed after save/load"
    assert torch.allclose(std1, std2), "Policy std changed after save/load"
    assert torch.allclose(value1, value2), "Value changed after save/load"


def test_policy_network_shapes(observation_space, action_space):
    """Test policy network handling of different input shapes."""
    network = PolicyNetwork(observation_space, action_space)
    
    # Test single observation
    obs = torch.randn(1, 10, 5)  # (batch_size=1, window_size=10, features=5)
    mean, std = network(obs)
    
    assert mean.shape == (1, 1), f"Expected action mean shape (1, 1), got {mean.shape}"
    assert std.shape == (1, 1), f"Expected action std shape (1, 1), got {std.shape}"
    
    # Test batch of observations
    batch_size = 32
    obs_batch = torch.randn(batch_size, 10, 5)
    mean_batch, std_batch = network(obs_batch)
    
    assert mean_batch.shape == (batch_size, 1), f"Expected batch mean shape ({batch_size}, 1), got {mean_batch.shape}"
    assert std_batch.shape == (batch_size, 1), f"Expected batch std shape ({batch_size}, 1), got {std_batch.shape}"
    
    # Test with flattened input
    obs_flat = torch.randn(batch_size, 50)  # (batch_size, window_size * features)
    mean_flat, std_flat = network(obs_flat)
    
    assert mean_flat.shape == (batch_size, 1), "Network should handle flattened input"
    assert std_flat.shape == (batch_size, 1), "Network should handle flattened input"


def test_value_network_shapes(observation_space):
    """Test value network handling of different input shapes."""
    network = ValueNetwork(observation_space)
    
    # Test single observation
    obs = torch.randn(1, 10, 5)
    value = network(obs)
    
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"
    
    # Test batch of observations
    batch_size = 32
    obs_batch = torch.randn(batch_size, 10, 5)
    value_batch = network(obs_batch)
    
    assert value_batch.shape == (batch_size, 1), f"Expected batch value shape ({batch_size}, 1), got {value_batch.shape}"
    
    # Test with flattened input
    obs_flat = torch.randn(batch_size, 50)
    value_flat = network(obs_flat)
    
    assert value_flat.shape == (batch_size, 1), "Network should handle flattened input"


def test_network_nan_handling():
    """Test network behavior with NaN inputs."""
    obs_space = Box(low=-np.inf, high=np.inf, shape=(10, 5), dtype=np.float32)
    act_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    policy_net = PolicyNetwork(obs_space, act_space)
    value_net = ValueNetwork(obs_space)
    
    # Create input with some NaN values
    obs_with_nan = torch.randn(1, 10, 5)
    obs_with_nan[0, 0, 0] = float('nan')
    
    # Policy network should handle NaN
    mean, std = policy_net(obs_with_nan)
    assert not torch.isnan(mean).any(), "Policy mean contains NaN"
    assert not torch.isnan(std).any(), "Policy std contains NaN"
    
    # Value network should handle NaN
    value = value_net(obs_with_nan)
    assert not torch.isnan(value).any(), "Value contains NaN"


if __name__ == "__main__":
    pytest.main([__file__])
