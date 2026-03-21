import pytest
import numpy as np
import torch
# PPOBuffer was replaced by SB3's internal buffer in Week 19; skip this module.
pytest.importorskip(
    "buffers.ppo_buffer",
    reason="Legacy PPOBuffer removed in Week 19 (replaced by SB3 internal buffer).",
)
from buffers.ppo_buffer import PPOBuffer

@pytest.fixture
def buffer_config():
    """Create basic buffer configuration."""
    return {
        "obs_shape": (10, 5),  # (window_size, features)
        "action_shape": (1,),
        "size": 64,  # batch size
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "device": "cpu"
    }

def test_buffer_initialization(buffer_config):
    """Test buffer initialization and empty state."""
    buffer = PPOBuffer(**buffer_config)
    
    assert len(buffer) == 0, "Buffer should be empty after initialization"
    assert buffer.states == [], "States list should be empty"
    assert buffer.advantages is None, "Advantages should be None"
    assert buffer.returns is None, "Returns should be None"

def test_buffer_append(buffer_config):
    """Test appending experiences to buffer."""
    buffer = PPOBuffer(**buffer_config)
    
    # Create sample experience
    experience = {
        "state": np.random.randn(10, 5),
        "action": np.array([0.5]),
        "reward": 1.0,
        "done": False,
        "value": np.array([[0.5]]),
        "log_prob": np.array([[0.1]])
    }
    
    # Append experience
    buffer.append(experience)
    
    assert len(buffer) == 1, "Buffer should have length 1"
    assert len(buffer.states) == 1, "States list should have length 1"
    assert isinstance(buffer.states[0], np.ndarray), "Stored state should be numpy array"
    assert buffer.states[0].shape == (10, 5), "State shape should be preserved"

def test_buffer_compute_advantages(buffer_config):
    """Test GAE computation with various scenarios."""
    buffer = PPOBuffer(**buffer_config)
    
    # Add multiple experiences
    n_steps = 5
    for i in range(n_steps):
        experience = {
            "state": np.random.randn(10, 5),
            "action": np.array([0.5]),
            "reward": float(i),  # Increasing rewards
            "done": i == n_steps-1,  # Last step is done
            "value": np.array([[float(i)]]),
            "log_prob": np.array([[0.1]])
        }
        buffer.append(experience)
    
    # Compute advantages
    last_value = np.array([[5.0]])
    buffer.compute_advantages(last_value)
    
    assert buffer.advantages is not None, "Advantages should be computed"
    assert buffer.returns is not None, "Returns should be computed"
    assert len(buffer.advantages) == n_steps, "Should have advantage for each step"
    assert len(buffer.returns) == n_steps, "Should have return for each step"
    assert not np.isnan(buffer.advantages).any(), "Advantages should not contain NaN"
    assert not np.isnan(buffer.returns).any(), "Returns should not contain NaN"

def test_buffer_get_batch(buffer_config):
    """Test batch sampling from buffer."""
    buffer = PPOBuffer(**buffer_config)
    
    # Fill buffer
    n_steps = buffer_config["size"]
    for i in range(n_steps):
        experience = {
            "state": np.random.randn(10, 5),
            "action": np.array([0.5]),
            "reward": float(i % 3),  # Cycling rewards
            "done": False,
            "value": np.array([[1.0]]),
            "log_prob": np.array([[0.1]])
        }
        buffer.append(experience)
    
    # Compute advantages before getting batch
    buffer.compute_advantages(np.array([[1.0]]))
    
    # Get full batch
    states, actions, log_probs, returns, advantages, values = buffer.get_batch(
        batch_size=n_steps,
        shuffle=False
    )
    
    # Verify shapes and types
    assert isinstance(states, torch.Tensor), "States should be torch tensor"
    assert isinstance(advantages, torch.Tensor), "Advantages should be torch tensor"
    assert states.shape == (n_steps, 10, 5), f"Expected states shape ({n_steps}, 10, 5), got {states.shape}"
    assert actions.shape == (n_steps, 1), f"Expected actions shape ({n_steps}, 1), got {actions.shape}"
    assert advantages.shape == (n_steps,), f"Expected advantages shape ({n_steps},), got {advantages.shape}"
    
    # Test smaller batch
    small_batch_size = 32
    states, actions, log_probs, returns, advantages, values = buffer.get_batch(
        batch_size=small_batch_size,
        shuffle=True
    )
    
    assert states.shape == (small_batch_size, 10, 5), "Should return requested batch size"

def test_buffer_edge_cases(buffer_config):
    """Test buffer behavior in edge cases."""
    buffer = PPOBuffer(**buffer_config)
    
    # Test computing advantages on empty buffer
    buffer.compute_advantages(np.array([[0.0]]))
    assert buffer.advantages is None, "Empty buffer should have no advantages"
    
    # Test getting batch from empty buffer
    batch = buffer.get_batch(batch_size=32, shuffle=True)
    assert batch is None, "Empty buffer should return None for get_batch"
    
    # Test buffer with single experience
    experience = {
        "state": np.random.randn(10, 5),
        "action": np.array([0.5]),
        "reward": 1.0,
        "done": True,
        "value": np.array([[1.0]]),
        "log_prob": np.array([[0.1]])
    }
    buffer.append(experience)
    buffer.compute_advantages(np.array([[0.0]]))
    
    assert len(buffer.advantages) == 1, "Should compute advantage for single experience"
    assert not np.isnan(buffer.advantages).any(), "Single experience advantage should not be NaN"

def test_buffer_reset(buffer_config):
    """Test buffer reset functionality."""
    buffer = PPOBuffer(**buffer_config)
    
    # Add some experiences
    for _ in range(5):
        experience = {
            "state": np.random.randn(10, 5),
            "action": np.array([0.5]),
            "reward": 1.0,
            "done": False,
            "value": np.array([[1.0]]),
            "log_prob": np.array([[0.1]])
        }
        buffer.append(experience)
    
    buffer.compute_advantages(np.array([[1.0]]))
    buffer.reset()
    
    assert len(buffer) == 0, "Buffer should be empty after reset"
    assert buffer.advantages is None, "Advantages should be None after reset"
    assert buffer.returns is None, "Returns should be None after reset" 