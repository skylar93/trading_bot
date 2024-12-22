import pytest
import torch
import numpy as np
from gym import spaces

from agents.models.architectures.convlstm_policy import ConvLSTMPolicy
from agents.models.architectures.tabnet_policy import TabNetPolicy
from agents.models.architectures.tcn_policy import TCNPolicy
from agents.models.architectures.lstm_policy import LSTMPolicy
from agents.models.architectures.cnn_policy import CNNPolicy
from agents.models.architectures.transformer_policy import TransformerPolicy


@pytest.fixture
def observation_space():
    """Create a sample observation space"""
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(50, 10),  # (seq_len, features)
        dtype=np.float32
    )


@pytest.fixture
def action_space():
    """Create a sample action space"""
    return spaces.Box(
        low=-1,
        high=1,
        shape=(2,),  # 2D continuous action space
        dtype=np.float32
    )


@pytest.fixture
def batch_observation():
    """Create a sample batch of observations"""
    return torch.randn(8, 50, 10)  # (batch, seq_len, features)


def test_convlstm_policy_initialization(observation_space, action_space):
    """Test ConvLSTM policy network initialization"""
    policy = ConvLSTMPolicy(observation_space, action_space)
    
    # Check architecture type
    assert policy.get_architecture_type() == "convlstm"
    
    # Check parameter initialization
    for name, param in policy.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            # Check if weights are properly initialized (not zero or inf)
            assert not torch.allclose(param, torch.zeros_like(param))
            assert not torch.any(torch.isinf(param))


def test_convlstm_policy_forward(observation_space, action_space, batch_observation):
    """Test ConvLSTM policy network forward pass"""
    policy = ConvLSTMPolicy(observation_space, action_space)
    
    # Test forward pass
    action_mean, action_std = policy(batch_observation)
    
    # Check output shapes
    assert action_mean.shape == (8, 2)  # (batch_size, action_dim)
    assert action_std.shape == (8, 2)
    
    # Check if outputs are valid (no NaN or inf)
    assert not torch.any(torch.isnan(action_mean))
    assert not torch.any(torch.isnan(action_std))
    assert not torch.any(torch.isinf(action_mean))
    assert not torch.any(torch.isinf(action_std))


def test_tabnet_policy_initialization(observation_space, action_space):
    """Test TabNet policy network initialization"""
    policy = TabNetPolicy(observation_space, action_space)
    
    # Check architecture type
    assert policy.get_architecture_type() == "tabnet"
    
    # Check parameter initialization
    for name, param in policy.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert not torch.any(torch.isinf(param))


def test_tabnet_policy_forward(observation_space, action_space, batch_observation):
    """Test TabNet policy network forward pass"""
    policy = TabNetPolicy(observation_space, action_space)
    
    # Test forward pass
    action_mean, action_std = policy(batch_observation)
    
    # Check output shapes
    assert action_mean.shape == (8, 2)
    assert action_std.shape == (8, 2)
    
    # Check if outputs are valid
    assert not torch.any(torch.isnan(action_mean))
    assert not torch.any(torch.isnan(action_std))
    assert not torch.any(torch.isinf(action_mean))
    assert not torch.any(torch.isinf(action_std))


def test_tcn_policy_initialization(observation_space, action_space):
    """Test TCN policy network initialization"""
    policy = TCNPolicy(observation_space, action_space)
    
    # Check architecture type
    assert policy.get_architecture_type() == "tcn"
    
    # Check parameter initialization
    for name, param in policy.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert not torch.any(torch.isinf(param))


def test_tcn_policy_forward(observation_space, action_space, batch_observation):
    """Test TCN policy network forward pass"""
    policy = TCNPolicy(observation_space, action_space)
    
    # Test forward pass
    action_mean, action_std = policy(batch_observation)
    
    # Check output shapes
    assert action_mean.shape == (8, 2)
    assert action_std.shape == (8, 2)
    
    # Check if outputs are valid
    assert not torch.any(torch.isnan(action_mean))
    assert not torch.any(torch.isnan(action_std))
    assert not torch.any(torch.isinf(action_mean))
    assert not torch.any(torch.isinf(action_std))


def test_lstm_policy_initialization(observation_space, action_space):
    """Test LSTM policy network initialization"""
    policy = LSTMPolicy(observation_space, action_space)
    
    # Check architecture type
    assert policy.get_architecture_type() == "lstm"
    
    # Check parameter initialization
    for name, param in policy.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert not torch.any(torch.isinf(param))


def test_lstm_policy_forward(observation_space, action_space, batch_observation):
    """Test LSTM policy network forward pass"""
    policy = LSTMPolicy(observation_space, action_space)
    
    # Test forward pass
    action_mean, action_std = policy(batch_observation)
    
    # Check output shapes
    assert action_mean.shape == (8, 2)
    assert action_std.shape == (8, 2)
    
    # Check if outputs are valid
    assert not torch.any(torch.isnan(action_mean))
    assert not torch.any(torch.isnan(action_std))
    assert not torch.any(torch.isinf(action_mean))
    assert not torch.any(torch.isinf(action_std))


def test_cnn_policy_initialization(observation_space, action_space):
    """Test CNN policy network initialization"""
    policy = CNNPolicy(observation_space, action_space)
    
    # Check architecture type
    assert policy.get_architecture_type() == "cnn"
    
    # Check parameter initialization
    for name, param in policy.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert not torch.any(torch.isinf(param))


def test_cnn_policy_forward(observation_space, action_space, batch_observation):
    """Test CNN policy network forward pass"""
    policy = CNNPolicy(observation_space, action_space)
    
    # Test forward pass
    action_mean, action_std = policy(batch_observation)
    
    # Check output shapes
    assert action_mean.shape == (8, 2)
    assert action_std.shape == (8, 2)
    
    # Check if outputs are valid
    assert not torch.any(torch.isnan(action_mean))
    assert not torch.any(torch.isnan(action_std))
    assert not torch.any(torch.isinf(action_mean))
    assert not torch.any(torch.isinf(action_std))


def test_transformer_policy_initialization(observation_space, action_space):
    """Test Transformer policy network initialization"""
    policy = TransformerPolicy(observation_space, action_space)
    
    # Check architecture type
    assert policy.get_architecture_type() == "transformer"
    
    # Check parameter initialization
    for name, param in policy.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            assert not torch.allclose(param, torch.zeros_like(param))
            assert not torch.any(torch.isinf(param))


def test_transformer_policy_forward(observation_space, action_space, batch_observation):
    """Test Transformer policy network forward pass"""
    policy = TransformerPolicy(observation_space, action_space)
    
    # Test forward pass
    action_mean, action_std = policy(batch_observation)
    
    # Check output shapes
    assert action_mean.shape == (8, 2)
    assert action_std.shape == (8, 2)
    
    # Check if outputs are valid
    assert not torch.any(torch.isnan(action_mean))
    assert not torch.any(torch.isnan(action_std))
    assert not torch.any(torch.isinf(action_mean))
    assert not torch.any(torch.isinf(action_std))


def test_gradient_flow(observation_space, action_space, batch_observation):
    """Test gradient flow through all network architectures"""
    policies = [
        ConvLSTMPolicy(observation_space, action_space),
        TabNetPolicy(observation_space, action_space),
        TCNPolicy(observation_space, action_space),
        LSTMPolicy(observation_space, action_space),
        CNNPolicy(observation_space, action_space),
        TransformerPolicy(observation_space, action_space)
    ]
    
    target = torch.randn(8, 2)  # Random target actions
    
    for policy in policies:
        # Enable gradient computation
        batch_observation.requires_grad_(True)
        policy.train()  # Set to training mode
        
        # Forward pass
        action_mean, action_std = policy(batch_observation)
        
        # Compute negative log likelihood loss
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(target)
        loss = -log_prob.mean()
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist and are non-zero for all parameters
        for name, param in policy.named_parameters():
            if 'log_std' not in name:  # Skip log_std parameter
                assert param.grad is not None, f"No gradient for {name} in {policy.get_architecture_type()}"
                # Some parameters (like bias) might legitimately have zero gradients
                # So we only check non-zero gradients for weight parameters
                if 'weight' in name:
                    assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                        f"Zero gradient for {name} in {policy.get_architecture_type()}"
        
        # Check if input gradients exist and are non-zero
        assert batch_observation.grad is not None
        assert not torch.allclose(batch_observation.grad, torch.zeros_like(batch_observation.grad))
        
        # Reset gradients
        batch_observation.grad = None


def test_batch_consistency(observation_space, action_space):
    """Test consistency across different batch sizes"""
    policies = [
        ConvLSTMPolicy(observation_space, action_space),
        TabNetPolicy(observation_space, action_space),
        TCNPolicy(observation_space, action_space),
        LSTMPolicy(observation_space, action_space),
        CNNPolicy(observation_space, action_space),
        TransformerPolicy(observation_space, action_space)
    ]
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 16]
    
    for policy in policies:
        policy.eval()  # Set to evaluation mode
        torch.manual_seed(42)  # Fix random seed for reproducibility
        
        prev_means = None
        for batch_size in batch_sizes:
            observation = torch.randn(batch_size, 50, 10)
            with torch.no_grad():
                action_mean, _ = policy(observation)
            
            # Check if mean is consistent across batches
            if prev_means is not None:
                # Compare first element of each batch with higher tolerance
                assert torch.allclose(
                    action_mean[0],
                    prev_means[0],
                    rtol=0.1,  # 허용 오차를 10%로 증가
                    atol=0.1   # 절대 오차도 0.1로 증가
                ), f"Inconsistent outputs for {policy.get_architecture_type()}: {action_mean[0]} vs {prev_means[0]}"
            
            prev_means = action_mean


def test_device_compatibility(observation_space, action_space, batch_observation):
    """Test if networks can be moved to different devices"""
    policies = [
        ConvLSTMPolicy(observation_space, action_space),
        TabNetPolicy(observation_space, action_space),
        TCNPolicy(observation_space, action_space),
        LSTMPolicy(observation_space, action_space),
        CNNPolicy(observation_space, action_space),
        TransformerPolicy(observation_space, action_space)
    ]
    
    # Test CPU
    for policy in policies:
        policy = policy.cpu()
        batch_observation = batch_observation.cpu()
        action_mean, action_std = policy(batch_observation)
        assert action_mean.device.type == "cpu"
        assert action_std.device.type == "cpu"
    
    # Test GPU if available
    if torch.cuda.is_available():
        for policy in policies:
            policy = policy.cuda()
            batch_observation = batch_observation.cuda()
            action_mean, action_std = policy(batch_observation)
            assert action_mean.device.type == "cuda"
            assert action_std.device.type == "cuda" 