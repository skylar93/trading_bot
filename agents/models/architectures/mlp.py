import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork
import logging
import gymnasium as gym
from gymnasium.spaces import Box


class PolicyNetwork(BaseNetwork):
    """Policy network for continuous action space using MLPs.
    
    Features:
    - Handles 1D inputs as (features,) -> (1, features)
    - Handles 2D inputs in two ways:
      a) (batch_size, input_size) passed through directly
      b) (window_size, features) flattened to (1, window_size*features)
    - Handles 3D inputs as (batch_size, window_size, features) -> (batch_size, window_size*features)
    """
    
    def __init__(self, observation_space: Box, action_space: Box):
        """Initialize policy network.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            
        The network expects inputs to match observation_space.shape:
        - If shape is (features,): input_size = features
        - If shape is (window_size, features): input_size = window_size * features
        """
        super().__init__()
        
        # Store spaces
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Calculate input size
        if len(observation_space.shape) == 1:  # (features,)
            self.input_size = observation_space.shape[0]
        elif len(observation_space.shape) == 2:  # (window_size, features)
            self.input_size = observation_space.shape[0] * observation_space.shape[1]
        else:
            raise ValueError(
                f"PolicyNetwork only supports 1D or 2D Box shapes, got shape={observation_space.shape}"
            )
            
        # Network architecture
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(256, action_space.shape[0])
        self.std_head = nn.Linear(256, action_space.shape[0])
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store latest outputs
        self._mean = None
        self._std = None
        
        self.logger.info(
            f"Initialized PolicyNetwork with input_size={self.input_size}, "
            f"action_size={action_space.shape[0]}"
        )
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through network.
        
        Handles input shapes:
          - 1D: (features,) -> (1, features)
          - 2D: (batch_size, input_size) or (window_size, features)
          - 3D: (batch_size, window_size, features)
        
        Args:
            x: Input tensor with supported shapes
               
        Returns:
            Tuple of (action_mean, action_std)
            
        Raises:
            ValueError: If input shape cannot be interpreted as valid format
        """
        # Handle NaN values
        if torch.isnan(x).any():
            self.logger.warning("NaN in policy network input; replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)
            
        original_shape = x.shape
        
        # Handle different input dimensions
        if x.dim() == 1:
            # (features,) -> (1, features)
            if x.shape[0] == self.input_size:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"PolicyNetwork expects input_size={self.input_size} but got 1D with shape {x.shape}"
                )
                
        elif x.dim() == 2:
            # (batch_size, input_size) or (window_size, features)
            if x.shape[1] == self.input_size:
                # Already (batch_size, input_size)
                pass
            elif x.shape[0] * x.shape[1] == self.input_size:
                # Single sample => flatten to (1, window_size*features)
                x = x.reshape(1, -1)
            else:
                raise ValueError(
                    f"PolicyNetwork expects input_size={self.input_size}, but got 2D {x.shape}. "
                    f"Cannot interpret as (batch_size, input_size) or single sample (window_size, features)."
                )
                
        elif x.dim() == 3:
            # (batch_size, window_size, features)
            b, w, f = x.shape
            if w * f != self.input_size:
                raise ValueError(
                    f"PolicyNetwork expects window_size*features={self.input_size}, "
                    f"but got shape {x.shape} => w*f={w*f}."
                )
            x = x.reshape(b, w*f)
            
        else:
            raise ValueError(
                f"PolicyNetwork doesn't accept {x.dim()}D input: shape={original_shape}"
            )
            
        # Final shape validation
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"PolicyNetwork final check failed: expecting input_size={self.input_size}, got shape={x.shape}"
            )
            
        # Forward pass through shared layers
        features = self.shared(x)
        
        # Get action distribution parameters
        # Use tanh for [-1, 1] range to match environment action space
        self._mean = torch.tanh(self.mean_head(features))  # [-1, 1] range
        # Use softplus for std to ensure positive values, scaled for stable training
        self._std = torch.nn.functional.softplus(self.std_head(features)) * 0.1 + 1e-6
        
        return self._mean, self._std
        
    @property
    def mean(self) -> torch.Tensor:
        """Get latest action mean."""
        if self._mean is None:
            raise ValueError("No forward pass has been performed yet")
        return self._mean
        
    @property
    def std(self) -> torch.Tensor:
        """Get latest action standard deviation."""
        if self._std is None:
            raise ValueError("No forward pass has been performed yet")
        return self._std
        
    def save(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path))

    @property
    def network_type(self):
        """Get network type"""
        return "mlp"

    def get_architecture_type(self) -> str:
        """Get architecture type.
        
        Returns:
            String identifier for architecture type
        """
        return "mlp"


class ValueNetwork(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        
        # Calculate input dimension based on observation space
        if isinstance(observation_space, gym.spaces.Box):
            # For batched observations: (batch_size, window_size, features)
            input_dim = observation_space.shape[0] * observation_space.shape[1]  # window_size * features
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        # Reshape input: (batch_size, window_size, features) -> (batch_size, window_size * features)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.net(x)
