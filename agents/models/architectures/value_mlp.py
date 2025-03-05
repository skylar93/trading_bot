import torch
import torch.nn as nn
from gymnasium.spaces import Box
import numpy as np
from agents.models.architectures.base import BaseNetwork
import logging

class ValueNetwork(BaseNetwork):
    """Value network using MLPs.
    
    Features:
    - Handles 1D inputs: (features,) -> (1, features)
    - Handles 2D inputs: 
      a) (batch_size, input_dim) passed through directly
      b) (window_size, features) flattened to (1, window_size*features)
    - Handles 3D inputs: (batch_size, window_size, features) -> (batch_size, window_size*features)
    """
    
    def __init__(self, observation_space: Box):
        """Initialize value network.
        
        Args:
            observation_space: Observation space (Box)
            
        The network expects inputs to match observation_space.shape:
        - If shape is (features,): input_dim = features
        - If shape is (window_size, features): input_dim = window_size * features
        """
        super().__init__()
        
        # Validate observation space type
        if not isinstance(observation_space, Box):
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")
            
        # Calculate input dimension based on observation shape
        obs_shape = observation_space.shape
        if len(obs_shape) == 1:
            # (features,)
            self.input_dim = obs_shape[0]
        elif len(obs_shape) == 2:
            # (window_size, features)
            self.input_dim = obs_shape[0] * obs_shape[1]
        else:
            raise ValueError(
                f"ValueNetwork only supports 1D or 2D Box shapes, got shape={obs_shape}"
            )
            
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output single value
        )
        
        # Initialize weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized ValueNetwork with input_dim={self.input_dim}, obs_shape={obs_shape}"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.
        
        Handles input shapes:
          - 1D: (features,) -> (1, features)
          - 2D: (batch_size, input_dim) or (window_size, features)
          - 3D: (batch_size, window_size, features)
        
        Args:
            x: Input tensor with supported shapes
               
        Returns:
            Value predictions with shape (batch_size, 1)
            
        Raises:
            ValueError: If input shape cannot be interpreted as valid format
        """
        original_shape = x.shape
        
        # 1) Handle NaN values
        if torch.isnan(x).any():
            self.logger.warning("NaN in value network input; replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)
            
        # 2) Handle different input dimensions
        if x.dim() == 1:
            # (features,) -> (1, features)
            if x.shape[0] == self.input_dim:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"ValueNetwork expects input_dim={self.input_dim} but got 1D with shape {x.shape}"
                )
                
        elif x.dim() == 2:
            # (batch_size, input_dim) or (window_size, features)
            if x.shape[1] == self.input_dim:
                # Already (batch_size, input_dim)
                pass
            elif x.shape[0] * x.shape[1] == self.input_dim:
                # Single sample => flatten to (1, window_size*features)
                x = x.reshape(1, -1)
            else:
                raise ValueError(
                    f"ValueNetwork expects input_dim={self.input_dim}, but got 2D {x.shape}. "
                    f"Cannot interpret as (batch_size, input_dim) or single sample (window_size, features)."
                )
                
        elif x.dim() == 3:
            # (batch_size, window_size, features)
            b, w, f = x.shape
            if w * f != self.input_dim:
                raise ValueError(
                    f"ValueNetwork expects window_size*features={self.input_dim}, "
                    f"but got shape {x.shape} => w*f={w*f}."
                )
            x = x.reshape(b, w*f)
            
        else:
            raise ValueError(
                f"ValueNetwork doesn't accept {x.dim()}D input: shape={original_shape}"
            )
            
        # Final shape validation
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"ValueNetwork final check failed: expecting input_dim={self.input_dim}, got shape={x.shape}"
            )
            
        return self.network(x)
        
    def get_architecture_type(self) -> str:
        """Get architecture type.
        
        Returns:
            String identifier for architecture type
        """
        return "mlp"
