import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, Any, Tuple, List, Optional, Union, Type
from gymnasium.spaces import Box
from agents.models.architectures.base import BaseNetwork


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
            x = x.unsqueeze(0)  # 항상 배치 차원 추가
            
        elif x.dim() == 2:
            # 이미 2D 형태이므로 추가 처리 필요 없음
            pass
                
        elif x.dim() == 3:
            # (batch_size, window_size, features)
            b = x.shape[0]
            orig_shape = x.shape
            x = x.reshape(b, -1)
            self.logger.debug(f"Reshaped 3D input from {orig_shape} to {x.shape}")
            
        else:
            self.logger.warning(f"Unexpected input dimensions: {x.dim()}D. Attempting to reshape.")
            x = x.reshape(1, -1)  # 일단 평면화 시도
            
        # 차원 조정 (입력 크기 불일치 처리)
        if x.shape[1] != self.input_size:
            self.logger.warning(
                f"Input dimension mismatch: got {x.shape[1]}, expected {self.input_size}. "
                f"Original shape: {original_shape}. Applying adaptive sizing."
            )
            
            if x.shape[1] > self.input_size:
                # 큰 입력은 적응형 풀링으로 처리
                x_reshaped = x.unsqueeze(1)  # [batch, 1, features]
                pool = nn.AdaptiveAvgPool1d(self.input_size)
                x = pool(x_reshaped).squeeze(1)
                self.logger.debug(f"Applied pooling to reduce input size to {x.shape}")
            else:
                # 작은 입력은 0으로 패딩
                padding = torch.zeros(x.shape[0], self.input_size - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
                self.logger.debug(f"Applied padding to increase input size to {x.shape}")
            
        # Forward pass through shared layers
        features = self.shared(x)
        
        # Get action distribution parameters
        self._mean = torch.sigmoid(self.mean_head(features))  # Ensure [0, 1] range
        
        # Fix: Increase minimum standard deviation to prevent negative entropy
        raw_std = torch.sigmoid(self.std_head(features))
        min_std = 0.1  # Increased from 0.01 to prevent negative entropy
        max_std = 0.5  # Set a reasonable maximum
        self._std = min_std + raw_std * (max_std - min_std)
        
        # NaN 확인 및 처리
        if torch.isnan(self._mean).any() or torch.isnan(self._std).any():
            self.logger.warning("NaN in policy network output; replacing with safe values")
            self._mean = torch.nan_to_num(self._mean, nan=0.5)
            self._std = torch.nan_to_num(self._std, nan=0.1)
            
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
            self.input_dim = observation_space.shape[0] * observation_space.shape[1]  # window_size * features
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")
            
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
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
        # Record original shape for debugging
        original_shape = x.shape
        
        # Handle different input dimensions
        if len(x.shape) == 1:  # Single vector
            x = x.unsqueeze(0)  # Add batch dimension
            flattened = True
        else:
            flattened = False
            
        # Reshape input: (batch_size, window_size, features) -> (batch_size, window_size * features)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Check for dimension mismatch and handle it
        if x.size(1) != self.input_dim:
            logging.warning(
                f"ValueNetwork dimension mismatch: got {x.size(1)}, expected {self.input_dim}. "
                f"Original shape: {original_shape}. Reshaping..."
            )
            
            if x.size(1) > self.input_dim:
                # For larger inputs, use adaptive pooling
                if x.size(1) > self.input_dim * 1.5:
                    # Reshape for 1D adaptive pooling
                    x_reshaped = x.unsqueeze(1)  # [batch, 1, features]
                    pool = nn.AdaptiveAvgPool1d(self.input_dim)
                    x = pool(x_reshaped).squeeze(1)
                else:
                    # Truncate to expected size
                    x = x[:, :self.input_dim]
            else:
                # Pad with zeros if input is smaller
                padding = torch.zeros(batch_size, self.input_dim - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
                
        # Check for NaN values
        if torch.isnan(x).any():
            logging.warning("NaN values in ValueNetwork input. Replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)
            
        value = self.net(x)
        
        # If input was a single vector, return single value
        if flattened:
            value = value.squeeze(0)
            
        return value
