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
    - Supports customizable hidden layer sizes
    """
    
    def __init__(self, observation_space: Box, hidden_sizes=None):
        """Initialize value network.
        
        Args:
            observation_space: Observation space (Box)
            hidden_sizes: List of hidden layer sizes (default: [256, 256])
            
        The network expects inputs to match observation_space.shape:
        - If shape is (features,): input_dim = features
        - If shape is (window_size, features): input_dim = window_size * features
        """
        super().__init__()
        
        # Use default hidden sizes if none provided
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        
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
            
        # Build dynamic network based on hidden_sizes
        layers = []
        prev_size = self.input_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        # Add final output layer
        layers.append(nn.Linear(prev_size, 1))  # Output single value
        
        # Create sequential network
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized ValueNetwork with input_dim={self.input_dim}, "
            f"hidden_sizes={hidden_sizes}, obs_shape={obs_shape}"
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
            x = x.unsqueeze(0)  # 항상 배치 차원 추가
                
        elif x.dim() == 2:
            # 이미 2D 형태이므로 추가 처리 필요 없음
            pass
                
        elif x.dim() == 3:
            # (batch_size, window_size, features)
            b = x.shape[0]
            x = x.reshape(b, -1)  # 평면화
            
        else:
            self.logger.warning(f"Unexpected input dimensions: {x.dim()}D. Attempting to reshape.")
            x = x.reshape(1, -1)  # 일단 평면화 시도
            
        # 차원 조정 (입력 크기 불일치 처리)
        if x.shape[1] != self.input_dim:
            self.logger.warning(f"Input dimension mismatch: got {x.shape[1]}, expected {self.input_dim}")
            
            if x.shape[1] > self.input_dim:
                # 큰 입력은 적응형 풀링으로 처리
                x_reshaped = x.unsqueeze(1)  # [batch, 1, features]
                pool = nn.AdaptiveAvgPool1d(self.input_dim)
                x = pool(x_reshaped).squeeze(1)
            else:
                # 작은 입력은 0으로 패딩
                padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
                
        # 최종 값 계산
        value = self.network(x)
        
        # NaN 확인 및 처리
        if torch.isnan(value).any():
            self.logger.warning("NaN in value network output; replacing with 0.0")
            value = torch.nan_to_num(value, nan=0.0)
            
        return value
        
    def get_architecture_type(self) -> str:
        """Get architecture type.
        
        Returns:
            String identifier for architecture type
        """
        return "mlp"
