import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork


class CNNPolicy(BaseNetwork):
    """1D CNN-based policy network for handling sequential data"""

    def __init__(
        self,
        observation_space,
        action_space,
        base_filters: int = 64,
        kernel_sizes: list = [3, 5, 7],
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize CNN policy network

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            base_filters: Base number of convolutional filters
            kernel_sizes: List of kernel sizes for parallel convolutions
            n_blocks: Number of residual blocks
            dropout: Dropout probability
        """
        super().__init__()

        # Get input and output dimensions
        if hasattr(observation_space, "shape"):
            self.seq_len = observation_space.shape[0]
            input_dim = observation_space.shape[1]
        else:
            raise ValueError("Observation space must have shape attribute")

        if hasattr(action_space, "shape"):
            output_dim = int(np.prod(action_space.shape))
        else:
            output_dim = action_space.n

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, base_filters),
            nn.ReLU(),
            nn.LayerNorm(base_filters)
        )

        # Parallel convolution layers with proper initialization
        self.parallel_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=base_filters,
                    out_channels=base_filters,
                    kernel_size=k,
                    padding='same',
                    padding_mode='replicate'
                ),
                nn.ReLU(),
                nn.LayerNorm([base_filters, self.seq_len])
            ) for k in kernel_sizes
        ])

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                channels=base_filters * len(kernel_sizes),
                kernel_size=5,
                dropout=dropout
            ) for _ in range(n_blocks)
        ])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output layers for mean with proper initialization
        hidden_dim = base_filters * len(kernel_sizes)
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize log_std as a parameter
        self.log_std = nn.Parameter(torch.ones(output_dim) * -0.5)

        # Initialize weights with proper scaling
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if 'conv' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif len(param.shape) >= 2:
                    if 'mean_net' in name:
                        if name.endswith('.0.weight'):  # First layer
                            nn.init.orthogonal_(param, gain=np.sqrt(2))
                        else:  # Output layer
                            nn.init.orthogonal_(param, gain=0.01)
                    else:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'log_std' in name:
                nn.init.constant_(param, -0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network

        Args:
            x: Input tensor of shape (batch_size, seq_len, features)

        Returns:
            Tuple of (action mean, action standard deviation)
        """
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # Project input
        x = self.input_proj(x)  # Shape: (batch, seq, base_filters)
        x = x.transpose(1, 2)  # Shape: (batch, base_filters, seq)

        # Parallel convolutions
        conv_outputs = []
        for conv in self.parallel_convs:
            conv_outputs.append(conv(x))
        x = torch.cat(conv_outputs, dim=1)  # Shape: (batch, base_filters * len(kernel_sizes), seq)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Global pooling
        x = self.global_pool(x)  # Shape: (batch, channels, 1)
        x = x.squeeze(-1)  # Shape: (batch, channels)

        # Get mean and standard deviation
        action_mean = self.mean_net(x)
        action_std = torch.exp(self.log_std)

        # Expand std to match batch size if needed
        if len(action_mean.shape) > 1:
            action_std = action_std.expand_as(action_mean)

        return action_mean, action_std

    def get_architecture_type(self) -> str:
        """Get architecture type

        Returns:
            String identifier for architecture type
        """
        return "cnn"


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions"""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        """Initialize residual block

        Args:
            channels: Number of input/output channels
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding='same',
            padding_mode='replicate',
            dilation=1
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding='same',
            padding_mode='replicate',
            dilation=2
        )

        self.norm1 = nn.LayerNorm([channels, 50])  # 50 is seq_len
        self.norm2 = nn.LayerNorm([channels, 50])
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of same shape
        """
        # First convolution block
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.dropout(x)

        # Second convolution block
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.dropout(x)

        # Residual connection with scaling
        return (x + residual) / np.sqrt(2)