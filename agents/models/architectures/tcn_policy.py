import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from agents.models.architectures.base import BaseNetwork


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolutions"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout probability
        """
        super().__init__()
        
        # 가중치 초기화를 위한 시드 고정
        torch.manual_seed(42)
        
        # First dilated convolution with gradient scaling
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode='replicate',
            bias=True
        )
        
        # Layer normalization for better stability
        self.norm1 = nn.GroupNorm(1, n_outputs, eps=1e-5)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated convolution with gradient scaling
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode='replicate',
            bias=True
        )
        
        # Layer normalization
        self.norm2 = nn.GroupNorm(1, n_outputs, eps=1e-5)
        
        # Dropout
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection if input and output dimensions differ
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, 1, bias=True),
            nn.GroupNorm(1, n_outputs, eps=1e-5)
        ) if n_inputs != n_outputs else None
        
        # Activation function
        self.activation = nn.GELU()
        
        # Save parameters for sequence length adjustment
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient scaling

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, channels, seq_len)
        """
        # Save original sequence length
        seq_len = x.size(2)
        
        # Save original for residual
        identity = x if self.downsample is None else self.downsample(x)
        
        # First conv block with gradient scaling
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        # Second conv block with gradient scaling
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        # Adjust sequence length if needed
        if out.size(2) != seq_len:
            if out.size(2) > seq_len:
                out = out[:, :, :seq_len]
            else:
                pad_size = seq_len - out.size(2)
                out = torch.nn.functional.pad(out, (0, pad_size), mode='replicate')
        
        # Residual connection with gradient scaling
        out = out + identity
        out = out / np.sqrt(2)
        
        return out


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with improved gradient flow"""

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """Initialize TCN

        Args:
            num_inputs: Number of input channels
            num_channels: Number of channels in each layer
            kernel_size: Kernel size for all convolutions
            dropout: Dropout probability
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Input projection with skip connection
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_inputs, num_channels[0], 1),
            nn.GroupNorm(min(8, num_channels[0]), num_channels[0]),
            nn.GELU()
        )
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            
            # Calculate padding for causal convolution
            padding = (kernel_size - 1) * dilation
            
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout
                )
            )
        
        self.network = nn.ModuleList(layers)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient scaling and residual connections

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, channels, seq_len)
        """
        # Initial projection
        out = self.input_proj(x)
        
        # Process through temporal blocks
        for layer in self.network:
            out = layer(out)
        
        return out


class TCNPolicy(BaseNetwork):
    """TCN-based policy network for handling sequential data"""

    def __init__(
        self,
        observation_space,
        action_space,
        num_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize TCN policy network

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            num_channels: Number of channels in each TCN layer
            kernel_size: Kernel size for all convolutions
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

        # Create TCN network with memory-efficient channel sizes
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Global pooling with learnable weights
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.GELU()
        )

        # Output layers for mean with proper initialization
        self.mean_net = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.GELU(),
            nn.Linear(num_channels[-1], output_dim)
        )

        # Initialize log_std as a parameter
        self.log_std = nn.Parameter(torch.ones(output_dim) * -0.5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'log_std' in name:
                nn.init.constant_(param, -0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network with gradient scaling

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

        # Transpose for TCN (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Process in chunks for memory efficiency
        chunk_size = min(50, x.size(0))  # Process 50 samples at a time
        outputs = []
        
        for i in range(0, x.size(0), chunk_size):
            chunk = x[i:i + chunk_size]
            # Forward through TCN
            chunk_output = self.tcn(chunk)
            # Apply global pooling
            chunk_pooled = self.global_pool(chunk_output)
            outputs.append(chunk_pooled)

        # Combine chunk outputs
        pooled = torch.cat(outputs, dim=0)

        # Get action mean with gradient scaling
        action_mean = self.mean_net(pooled)
        action_mean = action_mean / np.sqrt(2)  # Scale gradients

        # Get action standard deviation
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
        return "tcn" 