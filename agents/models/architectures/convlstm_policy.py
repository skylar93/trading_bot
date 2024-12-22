import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for processing sequential data with spatial structure"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int):
        """Initialize ConvLSTM cell

        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels
            kernel_size: Size of the convolutional kernel
        """
        super().__init__()

        # 가중치 초기화를 위한 시드 고정
        torch.manual_seed(42)

        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Separate convolutions for better gradient flow
        self.conv_ih = nn.Conv1d(
            in_channels=input_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='replicate',
            bias=True
        )
        
        self.conv_hh = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='replicate',
            bias=True
        )

        # Layer normalization for better stability
        self.norm_ih = nn.LayerNorm(4 * hidden_dim, eps=1e-5)
        self.norm_hh = nn.LayerNorm(4 * hidden_dim, eps=1e-5)

        # Initialize parameters after creation
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                if 'conv' in name:
                    # LSTM bias initialization
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)  # forget gate bias = 1
                    param.data[:start].fill_(0.)     # input gate bias = 0
                    param.data[end:].fill_(0.)       # output gate bias = 0
                else:
                    nn.init.zeros_(param)
            elif 'norm' in name and 'weight' in name:
                nn.init.ones_(param)
            elif 'norm' in name and 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through ConvLSTM cell

        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            hidden: Tuple of (hidden state, cell state)

        Returns:
            Tuple of output and new hidden state
        """
        h_prev, c_prev = hidden
        
        # Process input and hidden state separately
        gates_ih = self.conv_ih(x)
        gates_ih = gates_ih.transpose(1, 2)  # (batch, seq, channels)
        gates_ih = self.norm_ih(gates_ih)
        gates_ih = gates_ih.transpose(1, 2)  # (batch, channels, seq)
        
        gates_hh = self.conv_hh(h_prev)
        gates_hh = gates_hh.transpose(1, 2)  # (batch, seq, channels)
        gates_hh = self.norm_hh(gates_hh)
        gates_hh = gates_hh.transpose(1, 2)  # (batch, channels, seq)
        
        # Combine gates
        gates = gates_ih + gates_hh
        
        # Split gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply non-linearities with gradient-friendly functions
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1.0)  # Add 1.0 for better gradient flow
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Update cell state with gradient scaling
        c_next = (f * c_prev + i * g).clamp(-1, 1)  # Clamp to prevent exploding gradients
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class ConvLSTMPolicy(BaseNetwork):
    """ConvLSTM-based policy network for handling sequential data"""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize ConvLSTM policy network

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            hidden_dim: Number of hidden channels
            kernel_size: Size of the convolutional kernel
            num_layers: Number of ConvLSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        # 가중치 초기화를 위한 시드 고정
        torch.manual_seed(42)

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

        # Input projection with skip connection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
            nn.GELU()
        )

        # ConvLSTM layers with residual connections
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        
        # Global pooling with learnable weights
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
            nn.GELU()
        )

        # Output layers for mean with proper initialization
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
            nn.GELU(),
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
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'norm' in name and 'weight' in name:
                nn.init.ones_(param)
            elif 'norm' in name and 'bias' in name:
                nn.init.zeros_(param)
            elif 'log_std' in name:
                nn.init.constant_(param, -0.5)

    def _init_hidden(self, batch_size: int, device: torch.device) -> list:
        """Initialize hidden states with fixed values

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            List of initial hidden states for each layer
        """
        hidden_states = []
        for layer in self.convlstm_layers:
            h = torch.zeros(batch_size, layer.hidden_dim, 1, device=device)
            c = torch.zeros(batch_size, layer.hidden_dim, 1, device=device)
            hidden_states.append((h, c))
        return hidden_states

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        device = x.device

        # Project input with residual connection
        x_proj = self.input_proj(x)  # Shape: (batch, seq, hidden)
        x_proj = x_proj.transpose(1, 2)  # Shape: (batch, hidden, seq)

        # Initialize hidden states
        hidden_states = self._init_hidden(batch_size, device)

        # Process sequence with residual connections
        layer_outputs = []
        for i, convlstm_cell in enumerate(self.convlstm_layers):
            output_sequence = []
            h, c = hidden_states[i]
            
            # Process each time step
            for t in range(self.seq_len):
                current_input = x_proj[:, :, t:t+1]
                h, (h, c) = convlstm_cell(current_input, (h, c))
                output_sequence.append(h)
            
            # Concatenate outputs and apply dropout
            layer_output = torch.cat(output_sequence, dim=2)
            layer_output = self.dropout(layer_output)
            
            # Add residual connection if dimensions match
            if layer_output.size(1) == x_proj.size(1):
                layer_output = layer_output + x_proj
                layer_output = layer_output / np.sqrt(2)  # Gradient scaling
            
            # Update input for next layer
            x_proj = layer_output
            layer_outputs.append(layer_output)

        # Global pooling
        pooled = self.global_pool(layer_outputs[-1])

        # Get action mean and standard deviation
        action_mean = self.mean_net(pooled)
        action_std = torch.exp(self.log_std)

        # Expand std to match batch size if needed
        if len(action_mean.shape) > 1:
            action_std = action_std.expand_as(action_mean)

        return action_mean, action_std

    def get_architecture_type(self) -> str:
        return "convlstm"