import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork


class TransformerPolicy(BaseNetwork):
    """Transformer-based policy network for handling sequential data"""

    def __init__(
        self,
        observation_space,
        action_space,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "ReLU",
    ):
        """Initialize Transformer policy network

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            d_model: Dimension of model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
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

        # Input embedding with scaled initialization
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            getattr(nn, activation)(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation.lower(),
            batch_first=True,
            norm_first=False  # Changed to improve gradient flow
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output layers for mean and standard deviation
        self.mean_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )

        self.log_std = nn.Parameter(torch.ones(output_dim) * -0.5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with special attention to Transformer components"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'input_embedding.0' in name:  # Input projection
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                elif 'transformer' in name and len(param.shape) >= 2:
                    # Transformer weights initialization
                    nn.init.xavier_uniform_(param, gain=1/np.sqrt(2))
                elif 'mean_net' in name and len(param.shape) >= 2:
                    if 'mean_net.0' in name:  # First layer
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                    else:  # Output layer
                        nn.init.orthogonal_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

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

        # Input embedding with residual connection
        embedded = self.input_embedding(x)
        x = embedded + x if x.size(-1) == embedded.size(-1) else embedded

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer forward pass with gradient scaling
        x = x * np.sqrt(x.size(-1))  # Scale input to improve gradient flow
        transformer_out = self.transformer_encoder(x)

        # Global pooling (instead of just taking the last token)
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)

        # Get mean and standard deviation
        action_mean = self.mean_net(pooled)
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
        return "transformer"


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding

        Args:
            d_model: Dimension of model
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 