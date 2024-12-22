import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork


class LSTMPolicy(BaseNetwork):
    """LSTM-based policy network for handling sequential data"""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize LSTM policy network

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            hidden_size: Size of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        # Get input and output dimensions
        if hasattr(observation_space, "shape"):
            # For (window_size, features) shape
            self.seq_len = observation_space.shape[0]
            input_dim = observation_space.shape[1]
        else:
            raise ValueError("Observation space must have shape attribute")

        if hasattr(action_space, "shape"):
            output_dim = int(np.prod(action_space.shape))
        else:
            output_dim = action_space.n

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Output layers for mean and standard deviation
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_dim),
        )
        
        self.log_std = nn.Parameter(torch.zeros(output_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'log_std' in name:
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

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq, hidden)

        # Apply attention
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # Self-attention
        
        # Use the last output with attention
        combined = lstm_out[:, -1] + attn_out[:, -1]  # Shape: (batch, hidden)

        # Get mean and standard deviation
        action_mean = self.mean_net(combined)
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
        return "lstm" 