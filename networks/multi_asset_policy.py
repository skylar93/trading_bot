"""
Multi-asset policy network for reinforcement learning trading agents.

This module provides neural network architectures specifically designed to process
observations from multiple assets simultaneously and output trading decisions for each.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Type, Union
from gymnasium.spaces import Box, Space

class MultiAssetLSTMPolicy(nn.Module):
    """
    LSTM-based policy network for handling multiple assets simultaneously.
    
    Features:
    - Processes time series data for multiple assets
    - Uses shared feature extractor followed by asset-specific heads
    - Outputs actions for all assets in the portfolio
    
    Implementation Notes:
    - Input shape: (batch_size, window_size, n_assets * n_features)
    - Output shape: (batch_size, n_assets * action_dim)
    - Uses LSTM for temporal processing followed by MLP for action prediction
    
    Recent Changes:
    - Initial implementation with shared LSTM encoder
    - Added asset-specific heads for action generation
    - Implemented flexible action space handling for different trading strategies
    """
    
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        n_assets: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        window_size: int = 30,
        features_per_asset: int = 8,
        action_dim_per_asset: int = 1,
        dropout: float = 0.1,
        activation: Type[nn.Module] = nn.ReLU,
        min_std: float = 1e-6,
        max_std: float = 1.0,
    ):
        """
        Initialize the multi-asset LSTM policy network.
        
        Args:
            observation_space: Gym space representing observations
            action_space: Gym space representing actions
            n_assets: Number of assets to manage
            hidden_size: Size of LSTM and linear layer features
            lstm_layers: Number of LSTM layers
            window_size: Number of time steps in each observation
            features_per_asset: Number of features per asset
            action_dim_per_asset: Action dimensions per asset
            dropout: Dropout probability
            activation: Activation function
            min_std: Minimum standard deviation for action distribution
            max_std: Maximum standard deviation for action distribution
        """
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_assets = n_assets
        self.window_size = window_size
        self.features_per_asset = features_per_asset
        self.total_features = n_assets * features_per_asset
        self.action_dim_per_asset = action_dim_per_asset
        self.total_action_dim = n_assets * action_dim_per_asset
        self.min_std = min_std
        self.max_std = max_std
        
        # LSTM for processing temporal data
        self.lstm = nn.LSTM(
            input_size=self.total_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Shared feature extractor after LSTM
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Dropout(dropout)
        )
        
        # Asset-specific action heads
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                activation(),
                nn.Linear(hidden_size // 2, action_dim_per_asset * 2)  # Mean and log_std
            )
            for _ in range(n_assets)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights properly for stable training."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, n_assets * n_features)
            
        Returns:
            Tuple containing:
                - action_means: Mean actions for all assets
                - action_stds: Standard deviations for action distributions
        """
        batch_size = x.shape[0]
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        lstm_features = lstm_out[:, -1, :]
        
        # Process through shared layers
        shared_features = self.shared_layers(lstm_features)
        
        # Process through asset-specific heads
        action_outputs = []
        for i in range(self.n_assets):
            asset_output = self.action_heads[i](shared_features)
            action_outputs.append(asset_output)
        
        # Combine all asset outputs
        combined_output = torch.cat(action_outputs, dim=1)
        
        # Split into mean and log_std
        means, log_stds = torch.chunk(combined_output, 2, dim=1)
        
        # Apply tanh to bound means to action space
        means = torch.tanh(means)
        
        # Process log_stds with softplus and clamp
        stds = F.softplus(log_stds) + self.min_std
        stds = torch.clamp(stds, self.min_std, self.max_std)
        
        return means, stds
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get actions for given observations.
        
        Args:
            x: Observation tensor
            deterministic: If True, return deterministic actions (means)
            
        Returns:
            Actions tensor
        """
        means, stds = self.forward(x)
        
        if deterministic:
            return means
        
        # Sample from Normal distribution
        normal = torch.distributions.Normal(means, stds)
        actions = normal.sample()
        
        # Clip actions to valid range
        actions = torch.clamp(actions, -1.0, 1.0)
        
        return actions

class MultiAssetAttentionPolicy(nn.Module):
    """
    Attention-based policy network for handling multiple assets simultaneously.
    
    Features:
    - Uses self-attention to capture relationships between assets
    - Processes time series data with transformer architecture
    - Outputs coordinated actions for all assets in the portfolio
    
    Implementation Notes:
    - Reshapes input to consider assets as sequence elements for cross-asset attention
    - Uses positional encoding for temporal information
    - Outputs action distribution parameters for each asset
    """
    
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        n_assets: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        window_size: int = 30,
        features_per_asset: int = 8,
        action_dim_per_asset: int = 1,
        dropout: float = 0.1,
        activation: Type[nn.Module] = nn.ReLU,
        min_std: float = 1e-6,
        max_std: float = 1.0,
    ):
        """Initialize the multi-asset attention policy network."""
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_assets = n_assets
        self.window_size = window_size
        self.features_per_asset = features_per_asset
        self.action_dim_per_asset = action_dim_per_asset
        self.total_action_dim = n_assets * action_dim_per_asset
        self.hidden_size = hidden_size
        self.min_std = min_std
        self.max_std = max_std
        
        # Feature embedding for each asset
        self.feature_embedding = nn.Linear(features_per_asset, hidden_size)
        
        # Positional encoding for temporal information
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(window_size, hidden_size)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Asset attention mechanism (to learn relationships between assets)
        self.asset_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Asset-specific action heads
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                activation(),
                nn.Linear(hidden_size // 2, action_dim_per_asset * 2)  # Mean and log_std
            )
            for _ in range(n_assets)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding for transformer."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _init_weights(self, module):
        """Initialize network weights properly for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, n_assets * features_per_asset)
            
        Returns:
            Tuple containing:
                - action_means: Mean actions for all assets
                - action_stds: Standard deviations for action distributions
        """
        batch_size = x.shape[0]
        
        # Reshape input to (batch_size, window_size, n_assets, features_per_asset)
        x_reshaped = x.reshape(batch_size, self.window_size, self.n_assets, self.features_per_asset)
        
        # Process each asset's features through embedding
        embeddings = []
        for i in range(self.n_assets):
            asset_features = x_reshaped[:, :, i, :]
            asset_embedding = self.feature_embedding(asset_features)
            embeddings.append(asset_embedding)
        
        # Stack asset embeddings to (batch_size, n_assets, window_size, hidden_size)
        asset_embeddings = torch.stack(embeddings, dim=1)
        
        # Add positional encoding
        for i in range(self.n_assets):
            asset_embeddings[:, i, :, :] += self.positional_encoding
        
        # Process each asset's time series with transformer
        asset_encodings = []
        for i in range(self.n_assets):
            asset_seq = asset_embeddings[:, i, :, :]
            asset_encoded = self.transformer_encoder(asset_seq)
            # Use the last time step as the asset representation
            asset_encodings.append(asset_encoded[:, -1, :])
        
        # Stack and reshape for asset attention (batch_size, n_assets, hidden_size)
        asset_features = torch.stack(asset_encodings, dim=1)
        
        # Apply cross-asset attention
        attn_output, _ = self.asset_attention(
            asset_features, asset_features, asset_features
        )
        
        # Process through asset-specific heads
        action_outputs = []
        for i in range(self.n_assets):
            asset_output = self.action_heads[i](attn_output[:, i, :])
            action_outputs.append(asset_output)
        
        # Combine all asset outputs
        combined_output = torch.cat(action_outputs, dim=1)
        
        # Split into mean and log_std
        means, log_stds = torch.chunk(combined_output, 2, dim=1)
        
        # Apply tanh to bound means to action space
        means = torch.tanh(means)
        
        # Process log_stds with softplus and clamp
        stds = F.softplus(log_stds) + self.min_std
        stds = torch.clamp(stds, self.min_std, self.max_std)
        
        return means, stds
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get actions for given observations."""
        means, stds = self.forward(x)
        
        if deterministic:
            return means
        
        # Sample from Normal distribution
        normal = torch.distributions.Normal(means, stds)
        actions = normal.sample()
        
        # Clip actions to valid range
        actions = torch.clamp(actions, -1.0, 1.0)
        
        return actions 