"""Custom SB3 feature extractors for trading data."""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TradingWindowExtractor(BaseFeaturesExtractor):
    """
    Conv1D feature extractor for 2D trading observation windows.

    Input: (batch, window_size, n_features)
    Output: flat feature vector of size features_dim

    Architecture: Conv1D(n_features→64) → ReLU → Conv1D(64→128) → ReLU → Flatten → Linear
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # observation_space.shape: (window_size, n_features)
        if len(observation_space.shape) != 2:
            raise ValueError(
                f"TradingWindowExtractor expects 2D obs (window_size, n_features), "
                f"got shape {observation_space.shape}"
            )
        n_features = observation_space.shape[1]  # channels for Conv1D

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)  # (1, window_size, n_features)
            sample = sample.transpose(1, 2)  # (1, n_features, window_size)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, window_size, n_features)
        # Conv1D expects: (batch, channels, length) = (batch, n_features, window_size)
        x = observations.transpose(1, 2)
        return self.linear(self.cnn(x))


class LSTMTradingExtractor(BaseFeaturesExtractor):
    """
    LSTM feature extractor for 2D trading observation windows.

    Input: (batch, window_size, n_features)
    Output: flat feature vector of size features_dim (last LSTM hidden state)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 2:
            raise ValueError(
                f"LSTMTradingExtractor expects 2D obs (window_size, n_features), "
                f"got shape {observation_space.shape}"
            )
        n_features = observation_space.shape[1]

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, window_size, n_features)
        lstm_out, _ = self.lstm(observations)
        last_hidden = lstm_out[:, -1, :]  # Take last timestep
        return self.linear(last_hidden)
