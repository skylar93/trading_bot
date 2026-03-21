"""
GTrXL (Gated Transformer-XL) feature extractor for Stable-Baselines3.

Reference: Parisotto et al., "Stabilizing Transformers for Reinforcement Learning" (2019)
           Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (2019)

Week 21 implementation.
"""

import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUGate(nn.Module):
    """GRU-style gating to replace residual connections in transformer layers.

    Stabilises training by preventing gradient explosion/vanishing through
    identity-initialised forget gates (z bias = -2).

    Gate equations (Parisotto et al. 2019):
        r = σ(W_r [x; y])
        z = σ(W_z [x; y] + b_z)      ← b_z initialised to -2 → near-identity at init
        h = tanh(W_h [x; r·y])
        out = (1 - z) · x + z · h
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W_r = nn.Linear(2 * d_model, d_model, bias=True)
        self.W_z = nn.Linear(2 * d_model, d_model, bias=True)
        self.W_h = nn.Linear(2 * d_model, d_model, bias=True)
        # Initialise z gate bias to -2 → gates start near 0 → near-identity
        nn.init.constant_(self.W_z.bias, -2.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: residual (input to sublayer), shape (..., d_model)
            y: sublayer output (attention or FFN), shape (..., d_model)
        Returns:
            gated output, shape (..., d_model)
        """
        xcat = torch.cat([x, y], dim=-1)
        r = torch.sigmoid(self.W_r(xcat))
        z = torch.sigmoid(self.W_z(xcat))
        h = torch.tanh(self.W_h(torch.cat([x, r * y], dim=-1)))
        return (1.0 - z) * x + z * h


class GTrXLLayer(nn.Module):
    """Single GTrXL layer: pre-norm multi-head attention + GRU gate + pre-norm FFN + GRU gate."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate1 = GRUGate(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.gate2 = GRUGate(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            memory: optional Transformer-XL memory segment, (batch, mem_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Prepend memory to key/value for extended context
        normed_x = self.norm1(x)
        if memory is not None:
            kv = torch.cat([memory, normed_x], dim=1)
        else:
            kv = normed_x
        attn_out, _ = self.attn(normed_x, kv, kv)
        x = self.gate1(x, attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = self.gate2(x, ffn_out)
        return x


class GTrXLExtractor(BaseFeaturesExtractor):
    """Gated Transformer-XL feature extractor for SB3 policies.

    Processes sequential observations of shape (window_size, n_features) and
    returns a fixed-size feature vector of length ``features_dim``.

    Architecture:
        1. Linear input projection: n_features → d_model
        2. Sinusoidal positional encoding
        3. n_layers of GTrXLLayer (pre-norm + GRU gating + optional XL memory)
        4. Last-position readout: d_model → features_dim

    Args:
        observation_space: Must have shape (window_size, n_features).
        features_dim: Output feature dimension.
        n_layers: Number of GTrXL layers.
        d_model: Internal transformer dimension.
        n_heads: Number of attention heads (d_model must be divisible by n_heads).
        memory_len: Number of past hidden states kept as Transformer-XL memory.
                    Set to 0 to disable XL memory (standard transformer).
        gate_type: Gating mechanism — only "gru" is supported.
        dropout: Dropout probability inside attention.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        memory_len: int = 64,
        gate_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        if len(obs_shape) != 2:
            raise ValueError(
                f"GTrXLExtractor expects 2-D observations (window_size, n_features), "
                f"got shape {obs_shape}"
            )
        _window_size, n_features = obs_shape

        if gate_type != "gru":
            raise ValueError(f"Unsupported gate_type '{gate_type}'. Only 'gru' is supported.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.memory_len = memory_len

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Sinusoidal positional encoding (covers window + potential memory)
        max_len = _window_size + memory_len + 1
        self._build_pos_encoding(max_len, d_model)

        # GTrXL layers
        ffn_dim = 4 * d_model
        self.layers = nn.ModuleList([
            GTrXLLayer(d_model, n_heads, ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection from last token
        self.output_proj = nn.Linear(d_model, features_dim)

        # Transformer-XL memory (list of per-layer hidden states)
        # Stored as a plain Python list of tensors; reset on demand.
        self._memories: list[torch.Tensor | None] = [None] * n_layers
        self._n_layers = n_layers

    def _build_pos_encoding(self, max_len: int, d_model: int) -> None:
        """Register sinusoidal positional encoding buffer."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) for broadcasting over batch
        self.register_buffer("pos_encoding", pe.unsqueeze(0))

    def reset_memory(self) -> None:
        """Clear Transformer-XL memory segments (call at episode start)."""
        self._memories = [None] * self._n_layers

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch, window_size, n_features)
        Returns:
            features: (batch, features_dim)
        """
        # observations may arrive as float32 (SB3 normalises externally)
        x = self.input_proj(observations)  # (B, T, d_model)

        # Add positional encoding to input tokens
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]  # type: ignore[index]

        new_memories: list[torch.Tensor | None] = []
        for i, layer in enumerate(self.layers):
            # Retrieve memory for this layer
            mem = self._memories[i]

            # Apply GTrXL layer (memory extends key/value context)
            x_out = layer(x, memory=mem)

            # Update memory: keep last memory_len hidden states
            if self.memory_len > 0:
                detached = x.detach()  # stop gradient through memory
                if mem is not None:
                    new_mem = torch.cat([mem, detached], dim=1)[:, -self.memory_len:, :]
                else:
                    new_mem = detached[:, -self.memory_len:, :]
                new_memories.append(new_mem)
            else:
                new_memories.append(None)

            x = x_out

        self._memories = new_memories

        # Readout from last sequence position
        last_hidden = x[:, -1, :]           # (B, d_model)
        features = self.output_proj(last_hidden)  # (B, features_dim)
        return features
