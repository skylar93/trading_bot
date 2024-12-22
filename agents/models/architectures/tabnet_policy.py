import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork


class GhostBatchNorm(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
        # 배치 정규화 레이어
        self.bn = nn.BatchNorm1d(
            self.input_dim,
            momentum=momentum,
            affine=True,
            track_running_stats=True,
            eps=1e-5
        )
        
        # 가중치 초기화
        if self.bn.weight is not None:
            nn.init.ones_(self.bn.weight)
        if self.bn.bias is not None:
            nn.init.zeros_(self.bn.bias)
        
    def forward(self, x):
        if self.training:
            chunks = x.chunk(max(1, int(np.ceil(x.shape[0] / self.virtual_batch_size))))
            res = []
            for chunk in chunks:
                # 작은 배치에 대해 running stats 업데이트 방지
                if chunk.size(0) < 2:
                    chunk = torch.cat([chunk, chunk], dim=0)
                res.append(self.bn(chunk))
            return torch.cat(res[:len(chunks)], dim=0)
        else:
            return self.bn(x)


class GLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128):
        super().__init__()
        self.output_dim = output_dim
        
        # 선형 레이어
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=True)
        
        # 배치 정규화
        self.bn = GhostBatchNorm(output_dim * 2, virtual_batch_size=virtual_batch_size)
        
        # 가중치 초기화
        torch.manual_seed(42)  # 재현성을 위한 시드 고정
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        chunks = x.chunk(2, dim=1)
        return chunks[0] * torch.sigmoid(chunks[1])


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent, virtual_batch_size=128):
        super().__init__()
        
        self.shared = nn.ModuleList()
        for i in range(shared_layers):
            if i == 0:
                self.shared.append(GLU_Block(input_dim, output_dim, virtual_batch_size))
            else:
                self.shared.append(GLU_Block(output_dim, output_dim, virtual_batch_size))
                
        self.independent = nn.ModuleList()
        for i in range(n_glu_independent):
            self.independent.append(GLU_Block(output_dim, output_dim, virtual_batch_size))
            
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        for layer in self.shared:
            x = layer(x)
        for layer in self.independent:
            x = layer(x)
        return x


class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 n_shared=2, n_independent=2, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.virtual_batch_size = virtual_batch_size
        
        self.shared = FeatureTransformer(
            self.input_dim,
            self.n_d + self.n_a,
            n_shared,
            n_independent,
            virtual_batch_size
        )
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.reshape(batch_size * seq_len, features)  # Flatten sequence dimension
        
        # Process in chunks for memory efficiency
        chunk_size = self.virtual_batch_size
        outputs = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i + chunk_size]
            chunk_output = self.shared(chunk)
            outputs.append(chunk_output)
        
        x_processed = torch.cat(outputs, dim=0)
        x_processed = x_processed.reshape(batch_size, seq_len, -1)  # Restore sequence dimension
        x_processed = x_processed.mean(dim=1)  # Average over sequence dimension
        return x_processed


class TabNetPolicy(BaseNetwork):
    def __init__(self, observation_space, action_space, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 n_shared=2, n_independent=2, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        
        # Get input and output dimensions from spaces
        if len(observation_space.shape) > 1:
            input_dim = observation_space.shape[-1]  # Last dimension for features
            self.seq_len = observation_space.shape[-2]  # Second to last dimension for sequence length
        else:
            input_dim = observation_space.shape[0]
            self.seq_len = 1
        
        if hasattr(action_space, "shape"):
            output_dim = int(np.prod(action_space.shape))
        else:
            output_dim = action_space.n
        
        self.encoder = TabNetEncoder(
            input_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_shared,
            n_independent,
            virtual_batch_size,
            momentum
        )
        
        # Use smaller hidden dimensions for memory efficiency
        hidden_dim = 32
        
        self.head_mean = nn.Sequential(
            nn.Linear(n_d + n_a, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize log_std as a parameter
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize weights with proper scaling
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
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if len(x.shape) == 2:  # (seq_len, features)
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Process through encoder
        features = self.encoder(x)
        
        # Get mean and std
        action_mean = self.head_mean(features)
        action_std = torch.exp(self.log_std)
        
        # Expand std to match batch size if needed
        if len(action_mean.shape) > 1:
            action_std = action_std.expand_as(action_mean)
            
        return action_mean, action_std
    
    def get_architecture_type(self) -> str:
        return "tabnet"