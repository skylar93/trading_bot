"""
Simplified agent for hyperparameter optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from agents.base.base_agent import BaseAgent

class SimpleNetwork(nn.Module):
    """Simple neural network for rapid experimentation"""
    
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU()
            ])
            current_dim = hidden_size
            
        # Output layer (single action)
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh())  # Scale output to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class HyperoptAgent(BaseAgent):
    """Simplified trading agent for hyperparameter tuning"""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 1e-3):
        super().__init__(observation_space, action_space)
        
        # Calculate input dimensions from observation space
        input_dim = observation_space.shape[0] * observation_space.shape[1]
        
        # Initialize network and optimizer
        self.network = SimpleNetwork(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )
        
        # Store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def get_action(self, state: np.ndarray) -> float:
        """Get trading action from state"""
        # Flatten and convert state to tensor
        x = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        
        with torch.no_grad():
            action = self.network(x)
        
        return action.cpu().item()
    
    def train(self, state, action, reward, next_state, done) -> Dict:
        """Simple training step using mean squared error"""
        # Prepare tensors
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        next_state = torch.FloatTensor(next_state).reshape(1, -1).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([float(done)]).to(self.device)
        
        # Predict action
        predicted_action = self.network(state)
        
        # Use reward as target (simple value-based learning)
        # Higher rewards should encourage similar actions in similar states
        target = torch.clamp(reward, -1, 1)  # Clamp reward to action range
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_action, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'predicted_action': predicted_action.item(),
            'target': target.item()
        }
    
    def get_hyperparameters(self) -> Dict:
        """Return current hyperparameters"""
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate
        }