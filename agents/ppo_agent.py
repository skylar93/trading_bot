import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from agents.base.base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Feature extractor (shared layers)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Output mean and log_std
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        actor_output = self.actor(features)
        value = self.critic(features)
        
        # Split actor output into mean and log_std
        mean, log_std = actor_output[:, 0], actor_output[:, 1]
        return mean, log_std, value

class PPOAgent(BaseAgent):
    def __init__(self, observation_space, action_space, hidden_dim: int = 256,
                 lr: float = 3e-4, gamma: float = 0.99, epsilon: float = 0.2):
        super(PPOAgent, self).__init__(observation_space, action_space)
        
        # Initialize hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize neural network
        input_dim = observation_space.shape[0] * observation_space.shape[1]  # Flatten the input
        self.network = ActorCritic(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """Get action from the agent"""
        # Prepare input
        x = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        
        # Get action distribution parameters
        with torch.no_grad():
            mean, log_std, _ = self.network(x)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                action = torch.tanh(normal.sample())
        
        return action.cpu().numpy()[0]
    
    def train(self, state, action, reward, next_state, done) -> Dict[str, float]:
        """Train the agent using PPO with single transition"""
        # Convert to tensors
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        action = torch.FloatTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).reshape(1, -1).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # Simple policy gradient update for MVP
        # Forward pass
        mean, log_std, value = self.network(state)
        
        # Compute value loss
        with torch.no_grad():
            _, _, next_value = self.network(next_state)
            target_value = reward + (1 - done) * self.gamma * next_value

        value_loss = 0.5 * (target_value - value).pow(2).mean()
        
        # Compute policy loss
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action)
        policy_loss = -log_prob * (target_value - value).detach()
        
        # Total loss
        loss = policy_loss.mean() + value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.mean().item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item()
        }
    
    def save(self, path: str):
        """Save the agent to disk"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load the agent from disk"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])