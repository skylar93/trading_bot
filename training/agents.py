"""Reinforcement Learning Agents"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pickle
from typing import Dict, Any, Tuple
import os

class BaseNetwork(nn.Module):
    """Base neural network architecture"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(BaseNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions: sell, hold, buy
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent"""
    
    def __init__(self, env, learning_rate: float = 0.001, gamma: float = 0.99, 
                 epsilon: float = 0.2, batch_size: int = 64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        # Initialize networks
        input_dim = env.observation_space.shape[0]
        self.policy = BaseNetwork(input_dim)
        self.value = BaseNetwork(input_dim)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        
        # Initialize memory
        self.reset_memory()
    
    def reset_memory(self):
        """Reset experience memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
    
    def get_action(self, state: np.ndarray) -> int:
        """Get action from policy network"""
        state = torch.FloatTensor(state)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        self.values.append(self.value(state))
        
        return action.item()
    
    def train(self, state, action, reward, next_state, done):
        """Train the agent"""
        self.rewards.append(reward)
        
        if done:
            self._update()
            self.reset_memory()
    
    def _update(self):
        """Update policy and value networks"""
        # Convert to tensor
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach()
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        advantages = returns - old_values
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            # Get new log probs and values
            logits = self.policy(states)
            new_values = self.value(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
            
            # Calculate losses
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def evaluate(self, env) -> Dict[str, float]:
        """Evaluate the agent"""
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        portfolio_values = []
        
        while not (done or truncated):
            with torch.no_grad():
                state = torch.FloatTensor(state)
                logits = self.policy(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
            
            state, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        max_drawdown = np.min(portfolio_values) / np.max(portfolio_values) - 1
        
        return {
            'total_reward': total_reward,
            'final_portfolio_value': portfolio_values[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def save(self, path: str):
        """Save the agent"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load the agent"""
        state = torch.load(path)
        self.policy.load_state_dict(state['policy_state_dict'])
        self.value.load_state_dict(state['value_state_dict'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(state['value_optimizer_state_dict'])

def create_agent(model_type: str, env, **kwargs) -> Any:
    """Create an agent based on the model type"""
    if model_type == "PPO":
        return PPOAgent(env, **kwargs)
    elif model_type == "DQN":
        # TODO: Implement DQN agent
        raise NotImplementedError("DQN agent not implemented yet")
    elif model_type == "A2C":
        # TODO: Implement A2C agent
        raise NotImplementedError("A2C agent not implemented yet")
    else:
        raise ValueError(f"Unknown model type: {model_type}") 