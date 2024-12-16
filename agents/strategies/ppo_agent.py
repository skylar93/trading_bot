import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import mlflow
from agents.base.base_agent import BaseAgent
from agents.models.policy_network import PolicyNetwork
from agents.models.value_network import ValueNetwork

class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) Agent"""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 batch_size: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize PPO agent
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            hidden_dim: Hidden dimension of networks
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            batch_size: Batch size for training
            device: Device to use for training
        """
        super(PPOAgent, self).__init__(observation_space, action_space)
        
        # Initialize parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Initialize networks
        input_dim = observation_space.shape[0]
        self.policy = PolicyNetwork(input_dim, hidden_dim).to(self.device)
        self.value = ValueNetwork(input_dim, hidden_dim).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)
        
        # Initialize memory
        self.reset_memory()
    
    def reset_memory(self):
        """Reset experience memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []  # For handling episode termination
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from policy network
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic action
            
        Returns:
            Action array
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, log_std = self.policy(state)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                action = torch.tanh(normal.sample())
                
                # Store experience
                value = self.value(state)
                self.states.append(state)
                self.actions.append(action)
                self.values.append(value)
                self.log_probs.append(normal.log_prob(action))
        
        return action.cpu().numpy()
    
    def train(self, state, action, reward, next_state, done) -> Dict[str, float]:
        """Train the agent
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Store experience
        self.rewards.append(reward)
        self.masks.append(1.0 - float(done))
        
        # Train if episode is done
        metrics = {}
        if done:
            metrics = self._update()
            self.reset_memory()
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'policy_loss': metrics['policy_loss'],
                'value_loss': metrics['value_loss'],
                'total_loss': metrics['total_loss'],
                'mean_reward': np.mean(self.rewards)
            })
        
        return metrics
    
    def _update(self) -> Dict[str, float]:
        """Update policy and value networks
        
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensor
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach()
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        masks = torch.FloatTensor(self.masks).to(self.device)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for r, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = r + self.gamma * R * mask
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - old_values
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(10):  # Multiple epochs
            # Process in batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                # Get batch indices
                idx = indices[start:start + self.batch_size]
                
                # Get batch data
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Get new log probs and values
                mean, log_std = self.policy(batch_states)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                new_log_probs = normal.log_prob(batch_actions)
                new_values = self.value(batch_states)
                
                # Calculate ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
                
                # Calculate losses
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                value_loss = (new_values - batch_returns).pow(2).mean()
                
                # Update networks
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # Calculate average losses
        num_updates = len(states) // self.batch_size * 10
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_policy_loss + avg_value_loss
        }
    
    def save(self, path: str):
        """Save the agent
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load the agent
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])