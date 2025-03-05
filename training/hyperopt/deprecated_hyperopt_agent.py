"""
Simplified agent for hyperparameter optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MinimalNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head (action)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # [-1, 1] action range
        )

        # Value head (state value)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        return self.policy(shared_features), self.value(shared_features)


class MinimalPPOAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
    ):

        input_dim = observation_space.shape[0] * observation_space.shape[1]

        self.network = MinimalNetwork(input_dim, hidden_size)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network.to(self.device)

        self.gamma = gamma
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

    def get_action(self, state: np.ndarray) -> float:
        x = torch.FloatTensor(state).reshape(1, -1).to(self.device)

        with torch.no_grad():
            action, _ = self.network(x)

        return action.cpu().item()

    def train(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict:

        # Convert to tensors
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        next_state = (
            torch.FloatTensor(next_state).reshape(1, -1).to(self.device)
        )
        action = torch.FloatTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([float(done)]).to(self.device)

        # Get values and advantage
        with torch.no_grad():
            _, next_value = self.network(next_state)
            target_value = reward + (1 - done) * self.gamma * next_value

        # Get current predictions
        current_action, current_value = self.network(state)

        # Calculate advantage
        advantage = (target_value - current_value).detach()

        # Policy loss (PPO-Clip)
        ratio = torch.exp(
            torch.log(current_action + 1e-8) - torch.log(action + 1e-8)
        )
        policy_loss_1 = ratio * advantage
        policy_loss_2 = (
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss
        value_loss = nn.MSELoss()(current_value, target_value.detach())

        # Combined loss
        loss = policy_loss + 0.5 * value_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "advantage": advantage.mean().item(),
        }

    def save(self, path: str):
        torch.save(
            {
                "network_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "hyperparameters": {
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                },
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
