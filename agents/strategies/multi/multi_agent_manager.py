"""Multi-agent manager for coordinating different trading strategies"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from gymnasium import spaces

logger = logging.getLogger(__name__)

@dataclass
class ExperienceMetadata:
    """Metadata for shared experiences"""
    timestamp: datetime
    strategy_type: str
    reward: float
    volatility: float
    market_trend: float

class MultiAgentManager:
    """
    Multi-agent manager that handles multiple trading agents with different strategies.
    Coordinates training, experience sharing, and agent interactions.
    """
    
    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the multi-agent manager.
        
        Args:
            agent_configs: List of agent configurations
            device: Device to use for computations
        """
        self.device = device
        self.agents = {}
        self.shared_buffer = []
        self.shared_buffer_size = 10000
        self.min_share_reward = 0.5  # Minimum reward threshold for sharing
        
        # Initialize agents based on configs
        for config in agent_configs:
            agent_id = config["id"]
            # Support both 'type' and 'strategy' keys for backward compatibility
            strategy = config.get("type", config.get("strategy"))
            if not strategy:
                raise ValueError(f"Agent config must specify either 'type' or 'strategy': {config}")
            
            # Add device to config
            config["device"] = device
            
            # Get observation and action spaces
            observation_space = config.get("observation_space")
            action_space = config.get("action_space")
            
            # Normalize strategy name
            strategy = strategy.lower().replace("_", "")
            
            if strategy == "momentum":
                from .momentum_ppo_agent import MomentumPPOAgent
                self.agents[agent_id] = MomentumPPOAgent(
                    observation_space=observation_space,
                    action_space=action_space,
                    **{k: v for k, v in config.items() if k not in ["id", "type", "strategy", "observation_space", "action_space"]}
                )
            elif strategy == "meanreversion":
                from .mean_reversion_ppo_agent import MeanReversionPPOAgent
                self.agents[agent_id] = MeanReversionPPOAgent(
                    observation_space=observation_space,
                    action_space=action_space,
                    **{k: v for k, v in config.items() if k not in ["id", "type", "strategy", "observation_space", "action_space"]}
                )
            else:
                raise ValueError(f"Unknown strategy type: {strategy}")
        
        logger.info(f"Initialized MultiAgentManager with {len(self.agents)} agents")
    
    def act(self, observations: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        """
        Get actions from all agents based on their observations.
        
        Args:
            observations: Dictionary mapping agent_id to their observations
            deterministic: Whether to use deterministic action selection
        
        Returns:
            Dictionary mapping agent_id to their selected actions
        """
        actions = {}
        for agent_id, obs in observations.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].get_action(obs, deterministic)
        return actions
    
    def _calculate_trend(self, state: np.ndarray) -> float:
        """Calculate market trend from state data"""
        # Assuming state contains OHLCV data with Close price at index 3
        close_prices = state[:, 3]
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)
        return slope
    
    def _is_valuable_experience(self, experience: Dict[str, Any]) -> bool:
        """
        Determine if an experience is valuable enough to share.
        
        Args:
            experience: Experience dictionary containing state, action, reward, etc.
        
        Returns:
            bool: Whether the experience should be shared
        """
        # Check reward threshold
        if experience["reward"] <= self.min_share_reward:
            return False
        
        # Calculate market trend
        trend = self._calculate_trend(experience["state"])
        
        # Experience is valuable if there's a significant trend (up or down)
        # and the reward is above threshold
        return abs(trend) > 0.001 and experience["reward"] > self.min_share_reward
    
    def train_step(self, experiences: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Train all agents using their experiences.
        
        Args:
            experiences: Dictionary mapping agent_id to their experiences
        
        Returns:
            Dictionary of training metrics for each agent
        """
        metrics = {agent_id: {} for agent_id in self.agents.keys()}
        
        # First, add valuable experiences to shared buffer
        for agent_id, exp in experiences.items():
            if self._is_valuable_experience(exp):
                self._add_to_shared_buffer(agent_id, exp)
        
        # Then, train each agent
        for agent_id, agent in self.agents.items():
            if agent_id in experiences:
                # Train on own experience
                own_metrics = agent.train_step(**experiences[agent_id])
                if own_metrics is not None:
                    metrics[agent_id].update(own_metrics)
            
            # Learn from shared experiences if available
            if len(self.shared_buffer) > 0:
                try:
                    shared_metrics = agent.learn_from_shared_experience(self.shared_buffer)
                    if shared_metrics is not None:
                        metrics[agent_id].update({
                            f"shared_{k}": v for k, v in shared_metrics.items()
                        })
                except Exception as e:
                    logger.warning(f"Error during shared experience learning for agent {agent_id}: {str(e)}")
                    metrics[agent_id].update({
                        "shared_policy_loss": 0.0,
                        "shared_value_loss": 0.0,
                        "shared_entropy": 0.0
                    })
        
        return metrics
    
    def _add_to_shared_buffer(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """Add experience to shared buffer with source agent id."""
        if len(self.shared_buffer) >= self.shared_buffer_size:
            self.shared_buffer.pop(0)  # Remove oldest experience
        
        # Add agent ID to experience
        experience["agent_id"] = agent_id
        self.shared_buffer.append(experience)
    
    def save(self, path: str) -> None:
        """Save all agents' models."""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}/{agent_id}"
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            agent.save(agent_path)
            logger.info(f"Saved agent {agent_id} to {agent_path}")
    
    def load(self, path: str) -> None:
        """Load all agents' models."""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}/{agent_id}"
            agent.load(agent_path)
            logger.info(f"Loaded agent {agent_id} from {agent_path}")
