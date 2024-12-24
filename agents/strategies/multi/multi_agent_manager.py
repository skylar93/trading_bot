"""Multi-agent manager for coordinating different trading strategies"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

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
    
    def __init__(self, agent_configs: List[Dict[str, Any]], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
        self.experience_ttl = timedelta(hours=24)  # Time-to-live for shared experiences
        
        # Initialize agents based on configs
        for config in agent_configs:
            agent_id = config["id"]
            strategy = config["strategy"]
            
            config["device"] = device
            
            if strategy == "momentum":
                from .momentum_ppo_agent import MomentumPPOAgent
                self.agents[agent_id] = MomentumPPOAgent(config)
            elif strategy == "mean_reversion":
                from .mean_reversion_ppo_agent import MeanReversionPPOAgent
                self.agents[agent_id] = MeanReversionPPOAgent(config)
            elif strategy == "market_making":
                from .market_maker_ppo_agent import MarketMakerPPOAgent
                self.agents[agent_id] = MarketMakerPPOAgent(config)
            else:
                raise ValueError(f"Unknown strategy type: {strategy}")
        
        logger.info(f"Initialized MultiAgentManager with {len(self.agents)} agents")
    
    def _calculate_market_metrics(self, state: np.ndarray) -> Tuple[float, float]:
        """Calculate market volatility and trend from state"""
        if len(state.shape) == 3:  # Batch of states
            close_prices = state[0, :, 3]  # Use first batch
        else:
            close_prices = state[:, 3]
        
        # Calculate volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate trend using linear regression
        x = np.arange(len(close_prices))
        coeffs = np.polyfit(x, close_prices, 1)
        trend = coeffs[0]  # Slope of the line
        
        return volatility, trend
    
    def _is_valuable_experience(self, experience: Dict[str, Any], agent_id: str) -> bool:
        """
        Determine if an experience is valuable enough to share.
        
        Args:
            experience: The experience to evaluate
            agent_id: ID of the agent that generated the experience
        
        Returns:
            Whether the experience should be shared
        """
        # Basic reward threshold
        if experience["reward"] < self.min_share_reward:
            return False
        
        # Calculate market metrics
        volatility, trend = self._calculate_market_metrics(experience["state"])
        
        # Get agent's strategy type
        strategy = next(agent.strategy for aid, agent in self.agents.items() if aid == agent_id)
        
        # Strategy-specific criteria
        if strategy == "momentum":
            # Share momentum experiences in trending markets
            return abs(trend) > 0.001 and experience["reward"] > 0
        elif strategy == "mean_reversion":
            # Share mean reversion experiences in ranging markets
            return abs(trend) < 0.001 and volatility > 0.1 and experience["reward"] > 0
        
        return False
    
    def _add_to_shared_buffer(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """
        Add experience to shared buffer with metadata.
        
        Args:
            agent_id: ID of the agent sharing the experience
            experience: The experience to share
        """
        volatility, trend = self._calculate_market_metrics(experience["state"])
        
        metadata = ExperienceMetadata(
            timestamp=datetime.now(),
            strategy_type=next(agent.strategy for aid, agent in self.agents.items() if aid == agent_id),
            reward=experience["reward"],
            volatility=volatility,
            market_trend=trend
        )
        
        self.shared_buffer.append({
            "agent_id": agent_id,
            "metadata": metadata,
            **experience
        })
        
        # Clean old experiences and maintain buffer size
        self._clean_shared_buffer()
    
    def _clean_shared_buffer(self) -> None:
        """Clean expired experiences and maintain buffer size"""
        current_time = datetime.now()
        
        # Remove expired experiences
        self.shared_buffer = [
            exp for exp in self.shared_buffer
            if current_time - exp["metadata"].timestamp < self.experience_ttl
        ]
        
        # Maintain buffer size
        if len(self.shared_buffer) > self.shared_buffer_size:
            # Remove oldest experiences
            self.shared_buffer = sorted(
                self.shared_buffer,
                key=lambda x: x["metadata"].reward,
                reverse=True
            )[:self.shared_buffer_size]
    
    def _filter_relevant_experiences(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Filter experiences relevant for a specific agent.
        
        Args:
            agent_id: ID of the agent to filter experiences for
            
        Returns:
            List of relevant experiences
        """
        agent_strategy = next(agent.strategy for aid, agent in self.agents.items() if aid == agent_id)
        current_time = datetime.now()
        
        filtered_experiences = []
        for exp in self.shared_buffer:
            if exp["agent_id"] == agent_id:
                continue
                
            metadata = exp["metadata"]
            
            # Skip expired experiences
            if current_time - metadata.timestamp >= self.experience_ttl:
                continue
            
            # Strategy-specific filtering
            if agent_strategy == "momentum" and abs(metadata.market_trend) > 0.001:
                filtered_experiences.append(exp)
            elif agent_strategy == "mean_reversion" and abs(metadata.market_trend) < 0.001:
                filtered_experiences.append(exp)
        
        return filtered_experiences

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
            if self._is_valuable_experience(exp, agent_id):
                self._add_to_shared_buffer(agent_id, exp)
        
        # Then, train each agent
        for agent_id, agent in self.agents.items():
            # Train on own experience if available
            if agent_id in experiences:
                own_metrics = agent.train_step(**experiences[agent_id])
                if own_metrics is not None:
                    metrics[agent_id].update(own_metrics)
            
            # Learn from relevant shared experiences
            relevant_experiences = self._filter_relevant_experiences(agent_id)
            
            if relevant_experiences:
                try:
                    shared_metrics = agent.learn_from_shared_experience(relevant_experiences)
                    if shared_metrics is not None:
                        metrics[agent_id].update({
                            f"shared_{k}": v for k, v in shared_metrics.items()
                        })
                    else:
                        metrics[agent_id].update({
                            "shared_policy_loss": 0.0,
                            "shared_value_loss": 0.0,
                            "shared_entropy": 0.0
                        })
                except Exception as e:
                    logger.warning(f"Error during shared experience learning for agent {agent_id}: {str(e)}")
                    metrics[agent_id].update({
                        "shared_policy_loss": 0.0,
                        "shared_value_loss": 0.0,
                        "shared_entropy": 0.0
                    })
            else:
                # Add zero metrics when no shared experiences
                metrics[agent_id].update({
                    "shared_policy_loss": 0.0,
                    "shared_value_loss": 0.0,
                    "shared_entropy": 0.0
                })
        
        return metrics
    
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
    
    def save(self, path: str) -> None:
        """Save all agents' models."""
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}/{agent_id}"
            # Create agent-specific directory
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            agent.save(agent_path)
            logger.info(f"Saved agent {agent_id} to {agent_path}")
    
    def load(self, path: str) -> None:
        """Load all agents' models."""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}/{agent_id}"
            agent.load(agent_path)
            logger.info(f"Loaded agent {agent_id} from {agent_path}")
