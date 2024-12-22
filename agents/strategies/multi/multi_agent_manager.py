import logging
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import os

logger = logging.getLogger(__name__)

class MultiAgentManager:
    """
    Multi-agent manager that handles multiple trading agents with different strategies.
    Coordinates training, experience sharing, and agent interactions.
    """
    
    def __init__(self, agent_configs: List[Dict[str, Any]], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the multi-agent manager.
        
        Args:
            agent_configs: List of agent configurations, each containing:
                - id: Unique identifier for the agent
                - strategy: Strategy type (momentum, mean_reversion, market_making)
                - Other strategy-specific parameters
            device: Device to use for computations
        """
        self.device = device
        self.agents = {}
        self.shared_buffer = []
        self.shared_buffer_size = 10000
        
        # Initialize agents based on configs
        for config in agent_configs:
            agent_id = config["id"]
            strategy = config["strategy"]
            
            # Set device in config
            config["device"] = device
            
            # Initialize appropriate agent based on strategy
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
    
    def train_step(self, experiences: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Train all agents using their experiences.
        
        Args:
            experiences: Dictionary mapping agent_id to their experiences
                Each experience contains: state, action, reward, next_state, done
        
        Returns:
            Dictionary of training metrics for each agent
        """
        metrics = {}
        
        # First, add experiences to shared buffer if they're valuable
        for agent_id, exp in experiences.items():
            if self._is_valuable_experience(exp):
                self._add_to_shared_buffer(agent_id, exp)
        
        # Then, train each agent with their experiences
        for agent_id, agent in self.agents.items():
            if agent_id in experiences:
                # Train on own experience
                metrics[agent_id] = agent.train_step(**experiences[agent_id])
                
                # Learn from shared experiences if available
                if len(self.shared_buffer) > 0:
                    # Filter shared experiences to avoid learning from own experiences
                    filtered_buffer = [
                        exp for exp in self.shared_buffer 
                        if exp["agent_id"] != agent_id
                    ]
                    
                    if filtered_buffer:  # Only learn if there are experiences from other agents
                        try:
                            shared_metrics = agent.learn_from_shared_experience(filtered_buffer)
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
    
    def _is_valuable_experience(self, experience: Dict[str, Any]) -> bool:
        """Determine if an experience is valuable enough to share."""
        # For now, consider experiences with positive rewards as valuable
        return experience["reward"] > 0
    
    def _add_to_shared_buffer(self, agent_id: str, experience: Dict[str, Any]) -> None:
        """Add experience to shared buffer with source agent id."""
        self.shared_buffer.append({
            "agent_id": agent_id,
            **experience
        })
        
        # Maintain buffer size
        if len(self.shared_buffer) > self.shared_buffer_size:
            self.shared_buffer.pop(0)
    
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
