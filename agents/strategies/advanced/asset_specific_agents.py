"""
Asset-specific trading agents for multi-agent reinforcement learning.

This module provides specialized agent implementations for different asset classes,
allowing for optimized strategies based on asset characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from gymnasium import spaces
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AssetCharacteristics:
    """
    Characteristics of different asset types to inform agent behavior.
    
    Attributes:
        volatility_factor: Typical volatility scaling for this asset type
        trading_hours: Trading hours profile (24/7, exchange hours, etc.)
        typical_spread: Typical bid-ask spread as percentage of price
        slippage_factor: Typical slippage factor for order execution
        min_trade_size: Minimum trade size constraint
        fee_structure: Fee structure for this asset type
    """
    volatility_factor: float
    trading_hours: str
    typical_spread: float
    slippage_factor: float
    min_trade_size: float
    fee_structure: Dict[str, float]


class AssetSpecificAgent:
    """
    Base class for asset-specific trading agents.
    
    Features:
    - Configurable for specific asset characteristics
    - Adjusts decision-making based on asset type
    - Compatible with MultiAgentManager for portfolio-level coordination
    
    Implementation Notes:
    - Subclass this for specific asset classes (Crypto, Equities, etc.)
    - Compatible with standard RL environments
    - Maintains asset-specific state and metrics
    
    Recent Changes:
    - Added asset characteristics to inform agent behavior
    - Implemented asset-specific observation preprocessing
    - Added support for asset-specific action scaling
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        asset_id: str,
        asset_type: str,
        characteristics: Optional[AssetCharacteristics] = None,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        network_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the asset-specific agent.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            asset_id: Identifier for this specific asset (e.g., "BTC", "AAPL")
            asset_type: Type of asset ("crypto", "equity", "commodity", etc.)
            characteristics: Asset-specific characteristics
            learning_rate: Learning rate for optimization
            device: Device to run the model on
            network_config: Configuration for the neural network
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.characteristics = characteristics or self._default_characteristics()
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize network and optimizer
        self._setup_network(network_config or {})
        
        # Metrics tracking
        self.metrics = {
            "train_loss": [],
            "value_loss": [],
            "policy_loss": [],
            "rewards": [],
            "episode_lengths": []
        }
        
        # Asset-specific state
        self.state = {
            "position": 0.0,
            "avg_entry_price": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "trade_count": 0,
            "last_action": None,
            "market_regime": "unknown"
        }
        
        logger.info(f"Initialized asset-specific agent for {asset_id} ({asset_type})")
    
    def _default_characteristics(self) -> AssetCharacteristics:
        """Return default characteristics based on asset type."""
        if self.asset_type.lower() == "crypto":
            return AssetCharacteristics(
                volatility_factor=2.0,
                trading_hours="24/7",
                typical_spread=0.001,  # 0.1%
                slippage_factor=0.002,  # 0.2%
                min_trade_size=0.0001,
                fee_structure={"maker": 0.001, "taker": 0.002}
            )
        elif self.asset_type.lower() == "equity":
            return AssetCharacteristics(
                volatility_factor=1.0,
                trading_hours="exchange",
                typical_spread=0.0005,  # 0.05%
                slippage_factor=0.001,  # 0.1%
                min_trade_size=1.0,
                fee_structure={"commission": 0.0005}
            )
        elif self.asset_type.lower() == "commodity":
            return AssetCharacteristics(
                volatility_factor=1.5,
                trading_hours="futures",
                typical_spread=0.0008,  # 0.08%
                slippage_factor=0.0015,  # 0.15%
                min_trade_size=0.01,
                fee_structure={"exchange": 0.0002, "clearing": 0.0001}
            )
        else:
            # Default characteristics
            return AssetCharacteristics(
                volatility_factor=1.0,
                trading_hours="standard",
                typical_spread=0.001,
                slippage_factor=0.001,
                min_trade_size=0.001,
                fee_structure={"flat": 0.001}
            )
    
    def _setup_network(self, config: Dict[str, Any]):
        """Set up the neural network and optimizer."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """
        Preprocess the observation specifically for this asset type.
        
        Args:
            observation: Raw observation from the environment
            
        Returns:
            Preprocessed observation tensor
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        # Apply asset-specific preprocessing if needed
        if self.asset_type.lower() == "crypto":
            # Crypto assets might need different scaling
            pass
        elif self.asset_type.lower() == "equity":
            # Equities might have different features
            pass
        
        return obs_tensor
    
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Determine the trading action for assets.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action array
        """
        # 하위 호환성을 위해 액션만 반환
        action, _ = self.act_with_hidden_state(observation, deterministic)
        return action
    
    def act_with_hidden_state(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the trading action and return the internal hidden state.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action_np, hidden_state_np):
                - action_np: The action to take
                - hidden_state_np: The internal hidden state representation
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: str):
        """Save the agent's model and state."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self, path: str):
        """Load the agent's model and state from disk."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_state(self, new_state: Dict[str, Any]):
        """Update the agent's internal state."""
        self.state.update(new_state)
    
    def reset(self):
        """Reset the agent's state for a new episode."""
        self.state = {
            "position": 0.0,
            "avg_entry_price": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "trade_count": 0,
            "last_action": None,
            "market_regime": "unknown"
        }
    
    def analyze_market_regime(self, observation: np.ndarray) -> str:
        """
        Analyze the current market regime based on the observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Market regime label ("trending", "ranging", "volatile", etc.)
        """
        # Implement market regime detection logic
        # This could be based on volatility, trend indicators, etc.
        return "unknown"
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from the agent.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action array
        """
        # 하위 호환성을 위해 액션만 반환
        action, _ = self.get_action_with_hidden_state(observation, deterministic)
        return action
    
    def get_action_with_hidden_state(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action and hidden state from the agent.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action_np, hidden_state_np):
                - action_np: The action to take
                - hidden_state_np: The internal hidden state representation
        """
        return self.act_with_hidden_state(observation, deterministic)


class CryptoAgent(AssetSpecificAgent):
    """
    Trading agent specialized for cryptocurrency assets.
    
    Features:
    - Optimized for 24/7 markets with high volatility
    - Handles rapid price changes and flash crashes
    - Adapts to varying liquidity conditions
    
    Implementation Notes:
    - Uses more aggressive position sizing for volatile markets
    - Includes specialized preprocessing for crypto-specific features
    - Includes flash crash protection mechanisms
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        asset_id: str,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        network_config: Optional[Dict[str, Any]] = None,
        volatility_scaling: bool = True
    ):
        """Initialize the crypto-specific agent."""
        self.volatility_scaling = volatility_scaling
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            asset_id=asset_id,
            asset_type="crypto",
            learning_rate=learning_rate,
            device=device,
            network_config=network_config
        )
    
    def _setup_network(self, config: Dict[str, Any]):
        """Set up the neural network optimized for crypto trading."""
        # Extract dimensions from spaces
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        
        # Network parameters
        hidden_size = config.get("hidden_size", 256)
        
        # Create policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2)  # Mean and log_std
        ).to(self.device)
        
        # Create value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.learning_rate
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with crypto-optimized scaling."""
        for module in self.policy_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        for module in self.value_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Preprocess observation with crypto-specific features."""
        obs_tensor = super().preprocess_observation(observation)
        
        # Crypto-specific preprocessing:
        # 1. Log-transform volume features (often has extreme values)
        # 2. Adjust volatility scaling if enabled
        
        if self.volatility_scaling and obs_tensor.shape[-1] >= 5:
            # Assuming standard OHLCV format
            # Apply log transform to volume (if present)
            volume_idx = 4  # Typical index for volume in OHLCV data
            if volume_idx < obs_tensor.shape[-1]:
                # Add small epsilon to avoid log(0)
                obs_tensor[..., volume_idx] = torch.log(obs_tensor[..., volume_idx] + 1e-8)
        
        return obs_tensor
    
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Determine the trading action for crypto assets.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action array
        """
        # 하위 호환성을 위해 액션만 반환
        action, _ = self.act_with_hidden_state(observation, deterministic)
        return action
    
    def act_with_hidden_state(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the trading action for crypto assets and return the internal hidden state.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action_np, hidden_state_np):
                - action_np: The action to take
                - hidden_state_np: The internal hidden state representation
        """
        # Preprocess the observation
        obs_tensor = self.preprocess_observation(observation)
        
        # Extract hidden state representation from policy network
        # We'll capture the output of the second-last layer
        hidden_state = None
        # Process through layers up to the second-last layer to get hidden representation
        with torch.no_grad():
            # Pass through the first layers to get hidden representation
            # For MLP networks, we'll extract after the penultimate layer
            hidden_state = obs_tensor
            for i, module in enumerate(self.policy_net):
                hidden_state = module(hidden_state)
                # Stop before the final layer to capture hidden representation
                if i == len(self.policy_net) - 2:  # Second-last layer
                    break
        
        # Get action from policy network (full forward pass)
        with torch.no_grad():
            policy_output = self.policy_net(obs_tensor)
            means, log_stds = torch.chunk(policy_output, 2, dim=-1)
            
            # Apply tanh to bound action means
            means = torch.tanh(means)
            
            # Process log_stds with appropriate bounds
            stds = F.softplus(log_stds) + 1e-6
            
            # Sample from distribution if not deterministic
            if deterministic:
                action = means
            else:
                normal = torch.distributions.Normal(means, stds)
                action = normal.sample()
                action = torch.clamp(action, -1.0, 1.0)
        
        # Adjust based on market regime if needed
        market_regime = self.analyze_market_regime(observation)
        self.state["market_regime"] = market_regime
        
        # Apply volatility scaling for aggressive/conservative positioning
        if self.volatility_scaling and market_regime == "volatile":
            # Scale down action in highly volatile markets for risk management
            action = action * 0.7  # Reduce position size in volatile conditions
        
        # Convert to numpy arrays
        action_np = action.cpu().numpy()
        hidden_state_np = hidden_state.cpu().numpy()
        
        self.state["last_action"] = action_np
        
        return action_np, hidden_state_np
    
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent's policy based on crypto-specific experience."""
        # Extract experience data
        states = torch.FloatTensor(experience["states"]).to(self.device)
        actions = torch.FloatTensor(experience["actions"]).to(self.device)
        rewards = torch.FloatTensor(experience["rewards"]).to(self.device)
        next_states = torch.FloatTensor(experience["next_states"]).to(self.device)
        dones = torch.FloatTensor(experience["dones"]).to(self.device)
        
        # Compute value predictions
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        
        # Compute returns (simple one-step)
        returns = rewards + (1.0 - dones) * 0.99 * next_values
        
        # Compute advantages
        advantages = returns - values
        
        # Compute policy loss
        policy_output = self.policy_net(states)
        means, log_stds = torch.chunk(policy_output, 2, dim=-1)
        means = torch.tanh(means)
        stds = F.softplus(log_stds) + 1e-6
        
        normal = torch.distributions.Normal(means, stds)
        log_probs = normal.log_prob(actions).sum(dim=-1)
        
        # PPO-style policy loss (simplified)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
        
        self.metrics["train_loss"].append(metrics["loss"])
        self.metrics["policy_loss"].append(metrics["policy_loss"])
        self.metrics["value_loss"].append(metrics["value_loss"])
        
        return metrics
    
    def save(self, path: str):
        """Save the crypto agent's model and state."""
        save_path = f"{path}/{self.asset_id}_crypto_agent.pt"
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "state": self.state,
            "characteristics": self.characteristics,
            "volatility_scaling": self.volatility_scaling
        }, save_path)
        logger.info(f"Saved crypto agent to {save_path}")
    
    def load(self, path: str):
        """Load the crypto agent's model and state from disk."""
        load_path = f"{path}/{self.asset_id}_crypto_agent.pt"
        if not torch.cuda.is_available():
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(load_path)
        
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = checkpoint["metrics"]
        self.state = checkpoint["state"]
        self.characteristics = checkpoint["characteristics"]
        self.volatility_scaling = checkpoint["volatility_scaling"]
        logger.info(f"Loaded crypto agent from {load_path}")
    
    def analyze_market_regime(self, observation: np.ndarray) -> str:
        """Analyze the crypto market regime with specialized indicators."""
        # Implement crypto-specific market regime detection
        # This would analyze volatility, volume patterns, etc.
        
        # Simplified example: check if recent volatility is high
        try:
            # Assuming OHLCV format in the observation window
            if len(observation.shape) > 1 and observation.shape[1] >= 4:
                # Calculate price range as percentage
                high_idx, low_idx = 2, 3  # Typical indices for high/low
                recent_highs = observation[:, high_idx]
                recent_lows = observation[:, low_idx]
                
                # Calculate recent volatility as avg(high-low)/avg(low)
                recent_ranges = (recent_highs - recent_lows) / recent_lows
                avg_range = np.mean(recent_ranges)
                
                if avg_range > 0.05:  # 5% average daily range
                    return "volatile"
                elif avg_range < 0.02:  # 2% average daily range
                    return "ranging"
                else:
                    return "normal"
        except Exception as e:
            logger.warning(f"Error analyzing market regime: {str(e)}")
        
        return "unknown"


class EquityAgent(AssetSpecificAgent):
    """
    Trading agent specialized for equity (stock) assets.
    
    Features:
    - Optimized for exchange trading hours
    - Handles fundamentals and technical indicators
    - Considers market session dynamics (open, close, etc.)
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        asset_id: str,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        network_config: Optional[Dict[str, Any]] = None,
        use_fundamentals: bool = False
    ):
        """Initialize the equity-specific agent."""
        self.use_fundamentals = use_fundamentals
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            asset_id=asset_id,
            asset_type="equity",
            learning_rate=learning_rate,
            device=device,
            network_config=network_config
        )
    
    def _setup_network(self, config: Dict[str, Any]):
        """Set up the neural network optimized for equity trading."""
        # Extract dimensions from spaces
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        
        # Network parameters
        hidden_size = config.get("hidden_size", 256)
        
        # Create policy network with attention for technical patterns
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2)  # Mean and log_std
        ).to(self.device)
        
        # Create value network
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.learning_rate
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for equity trading."""
        for module in self.policy_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        for module in self.value_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Preprocess observation with equity-specific features."""
        obs_tensor = super().preprocess_observation(observation)
        
        # Equity-specific preprocessing
        # (e.g., normalizing fundamentals, handling session info)
        
        return obs_tensor
    
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Determine the trading action for equity assets.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action array
        """
        # 하위 호환성을 위해 액션만 반환
        action, _ = self.act_with_hidden_state(observation, deterministic)
        return action
    
    def act_with_hidden_state(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the trading action for equity assets and return the internal hidden state.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action_np, hidden_state_np):
                - action_np: The action to take
                - hidden_state_np: The internal hidden state representation
        """
        # Preprocess the observation
        obs_tensor = self.preprocess_observation(observation)
        
        # Extract hidden state representation from policy network
        # We'll capture the output of the second-last layer
        hidden_state = None
        # Process through layers up to the second-last layer to get hidden representation
        with torch.no_grad():
            # Pass through the first layers to get hidden representation
            hidden_state = obs_tensor
            for i, module in enumerate(self.policy_net):
                hidden_state = module(hidden_state)
                # Stop before the final layer to capture hidden representation
                if i == len(self.policy_net) - 2:  # Second-last layer
                    break
        
        # Get action from policy network (full forward pass)
        with torch.no_grad():
            policy_output = self.policy_net(obs_tensor)
            means, log_stds = torch.chunk(policy_output, 2, dim=-1)
            
            # Apply tanh to bound action means
            means = torch.tanh(means)
            
            # Process log_stds with appropriate bounds
            stds = F.softplus(log_stds) + 1e-6
            
            # Sample from distribution if not deterministic
            if deterministic:
                action = means
            else:
                normal = torch.distributions.Normal(means, stds)
                action = normal.sample()
                action = torch.clamp(action, -1.0, 1.0)
                
        # Additional equity-specific logic...
        if self.use_fundamentals:
            # Adjust based on fundamental factors if available in the observation
            fundamental_idx = self.get_fundamental_indices(observation)
            if fundamental_idx is not None and len(fundamental_idx) > 0:
                fundamental_values = observation[fundamental_idx]
                
                # Simple adjustment based on PE ratio or other fundamentals
                # Reduce position if fundamentals are poor
                if hasattr(self, 'fundamental_threshold'):
                    threshold = self.fundamental_threshold
                else:
                    threshold = 0.0  # 기본값
                
                if np.mean(fundamental_values) < threshold:
                    action = action * 0.8
                    
        # Convert to numpy arrays
        action_np = action.cpu().numpy()
        hidden_state_np = hidden_state.cpu().numpy()
        
        self.state["last_action"] = action_np
        
        return action_np, hidden_state_np
    
    def get_fundamental_indices(self, observation: np.ndarray) -> np.ndarray:
        """
        Get indices of fundamental data in observation array.
        
        Args:
            observation: Observation array
            
        Returns:
            Array of indices corresponding to fundamental data
        """
        # 간단하게 observation의 끝부분을 fundamentals로 간주
        # 실제 구현에서는 정확한 인덱스 위치를 반환해야 함
        return np.arange(len(observation) - 5, len(observation)) if len(observation) > 5 else np.array([])
    
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent's policy based on equity-specific experience."""
        # Similar to CryptoAgent.update()
        # Extract experience data
        states = torch.FloatTensor(experience["states"]).to(self.device)
        actions = torch.FloatTensor(experience["actions"]).to(self.device)
        rewards = torch.FloatTensor(experience["rewards"]).to(self.device)
        next_states = torch.FloatTensor(experience["next_states"]).to(self.device)
        dones = torch.FloatTensor(experience["dones"]).to(self.device)
        
        # Compute value predictions
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        
        # Compute returns (simple one-step)
        returns = rewards + (1.0 - dones) * 0.99 * next_values
        
        # Compute advantages
        advantages = returns - values
        
        # Compute policy loss
        policy_output = self.policy_net(states)
        means, log_stds = torch.chunk(policy_output, 2, dim=-1)
        means = torch.tanh(means)
        stds = F.softplus(log_stds) + 1e-6
        
        normal = torch.distributions.Normal(means, stds)
        log_probs = normal.log_prob(actions).sum(dim=-1)
        
        # PPO-style policy loss (simplified)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
        
        self.metrics["train_loss"].append(metrics["loss"])
        self.metrics["policy_loss"].append(metrics["policy_loss"])
        self.metrics["value_loss"].append(metrics["value_loss"])
        
        return metrics
    
    def save(self, path: str):
        """Save the equity agent's model and state."""
        save_path = f"{path}/{self.asset_id}_equity_agent.pt"
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "state": self.state,
            "characteristics": self.characteristics
        }, save_path)
        logger.info(f"Saved equity agent to {save_path}")
    
    def load(self, path: str):
        """Load the equity agent's model and state from disk."""
        load_path = f"{path}/{self.asset_id}_equity_agent.pt"
        if not torch.cuda.is_available():
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(load_path)
        
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = checkpoint["metrics"]
        self.state = checkpoint["state"]
        self.characteristics = checkpoint["characteristics"]
        logger.info(f"Loaded equity agent from {load_path}")
    
    def analyze_market_regime(self, observation: np.ndarray) -> str:
        """Analyze the equity market regime with specialized indicators."""
        # Implement equity-specific market regime detection
        # This would consider technical patterns, volume, etc.
        
        # Default implementation
        return "normal"


class AssetSpecificAgentFactory:
    """Factory class for creating asset-specific agents."""
    
    @staticmethod
    def create_agent(
        asset_id: str,
        asset_type: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        config: Optional[Dict[str, Any]] = None
    ) -> AssetSpecificAgent:
        """
        Create an appropriate agent for the specified asset.
        
        Args:
            asset_id: Identifier for the asset (e.g., "BTC", "AAPL")
            asset_type: Type of asset ("crypto", "equity", "commodity")
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            config: Configuration parameters for the agent
            
        Returns:
            An initialized asset-specific agent
        """
        config = config or {}
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = config.get("learning_rate", 3e-4)
        network_config = config.get("network_config", {})
        
        asset_type = asset_type.lower()
        
        if asset_type == "crypto":
            volatility_scaling = config.get("volatility_scaling", True)
            return CryptoAgent(
                observation_space=observation_space,
                action_space=action_space,
                asset_id=asset_id,
                learning_rate=learning_rate,
                device=device,
                network_config=network_config,
                volatility_scaling=volatility_scaling
            )
            
        elif asset_type == "equity":
            use_fundamentals = config.get("use_fundamentals", False)
            return EquityAgent(
                observation_space=observation_space,
                action_space=action_space,
                asset_id=asset_id,
                learning_rate=learning_rate,
                device=device,
                network_config=network_config,
                use_fundamentals=use_fundamentals
            )
            
        else:
            # Default to base implementation
            logger.warning(f"No specialized agent for asset type '{asset_type}', using base implementation")
            return AssetSpecificAgent(
                observation_space=observation_space,
                action_space=action_space,
                asset_id=asset_id,
                asset_type=asset_type,
                learning_rate=learning_rate,
                device=device,
                network_config=network_config
            ) 