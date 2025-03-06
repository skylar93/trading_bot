"""
Environment Factory for Trading Environments.

This module provides a centralized factory for creating trading environments
based on configuration. It supports both single-agent and multi-agent
reinforcement learning environments.

Features:
- Create environments from configuration
- Support for single-asset and multi-agent environments
- Automatic data loading and preprocessing
- Environment wrapping with normalization and stacking

Implementation Notes:
- Uses a unified configuration format for all environment types
- Automatically handles data format conversions
- Enforces consistent column naming with $ prefix
- Applies appropriate environment wrappers based on config

Recent Changes:
- Added support for multi-agent environment creation
- Enhanced data preprocessing for trading environments
- Improved error handling and validation
- Added support for risk-adjusted rewards and realistic frictions
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from envs.multi_agent_env import MultiAgentTradingEnv
from envs.wrap_env import make_env

logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a file path.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        DataFrame with the loaded data
        
    Raises:
        FileNotFoundError: If the data file cannot be found
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Determine file type from extension
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(data_path)
    elif ext in ['.pkl', '.pickle']:
        df = pd.read_pickle(data_path)
    elif ext == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Ensure column names have '$' prefix for OHLCV data
    rename_dict = {}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            rename_dict[col] = f"${col}"
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Renamed columns: {rename_dict}")
    
    logger.info(f"Loaded data from {data_path} with shape {df.shape}")
    return df

def normalize_data_format(data: Any) -> pd.DataFrame:
    """
    Normalize data into a pandas DataFrame with proper format.
    
    Args:
        data: Data in various formats (DataFrame, list of dicts, etc.)
        
    Returns:
        Normalized DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data
    
    # Handle list of dictionaries
    if isinstance(data, (list, tuple)) and all(isinstance(d, dict) for d in data):
        max_len = max(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for d in data
            for v in d.values()
        )
        normalized_data = {}
        for k, v in data[0].items():
            if not isinstance(v, (list, np.ndarray)):
                v = [v] * max_len
            normalized_data[k] = v
        return pd.DataFrame(normalized_data)
    
    raise ValueError("Unsupported data format. Expected DataFrame or list of dicts.")

def create_env(
    config: Dict[str, Any], 
    data: Optional[Union[pd.DataFrame, List[Dict]]] = None
) -> Union[SingleAssetRLTradingEnv, MultiAgentTradingEnv]:
    """
    Create a trading environment based on configuration.
    
    Args:
        config: Configuration dictionary with environment settings
        data: Optional data to use (if not provided, will be loaded from config)
        
    Returns:
        An instance of a trading environment
        
    Raises:
        ValueError: If the configuration is invalid or the environment type is unsupported
    """
    # Extract environment configuration
    env_config = config.get("env", {})
    env_type = env_config.get("type", "single_asset_rl")
    
    # Load data if not provided
    if data is None:
        data_config = config.get("data", {})
        data_path = data_config.get("data_path")
        
        if not data_path:
            raise ValueError("No data path specified in configuration")
        
        data = load_data(data_path)
    else:
        # Normalize data format if provided
        data = normalize_data_format(data)
    
    # Create the appropriate environment
    if env_type == "single_asset_rl":
        # Basic parameters
        window_size = env_config.get("window_size", 10)
        initial_capital = env_config.get("initial_capital", 10000.0)
        trading_fee = env_config.get("trading_fee", 0.001)
        max_position_size = env_config.get("max_position_size", 1.0)
        
        # Risk-oriented reward parameters
        risk_config = env_config.get("risk_reward", {})
        risk_adjusted_reward = risk_config.get("enabled", True)
        sharpe_lookback = risk_config.get("sharpe_lookback", 30)
        sharpe_weight = risk_config.get("sharpe_weight", 0.5)
        drawdown_penalty = risk_config.get("drawdown_penalty", True)
        max_drawdown_penalty_threshold = risk_config.get("max_drawdown_threshold", 0.1)
        
        # Market friction parameters
        friction_config = env_config.get("friction", {})
        apply_slippage = friction_config.get("apply_slippage", True)
        slippage_factor = friction_config.get("slippage_factor", 0.0005)
        partial_fills = friction_config.get("partial_fills", True)
        min_fill_rate = friction_config.get("min_fill_rate", 0.8)
        volume_slippage_factor = friction_config.get("volume_slippage_factor", 0.1)
        
        logger.info(
            f"Creating SingleAssetRLTradingEnv with window_size={window_size}, "
            f"initial_capital={initial_capital}, trading_fee={trading_fee}, "
            f"risk_adjusted_reward={risk_adjusted_reward}, apply_slippage={apply_slippage}"
        )
        
        env = SingleAssetRLTradingEnv(
            data=data,
            window_size=window_size,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            max_position_size=max_position_size,
            # Risk reward parameters
            risk_adjusted_reward=risk_adjusted_reward,
            sharpe_lookback=sharpe_lookback,
            sharpe_weight=sharpe_weight,
            drawdown_penalty=drawdown_penalty,
            max_drawdown_penalty_threshold=max_drawdown_penalty_threshold,
            # Friction parameters
            apply_slippage=apply_slippage,
            slippage_factor=slippage_factor,
            partial_fills=partial_fills,
            min_fill_rate=min_fill_rate,
            volume_slippage_factor=volume_slippage_factor,
        )
        
        # Apply wrappers if specified
        if env_config.get("normalize", False) or env_config.get("stack_size", 0) > 0:
            env = make_env(
                env,
                normalize=env_config.get("normalize", False),
                stack_size=env_config.get("stack_size", 4)
            )
        
        logger.info(f"Created single-agent environment: {env}")
        return env
    
    elif env_type == "multi_agent_rl":
        # Get multi-agent configurations
        multi_agent_configs = env_config.get("multi_agent_configs", [])
        
        if not multi_agent_configs:
            raise ValueError("No agent configurations provided for multi-agent environment")
        
        # Get total environment balance
        total_balance = env_config.get("initial_balance", 10000.0)
        
        # Convert configurations to format expected by MultiAgentTradingEnv
        agent_configs = []
        for agent_cfg in multi_agent_configs:
            # Calculate agent's initial balance based on percentage
            initial_capital_percentage = agent_cfg.get("initial_capital_percentage", 1.0)
            agent_balance = total_balance * initial_capital_percentage
            
            agent_configs.append({
                "id": agent_cfg["id"],
                "type": agent_cfg["type"],
                "strategy": agent_cfg["strategy"],
                "initial_balance": agent_balance,  # Added calculated initial balance
                "initial_capital_percentage": initial_capital_percentage,
                "priority": agent_cfg.get("priority", 1),
                # Additional environment-related agent parameters
            })
        
        # Create multi-agent environment
        env = MultiAgentTradingEnv(
            data=data,
            agent_configs=agent_configs,
            window_size=env_config.get("window_size", 20),
            trading_fee=env_config.get("trading_fee", 0.001)
        )
        
        logger.info(f"Created multi-agent environment with {len(agent_configs)} agents")
        return env
    
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

def create_eval_env(
    config: Dict[str, Any], 
    data: pd.DataFrame
) -> Union[SingleAssetRLTradingEnv, MultiAgentTradingEnv]:
    """
    Create an evaluation environment with the same configuration as the training environment.
    
    Args:
        config: Configuration dictionary
        data: Evaluation data
        
    Returns:
        An environment instance for evaluation
    """
    # Create a deep copy of the config to avoid modifying the original
    import copy
    eval_config = copy.deepcopy(config)
    
    # Turn off normalization and stacking for evaluation if present
    # This ensures we get raw observations and rewards for accurate evaluation
    if "env" in eval_config and "normalize" in eval_config["env"]:
        eval_config["env"]["normalize"] = False
    
    # Create the environment with evaluation data
    return create_env(eval_config, data) 