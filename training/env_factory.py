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
from envs.multi_asset_env import MultiAssetTradingEnv
from envs.wrap_env import make_env

# MultiAgentMultiAssetEnv will be imported once implemented
# from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv

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
    
    # Process datetime columns: convert to numeric or drop
    for col in df.columns:
        # Check for datetime-like string columns
        if df[col].dtype == 'object':
            try:
                # Try to convert to datetime
                test_datetime = pd.to_datetime(df[col], errors='coerce')
                if not test_datetime.isna().all():
                    # Convert successful datetime columns to numeric (days since epoch)
                    df[col] = test_datetime.astype(np.int64) // 10**9 // 86400
                    logger.info(f"Converted datetime column '{col}' to numeric days since epoch")
            except Exception:
                # If conversion fails, leave as is
                pass
    
    # Reset datetime index if present
    if isinstance(df.index, pd.DatetimeIndex):
        logger.info("Converting datetime index to numeric index")
        df = df.reset_index(drop=True)
    
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
) -> Union[SingleAssetRLTradingEnv, MultiAgentTradingEnv, MultiAssetTradingEnv]:
    """
    Create a trading environment based on configuration.
    
    Args:
        config: Configuration dictionary
        data: Optional data to use (if not provided, will load from config)
        
    Returns:
        Trading environment instance
    """
    # Get environment type
    env_type = config["env"]["type"]
    env_config = config["env"]
    
    # Load data if not provided
    if data is None:
        data_path = config.get("data", {}).get("data_path", None)
        if data_path:
            data = load_data(data_path)
        else:
            raise ValueError("No data provided and no data_path in config")
    
    # Ensure data is in the correct format
    data = normalize_data_format(data)
    
    # Create environment based on type
    if env_type == "single_asset_rl":
        # Create single-asset environment with only supported parameters
        env = SingleAssetRLTradingEnv(
            data=data,
            window_size=env_config.get("window_size", 20),
            initial_capital=env_config.get("initial_balance", 10000.0),
            trading_fee=env_config.get("trading_fee", 0.001),
            max_position_size=env_config.get("max_position_size", 1.0),
            # Risk reward parameters
            risk_adjusted_reward=env_config.get("risk_adjusted_reward", True),
            sharpe_lookback=env_config.get("sharpe_lookback", 30),
            sharpe_weight=env_config.get("sharpe_weight", 0.5),
            drawdown_penalty=env_config.get("drawdown_penalty", True),
            # Friction parameters
            apply_slippage=env_config.get("apply_slippage", True),
            slippage_factor=env_config.get("slippage_factor", 0.0005),
            partial_fills=env_config.get("partial_fills", True)
        )
        
        logger.info(f"Created single-agent environment: {env}")
        return env
    
    elif env_type == "multi_asset_rl":
        # Create multi-asset environment with only supported parameters
        env = MultiAssetTradingEnv(
            df=data,
            window_size=env_config.get("window_size", 20),
            initial_balance=env_config.get("initial_balance", 10000.0),
            trading_fee=env_config.get("trading_fee", 0.001),
            max_position_size=env_config.get("max_position_size", 1.0),
            action_type=env_config.get("action_type", "portfolio_weights"),
            allow_short=env_config.get("allow_short", False),
            rebalance_freq=env_config.get("rebalance_freq", 1)
        )
        
        logger.info(f"Created multi-asset environment with {len(env.assets)} assets")
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
                "type": agent_cfg.get("agent_type", "ppo"),  # Changed from type to agent_type
                "strategy": agent_cfg.get("strategy", ""),  # Made strategy optional
                "initial_balance": agent_cfg.get("initial_balance", agent_balance),  # Use provided balance or calculate
                "initial_capital_percentage": initial_capital_percentage,
                "priority": agent_cfg.get("priority", 1),
                # Additional environment-related agent parameters
            })
        
        # Create multi-agent environment
        env = MultiAgentTradingEnv(
            data=data,
            agent_configs=agent_configs,
            window_size=env_config.get("window_size", 20),
            trading_fee=env_config.get("trading_fee", 0.001),
            shared_capital=env_config.get("shared_capital", False),
            capital_reallocation_freq=env_config.get("capital_reallocation_freq", 20)
        )
        
        logger.info(f"Created multi-agent environment with {len(agent_configs)} agents")
        return env
    
    elif env_type == "multi_asset_multi_agent_rl":
        # Get multi-agent configurations
        multi_agent_configs = env_config.get("multi_agent_configs", [])
        
        if not multi_agent_configs:
            raise ValueError("No agent configurations provided for multi-agent environment")
        
        # Get total environment balance
        total_balance = env_config.get("initial_balance", 10000.0)
        
        # Convert configurations to format expected by MultiAgentMultiAssetEnv
        agent_configs = []
        for agent_cfg in multi_agent_configs:
            # Calculate agent's initial balance based on percentage
            initial_capital_percentage = agent_cfg.get("initial_capital_percentage", 1.0)
            agent_balance = total_balance * initial_capital_percentage
            
            # Get asset assignment for this agent (if specified)
            assigned_assets = agent_cfg.get("assigned_assets", None)
            
            agent_configs.append({
                "id": agent_cfg["id"],
                "type": agent_cfg.get("agent_type", "ppo"),
                "strategy": agent_cfg.get("strategy", ""),
                "initial_balance": agent_cfg.get("initial_balance", agent_balance),
                "initial_capital_percentage": initial_capital_percentage,
                "priority": agent_cfg.get("priority", 1),
                "assigned_assets": assigned_assets  # Assets this agent is responsible for
            })
        
        logger.warning("MultiAgentMultiAssetEnv not yet implemented - importing placeholder implementation")
        try:
            # Try to import MultiAgentMultiAssetEnv
            from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv
            
            # Create multi-agent multi-asset environment
            env = MultiAgentMultiAssetEnv(
                data=data,
                agent_configs=agent_configs,
                window_size=env_config.get("window_size", 20),
                trading_fee=env_config.get("trading_fee", 0.001),
                action_type=env_config.get("action_type", "portfolio_weights"),
                shared_capital=env_config.get("shared_capital", True),
                capital_reallocation_freq=env_config.get("capital_reallocation_freq", 20)
            )
            
            logger.info(f"Created multi-agent multi-asset environment with {len(agent_configs)} agents")
            return env
            
        except ImportError:
            logger.error("MultiAgentMultiAssetEnv not found - falling back to MultiAssetTradingEnv with warning")
            logger.warning("Using MultiAssetTradingEnv as fallback - multi-agent functionality will not be available")
            
            # Create multi-asset environment as fallback
            env = MultiAssetTradingEnv(
                df=data,
                window_size=env_config.get("window_size", 20),
                initial_balance=env_config.get("initial_balance", 10000.0),
                trading_fee=env_config.get("trading_fee", 0.001),
                action_type=env_config.get("action_type", "portfolio_weights")
            )
            
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