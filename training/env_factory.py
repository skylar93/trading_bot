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
- Data validation (NaN, negative prices, minimum row count)
- Train / val / test split with configurable ratios
- Optional feature engineering (RSI, MACD, BB, ATR, OBV, VWAP)

Implementation Notes:
- Uses a unified configuration format for all environment types
- Automatically handles data format conversions
- Enforces consistent column naming with $ prefix
- Applies appropriate environment wrappers based on config
"""

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from envs.multi_agent_env import MultiAgentTradingEnv
from envs.multi_asset_env import MultiAssetTradingEnv
from envs.wrap_env import make_env

# MultiAgentMultiAssetEnv will be imported once implemented
# from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

@dataclass
class DataValidationResult:
    """Result of a data quality check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"valid={self.is_valid}"]
        if self.errors:
            lines.append("ERRORS: " + "; ".join(self.errors))
        if self.warnings:
            lines.append("WARNINGS: " + "; ".join(self.warnings))
        return " | ".join(lines)


def validate_data(
    df: pd.DataFrame,
    min_rows: int = 50,
    price_cols: Optional[List[str]] = None,
) -> DataValidationResult:
    """
    Validate a trading DataFrame.

    Checks:
    - Minimum row count
    - NaN values in OHLCV columns
    - Non-positive prices (open/high/low/close)
    - High < Low inconsistency
    - Zero-volume rows (warning only)

    Returns :class:`DataValidationResult` with ``is_valid=True`` iff no errors.
    """
    if price_cols is None:
        price_cols = ["$open", "$high", "$low", "$close"]

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {"n_rows": len(df), "n_cols": len(df.columns)}

    # --- Minimum rows ---
    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} < {min_rows}")

    # --- Required columns ---
    required = {"$open", "$high", "$low", "$close", "$volume"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        # Can't do further price checks
        return DataValidationResult(is_valid=False, errors=errors,
                                    warnings=warnings, stats=stats)

    # --- NaN check ---
    for col in required:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            errors.append(f"NaN in '{col}': {n_nan} rows")
    stats["nan_counts"] = {c: int(df[c].isna().sum()) for c in required}

    # --- Non-positive prices ---
    for col in price_cols:
        n_nonpos = int((df[col] <= 0).sum())
        if n_nonpos > 0:
            errors.append(f"Non-positive price in '{col}': {n_nonpos} rows")
    stats["price_min"] = {c: float(df[c].min()) for c in price_cols}

    # --- High < Low ---
    bad_hl = int((df["$high"] < df["$low"]).sum())
    if bad_hl > 0:
        errors.append(f"High < Low in {bad_hl} rows")

    # --- Zero volume (warning) ---
    zero_vol = int((df["$volume"] == 0).sum())
    if zero_vol > 0:
        warnings.append(f"Zero volume in {zero_vol} rows")

    # --- Stats ---
    stats["price_stats"] = {
        "close_mean": float(df["$close"].mean()),
        "close_std": float(df["$close"].std()),
        "close_min": float(df["$close"].min()),
        "close_max": float(df["$close"].max()),
    }
    stats["volume_stats"] = {
        "mean": float(df["$volume"].mean()),
        "zero_count": zero_vol,
    }

    return DataValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def log_data_quality_report(
    result: DataValidationResult,
    mlflow_manager=None,
) -> None:
    """Log validation result to Python logger and optionally to MLflow."""
    if result.is_valid:
        logger.info("Data validation PASSED. %s", result.summary())
    else:
        logger.warning("Data validation FAILED. %s", result.summary())

    if mlflow_manager is not None:
        try:
            flat = {
                "data/n_rows": result.stats.get("n_rows", 0),
                "data/validation_passed": int(result.is_valid),
                "data/n_errors": len(result.errors),
                "data/n_warnings": len(result.warnings),
            }
            mlflow_manager.log_metrics(flat)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow data quality log failed: %s", exc)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into (train, val, test) DataFrames chronologically.

    Ratios must sum to 1.0 (±1e-6 tolerance).  The index of each split is
    reset to avoid confusion in downstream code.

    Returns:
        (train_df, val_df, test_df)
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1):
        raise ValueError("Each ratio must be in [0, 1) with train_ratio > 0")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train: n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)

    logger.info(
        "Data split: train=%d, val=%d, test=%d (total=%d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test


def split_data_from_config(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: reads split ratios from config["data"] and calls
    :func:`split_data`.
    """
    data_cfg = config.get("data", {})
    train_ratio = data_cfg.get("train_ratio", 0.7)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)
    return split_data(df, train_ratio=train_ratio, val_ratio=val_ratio,
                      test_ratio=test_ratio)


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
    data: Optional[Union[pd.DataFrame, List[Dict]]] = None,
    validate: bool = True,
    apply_features: bool = False,
    mlflow_manager=None,
) -> Union[SingleAssetRLTradingEnv, MultiAgentTradingEnv, MultiAssetTradingEnv]:
    """
    Create a trading environment based on configuration.

    Args:
        config: Configuration dictionary.
        data: Optional DataFrame (or list of dicts). Loaded from config if None.
        validate: Run :func:`validate_data` and log quality report.
        apply_features: Compute technical indicators via
            :class:`~training.data.feature_engineering.FeatureEngineer` and
            attach them to the DataFrame before passing to the env.
        mlflow_manager: Optional MLflow manager for logging the quality report.

    Returns:
        Trading environment instance.
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

    # Optional: data validation
    if validate:
        min_rows = config.get("data", {}).get("min_rows", 50)
        result = validate_data(data, min_rows=min_rows)
        log_data_quality_report(result, mlflow_manager=mlflow_manager)
        if not result.is_valid:
            raise ValueError(
                f"Data validation failed: {'; '.join(result.errors)}"
            )

    # Optional: feature engineering
    if apply_features:
        try:
            from training.data.feature_engineering import FeatureEngineer, FeatureConfig
            fe_cfg_raw = config.get("feature_engineering", {})
            fe_config = FeatureConfig(**{k: v for k, v in fe_cfg_raw.items()
                                         if hasattr(FeatureConfig, k)})
            fe = FeatureEngineer(fe_config)
            data = fe.compute_features(data)
            logger.info("Feature engineering applied: %s", fe_config.enabled_features)
        except ImportError:
            logger.warning("ta library not available; feature engineering skipped")

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
            sharpe_weight=env_config.get("sharpe_weight", 0.1),
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
    data: pd.DataFrame,
) -> Union[SingleAssetRLTradingEnv, MultiAgentTradingEnv]:
    """
    Create an evaluation environment with the same configuration as the
    training environment.

    Args:
        config: Configuration dictionary.
        data: Evaluation data.

    Returns:
        An environment instance for evaluation.
    """
    eval_config = copy.deepcopy(config)

    # Turn off normalization and stacking for evaluation if present
    # This ensures we get raw observations and rewards for accurate evaluation
    if "env" in eval_config and "normalize" in eval_config["env"]:
        eval_config["env"]["normalize"] = False

    # Create the environment with evaluation data
    return create_env(eval_config, data)
