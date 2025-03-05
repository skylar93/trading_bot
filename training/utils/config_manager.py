"""
Configuration Manager for Training Pipelines.

This module provides utilities to load, validate, and modify YAML configuration files
for both single-agent and multi-agent reinforcement learning training.

Features:
- Load and validate configuration files
- Merge default and custom configurations
- Programmatically modify configuration parameters
- Save configurations for experiment reproducibility
- Document configuration changes for version control

Implementation Notes:
- Uses deep_update pattern to merge nested configurations
- Validates required fields based on environment and agent types
- Supports dot notation for accessing nested config values
"""

import os
import yaml
import copy
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pprint

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for trading bot training pipelines.
    
    Handles loading, validation, and modification of configuration files
    for single-agent and multi-agent training setups.
    
    Features:
    - Load configuration from YAML files
    - Create configuration snapshots for experiment tracking
    - Validate configuration for required fields
    - Modify configuration programmatically with dot notation
    - Save modified configurations
    
    Implementation Notes:
    - Uses deep dictionary merging for overriding defaults
    - Supports both single-agent and multi-agent setups
    - Automatically validates configuration based on agent/env types
    
    Recent Changes:
    - Added HPC configuration support for UW Hyak
    - Added hyperparameter search space definitions
    - Enhanced multi-agent configuration handling
    
    Examples:
    ```python
    # Load default configuration
    config_mgr = ConfigManager()
    config = config_mgr.load_config()
    
    # Override specific parameters
    config_mgr.set("agent.learning_rate", 5e-4)
    config_mgr.set("training.total_timesteps", 200000)
    
    # Save configuration for experiment
    config_mgr.save_snapshot("experiment_001")
    ```
    """
    
    def __init__(self, default_config_path: str = "config/training_config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            default_config_path: Path to the default configuration file
        """
        self.default_config_path = default_config_path
        self.config = None
        self.config_history = []  # Track changes for documentation
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file, defaults to the default config path
            
        Returns:
            The loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file cannot be found
            ValueError: If the configuration is invalid
        """
        path = config_path if config_path else self.default_config_path
        
        # Get the absolute path of the project root
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Join with the config path
        abs_config_path = os.path.join(project_root, path)
        
        try:
            with open(abs_config_path, "r") as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {abs_config_path}")
                
                # Validate the configuration
                self._validate_config(self.config)
                
                # Record initial state for history
                self.config_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "load",
                    "details": f"Loaded configuration from {path}"
                })
                
                return self.config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {abs_config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration value using dot notation (e.g., "agent.learning_rate")
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default if not found
        """
        if not self.config:
            logger.warning("Configuration not loaded yet")
            return default
        
        keys = key_path.split(".")
        value = self.config
        
        try:
            for key in keys:
                if key.isdigit() and isinstance(value, list):
                    # Handle numeric indices for lists
                    value = value[int(key)]
                elif isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, IndexError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration value using dot notation
            value: Value to set
            
        Raises:
            ValueError: If the configuration is not loaded or the key path is invalid
        """
        if not self.config:
            raise ValueError("Configuration not loaded yet")
        
        keys = key_path.split(".")
        target = self.config
        
        # Navigate to the parent of the target key
        for i, key in enumerate(keys[:-1]):
            if key.isdigit() and isinstance(target, list):
                # Handle numeric indices for lists
                key_idx = int(key)
                if key_idx >= len(target):
                    # Extend the list if needed
                    target.extend([{} for _ in range(key_idx - len(target) + 1)])
                if i == len(keys) - 2 and not isinstance(target[key_idx], dict) and not isinstance(target[key_idx], list):
                    target[key_idx] = {}
                target = target[key_idx]
            elif key in target:
                if i == len(keys) - 2 and not isinstance(target[key], dict) and not isinstance(target[key], list):
                    target[key] = {}
                target = target[key]
            else:
                # Create nested dictionaries as needed
                target[key] = {}
                target = target[key]
        
        # Set the value
        last_key = keys[-1]
        if last_key.isdigit() and isinstance(target, list):
            key_idx = int(last_key)
            if key_idx >= len(target):
                # Extend the list if needed
                target.extend([None for _ in range(key_idx - len(target) + 1)])
            target[key_idx] = value
        else:
            target[last_key] = value
        
        # Record the change in history
        self.config_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "set",
            "key": key_path,
            "value": value
        })
        
        logger.debug(f"Set configuration {key_path} = {value}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with multiple values.
        
        Args:
            updates: Dictionary of dot notation paths to values
        """
        for key_path, value in updates.items():
            self.set(key_path, value)
    
    def save(self, config_path: str) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file
            
        Raises:
            ValueError: If the configuration is not loaded
        """
        if not self.config:
            raise ValueError("Configuration not loaded yet")
        
        # Get the absolute path of the project root
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Join with the config path
        abs_config_path = os.path.join(project_root, config_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(abs_config_path), exist_ok=True)
        
        with open(abs_config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Saved configuration to {abs_config_path}")
        
        # Record the save in history
        self.config_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "save",
            "details": f"Saved configuration to {config_path}"
        })
    
    def save_snapshot(self, experiment_id: str) -> str:
        """
        Save a snapshot of the current configuration for experiment tracking.
        
        Args:
            experiment_id: Identifier for the experiment
            
        Returns:
            Path to the saved snapshot
            
        Raises:
            ValueError: If the configuration is not loaded
        """
        if not self.config:
            raise ValueError("Configuration not loaded yet")
        
        # Create the snapshot filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"config_{experiment_id}_{timestamp}.yaml"
        snapshot_dir = "config/snapshots"
        snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
        
        # Get the absolute path of the project root
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Create the snapshots directory if it doesn't exist
        abs_snapshot_dir = os.path.join(project_root, snapshot_dir)
        os.makedirs(abs_snapshot_dir, exist_ok=True)
        
        # Save the snapshot
        abs_snapshot_path = os.path.join(project_root, snapshot_path)
        
        # Add metadata to the config
        snapshot_config = copy.deepcopy(self.config)
        if "metadata" not in snapshot_config:
            snapshot_config["metadata"] = {}
        
        snapshot_config["metadata"]["experiment_id"] = experiment_id
        snapshot_config["metadata"]["timestamp"] = timestamp
        snapshot_config["metadata"]["history"] = self.config_history
        
        with open(abs_snapshot_path, "w") as f:
            yaml.dump(snapshot_config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Saved configuration snapshot to {abs_snapshot_path}")
        
        return snapshot_path
    
    def reset(self) -> None:
        """Reset the configuration to the default."""
        self.config = None
        self.config_history = []
        self.load_config()
        
        logger.info("Reset configuration to default")
    
    def show(self) -> str:
        """
        Get a formatted string representation of the current configuration.
        
        Returns:
            A string representation of the configuration
            
        Raises:
            ValueError: If the configuration is not loaded
        """
        if not self.config:
            raise ValueError("Configuration not loaded yet")
        
        return pprint.pformat(self.config, indent=2)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration for required fields.
        
        Args:
            config: The configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Basic validation
        required_top_level = ["env", "agent", "training", "paths"]
        for key in required_top_level:
            if key not in config:
                raise ValueError(f"Missing required top-level configuration section: {key}")
        
        # Environment type specific validation
        env_type = config["env"].get("type", "single_asset_rl")
        
        if env_type == "single_asset_rl":
            # Validate single-agent environment config
            required_env_fields = ["initial_capital", "trading_fee", "window_size"]
            for field in required_env_fields:
                if field not in config["env"]:
                    raise ValueError(f"Missing required field in env configuration: {field}")
        
        elif env_type == "multi_agent_rl":
            # Validate multi-agent environment config
            if "multi_agent_configs" not in config["env"]:
                raise ValueError("Missing multi_agent_configs in multi-agent environment configuration")
            
            for i, agent_config in enumerate(config["env"]["multi_agent_configs"]):
                required_agent_fields = ["id", "type"]
                for field in required_agent_fields:
                    if field not in agent_config:
                        raise ValueError(f"Missing required field {field} in agent config at index {i}")
        
        # Agent validation
        agent_name = config["agent"].get("name")
        if not agent_name:
            raise ValueError("Missing agent name in agent configuration")
        
        # Training validation
        required_training_fields = ["total_timesteps"]
        for field in required_training_fields:
            if field not in config["training"]:
                raise ValueError(f"Missing required field in training configuration: {field}")
        
        logger.debug("Configuration validation passed")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        The loaded configuration dictionary
    """
    config_manager = ConfigManager()
    return config_manager.load_config(config_path)


def deep_update(source: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a nested dictionary.
    
    Args:
        source: The source dictionary to update
        updates: The updates to apply
        
    Returns:
        The updated dictionary
    """
    result = copy.deepcopy(source)
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    
    return result 