"""Configuration management utility"""

import os
import yaml

def load_config(project_root):
    """Load or create config"""
    config_path = os.path.join(project_root, "config/default_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        return create_default_config(config_path)

def create_default_config(config_path):
    """Create default config file"""
    config = {
        'env': {
            'initial_balance': 10000,
            'trading_fee': 0.001,
            'window_size': 20
        },
        'model': {
            'hidden_size': 256,
            'num_layers': 2
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.0003,
            'num_episodes': 1000
        }
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config