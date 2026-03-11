"""
Model Training Page for the Trading Bot.

This page provides a user interface for configuring and running
reinforcement learning model training for trading. It supports both
single-asset and multi-asset trading, as well as single-agent and 
multi-agent training scenarios.

Features:
- Environment configuration
  - Single or multi-asset trading selection
  - Asset symbol(s) selection
  - Timeframe and date range settings
- Agent/algorithm selection
- Training hyperparameter tuning
- Multi-agent setup with asset assignment
- Real-time training progress monitoring
- Integration with MLflow for experiment tracking

Implementation Notes:
- Uses the training_manager to interface with the training pipeline
- Provides real-time progress updates during training
- Visualizes training metrics using Streamlit components
- Supports configuration of complex multi-asset multi-agent setups
- Allows agents to be assigned specific assets in multi-asset mode
"""

import os
import sys
import asyncio
import time
from datetime import datetime, timedelta
import json
import yaml
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from deployment.web_interface.utils.state import init_session_state
from deployment.web_interface.training_manager import TrainingManager

def model_training_page():
    """
    Render the model training page.
    """
    st.title("Model Training")
    init_session_state()
    
    # Initialize training status if not present
    if "training_status" not in st.session_state:
        st.session_state.training_status = {
            "is_training": False,
            "start_time": None,
            "progress": 0.0,
            "metrics": {}
        }
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Environment Settings", 
        "Agent Settings", 
        "Training Parameters",
        "Run Training"
    ])
    
    # If no training config in session state, initialize it
    if "training_config" not in st.session_state:
        st.session_state.training_config = {
            "env": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "window_size": 20,
                "use_stop_loss": True,
                "stop_loss_pct": 5.0,
                "multi_agent": False,
                "agent_count": 2
            },
            "agent": {
                "algorithm": "ppo",
                "hidden_layers": [64, 64],
                "use_lstm": False,
                "lstm_hidden_size": 64
            },
            "training": {
                "total_timesteps": 100000,
                "batch_size": 64,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "eval_interval": 5000,
                "checkpoint_interval": 10000,
                "seed": 42
            }
        }
    
    # Environment Settings Tab
    with tab1:
        st.header("Environment Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Asset Selection
            st.subheader("Asset Selection")
            
            # Add Asset Mode selection
            asset_mode = st.selectbox(
                "Asset Mode",
                ["single", "multi"],
                index=0 if st.session_state.training_config["env"].get("asset_mode", "single") == "single" else 1
            )
            st.session_state.training_config["env"]["asset_mode"] = asset_mode
            
            if asset_mode == "single":
                # Single symbol selection
                symbol = st.selectbox(
                    "Symbol", 
                    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
                    index=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"].index(
                        st.session_state.training_config["env"]["symbol"]
                    )
                )
                st.session_state.training_config["env"]["symbol"] = symbol
                # Store as a list for consistency
                st.session_state.training_config["env"]["symbols"] = [symbol]
            else:
                # Multi symbol selection
                available_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
                default_symbols = st.session_state.training_config["env"].get("symbols", ["BTC/USDT", "ETH/USDT"])
                # Filter default symbols to ensure they're all in available_symbols
                default_symbols = [s for s in default_symbols if s in available_symbols]
                
                selected_symbols = st.multiselect(
                    "Select Multiple Assets",
                    available_symbols,
                    default=default_symbols
                )
                
                # Ensure at least one symbol is selected
                if not selected_symbols:
                    selected_symbols = ["BTC/USDT"]
                    st.warning("At least one symbol must be selected. Defaulting to BTC/USDT.")
                
                st.session_state.training_config["env"]["symbols"] = selected_symbols
                # For backwards compatibility, also set the symbol field to the first selected symbol
                st.session_state.training_config["env"]["symbol"] = selected_symbols[0]
            
            st.session_state.training_config["env"]["timeframe"] = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                index=["1m", "5m", "15m", "1h", "4h", "1d"].index(
                    st.session_state.training_config["env"]["timeframe"]
                )
            )
            
            # Date Range
            st.subheader("Date Range")
            start_date = datetime.strptime(st.session_state.training_config["env"]["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(st.session_state.training_config["env"]["end_date"], "%Y-%m-%d")
            
            new_start_date = st.date_input("Start Date", start_date)
            new_end_date = st.date_input("End Date", end_date)
            
            st.session_state.training_config["env"]["start_date"] = new_start_date.strftime("%Y-%m-%d")
            st.session_state.training_config["env"]["end_date"] = new_end_date.strftime("%Y-%m-%d")
            
            # Window Size
            st.subheader("Observation Window")
            st.session_state.training_config["env"]["window_size"] = st.slider(
                "Window Size",
                min_value=5,
                max_value=100,
                value=st.session_state.training_config["env"]["window_size"],
                step=5,
                help="Number of past time steps to include in each observation"
            )
        
        with col2:
            # Risk Management
            st.subheader("Risk Management")
            use_stop_loss = st.checkbox(
                "Use Stop Loss",
                value=st.session_state.training_config["env"]["use_stop_loss"]
            )
            st.session_state.training_config["env"]["use_stop_loss"] = use_stop_loss
            
            if use_stop_loss:
                stop_loss_pct = st.slider(
                    "Stop Loss (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=st.session_state.training_config["env"]["stop_loss_pct"],
                    step=0.5
                )
                st.session_state.training_config["env"]["stop_loss_pct"] = stop_loss_pct
            
            # Multi-Agent Configuration
            st.subheader("Multi-Agent Setup")
            multi_agent = st.checkbox(
                "Enable Multi-Agent Training",
                value=st.session_state.training_config["env"]["multi_agent"]
            )
            st.session_state.training_config["env"]["multi_agent"] = multi_agent
            
            # Determine environment type based on asset mode and multi-agent settings
            if asset_mode == "single" and not multi_agent:
                env_type = "single_asset_rl"
            elif asset_mode == "multi" and not multi_agent:
                env_type = "multi_asset_rl"
            elif asset_mode == "single" and multi_agent:
                env_type = "multi_agent_rl"
            elif asset_mode == "multi" and multi_agent:
                env_type = "multi_asset_multi_agent_rl"
            
            st.session_state.training_config["env"]["type"] = env_type
            
            if multi_agent:
                agent_count = st.slider(
                    "Number of Agents",
                    min_value=2,
                    max_value=5,
                    value=st.session_state.training_config["env"]["agent_count"]
                )
                st.session_state.training_config["env"]["agent_count"] = agent_count
                
                # Add manager configuration
                use_manager = st.checkbox(
                    "Enable Manager (Meta-Agent & Shared Buffer)",
                    value=st.session_state.training_config["env"].get("use_manager", False)
                )
                st.session_state.training_config["env"]["use_manager"] = use_manager
                
                ensemble_method = st.selectbox(
                    "Ensemble Method",
                    ["weighted", "best", "meta"],
                    index=["weighted", "best", "meta"].index(
                        st.session_state.training_config["env"].get("ensemble_method", "weighted")
                    )
                )
                st.session_state.training_config["env"]["ensemble_method"] = ensemble_method
                
                # Meta-agent configuration when ensemble method is "meta"
                if ensemble_method == "meta":
                    st.write("Meta-Agent Configuration")
                    
                    # Using a subsection with divider instead of expander
                    st.divider()
                    st.subheader("Meta-Agent Parameters", anchor=False)
                    
                    if "meta_config" not in st.session_state.training_config["env"]:
                        st.session_state.training_config["env"]["meta_config"] = {
                            "learning_rate": 3e-4,
                            "hidden_dim": 128,
                            "continuous_ensemble": True
                        }
                    
                    meta_lr = st.number_input(
                        "Meta Learning Rate",
                        min_value=1e-5,
                        max_value=1e-2,
                        value=float(st.session_state.training_config["env"]["meta_config"].get("learning_rate", 3e-4)),
                        format="%.5f"
                    )
                    st.session_state.training_config["env"]["meta_config"]["learning_rate"] = meta_lr
                    
                    meta_hidden_dim = st.number_input(
                        "Meta Hidden Dimension",
                        min_value=32,
                        max_value=512,
                        value=int(st.session_state.training_config["env"]["meta_config"].get("hidden_dim", 128)),
                        step=32
                    )
                    st.session_state.training_config["env"]["meta_config"]["hidden_dim"] = meta_hidden_dim
                    
                    continuous_ensemble = st.checkbox(
                        "Use Continuous Ensemble (weighted combination)",
                        value=st.session_state.training_config["env"]["meta_config"].get("continuous_ensemble", True)
                    )
                    st.session_state.training_config["env"]["meta_config"]["continuous_ensemble"] = continuous_ensemble
                
                # Shared experience buffer configuration
                if use_manager:
                    # Using a subsection with divider instead of expander
                    st.divider()
                    st.subheader("Shared Experience Buffer", anchor=False)
                    
                    if "shared_buffer" not in st.session_state.training_config["env"]:
                        st.session_state.training_config["env"]["shared_buffer"] = {
                            "enabled": True,
                            "min_share_reward": 0.2,
                            "max_buffer_size": 10000
                        }
                    
                    enable_shared_buffer = st.checkbox(
                        "Enable Shared Experience Buffer",
                        value=st.session_state.training_config["env"]["shared_buffer"].get("enabled", True)
                    )
                    st.session_state.training_config["env"]["shared_buffer"]["enabled"] = enable_shared_buffer
                    
                    if enable_shared_buffer:
                        min_share_reward = st.slider(
                            "Minimum Reward Threshold for Sharing",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(st.session_state.training_config["env"]["shared_buffer"].get("min_share_reward", 0.2)),
                            step=0.05
                        )
                        st.session_state.training_config["env"]["shared_buffer"]["min_share_reward"] = min_share_reward
                        
                        max_buffer_size = st.number_input(
                            "Maximum Buffer Size",
                            min_value=1000,
                            max_value=100000,
                            value=int(st.session_state.training_config["env"]["shared_buffer"].get("max_buffer_size", 10000)),
                            step=1000
                        )
                        st.session_state.training_config["env"]["shared_buffer"]["max_buffer_size"] = max_buffer_size
                
                # Configure each agent
                st.subheader("Agent Configuration")
                
                for i in range(agent_count):
                    with st.expander(f"Agent {i+1}"):
                        if f"agent_{i}" not in st.session_state.training_config["env"]:
                            st.session_state.training_config["env"][f"agent_{i}"] = {
                                "type": "ppo",
                                "strategy": None,
                                "capital_pct": 1.0 / agent_count,
                                "hidden_layers": [64, 64],
                                "learning_rate": 3e-4
                            }
                        
                        # Separate agent type (algorithm) and strategy
                        agent_type = st.selectbox(
                            "Learning Algorithm",
                            ["ppo", "sac", "dqn"],
                            key=f"agent_algo_{i}",
                            index=["ppo", "sac", "dqn"].index(
                                st.session_state.training_config["env"][f"agent_{i}"].get("type", "ppo")
                            ) if st.session_state.training_config["env"][f"agent_{i}"].get("type", "ppo") in ["ppo", "sac", "dqn"] else 0
                        )
                        st.session_state.training_config["env"][f"agent_{i}"]["type"] = agent_type
                        
                        strategy = st.selectbox(
                            "Trading Strategy",
                            [None, "momentum", "mean_reversion", "trend_following"],
                            key=f"agent_strategy_{i}",
                            index=[None, "momentum", "mean_reversion", "trend_following"].index(
                                st.session_state.training_config["env"][f"agent_{i}"].get("strategy")
                            ) if st.session_state.training_config["env"][f"agent_{i}"].get("strategy") in [None, "momentum", "mean_reversion", "trend_following"] else 0
                        )
                        st.session_state.training_config["env"][f"agent_{i}"]["strategy"] = strategy
                        
                        # Add assigned assets selection for multi-asset mode
                        if asset_mode == "multi":
                            available_symbols = st.session_state.training_config["env"]["symbols"]
                            default_assigned = st.session_state.training_config["env"][f"agent_{i}"].get("assigned_assets", available_symbols)
                            # Filter default_assigned to ensure they're all in available_symbols
                            default_assigned = [s for s in default_assigned if s in available_symbols]
                            
                            assigned_assets = st.multiselect(
                                "Assigned Assets",
                                available_symbols,
                                default=default_assigned,
                                key=f"assigned_assets_{i}"
                            )
                            
                            # Ensure at least one symbol is assigned
                            if not assigned_assets:
                                assigned_assets = [available_symbols[0]]
                                st.warning(f"Agent {i+1} must have at least one assigned asset. Defaulting to {available_symbols[0]}.")
                            
                            st.session_state.training_config["env"][f"agent_{i}"]["assigned_assets"] = assigned_assets
                        
                        capital_pct = st.slider(
                            "Capital Allocation (%)",
                            min_value=5.0,
                            max_value=100.0,
                            value=float(st.session_state.training_config["env"][f"agent_{i}"]["capital_pct"] * 100),
                            step=5.0,
                            key=f"capital_pct_{i}"
                        )
                        st.session_state.training_config["env"][f"agent_{i}"]["capital_pct"] = capital_pct / 100.0
                        
                        # Add hyperparameters section instead of nested expander
                        st.divider()
                        st.subheader("Hyperparameters", anchor=False)
                        
                        if "hyperparameters" not in st.session_state.training_config["env"][f"agent_{i}"]:
                            st.session_state.training_config["env"][f"agent_{i}"]["hyperparameters"] = {
                                "learning_rate": 3e-4,
                                "hidden_sizes": [64, 64]
                            }
                        
                        # Learning rate
                        learning_rate = st.number_input(
                            "Learning Rate",
                            min_value=1e-5,
                            max_value=1e-2,
                            value=float(st.session_state.training_config["env"][f"agent_{i}"]["hyperparameters"].get("learning_rate", 3e-4)),
                            format="%.5f",
                            key=f"learning_rate_{i}"
                        )
                        st.session_state.training_config["env"][f"agent_{i}"]["hyperparameters"]["learning_rate"] = learning_rate
                        
                        # Hidden layers
                        hidden_layers_options = [
                            [32, 32],
                            [64, 64],
                            [128, 128],
                            [64, 64, 64],
                            [128, 128, 128],
                            [256, 256]
                        ]
                        
                        current_hidden = st.session_state.training_config["env"][f"agent_{i}"]["hyperparameters"].get("hidden_sizes", [64, 64])
                        hidden_layers_index = 0
                        for idx, layers in enumerate(hidden_layers_options):
                            if layers == current_hidden:
                                hidden_layers_index = idx
                                break
                        
                        hidden_layers = st.selectbox(
                            "Hidden Layers",
                            [str(layers) for layers in hidden_layers_options],
                            index=hidden_layers_index,
                            key=f"hidden_layers_{i}"
                        )
                        selected_idx = ["[32, 32]", "[64, 64]", "[128, 128]", "[64, 64, 64]", "[128, 128, 128]", "[256, 256]"].index(hidden_layers)
                        st.session_state.training_config["env"][f"agent_{i}"]["hyperparameters"]["hidden_sizes"] = hidden_layers_options[selected_idx]

    # Agent Settings Tab
    with tab2:
        st.header("Agent Settings")
        
        # Only show these settings for single agent or if using meta-agent for multi-agent
        if not st.session_state.training_config["env"]["multi_agent"] or \
           st.session_state.training_config["env"].get("ensemble_method") == "meta":
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Algorithm")
                algorithm = st.selectbox(
                    "Algorithm Type",
                    ["ppo", "sac", "dqn", "a2c"],
                    index=["ppo", "sac", "dqn", "a2c"].index(
                        st.session_state.training_config["agent"]["algorithm"]
                    )
                )
                st.session_state.training_config["agent"]["algorithm"] = algorithm
                
                st.subheader("Network Structure")
                hidden_layers_options = [
                    [32, 32],
                    [64, 64],
                    [128, 128],
                    [64, 64, 64],
                    [128, 128, 128],
                    [256, 256]
                ]
                hidden_layers_index = 0
                for i, layers in enumerate(hidden_layers_options):
                    if layers == st.session_state.training_config["agent"]["hidden_layers"]:
                        hidden_layers_index = i
                        break
                
                hidden_layers = st.selectbox(
                    "Hidden Layers",
                    [str(layers) for layers in hidden_layers_options],
                    index=hidden_layers_index
                )
                st.session_state.training_config["agent"]["hidden_layers"] = \
                    hidden_layers_options[["[32, 32]", "[64, 64]", "[128, 128]", "[64, 64, 64]", "[128, 128, 128]", "[256, 256]"].index(hidden_layers)]
            
            with col2:
                st.subheader("LSTM Configuration")
                use_lstm = st.checkbox(
                    "Use LSTM",
                    value=st.session_state.training_config["agent"]["use_lstm"]
                )
                st.session_state.training_config["agent"]["use_lstm"] = use_lstm
                
                if use_lstm:
                    lstm_hidden_size = st.slider(
                        "LSTM Hidden Size",
                        min_value=16,
                        max_value=256,
                        value=st.session_state.training_config["agent"]["lstm_hidden_size"],
                        step=16
                    )
                    st.session_state.training_config["agent"]["lstm_hidden_size"] = lstm_hidden_size
        
        else:
            st.info("Agent settings are configured individually in the Multi-Agent Setup section.")
    
    # Training Parameters Tab
    with tab3:
        st.header("Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("General Parameters")
            
            total_timesteps = st.number_input(
                "Total Timesteps",
                min_value=10000,
                max_value=1000000,
                value=st.session_state.training_config["training"]["total_timesteps"],
                step=10000,
                help="Total number of environment steps for training"
            )
            st.session_state.training_config["training"]["total_timesteps"] = total_timesteps
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=16,
                max_value=2048,
                value=st.session_state.training_config["training"]["batch_size"],
                step=16
            )
            st.session_state.training_config["training"]["batch_size"] = batch_size
            
            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=st.session_state.training_config["training"]["seed"],
                help="Seed for reproducibility"
            )
            st.session_state.training_config["training"]["seed"] = seed
        
        with col2:
            st.subheader("Learning Parameters")
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-2,
                value=st.session_state.training_config["training"]["learning_rate"],
                format="%.5f",
                step=1e-5
            )
            st.session_state.training_config["training"]["learning_rate"] = learning_rate
            
            gamma = st.slider(
                "Gamma (Discount Factor)",
                min_value=0.8,
                max_value=0.999,
                value=st.session_state.training_config["training"]["gamma"],
                step=0.01
            )
            st.session_state.training_config["training"]["gamma"] = gamma
            
            st.subheader("Checkpoint Settings")
            
            eval_interval = st.number_input(
                "Evaluation Interval",
                min_value=1000,
                max_value=50000,
                value=st.session_state.training_config["training"]["eval_interval"],
                step=1000,
                help="How often to evaluate the agent"
            )
            st.session_state.training_config["training"]["eval_interval"] = eval_interval
            
            checkpoint_interval = st.number_input(
                "Checkpoint Interval",
                min_value=1000,
                max_value=50000,
                value=st.session_state.training_config["training"]["checkpoint_interval"],
                step=1000,
                help="How often to save model checkpoints"
            )
            st.session_state.training_config["training"]["checkpoint_interval"] = checkpoint_interval
    
    # Run Training Tab
    with tab4:
        st.header("Run Training")
        
        # Show configuration summary
        st.subheader("Configuration Summary")
        
        # Environment summary
        env_config = st.session_state.training_config["env"]
        st.markdown(f"**Environment**")
        if env_config.get("asset_mode", "single") == "single":
            st.markdown(f"- Asset Mode: Single")
            st.markdown(f"- Symbol: {env_config['symbol']}")
        else:
            st.markdown(f"- Asset Mode: Multi")
            st.markdown(f"- Symbols: {', '.join(env_config.get('symbols', []))}")
        st.markdown(f"- Timeframe: {env_config['timeframe']}")
        st.markdown(f"- Window Size: {env_config['window_size']}")
        st.markdown(f"- Date Range: {env_config['start_date']} to {env_config['end_date']}")
        
        # Multi-agent setup
        if env_config.get("multi_agent", False):
            st.markdown("**Multi-Agent Configuration**")
            st.markdown(f"- Number of Agents: {env_config['agent_count']}")
            st.markdown(f"- Ensemble Method: {env_config['ensemble_method']}")
            
            # Manager details
            if env_config.get("use_manager", False):
                st.markdown("**Manager Configuration**")
                st.markdown(f"- Manager Enabled: Yes")
                
                # Meta-agent details if applicable
                if env_config['ensemble_method'] == "meta" and "meta_config" in env_config:
                    meta_config = env_config["meta_config"]
                    st.markdown(f"- Meta-Agent Learning Rate: {meta_config.get('learning_rate', 3e-4)}")
                    st.markdown(f"- Meta-Agent Hidden Dim: {meta_config.get('hidden_dim', 128)}")
                    ensemble_type = "Continuous (weighted)" if meta_config.get("continuous_ensemble", True) else "Discrete (selection)"
                    st.markdown(f"- Ensemble Type: {ensemble_type}")
                
                # Shared buffer details
                if "shared_buffer" in env_config and env_config["shared_buffer"].get("enabled", True):
                    shared_buffer = env_config["shared_buffer"]
                    st.markdown("**Shared Experience Buffer**")
                    st.markdown(f"- Min Share Reward: {shared_buffer.get('min_share_reward', 0.2)}")
                    st.markdown(f"- Max Buffer Size: {shared_buffer.get('max_buffer_size', 10000)}")
            else:
                st.markdown(f"- Manager Enabled: No (using traditional multi-agent training)")
            
            # Agent configurations
            st.markdown("**Agents**")
            for i in range(env_config['agent_count']):
                agent_cfg = env_config.get(f"agent_{i}", {})
                st.markdown(f"**Agent {i+1}**")
                st.markdown(f"- Type: {agent_cfg['type']}")
                st.markdown(f"- Strategy: {agent_cfg['strategy']}")
                st.markdown(f"- Capital: {agent_cfg['capital_pct'] * 100:.1f}%")
                
                # Display assigned assets in multi-asset mode
                if env_config.get("asset_mode", "single") == "multi" and "assigned_assets" in agent_cfg:
                    st.markdown(f"- Assigned Assets: {', '.join(agent_cfg['assigned_assets'])}")
        
        # Single agent setup
        else:
            agent_config = st.session_state.training_config["agent"]
            st.markdown("**Agent Configuration**")
            st.markdown(f"- Algorithm: {agent_config['algorithm']}")
            st.markdown(f"- Hidden Layers: {agent_config['hidden_layers']}")
            st.markdown(f"- LSTM: {'Yes' if agent_config.get('use_lstm', False) else 'No'}")
        
        # Training parameters
        training_config = st.session_state.training_config["training"]
        st.markdown("**Training Parameters**")
        st.markdown(f"- Total Timesteps: {training_config['total_timesteps']:,}")
        st.markdown(f"- Batch Size: {training_config['batch_size']}")
        st.markdown(f"- Learning Rate: {training_config['learning_rate']}")
        st.markdown(f"- Gamma: {training_config['gamma']}")
        st.markdown(f"- Seed: {training_config.get('seed', 42)}")
        
        # Create status placeholders
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        chart_container = st.container()
        
        if st.button("Start Training"):
            if st.session_state.training_status["is_training"]:
                st.warning("Training is already in progress")
            else:
                # Create training manager
                manager = TrainingManager(st.session_state.training_config)
                st.session_state.training_manager = manager
                
                # Set up training status
                st.session_state.training_status = {
                    "is_training": True,
                    "start_time": datetime.now(),
                    "progress": 0.0,
                    "metrics": {}
                }
                
                # Run training synchronously (blocks UI until complete)
                run_training(manager, progress_bar, status_text, metrics_container, chart_container)

                # Force rerun to update UI
                st.rerun()

def run_training(manager, progress_bar, status_text, metrics_container, chart_container):
    """
    Run the training process and update the UI.
    
    Args:
        manager: TrainingManager instance
        progress_bar: Streamlit progress bar
        status_text: Streamlit text element for status
        metrics_container: Container for metrics display
        chart_container: Container for charts
    """
    try:
        # Create placeholders for charts
        with chart_container:
            st.subheader("Training Progress")
            
            # Add console output display
            console_output = st.expander("Console Output", expanded=True)
            console_text = console_output.empty()
            
            # Add MLflow link
            mlflow_link = st.empty()
            
            # Add timing information
            timing_info = st.empty()
            
            reward_chart = st.empty()
            eval_chart = st.empty()
            
            # Add multi-agent specific charts
            if st.session_state.training_config["env"].get("multi_agent", False):
                agent_comparison_chart = st.empty()
        
        # Training metrics history
        rewards_history = []
        eval_rewards_history = []
        steps_history = []
        
        # For multi-agent - track agent-specific metrics
        is_multi_agent = st.session_state.training_config["env"].get("multi_agent", False)
        agent_count = st.session_state.training_config["env"].get("agent_count", 2) if is_multi_agent else 0
        
        if is_multi_agent:
            agent_rewards_history = {f"agent_{i}": [] for i in range(agent_count)}
            agent_eval_rewards_history = {f"agent_{i}": [] for i in range(agent_count)}
        
        # Track console output for display
        current_console_output = []
        
        # Progress callback function
        def update_progress(progress, metrics):
            nonlocal current_console_output
            
            # Update session state
            st.session_state.training_status["progress"] = progress
            st.session_state.training_status["metrics"] = metrics
            
            # Update progress bar
            progress_bar.progress(progress)
            
            # Extract timing information
            elapsed_time = metrics.get("elapsed_time", 0)
            estimated_total_time = metrics.get("estimated_total_time", 0)
            remaining_time = max(0, estimated_total_time - elapsed_time)
            
            # Format times
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            remaining_str = str(datetime.timedelta(seconds=int(remaining_time)))
            
            # Update status text
            current_step = metrics.get("current_step", 0)
            total_steps = metrics.get("total_steps", manager.total_steps)
            status_text.text(
                f"Status: Running | Progress: {current_step:,}/{total_steps:,} steps ({progress*100:.1f}%) | Elapsed: {elapsed_str}"
            )
            
            # Update timing information
            timing_info.info(f"⏱️ Elapsed: {elapsed_str} | Estimated remaining: {remaining_str}")
            
            # Update console output if available
            if "console_output" in metrics and metrics["console_output"] != current_console_output:
                current_console_output = metrics["console_output"]
                console_text.code("\n".join(current_console_output))
            
            # Update MLflow link if available
            if hasattr(manager, 'mlflow_ui_url') and manager.mlflow_ui_url:
                mlflow_link.markdown(f"📊 [View training details in MLflow]({manager.mlflow_ui_url})")
            
            # Store metrics for plotting
            if current_step > 0:
                steps_history.append(current_step)
                
                # For single-agent, just track the main reward
                if not is_multi_agent:
                    if "episode_reward" in metrics:
                        rewards_history.append(metrics["episode_reward"])
                    if "eval_reward" in metrics:
                        eval_rewards_history.append(metrics["eval_reward"])
                    
                    # Update charts if enough data
                    if len(rewards_history) > 1:
                        # Rewards chart
                        rewards_df = pd.DataFrame({
                            "Step": steps_history[-len(rewards_history):],
                            "Reward": rewards_history
                        })
                        reward_chart.line_chart(rewards_df.set_index("Step"))
                        
                        # Evaluation rewards chart
                        if len(eval_rewards_history) > 0:
                            eval_steps = [steps_history[i] for i in range(0, len(steps_history), 
                                                                   manager.prepared_config["training"]["eval_interval"])]
                            eval_df = pd.DataFrame({
                                "Step": eval_steps[:len(eval_rewards_history)],
                                "Eval Reward": eval_rewards_history
                            })
                            eval_chart.line_chart(eval_df.set_index("Step"))
                
                # For multi-agent, track agent-specific rewards
                else:
                    # Track rewards for each agent
                    for i in range(agent_count):
                        agent_id = f"agent_{i}"
                        if f"{agent_id}/episode_reward" in metrics:
                            agent_rewards_history[agent_id].append(metrics[f"{agent_id}/episode_reward"])
                        if f"{agent_id}/eval_avg_return" in metrics:
                            agent_eval_rewards_history[agent_id].append(metrics[f"{agent_id}/eval_avg_return"])
                    
                    # Update multi-agent charts
                    if any(len(history) > 1 for history in agent_rewards_history.values()):
                        # Create a DataFrame for all agents
                        agents_data = {
                            "Step": steps_history[-min(len(h) for h in agent_rewards_history.values() if len(h) > 0):]
                        }
                        
                        for agent_id, history in agent_rewards_history.items():
                            if len(history) > 0:
                                agents_data[f"Agent {agent_id[-1]}"] = history[-len(agents_data["Step"]):]
                        
                        # Only create chart if we have data
                        if len(agents_data["Step"]) > 0 and len(agents_data) > 1:
                            agents_df = pd.DataFrame(agents_data)
                            reward_chart.line_chart(agents_df.set_index("Step"))
                            
                        # Evaluation reward comparison
                        if any(len(history) > 0 for history in agent_eval_rewards_history.values()):
                            eval_data = {}
                            
                            for agent_id, history in agent_eval_rewards_history.items():
                                if len(history) > 0:
                                    eval_data[f"Agent {agent_id[-1]}"] = history[-1]  # Latest eval
                            
                            # Update metrics display
                            with metrics_container:
                                st.subheader("Agent Performance")
                                cols = st.columns(min(len(eval_data), 4))
                                
                                for i, (agent_name, eval_reward) in enumerate(eval_data.items()):
                                    cols[i % len(cols)].metric(
                                        f"{agent_name} Eval", 
                                        f"{eval_reward:.2f}"
                                    )
                                    
                                # Display synergy metrics if using manager
                                if st.session_state.training_config["env"].get("use_manager", False):
                                    if "synergy_score" in metrics:
                                        st.metric("Synergy Score", f"{metrics['synergy_score']:.2f}")
            
        # Run the training with the progress callback
        result = asyncio.run(manager.run_training(update_progress))
        
        # Training completed
        # Update training status
        st.session_state.training_status["status"] = "completed"
        st.session_state.training_status["end_time"] = datetime.now()
        duration = st.session_state.training_status["end_time"] - st.session_state.training_status["start_time"]
        duration_str = str(duration).split(".")[0]
        
        # Show completion message
        progress_bar.progress(1.0)
        status_text.success(f"✅ Training completed successfully in {duration_str}")
        
        # Show MLflow link
        if hasattr(manager, 'mlflow_ui_url') and manager.mlflow_ui_url:
            st.markdown(f"📊 **[View complete training details and metrics in MLflow]({manager.mlflow_ui_url})**")
            st.info("To start the MLflow UI server (if not already running), open a terminal and run: `mlflow ui --port 5000`")
        
        # Get final metrics from MLflow
        try:
            metrics = manager.mlflow_manager.get_metric_history()
            if metrics:
                st.subheader("Final Training Metrics")
                for metric_name, values in metrics.items():
                    if values:
                        last_value = values[-1]["value"]
                        st.metric(label=metric_name.replace("_", " ").title(), value=f"{last_value:.4f}")
        except Exception as e:
            st.warning(f"Could not retrieve final metrics: {str(e)}")
        
        return result
        
    except Exception as e:
        # Update training status
        st.session_state.training_status["status"] = "failed"
        st.session_state.training_status["end_time"] = datetime.now()
        st.session_state.training_status["error"] = str(e)
        
        # Show error message
        status_text.error(f"❌ Training failed: {str(e)}")
        st.error(f"Error details: {str(e)}")
        
        # Log the error
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

def display_config_panel():
    """
    Display configuration panel for model training settings.
    """
    st.subheader("Environment")
    env_config = st.session_state.training_config["env"]
    
    # Asset setting
    asset_mode = env_config.get("asset_mode", "single")
    col1, col2 = st.columns(2)
    with col1:
        symbol_text = f"Symbol: {env_config['symbol']}" if asset_mode == "single" else f"Symbols: {', '.join(env_config.get('symbols', []))[:25]}..."
        st.write(symbol_text)
        st.write(f"Timeframe: {env_config['timeframe']}")
    with col2:
        st.write(f"Window Size: {env_config['window_size']}")
        st.write(f"Date Range: {env_config['start_date']} to {env_config['end_date']}")

    st.divider()
    
    # Agent setting
    st.subheader("Agent")
    agent_config = st.session_state.training_config["agent"]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Algorithm: {agent_config['algorithm']}")
        st.write(f"Hidden Layers: {agent_config['hidden_layers']}")
    with col2:
        lstm_status = "Yes" if agent_config.get("use_lstm", False) else "No"
        st.write(f"LSTM: {lstm_status}")
        if agent_config.get("use_lstm", False):
            st.write(f"LSTM Size: {agent_config.get('lstm_hidden_size', 64)}")
    
    st.divider()
            
    # Training parameters
    st.subheader("Training")
    training_config = st.session_state.training_config["training"]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Total Timesteps: {training_config['total_timesteps']:,}")
        st.write(f"Batch Size: {training_config['batch_size']}")
    with col2:
        st.write(f"Learning Rate: {training_config['learning_rate']}")
        st.write(f"Gamma: {training_config['gamma']}")
        st.write(f"Seed: {training_config.get('seed', 42)}")

if __name__ == "__main__":
    model_training_page()