"""Model Settings Component"""

import streamlit as st
import os
import yaml

def show_model_settings(config, project_root):
    """Show model settings interface"""
    st.header("Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment Settings")
        initial_balance = st.number_input(
            "Initial Balance",
            value=float(config['env']['initial_balance'])
        )
        trading_fee = st.number_input(
            "Trading Fee",
            value=float(config['env']['trading_fee'])
        )
        window_size = st.number_input(
            "Window Size",
            value=int(config['env']['window_size'])
        )
    
    with col2:
        st.subheader("Model Architecture")
        hidden_size = st.selectbox(
            "Hidden Layer Size",
            [64, 128, 256, 512],
            index=2
        )
        num_layers = st.selectbox(
            "Number of Layers",
            [1, 2, 3],
            index=1
        )
    
    if st.button("Save Settings"):
        config['env'].update({
            'initial_balance': initial_balance,
            'trading_fee': trading_fee,
            'window_size': window_size
        })
        config['model'].update({
            'hidden_size': hidden_size,
            'num_layers': num_layers
        })
        
        config_path = os.path.join(project_root, "config/default_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        st.success("Settings saved!")