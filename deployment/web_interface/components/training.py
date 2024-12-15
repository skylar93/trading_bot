"""Training Component"""

import streamlit as st
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

def show_training():
    """Show training interface"""
    st.title("Model Training")
    
    # Check for generated features
    if 'generated_features' not in st.session_state or st.session_state['generated_features'] is None:
        st.warning("Please generate features in the Data Management section first.")
        return
    
    features_df = st.session_state['generated_features']
    
    # Display features info
    st.subheader("Features Information")
    st.write(f"Number of samples: {len(features_df)}")
    st.write(f"Available features: {', '.join(features_df.columns)}")
    
    with st.expander("Preview Features"):
        st.dataframe(features_df.head())
    
    # Training settings
    st.subheader("Training Settings")
    
    # Split settings
    st.markdown("### Data Split")
    train_size = st.slider(
        "Training Data Size",
        min_value=50,
        max_value=90,
        value=80,
        step=5,
        help="Percentage of data to use for training"
    )
    
    # Model settings
    st.markdown("### Model Configuration")
    model_type = st.selectbox(
        "Model Type",
        ["PPO", "DQN", "A2C"],
        help="Select the reinforcement learning algorithm"
    )
    
    # Training parameters
    st.markdown("### Training Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        n_steps = st.number_input(
            "Training Steps",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Number of steps to train the model"
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=32,
            max_value=1024,
            value=64,
            step=32,
            help="Number of samples per training batch"
        )
    
    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%f",
            help="Model learning rate"
        )
        
        gamma = st.number_input(
            "Discount Factor (Gamma)",
            min_value=0.8,
            max_value=0.999,
            value=0.99,
            format="%f",
            help="Discount factor for future rewards"
        )
    
    # Training control
    st.markdown("### Training Control")
    
    if st.button("Start Training", use_container_width=True):
        with st.spinner("Training in progress..."):
            # Here you would implement the actual training logic
            st.info("Training functionality will be implemented in the next update.")