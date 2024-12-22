"""Training Component"""

import streamlit as st
import plotly.graph_objects as go
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def init_training_state():
    """Initialize training-related session state variables"""
    if "training_state" not in st.session_state:
        st.session_state["training_state"] = {
            "is_training": False,
            "current_episode": 0,
            "metrics": {
                "portfolio_values": [],
                "returns": [],
                "sharpe_ratios": [],
                "max_drawdowns": [],
            },
        }


def update_metrics_plot(metrics: Dict[str, list]):
    """Update training metrics visualization"""
    fig = go.Figure()

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            y=metrics["portfolio_values"],
            name="Portfolio Value",
            line=dict(color="blue"),
        )
    )

    # Add returns on secondary y-axis
    fig.add_trace(
        go.Scatter(
            y=metrics["returns"],
            name="Returns",
            line=dict(color="green"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Training Progress",
        xaxis_title="Episode",
        yaxis_title="Portfolio Value",
        yaxis2=dict(title="Returns", overlaying="y", side="right"),
        height=400,
    )

    return fig


def show_training():
    """Show training interface"""
    st.title("Model Training")

    # Initialize training state
    init_training_state()

    # Check for generated features
    if (
        "generated_features" not in st.session_state
        or st.session_state["generated_features"] is None
    ):
        st.warning(
            "Please generate features in the Data Management section first."
        )
        return

    features_df = st.session_state["generated_features"]

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
        help="Percentage of data to use for training",
    )

    # Model settings
    st.markdown("### Model Configuration")
    model_type = st.selectbox(
        "Model Type",
        ["PPO", "DQN", "A2C"],
        help="Select the reinforcement learning algorithm",
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
            help="Number of steps to train the model",
        )

        batch_size = st.number_input(
            "Batch Size",
            min_value=32,
            max_value=1024,
            value=64,
            step=32,
            help="Number of samples per training batch",
        )

    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%f",
            help="Model learning rate",
        )

        gamma = st.number_input(
            "Discount Factor (Gamma)",
            min_value=0.8,
            max_value=0.999,
            value=0.99,
            format="%f",
            help="Discount factor for future rewards",
        )

    # Training control
    st.markdown("### Training Control")

    # Create placeholders for training progress
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.button(
        "Start Training",
        use_container_width=True,
        disabled=st.session_state["training_state"]["is_training"],
    ):
        try:
            st.session_state["training_state"]["is_training"] = True

            # Split data
            train_idx = int(len(features_df) * (train_size / 100))
            train_data = features_df[:train_idx]
            val_data = features_df[train_idx:]

            # Initialize training environment and agent
            from training.environment import TradingEnvironment
            from training.agents import create_agent

            # Create environment
            env = TradingEnvironment(
                df=train_data,
                initial_balance=10000.0,
                trading_fee=0.001,
                window_size=20,
            )

            # Create agent
            agent = create_agent(
                model_type=model_type,
                env=env,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size,
            )

            # Training loop
            n_episodes = n_steps // batch_size
            best_sharpe = -np.inf
            best_model = None

            for episode in range(n_episodes):
                # Update progress
                progress = (episode + 1) / n_episodes
                progress_placeholder.progress(progress)

                # Train episode
                env.reset()
                episode_reward = 0
                done = False
                truncated = False

                while not (done or truncated):
                    state = env.get_state()
                    action = agent.get_action(state)
                    next_state, reward, done, truncated, info = env.step(
                        action
                    )
                    agent.train(state, action, reward, next_state, done)
                    episode_reward += reward

                # Evaluate on validation set
                val_env = TradingEnvironment(val_data, initial_balance=10000)
                val_metrics = agent.evaluate(val_env)

                # Update metrics
                st.session_state["training_state"]["metrics"][
                    "portfolio_values"
                ].append(info["portfolio_value"])
                st.session_state["training_state"]["metrics"][
                    "returns"
                ].append(episode_reward)
                st.session_state["training_state"]["metrics"][
                    "sharpe_ratios"
                ].append(val_metrics["sharpe_ratio"])
                st.session_state["training_state"]["metrics"][
                    "max_drawdowns"
                ].append(val_metrics["max_drawdown"])

                # Update plot
                fig = update_metrics_plot(
                    st.session_state["training_state"]["metrics"]
                )
                metrics_placeholder.plotly_chart(fig, use_container_width=True)

                # Update status
                status_text = f"""
                Episode {episode + 1}/{n_episodes}:
                Portfolio Value: ${info['portfolio_value']:.2f}
                Episode Return: {episode_reward:.2f}
                Sharpe Ratio: {val_metrics['sharpe_ratio']:.2f}
                Max Drawdown: {val_metrics['max_drawdown']*100:.1f}%
                """
                status_placeholder.text(status_text)

                # Save best model
                if val_metrics["sharpe_ratio"] > best_sharpe:
                    best_sharpe = val_metrics["sharpe_ratio"]
                    best_model = agent.save_state()

            # Training complete
            st.success("Training completed successfully!")

            # Save model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(
                model_dir,
                f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            )
            agent.save(model_path)
            st.info(f"Model saved to: {model_path}")

            # Show final metrics
            st.subheader("Final Performance")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Final Portfolio Value",
                    f"${info['portfolio_value']:.2f}",
                    f"{((info['portfolio_value'] / 10000) - 1) * 100:.1f}%",
                )
            with col2:
                st.metric("Best Sharpe Ratio", f"{best_sharpe:.2f}")
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{min(st.session_state['training_state']['metrics']['max_drawdowns'])*100:.1f}%",
                )

        except Exception as e:
            logger.error("Training error", exc_info=True)
            st.error(f"An error occurred during training: {str(e)}")

        finally:
            st.session_state["training_state"]["is_training"] = False
