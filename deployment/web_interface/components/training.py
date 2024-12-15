"""Training Component"""

import streamlit as st
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

def show_training():
    """Show training interface"""
    st.header("Training")
    
    if 'features' not in st.session_state:
        st.warning("Please generate features first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.selectbox(
            "Batch Size",
            [32, 64, 128, 256],
            index=2
        )
        learning_rate = st.selectbox(
            "Learning Rate",
            [0.0001, 0.0003, 0.001],
            index=1
        )
        num_episodes = st.number_input(
            "Number of Episodes",
            value=50
        )
        
    if st.button("Start Training"):
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_plot = st.empty()
            training_log = st.empty()

            # Training setup
            from training.train import TrainingPipeline
            pipeline = TrainingPipeline("config/default_config.yaml")
            
            # Split data
            features = st.session_state['features']
            train_size = int(len(features) * 0.7)
            val_size = int(len(features) * 0.15)
            
            train_data = features[:train_size]
            val_data = features[train_size:train_size+val_size]
            test_data = features[train_size+val_size:]

            # Initialize metrics storage
            if 'training_metrics' not in st.session_state:
                st.session_state['training_metrics'] = {
                    'portfolio_values': [],
                    'returns': [],
                    'episode_rewards': []
                }

            # Progress callback
            def update_training_progress(episode, train_metrics, val_metrics):
                # Update progress bar
                progress = (episode + 1) / num_episodes
                progress_bar.progress(progress)
                
                # Update status text
                status_text.text(f"Episode {episode + 1}/{num_episodes}")
                
                # Store metrics
                st.session_state['training_metrics']['portfolio_values'].append(
                    train_metrics['final_balance']
                )
                
                # Update plots every 5 episodes
                if episode % 5 == 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=st.session_state['training_metrics']['portfolio_values'],
                        name='Portfolio Value'
                    ))
                    fig.update_layout(
                        title='Training Progress',
                        xaxis_title='Episode',
                        yaxis_title='Portfolio Value'
                    )
                    metrics_plot.plotly_chart(fig)
                    
                    # Update log
                    log_text = f"""
                    Episode {episode + 1}:
                    - Train Return: {train_metrics['episode_reward']:.2f}
                    - Train Final Balance: {train_metrics['final_balance']:.2f}
                    - Val Return: {val_metrics['episode_reward']:.2f}
                    - Val Final Balance: {val_metrics['final_balance']:.2f}
                    """
                    training_log.text(log_text)

            # Run training
            agent = pipeline.train(
                train_data, 
                val_data,
                callback=update_training_progress
            )
            metrics = pipeline.evaluate(agent, test_data)

            # Show results
            st.success("Training completed!")
            st.subheader("Test Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Final Portfolio Value",
                    f"${metrics['final_balance']:.2f}",
                    f"{((metrics['final_balance'] / 10000) - 1) * 100:.1f}%"
                )
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")

        except Exception as e:
            st.error(f"Training error: {str(e)}")
            logger.error("Training error", exc_info=True)