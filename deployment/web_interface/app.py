import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import logging

from data.utils.data_loader import DataLoader
from data.utils.feature_generator import FeatureGenerator
from agents.ppo_agent import PPOAgent
from envs.trading_env import TradingEnvironment
from training.evaluation import TradingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBotUI:
    def __init__(self):
        st.set_page_config(page_title="Trading Bot", layout="wide")
        self.load_config()
        
    def load_config(self):
        config_path = os.path.join(project_root, "config/default_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.create_default_config(config_path)
    
    def create_default_config(self, config_path):
        self.config = {
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
            yaml.dump(self.config, f)
    
    def show_data_management(self):
        st.header("Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Collection")
            exchange = st.selectbox("Exchange", ["Binance", "Coinbase"])
            symbol = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT"])
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"])
            
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30)
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
            if st.button("Fetch Data"):
                with st.spinner("Fetching data..."):
                    try:
                        data_loader = DataLoader(
                            exchange_id=exchange.lower(),
                            symbol=symbol,
                            timeframe=timeframe
                        )
                        df = data_loader.fetch_data(
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d")
                        )
                        st.session_state['raw_data'] = df
                        st.success("Data fetched successfully!")
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
        
        with col2:
            st.subheader("Feature Generation")
            if 'raw_data' in st.session_state:
                df = st.session_state['raw_data']
                st.write("Raw Data Preview:")
                st.dataframe(df.head())
                
                if st.button("Generate Features"):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        log_container = st.empty()
                        logs = []

                        def update_progress(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(f"Progress: {progress*100:.0f}%")
                            logs.append(message)
                            log_container.text('\n'.join(logs))

                        # Generate features
                        generator = FeatureGenerator()
                        features = generator.generate_features(df, progress_callback=update_progress)

                        # Save and display results
                        st.session_state['features'] = features
                        st.success(f"Successfully generated {len(features.columns) - len(df.columns)} new features!")

                        # Feature preview
                        st.write("Features Preview:")
                        st.dataframe(features.head())

                        # Correlation heatmap
                        st.subheader("Feature Correlations")
                        fig = go.Figure(data=go.Heatmap(
                            z=features.corr(),
                            x=features.columns,
                            y=features.columns,
                            colorscale='RdBu'
                        ))
                        fig.update_layout(height=600)
                        st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Error generating features: {str(e)}")
                        st.error(f"Error details: {type(e).__name__}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.info("Please fetch data first")

    def show_model_settings(self):
        st.header("Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Settings")
            initial_balance = st.number_input(
                "Initial Balance",
                value=self.config['env']['initial_balance']
            )
            trading_fee = st.number_input(
                "Trading Fee",
                value=self.config['env']['trading_fee']
            )
            window_size = st.number_input(
                "Window Size",
                value=self.config['env']['window_size']
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
            self.config['env'].update({
                'initial_balance': initial_balance,
                'trading_fee': trading_fee,
                'window_size': window_size
            })
            self.config['model'].update({
                'hidden_size': hidden_size,
                'num_layers': num_layers
            })
            
            config_path = os.path.join(project_root, "config/default_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(self.config, f)
            st.success("Settings saved!")

    def show_training(self):
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
                value=50  # Reduced for testing
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
                
                df = st.session_state['features']
                train_size = int(len(df) * 0.7)
                val_size = int(len(df) * 0.15)
                
                train_data = df[:train_size]
                val_data = df[train_size:train_size+val_size]
                test_data = df[train_size+val_size:]

                # Initialize metrics storage
                if 'training_metrics' not in st.session_state:
                    st.session_state['training_metrics'] = {
                        'portfolio_values': [],
                        'returns': [],
                        'episode_rewards': []
                    }

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
                    
                    # Update plots in real-time
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=st.session_state['training_metrics']['portfolio_values'],
                        name='Portfolio Value',
                        line=dict(color='blue')
                    ))
                    fig.update_layout(
                        title='Training Progress',
                        xaxis_title='Episode',
                        yaxis_title='Portfolio Value'
                    )
                    metrics_plot.plotly_chart(fig)
                    
                    # Update training log
                    log_text = f"""
                    Episode {episode + 1}:
                    - Train Return: {train_metrics['episode_reward']:.2f}
                    - Train Final Balance: {train_metrics['final_balance']:.2f}
                    - Val Return: {val_metrics['episode_reward']:.2f}
                    - Val Final Balance: {val_metrics['final_balance']:.2f}
                    """
                    training_log.text(log_text)

                # Run training with progress updates
                agent = pipeline.train(train_data, val_data, callback=update_training_progress)
                metrics = pipeline.evaluate(agent, test_data)

                # Show final results
                st.success("Training completed!")
                st.subheader("Test Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Portfolio Value", f"${metrics['final_balance']:.2f}")
                with col2:
                    st.metric("Total Return", f"{((metrics['final_balance'] / 10000) - 1) * 100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")

            except Exception as e:
                st.error(f"Training error: {str(e)}")
                st.error("Error details:", str(type(e).__name__))
                import traceback
                st.error(traceback.format_exc())

    def show_monitoring(self):
        st.header("Live Monitoring")
        
        if 'training_metrics' not in st.session_state:
            st.info("No training data available yet. Please start training first.")
            return
        
        # Metrics Dashboard
        col1, col2, col3 = st.columns(3)
        
        metrics = st.session_state['training_metrics']
        with col1:
            current_value = metrics['portfolio_values'][-1] if metrics['portfolio_values'] else 10000
            initial_value = metrics['portfolio_values'][0] if metrics['portfolio_values'] else 10000
            return_pct = ((current_value / initial_value) - 1) * 100
            st.metric(
                "Portfolio Value",
                f"${current_value:.2f}",
                f"{return_pct:+.2f}%"
            )
        
        # Portfolio Value Chart
        st.subheader("Portfolio Performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=metrics['portfolio_values'],
            name='Portfolio Value',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Episode',
            yaxis_title='Value ($)'
        )
        st.plotly_chart(fig)
        
        # Trading Activity
        if 'trades' in st.session_state:
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(st.session_state['trades'])
            if not trades_df.empty:
                st.dataframe(trades_df)
                
                # Trade Distribution
                st.subheader("Trade Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=trades_df['profit_loss'],
                    name='Profit/Loss Distribution'
                ))
                fig.update_layout(
                    title='Trade Profit/Loss Distribution',
                    xaxis_title='Profit/Loss ($)',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig)

    def run(self):
        st.title("Trading Bot Control Panel")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Data Management", "Model Settings", "Training", "Live Monitoring"]
        )
        
        # Refresh Rate for Monitoring
        if page == "Live Monitoring":
            st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5
            )
        
        # Show selected page
        if page == "Data Management":
            self.show_data_management()
        elif page == "Model Settings":
            self.show_model_settings()
        elif page == "Training":
            self.show_training()
        elif page == "Live Monitoring":
            self.show_monitoring()

if __name__ == "__main__":
    app = TradingBotUI()
    app.run()

