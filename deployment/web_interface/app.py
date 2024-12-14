"""Streamlit UI for Trading Bot"""

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
from typing import Dict, Any
import queue
import threading

from data.utils.data_loader import DataLoader
from data.utils.feature_generator import FeatureGenerator
from agents.ppo_agent import PPOAgent
from envs.trading_env import TradingEnvironment
from training.evaluation import TradingMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackgroundFeatureGenerator:
    """Background worker for feature generation"""
    
    def __init__(self):
        self._generator = FeatureGenerator()
        self._running = False
        self._progress_queue = queue.Queue()
        self._result_queue = queue.Queue()
    
    def generate_features(self, df: pd.DataFrame) -> None:
        """Generate features in background thread"""
        self._running = True
        
        def progress_callback(progress: float, message: str) -> None:
            """Handle progress updates"""
            if self._running:
                self._progress_queue.put((progress, message))
        
        try:
            # Run feature generation
            result = self._generator.generate_features(df, progress_callback)
            self._result_queue.put(('success', result))
            
        except Exception as e:
            logger.error("Feature generation error", exc_info=True)
            self._result_queue.put(('error', str(e)))
            
        finally:
            self._running = False
    
    def start(self, df: pd.DataFrame) -> None:
        """Start feature generation in background"""
        threading.Thread(
            target=self.generate_features,
            args=(df,),
            daemon=True
        ).start()
    
    def get_progress(self) -> tuple:
        """Get latest progress update"""
        try:
            return self._progress_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_result(self) -> tuple:
        """Get generation result if complete"""
        try:
            return self._result_queue.get_nowait() 
        except queue.Empty:
            return None
    
    def stop(self) -> None:
        """Stop feature generation"""
        self._running = False

def init_session_state() -> None:
    """Initialize Streamlit session state"""
    if 'feature_generator' not in st.session_state:
        st.session_state['feature_generator'] = BackgroundFeatureGenerator()
    if 'progress' not in st.session_state:
        st.session_state['progress'] = 0.0
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []

class TradingBotUI:
    """Main UI class for Trading Bot"""
    
    def __init__(self):
        st.set_page_config(page_title="Trading Bot", layout="wide")
        self.load_config()
        init_session_state()
    
    def load_config(self) -> None:
        """Load or create config"""
        config_path = os.path.join(project_root, "config/default_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.create_default_config(config_path)
    
    def create_default_config(self, config_path: str) -> None:
        """Create default config file"""
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
        """Show data management page"""
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
                        logger.error("Data fetch error", exc_info=True)
                        st.error(f"Error fetching data: {str(e)}")
        
        with col2:
            st.subheader("Feature Generation")
            if 'raw_data' in st.session_state:
                df = st.session_state['raw_data']
                st.write("Raw Data Preview:")
                st.dataframe(df.head())
                
                if st.button("Generate Features"):
                    st.session_state['feature_generator'].start(df)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    logs = st.empty()
                    
                    while True:
                        # Check progress
                        progress_update = st.session_state['feature_generator'].get_progress()
                        if progress_update:
                            progress, message = progress_update
                            progress_bar.progress(progress)
                            status_text.text(f"Progress: {progress*100:.0f}%")
                            st.session_state['logs'].append(message)
                            logs.text("\n".join(st.session_state['logs'][-5:]))
                        
                        # Check completion
                        result = st.session_state['feature_generator'].get_result()
                        if result:
                            status, data = result
                            if status == 'success':
                                st.session_state['features'] = data
                                
                                st.success("Feature generation complete!")
                                st.write("Features Preview:")
                                st.dataframe(data.head())
                                
                                # Correlation heatmap
                                st.subheader("Feature Correlations")
                                fig = go.Figure(data=go.Heatmap(
                                    z=data.corr(),
                                    x=data.columns,
                                    y=data.columns,
                                    colorscale='RdBu'
                                ))
                                fig.update_layout(height=600)
                                st.plotly_chart(fig)
                                break
                            else:
                                st.error(f"Error generating features: {data}")
                                break
                        
                        # Small delay
                        import time
                        time.sleep(0.1)
                
                # Stop generation on page change
                def on_change():
                    if 'feature_generator' in st.session_state:
                        st.session_state['feature_generator'].stop()
                
                st.on_change(on_change)
            else:
                st.info("Please fetch data first")
    
    def show_model_settings(self):
        """Show model settings page"""
        st.header("Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Settings")
            initial_balance = st.number_input(
                "Initial Balance",
                value=float(self.config['env']['initial_balance'])
            )
            trading_fee = st.number_input(
                "Trading Fee",
                value=float(self.config['env']['trading_fee'])
            )
            window_size = st.number_input(
                "Window Size",
                value=int(self.config['env']['window_size'])
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
        """Show training page"""
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

    def run(self):
        """Run the Streamlit app"""
        st.title("Trading Bot Control Panel")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Data Management", "Model Settings", "Training"]
        )
        
        # Show selected page
        if page == "Data Management":
            self.show_data_management()
        elif page == "Model Settings":
            self.show_model_settings()
        elif page == "Training":
            self.show_training()

if __name__ == "__main__":
    app = TradingBotUI()
    app.run()