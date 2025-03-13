"""
Real-time trading functionality for web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from typing import Dict, List, Optional, Tuple
from envs.live_trading_env import LiveTradingEnvironment
from agents.strategies.single.ppo_agent import PPOAgent
import logging


class RealTimeTrading:
    """
    Manager for real-time trading that coordinates:
    1. Environment - LiveTradingEnvironment (RL interface, state, orders)
    2. Agent - Trained RL model that computes actions
    3. UI - Streamlit interface for visualization and control
    
    This class is responsible for:
    - Loading and initializing the trading environment
    - Loading trained agent models
    - Running the main trading loop
    - Updating the UI with latest trading information
    - Managing trading state (start/stop)
    
    It does NOT handle:
    - Order creation/execution (done by environment)
    - Market data fetching (done by environment)
    - Action space definition (defined by environment)
    
    Features:
    - Asynchronous trading loop
    - Real-time UI updates
    - Portfolio tracking and visualization
    - Trade history recording
    
    Implementation Notes:
    - Uses Streamlit for UI components
    - Manages asyncio tasks for the trading loop
    - Properly separates environment and manager responsibilities
    
    Recent Changes:
    - Updated to use LiveTradingEnvironment instead of RealtimeTradingEnvironment
    - Improved error handling and recovery
    - Enhanced UI components for better monitoring
    """

    def __init__(self):
        """Initialize real-time trading manager"""
        self.env = None
        self.agent = None
        self.trading_data = []
        self.portfolio_history = []
        self.is_trading = False
        self.trading_task = None

    def initialize_trading(
        self,
        symbol: str = "BTC/USDT",
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
        window_size: int = 60,
        exchange_id: str = "binance",
        test_mode: bool = True,
    ):
        """
        Initialize trading environment and agent
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            initial_balance: Starting account balance
            trading_fee: Fee per trade (e.g., 0.001 for 0.1%)
            window_size: Size of observation window
            exchange_id: Exchange to use (e.g., "binance")
            test_mode: Whether to run in test mode (paper trading)
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Initialize the environment
            self.env = LiveTradingEnvironment(
                symbol=symbol,
                initial_balance=initial_balance,
                trading_fee=trading_fee,
                window_size=window_size,
                exchange_id=exchange_id,
                test_mode=test_mode,
            )

            # Load trained agent
            self.agent = PPOAgent.load_from_checkpoint()
            
            # Reset trading data
            self.trading_data = []
            self.portfolio_history = []
            
            st.success(f"Trading initialized for {symbol}")
            return True

        except Exception as e:
            st.error(f"Failed to initialize trading: {str(e)}")
            return False

    async def start_trading(self):
        """Start the trading loop"""
        if not self.env or not self.agent:
            st.error("Trading environment not initialized. Please initialize first.")
            return False
        
        if self.is_trading:
            st.warning("Trading is already running")
            return True
            
        try:
            # Reset the environment to get initial observation
            obs, info = await self.env.reset()
            
            # Mark as trading
            self.is_trading = True
            
            # Start trading loop as a task
            self.trading_task = asyncio.create_task(self.trading_loop(obs))
            
            st.success("Trading started successfully")
            return True
            
        except Exception as e:
            st.error(f"Failed to start trading: {str(e)}")
            self.is_trading = False
            return False

    async def stop_trading(self):
        """Stop the trading loop and cleanup resources"""
        if not self.is_trading:
            st.warning("Trading is not running")
            return True
            
        try:
            # Stop trading loop
            self.is_trading = False
            
            # Wait for trading task to complete
            if self.trading_task and not self.trading_task.done():
                await asyncio.wait_for(self.trading_task, timeout=5.0)
                
            # Cleanup environment resources
            await self.env.cleanup()
            
            st.success("Trading stopped successfully")
            return True
            
        except asyncio.TimeoutError:
            st.warning("Timeout while waiting for trading to stop, resources may not be fully cleaned up")
            return False
        except Exception as e:
            st.error(f"Error stopping trading: {str(e)}")
            return False

    def _create_price_chart(self) -> Optional[go.Figure]:
        """Create real-time price chart"""
        if not self.env:
            return None

        data = self.env.data_stream.get_historical_data(self.env.symbol)
        if data.empty:
            return None

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price", "Volume"),
            row_heights=[0.7, 0.3],
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="OHLCV",
            ),
            row=1,
            col=1,
        )

        # Add volume bars
        colors = [
            "red" if row["open"] > row["close"] else "green"
            for _, row in data.iterrows()
        ]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                marker_color=colors,
                name="Volume",
            ),
            row=2,
            col=1,
        )

        # Add EMA lines
        ema20 = data["close"].ewm(span=20, adjust=False).mean()
        ema50 = data["close"].ewm(span=50, adjust=False).mean()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ema20,
                name="EMA20",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ema50,
                name="EMA50",
                line=dict(color="orange", width=1),
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            title="Real-time Market Data",
            xaxis_rangeslider_visible=False,
            height=600,
        )

        return fig

    def _create_portfolio_chart(self) -> Optional[go.Figure]:
        """Create portfolio value chart"""
        if not self.portfolio_history:
            return None

        df = pd.DataFrame(
            self.portfolio_history, columns=["timestamp", "portfolio_value"]
        )
        df.set_index("timestamp", inplace=True)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["portfolio_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2),
            )
        )

        # Add drawdown shading
        cummax = df["portfolio_value"].cummax()
        drawdown = (df["portfolio_value"] - cummax) / cummax * 100

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown,
                fill="tozeroy",
                name="Drawdown %",
                yaxis="y2",
                line=dict(color="red", width=1),
            )
        )

        fig.update_layout(
            title="Portfolio Performance",
            yaxis_title="Portfolio Value (USDT)",
            yaxis2=dict(
                title="Drawdown %",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            height=400,
            showlegend=True,
        )

        return fig

    def update_web_interface(self):
        """Update Streamlit web interface"""
        st.title("Real-time Trading Dashboard")

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            latest_price = self.env.data_stream.get_latest_data(
                self.env.symbol
            )["close"]
            st.metric(
                "Current Price",
                f"${latest_price:,.2f}",
                delta=(
                    f"{(latest_price / self.trading_data[-2]['price'] - 1) * 100:.2f}%"
                    if len(self.trading_data) > 1
                    else None
                ),
            )

        with col2:
            portfolio_value = self.env.balance + (
                self.env.position * latest_price
            )
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                delta=f"{(portfolio_value / self.initial_balance - 1) * 100:.2f}%",
            )

        with col3:
            st.metric("Position", f"{self.env.position:.6f}")

        with col4:
            st.metric("Balance", f"${self.env.balance:,.2f}")

        # Charts
        col1, col2 = st.columns([2, 1])

        with col1:
            price_chart = self._create_price_chart()
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)

        with col2:
            portfolio_chart = self._create_portfolio_chart()
            if portfolio_chart:
                st.plotly_chart(portfolio_chart, use_container_width=True)

        # Recent trades
        if self.trading_data:
            st.subheader("Recent Trades")

            # Convert to DataFrame for better display
            trades_df = pd.DataFrame(self.trading_data[-10:])  # Last 10 trades
            trades_df["profit"] = trades_df["portfolio_value"].diff()
            trades_df["return"] = trades_df["portfolio_value"].pct_change()

            # Style the DataFrame
            st.dataframe(
                trades_df.style.format(
                    {
                        "price": "${:.2f}",
                        "portfolio_value": "${:.2f}",
                        "profit": "${:.2f}",
                        "return": "{:.2%}",
                    }
                ).background_gradient(subset=["return"], cmap="RdYlGn")
            )

    async def trading_loop(self, initial_obs: np.ndarray):
        """
        Main trading loop that coordinates the environment and agent
        
        Args:
            initial_obs: Initial observation from environment reset
        """
        obs = initial_obs
        
        while self.is_trading:
            try:
                # 1. Get action from agent
                action = self.agent.compute_action(obs)
                
                # 2. Execute action in environment
                obs, reward, done, truncated, info = await self.env.step(action)
                
                # 3. Record trading data for UI
                self.trading_data.append(
                    {
                        "timestamp": datetime.now(),
                        "price": info.get("current_price", 0),
                        "action": float(action[0]),  # Convert to scalar
                        "position": info["position"],
                        "portfolio_value": info["portfolio_value"],
                        "reward": reward
                    }
                )
                
                # 4. Update portfolio history
                self.portfolio_history.append(
                    {
                        "timestamp": datetime.now(),
                        "portfolio_value": info["portfolio_value"],
                    }
                )
                
                # 5. Update UI (could be less frequent in production)
                self.update_web_interface()
                
                # 6. Reset if episode is done
                if done or truncated:
                    st.info("Trading episode complete, resetting environment")
                    obs, info = await self.env.reset()
                
                # 7. Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                st.error(f"Trading error: {str(e)}")
                logger = logging.getLogger(__name__)
                logger.exception("Exception in trading loop")
                await asyncio.sleep(5)  # Wait before retry
