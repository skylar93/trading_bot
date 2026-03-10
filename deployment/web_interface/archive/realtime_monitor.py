"""
Real-time trading monitor for Streamlit web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from envs.realtime_env import RealtimeTradingEnvironment
from typing import Dict, Optional


class TradingMonitor:
    """Real-time trading monitor for web interface"""

    def __init__(
        self, env: RealtimeTradingEnvironment, update_interval: int = 1
    ):
        """
        Initialize trading monitor

        Args:
            env: Real-time trading environment
            update_interval: Update interval in seconds
        """
        self.env = env
        self.update_interval = update_interval
        self.trading_data = []
        self.portfolio_history = []

    async def start_monitoring(self):
        """Start monitoring trading activity"""
        if not self.env.is_trading:
            await self.env.start_trading()

    async def stop_monitoring(self):
        """Stop monitoring"""
        if self.env.is_trading:
            await self.env.stop_trading()

    def _create_candlestick_chart(self, data: pd.DataFrame):
        """Create candlestick chart with Plotly"""
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
        fig.add_trace(
            go.Bar(x=data.index, y=data["volume"], name="Volume"), row=2, col=1
        )

        fig.update_layout(
            title="Real-time Market Data",
            xaxis_rangeslider_visible=False,
            height=600,
        )

        return fig

    def _create_portfolio_chart(self):
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
            )
        )

        fig.update_layout(
            title="Portfolio Value",
            xaxis_title="Time",
            yaxis_title="Value (USDT)",
            height=400,
        )

        return fig

    def update_web_interface(self):
        """Update Streamlit web interface"""
        st.title("Real-time Trading Monitor")

        # Status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Price",
                f"${self.env.data_stream.get_latest_data(self.env.symbol)['close']:,.2f}",
            )
        with col2:
            st.metric(
                "Portfolio Value",
                f"${(self.env.balance + self.env.position * self.env.data_stream.get_latest_data(self.env.symbol)['close']):,.2f}",
            )
        with col3:
            st.metric("Position", f"{self.env.position:.6f}")

        # Charts
        data = self.env.data_stream.get_historical_data(self.env.symbol)
        if not data.empty:
            st.plotly_chart(self._create_candlestick_chart(data))

        portfolio_chart = self._create_portfolio_chart()
        if portfolio_chart:
            st.plotly_chart(portfolio_chart)

        # Recent trades
        if self.trading_data:
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(self.trading_data[-10:])  # Last 10 trades
            st.dataframe(trades_df)

    async def update_data(self):
        """Update trading data"""
        current_data = self.env.data_stream.get_latest_data(self.env.symbol)
        if current_data:
            self.trading_data.append(
                {
                    "timestamp": current_data["timestamp"],
                    "price": current_data["close"],
                    "position": self.env.position,
                    "balance": self.env.balance,
                }
            )

            portfolio_value = self.env.balance + (
                self.env.position * current_data["close"]
            )
            self.portfolio_history.append(
                {
                    "timestamp": current_data["timestamp"],
                    "portfolio_value": portfolio_value,
                }
            )

    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        while self.env.is_trading:
            await self.update_data()
            await asyncio.sleep(self.update_interval)
