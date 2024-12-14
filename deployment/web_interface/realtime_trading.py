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
from typing import Dict, List, Optional
from envs.realtime_env import RealtimeTradingEnvironment
from agents.ppo_agent import PPOAgent

class RealTimeTrading:
    """Manages real-time trading functionality"""
    
    def __init__(self):
        """Initialize real-time trading manager"""
        self.env = None
        self.agent = None
        self.trading_data = []
        self.portfolio_history = []
        self.is_trading = False
    
    def initialize_trading(self,
                         symbol: str = 'BTC/USDT',
                         timeframe: str = '1m',
                         initial_balance: float = 10000.0):
        """Initialize trading environment and agent"""
        try:
            self.env = RealtimeTradingEnvironment(
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance
            )
            
            # Load trained agent
            self.agent = PPOAgent.load_from_checkpoint()
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize trading: {str(e)}")
            return False
            
    async def start_trading(self):
        """Start real-time trading"""
        if not self.env or not self.agent:
            st.error("Trading environment not initialized!")
            return False
            
        try:
            await self.env.start_trading()
            self.is_trading = True
            return True
            
        except Exception as e:
            st.error(f"Failed to start trading: {str(e)}")
            return False
            
    async def stop_trading(self):
        """Stop real-time trading"""
        if self.env and self.is_trading:
            await self.env.stop_trading()
            self.is_trading = False
            
    def _create_price_chart(self) -> Optional[go.Figure]:
        """Create real-time price chart"""
        if not self.env:
            return None
            
        data = self.env.data_stream.get_historical_data(self.env.symbol)
        if data.empty:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLCV'
            ),
            row=1, col=1
        )
        
        # Add volume bars
        colors = ['red' if row['open'] > row['close'] else 'green' 
                 for _, row in data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Add EMA lines
        ema20 = data['close'].ewm(span=20, adjust=False).mean()
        ema50 = data['close'].ewm(span=50, adjust=False).mean()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=ema20, name='EMA20',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=ema50, name='EMA50',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
        
        fig.update_layout(
            title='Real-time Market Data',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        return fig
        
    def _create_portfolio_chart(self) -> Optional[go.Figure]:
        """Create portfolio value chart"""
        if not self.portfolio_history:
            return None
            
        df = pd.DataFrame(self.portfolio_history,
                         columns=['timestamp', 'portfolio_value'])
        df.set_index('timestamp', inplace=True)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add drawdown shading
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax * 100
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown,
                fill='tozeroy',
                name='Drawdown %',
                yaxis='y2',
                line=dict(color='red', width=1)
            )
        )
        
        fig.update_layout(
            title='Portfolio Performance',
            yaxis_title='Portfolio Value (USDT)',
            yaxis2=dict(
                title='Drawdown %',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=400,
            showlegend=True
        )
        
        return fig
        
    def update_web_interface(self):
        """Update Streamlit web interface"""
        st.title('Real-time Trading Dashboard')
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_price = self.env.data_stream.get_latest_data(
                self.env.symbol
            )['close']
            st.metric(
                "Current Price",
                f"${latest_price:,.2f}",
                delta=f"{(latest_price / self.trading_data[-2]['price'] - 1) * 100:.2f}%" 
                if len(self.trading_data) > 1 else None
            )
            
        with col2:
            portfolio_value = self.env.balance + (self.env.position * latest_price)
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                delta=f"{(portfolio_value / self.initial_balance - 1) * 100:.2f}%"
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
            st.subheader('Recent Trades')
            
            # Convert to DataFrame for better display
            trades_df = pd.DataFrame(self.trading_data[-10:])  # Last 10 trades
            trades_df['profit'] = trades_df['portfolio_value'].diff()
            trades_df['return'] = trades_df['portfolio_value'].pct_change()
            
            # Style the DataFrame
            st.dataframe(
                trades_df.style.format({
                    'price': '${:.2f}',
                    'portfolio_value': '${:.2f}',
                    'profit': '${:.2f}',
                    'return': '{:.2%}'
                }).background_gradient(
                    subset=['return'],
                    cmap='RdYlGn'
                )
            )
        
    async def trading_loop(self):
        """Main trading loop"""
        while self.is_trading:
            try:
                # Get current state
                observation = await self.env._get_realtime_observation()
                
                # Get action from agent
                action = self.agent.compute_action(observation)
                
                # Execute trade
                observation, reward, done, _, info = await self.env.step(action)
                
                # Record trading data
                self.trading_data.append({
                    'timestamp': datetime.now(),
                    'price': info['current_price'],
                    'action': action,
                    'position': info['position'],
                    'portfolio_value': info['portfolio_value']
                })
                
                # Update portfolio history
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'portfolio_value': info['portfolio_value']
                })
                
                # UI update (in production, should be less frequent)
                self.update_web_interface()
                
                await asyncio.sleep(1)  # Rate limit
                
            except Exception as e:
                st.error(f"Trading error: {str(e)}")
                await asyncio.sleep(5)  # Wait before retry