"""OHLCV Chart Component"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class OHLCVChart:
    """Interactive OHLCV chart with real-time updates"""
    
    def __init__(self, height: int = 600):
        self.height = height
        self.fig = None
        self._init_figure()
        
    def _init_figure(self):
        """Initialize Plotly figure"""
        self.fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        self.fig.update_layout(
            height=self.height,
            title_text="Market Data",
            showlegend=True,
            xaxis_rangeslider_visible=False,
        )
        
    def update(self, data: pd.DataFrame, realtime_data: Optional[Dict[str, Any]] = None):
        """Update chart with new data"""
        if data.empty:
            return self.fig
            
        # Clear previous traces
        self.fig.data = []
        
        # Add candlestick chart
        self.fig.add_trace(
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
        
        # Add EMA lines
        for period in [20, 50]:
            ema = data['close'].ewm(span=period, adjust=False).mean()
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ema,
                    name=f'EMA{period}',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
            
    def _add_realtime_marker(self, data: pd.DataFrame, latest_data: Dict[str, Any]):
        """Add real-time price marker"""
        if latest_data and 'close' in latest_data:
            self.fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[latest_data['close']],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='yellow',
                        line=dict(color='black', width=2)
                    ),
                    name='Real-time'
                ),
                row=1, col=1
            )
    
    def update(self, data: pd.DataFrame, realtime_data: Optional[Dict[str, Any]] = None):
        """Update chart with new data"""
        if data.empty:
            return self.fig
        
        # Clear previous traces
        self.fig.data = []
        
        # Add basic chart components
        self._add_candlestick(data)
        self._add_volume(data)
        self._add_moving_averages(data)
        
        # Add real-time marker if available
        if realtime_data:
            self._add_realtime_marker(data, realtime_data)
        
        # Update axes labels
        self.fig.update_xaxes(title_text="Time", row=2, col=1)
        self.fig.update_yaxes(title_text="Price", row=1, col=1)
        self.fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return self.fig
        
        # Add volume bars
        colors = ['red' if row['open'] > row['close'] else 'green' 
                 for _, row in data.iterrows()]
        
        self.fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Add real-time marker if available
        if realtime_data:
            price = realtime_data.get('close', data['close'].iloc[-1])
            self.fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[price],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='yellow',
                        line=dict(color='black', width=2)
                    ),
                    name='Real-time'
                ),
                row=1, col=1
            )
        
        return self.fig