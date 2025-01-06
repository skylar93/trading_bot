"""
Chart components for the Trading Bot UI
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

def create_price_chart(data: pd.DataFrame, indicators: Optional[Dict] = None) -> Optional[go.Figure]:
    """Create price chart with error handling"""
    try:
        if data.empty:
            logger.warning("No data available for chart creation")
            return None

        # Get the first asset's columns
        if any("_$" in col for col in data.columns):
            # Multi-asset format
            asset = next(col.split("_")[0] for col in data.columns if "_$" in col)
            open_col = f"{asset}_$open"
            high_col = f"{asset}_$high"
            low_col = f"{asset}_$low"
            close_col = f"{asset}_$close"
            volume_col = f"{asset}_$volume"
        else:
            # Single asset format
            open_col = "$open"
            high_col = "$high"
            low_col = "$low"
            close_col = "$close"
            volume_col = "$volume"

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price", "Volume"),
            row_heights=[0.7, 0.3]
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data[open_col],
                high=data[high_col],
                low=data[low_col],
                close=data[close_col],
                name="OHLCV"
            ),
            row=1, col=1
        )

        # Add technical indicators if provided
        if indicators:
            for name, values in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=values,
                        name=name,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )

        # Add volume bars
        colors = ["red" if row[open_col] > row[close_col] else "green" 
                 for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[volume_col],
                name="Volume",
                marker_color=colors
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Market Data",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark",  # Dark theme
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        logger.info("Successfully created price chart")
        return fig
    except Exception as e:
        logger.error(f"Error creating price chart: {str(e)}", exc_info=True)
        return None

def create_portfolio_chart(portfolio_values: List[float]) -> Optional[go.Figure]:
    """Create portfolio value chart
    
    Args:
        portfolio_values: List of portfolio values
        
    Returns:
        Plotly figure or None if creation fails
    """
    try:
        if not portfolio_values:
            logger.warning("No portfolio values provided")
            return None
            
        # Create DataFrame with index
        df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=len(portfolio_values), freq='1H')
        })
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['portfolio_value'],
                mode='lines',
                name='Portfolio Value'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Time',
            yaxis_title='Portfolio Value (USDT)',
            height=400
        )
        
        logger.info("Successfully created portfolio chart")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating portfolio chart: {str(e)}", exc_info=True)
        return None

def create_metrics_chart(metrics_history: List[Dict]) -> Optional[go.Figure]:
    """Create performance metrics chart"""
    try:
        if not metrics_history:
            logger.warning("No metrics data available")
            return None

        df = pd.DataFrame(metrics_history)
        df.set_index("timestamp", inplace=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Sharpe Ratio
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sharpe_ratio"],
                name="Sharpe Ratio",
                line=dict(color="#00ff00", width=1)
            ),
            secondary_y=False
        )

        # Add Maximum Drawdown
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["max_drawdown"],
                name="Max Drawdown",
                line=dict(color="#ff0000", width=1)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Performance Metrics",
            template="plotly_dark",
            height=300,
            showlegend=True
        )

        fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=False)
        fig.update_yaxes(title_text="Maximum Drawdown (%)", secondary_y=True)

        return fig
    except Exception as e:
        logger.error(f"Error creating metrics chart: {str(e)}", exc_info=True)
        return None
