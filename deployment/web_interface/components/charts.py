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
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
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
        colors = ["red" if row["open"] > row["close"] else "green" 
                 for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
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

def create_portfolio_chart(portfolio_history: List[Dict]) -> Optional[go.Figure]:
    """Create portfolio performance chart"""
    try:
        if not portfolio_history:
            logger.warning("No portfolio data available")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df.set_index("timestamp", inplace=True)

        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["value"],
                name="Portfolio Value",
                line=dict(color="#00ff00", width=2)
            )
        )

        # Add annotations for significant points
        if len(df) > 1:
            initial_value = df["value"].iloc[0]
            current_value = df["value"].iloc[-1]
            pct_change = ((current_value - initial_value) / initial_value) * 100

            fig.add_annotation(
                x=df.index[-1],
                y=current_value,
                text=f"Current: ${current_value:.2f}<br>Change: {pct_change:.1f}%",
                showarrow=True,
                arrowhead=1
            )

        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            height=400,
            template="plotly_dark",
            showlegend=True
        )

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
