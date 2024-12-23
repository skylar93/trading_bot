"""
Metrics components for the Trading Bot UI
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def display_portfolio_metrics(portfolio_value: float, initial_balance: float):
    """Display current portfolio metrics"""
    try:
        # Calculate metrics
        pnl = portfolio_value - initial_balance
        pnl_pct = (pnl / initial_balance) * 100

        # Create three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                f"${pnl:,.2f} ({pnl_pct:.1f}%)",
                delta_color="normal"
            )

        with col2:
            st.metric(
                "Initial Balance",
                f"${initial_balance:,.2f}"
            )

        with col3:
            st.metric(
                "Return",
                f"{pnl_pct:.1f}%",
                delta_color="normal"
            )

    except Exception as e:
        logger.error(f"Error displaying portfolio metrics: {str(e)}", exc_info=True)
        st.error("Failed to display portfolio metrics")

def display_trading_metrics(metrics: Dict):
    """Display trading performance metrics"""
    try:
        # Create three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}"
            )
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.1f}%"
            )

        with col2:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.1f}%"
            )
            st.metric(
                "Profit Factor",
                f"{metrics.get('profit_factor', 0):.2f}"
            )

        with col3:
            st.metric(
                "Total Trades",
                metrics.get('total_trades', 0)
            )
            st.metric(
                "Average Trade",
                f"${metrics.get('avg_trade', 0):.2f}"
            )

    except Exception as e:
        logger.error(f"Error displaying trading metrics: {str(e)}", exc_info=True)
        st.error("Failed to display trading metrics")

def display_recent_trades(trades: List[Dict]):
    """Display recent trades table"""
    try:
        if not trades:
            st.info("No trades to display")
            return

        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Format the DataFrame
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["profit"] = df["profit"].map("${:,.2f}".format)
        df["price"] = df["price"].map("${:,.2f}".format)
        
        # Display with formatting
        st.dataframe(
            df,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="DD/MM/YY HH:mm:ss"
                ),
                "side": st.column_config.TextColumn(
                    "Side",
                    width="small"
                ),
                "price": st.column_config.TextColumn(
                    "Price",
                    width="medium"
                ),
                "profit": st.column_config.TextColumn(
                    "Profit/Loss",
                    width="medium"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:
        logger.error(f"Error displaying recent trades: {str(e)}", exc_info=True)
        st.error("Failed to display recent trades")
