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
    """
    Display trading performance metrics
    
    Args:
        metrics: Dictionary containing:
            - total_return: Total return as decimal (e.g., -0.0864 means -8.64%)
            - sharpe_ratio: Annualized Sharpe ratio (not percentage)
            - sortino_ratio: Annualized Sortino ratio (not percentage)
            - max_drawdown: Maximum drawdown as decimal (e.g., 0.152 means 15.2%)
            - total_trades: Number of successful trades (count)
            - win_rate: Win rate as decimal (e.g., 0.345 means 34.5%)
            - successful_trades: Number of successfully executed trades (count)
            - total_trade_attempts: Total number of trade attempts (count)
            - final_portfolio_value: Final total portfolio value (absolute value)
            - final_balance: Final cash balance (absolute value)
            
    Note:
        All percentage values are in decimal form and need to be multiplied by 100 for display
    """
    try:
        # Create four columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Convert decimal to percentage for display
            total_return = metrics.get('total_return', 0)
            final_portfolio_value = metrics.get('final_portfolio_value', 0)
            initial_value = final_portfolio_value / (1 + total_return) if abs(total_return) > 1e-12 else final_portfolio_value
            total_return_dollars = final_portfolio_value - initial_value
            
            st.metric(
                "Total Return ($)",
                f"${total_return_dollars:,.2f}",
                delta_color="normal"
            )
            st.metric(
                "Total Return (%)",
                f"{total_return * 100:.2f}%"  # Convert decimal to percentage
            )

        with col2:
            st.metric(
                "Final Portfolio",
                f"${metrics.get('final_portfolio_value', 0):,.2f}"
            )
            win_rate = metrics.get('win_rate', 0)
            st.metric(
                "Win Rate",
                f"{win_rate * 100:.2f}%"  # Convert decimal to percentage
            )

        with col3:
            max_dd = metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd * 100:.2f}%"  # Convert decimal to percentage
            )
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}"  # Not a percentage
            )

        with col4:
            st.metric(
                "Total Trades",
                f"{metrics.get('successful_trades', 0)} / {metrics.get('total_trade_attempts', 0)}"  # Counts
            )
            st.metric(
                "Sortino Ratio",
                f"{metrics.get('sortino_ratio', 0):.2f}"  # Not a percentage
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
