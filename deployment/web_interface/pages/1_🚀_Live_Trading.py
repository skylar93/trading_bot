"""
Live trading page for the Trading Bot UI
"""

import streamlit as st
import asyncio
import logging
from typing import Optional

from deployment.web_interface.components import (
    create_price_chart,
    create_portfolio_chart,
    display_portfolio_metrics,
    display_trading_metrics,
    display_recent_trades,
    trading_controls,
    debug_controls,
    indicator_controls
)
from deployment.web_interface.utils.data_stream import DataStream
from deployment.web_interface.utils.state import init_session_state, update_portfolio_history

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Live Trading - Trading Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

async def render_live_trading():
    """Render live trading page"""
    try:
        st.title("Live Trading")

        # Initialize session state if needed
        init_session_state()

        # Sidebar controls
        is_trading, settings = trading_controls()
        debug_mode, test_mode = debug_controls()
        selected_indicators = indicator_controls()

        # Main content area
        if test_mode:
            st.info("🧪 Running in test mode with simulated data")
        elif is_trading:
            st.warning("⚠️ Live Trading Mode - Real trades will be executed!")

        # Initialize data stream if not exists
        if "data_stream" not in st.session_state:
            st.session_state.data_stream = DataStream(
                symbol=settings["symbol"],
                timeframe=settings["timeframe"]
            )

        # Start data stream if not running
        if not st.session_state.data_stream.is_running:
            asyncio.create_task(st.session_state.data_stream.start())

        # Get current market data
        data = st.session_state.data_stream.get_current_data()
        
        if not data.empty:
            # Calculate indicators if selected
            indicators = {}
            if any(selected_indicators.values()):
                indicators = st.session_state.data_stream.calculate_indicators(data)
                
                # Filter selected indicators
                indicators = {
                    k: v for k, v in indicators.items()
                    if k.split("_")[0].lower() in selected_indicators
                    and selected_indicators[k.split("_")[0].lower()]
                }

            # Create and display price chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                price_chart = create_price_chart(data, indicators)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                else:
                    st.error("Failed to create price chart")

            with col2:
                # Display portfolio metrics
                current_price = st.session_state.data_stream.get_latest_price()
                if current_price:
                    portfolio_value = settings["initial_balance"]  # TODO: Calculate actual portfolio value
                    display_portfolio_metrics(portfolio_value, settings["initial_balance"])

                    # Update portfolio history
                    update_portfolio_history(portfolio_value)

                    # Display portfolio chart
                    if st.session_state.portfolio_history:
                        portfolio_chart = create_portfolio_chart(st.session_state.portfolio_history)
                        if portfolio_chart:
                            st.plotly_chart(portfolio_chart, use_container_width=True)

            # Display trading metrics
            st.subheader("Trading Performance")
            metrics = {  # TODO: Calculate actual metrics
                "sharpe_ratio": 1.5,
                "win_rate": 65.0,
                "max_drawdown": -5.2,
                "profit_factor": 1.8,
                "total_trades": 42,
                "avg_trade": 125.50
            }
            display_trading_metrics(metrics)

            # Display recent trades
            st.subheader("Recent Trades")
            trades = []  # TODO: Get actual trades
            display_recent_trades(trades)

        else:
            st.warning("Waiting for market data...")

        # Debug information
        if debug_mode:
            st.sidebar.subheader("Debug Information")
            st.sidebar.write("Last Update:", st.session_state.data_stream.last_update)
            st.sidebar.write("Data Buffer Size:", len(st.session_state.data_stream.data_buffer))

    except Exception as e:
        logger.error(f"Error in live trading page: {str(e)}", exc_info=True)
        st.error("An error occurred in the live trading page. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(render_live_trading())
