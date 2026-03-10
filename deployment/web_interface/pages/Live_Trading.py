"""
Live trading page for the Trading Bot UI

This page is responsible for UI presentation only. All trading logic is handled by RealTimeTradingManager.
"""

import streamlit as st
import asyncio
import logging
from typing import Optional
from components.charts import create_price_chart, create_portfolio_chart
from components.metrics import display_trading_metrics, display_portfolio_metrics, display_recent_trades
from components.controls import trading_controls, debug_controls, indicator_controls
from deployment.web_interface.utils.state import init_session_state
from deployment.web_interface.realtime_trading_manager import RealTimeTradingManager

logger = logging.getLogger(__name__)

async def render_live_trading():
    """Render live trading page (UI presentation only)"""
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

        # Initialize manager if not exists
        if "live_trading_manager" not in st.session_state:
            st.session_state.live_trading_manager = RealTimeTradingManager()

        manager = st.session_state.live_trading_manager
        
        # Configure manager with current settings
        manager.configure(settings)

        # Start or stop trading based on UI controls
        if is_trading and not manager.is_running:
            asyncio.create_task(manager.start())
        elif not is_trading and manager.is_running:
            asyncio.create_task(manager.stop())

        # Get UI update data from manager
        ui_data = manager.update_ui(selected_indicators, debug_mode)
        
        # Display UI elements based on the data provided by the manager
        if ui_data["price_data"] is not None and not ui_data["price_data"].empty:
            # Create and display price chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                price_chart = create_price_chart(ui_data["price_data"], ui_data["indicators"])
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                else:
                    st.error("Failed to create price chart")

            with col2:
                # Display portfolio metrics
                if ui_data["latest_price"]:
                    # Use portfolio value from manager
                    portfolio_value = ui_data["portfolio_history"][-1]["value"] if ui_data["portfolio_history"] else settings["initial_balance"]
                    display_portfolio_metrics(portfolio_value, settings["initial_balance"])

                    # Display portfolio chart
                    if ui_data["portfolio_history"]:
                        portfolio_chart = create_portfolio_chart(ui_data["portfolio_history"])
                        if portfolio_chart:
                            st.plotly_chart(portfolio_chart, use_container_width=True)

            # Display trading metrics
            st.subheader("Trading Performance")
            display_trading_metrics(ui_data["metrics"])

            # Display recent trades
            st.subheader("Recent Trades")
            display_recent_trades(ui_data["trade_history"])

        else:
            st.warning("Waiting for market data...")

        # Debug information
        if debug_mode:
            st.sidebar.subheader("Debug Information")
            st.sidebar.write("Last Update:", ui_data["last_update"])
            st.sidebar.write("Data Buffer Size:", ui_data["data_buffer_size"])

    except Exception as e:
        logger.error(f"Error in live trading page: {str(e)}", exc_info=True)
        st.error("An error occurred in the live trading page. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(render_live_trading())
