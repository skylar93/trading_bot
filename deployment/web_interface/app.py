"""
Main Streamlit application for Trading Bot with enhanced debugging capabilities
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import streamlit as st
from deployment.web_interface.pages.live_trading import render_live_trading
from deployment.web_interface.utils.state import init_session_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "app.log"), mode="a"),
    ],
)

logger = logging.getLogger(__name__)

async def main():
    """Main application entry point with error handling"""
    try:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Initialize session state
        init_session_state()

        # Configure page
        st.set_page_config(
            page_title="Trading Bot",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Live Trading", "Backtest Results", "Settings"]
        )

        # Content based on selected page
        if page == "Live Trading":
            await render_live_trading()

        elif page == "Backtest Results":
            st.subheader("Backtest Results")
            st.info("Backtest results page under construction")

        elif page == "Settings":
            st.subheader("Settings")
            st.info("Settings page under construction")

        # Log successful execution
        logger.info(f"Successfully rendered {page} page")

    except Exception as e:
        logger.error(f"Main application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for more information.")

if __name__ == "__main__":
    asyncio.run(main())
