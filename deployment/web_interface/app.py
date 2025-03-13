"""
Main Streamlit application for Trading Bot with enhanced debugging capabilities
"""

import os
import sys
import asyncio
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import streamlit as st
from pages.Live_Trading import render_live_trading
from pages.Backtest_Results import backtest_results_page
from pages.Model_Training import model_training_page
from utils.state import init_session_state
from utils.backtest import logger, get_log_filename  # Import the shared logger and filename function

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

        # Log that the application has started
        logger.info("Streamlit application started")

        # Display log file location in the sidebar for debugging
        log_path = os.path.join("logs", get_log_filename())
        
        with st.sidebar.expander("📋 Debug Info"):
            st.info(f"Log file: {os.path.abspath(log_path)}")
            st.info(f"Main log: {os.path.abspath('backtest_debug.log')}")

        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Backtest Results", "Live Trading", "Model Training", "Settings"]
        )

        # Content based on selected page
        if page == "Live Trading":
            await render_live_trading()

        elif page == "Backtest Results":
            backtest_results_page()
            
        elif page == "Model Training":
            await model_training_page()

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
