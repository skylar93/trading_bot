"""
Main Streamlit application for Trading Bot with enhanced debugging capabilities
"""

import os
import sys
import asyncio
from datetime import datetime
import pandas as pd
import logging
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import streamlit as st
from pages.Live_Trading import render_live_trading
from pages.Backtest_Results import backtest_results_page
from pages.Model_Training import model_training_page
from utils.state import init_session_state
from utils.backtest import logger, get_log_filename  # Import the shared logger and filename function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/backtest_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """
    Main entry point for the Streamlit application.
    """
    st.set_page_config(
        page_title="Trading Bot UI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    init_session_state()
    
    logger.info("Streamlit application started")
    
    # Set up the sidebar
    st.sidebar.title("Trading Bot UI")
    
    # Navigation in sidebar
    page = st.sidebar.radio(
        "Navigation", 
        ["Dashboard", "Backtest", "Model Training", "Live Trading"]
    )
    
    # Display the selected page
    if page == "Dashboard":
        st.title("Trading Bot Dashboard")
        # Dashboard content here
    
    elif page == "Backtest":
        from pages.Backtest import backtest_page
        await backtest_page()
    
    elif page == "Model Training":
        await model_training_page()
    
    elif page == "Live Trading":
        st.title("Live Trading")
        # Live trading content here
    
    # Add footer to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Trading Bot UI")
    st.sidebar.markdown("v0.1.0")
    
    logger.info("Successfully rendered page: " + page)

if __name__ == "__main__":
    asyncio.run(main())
