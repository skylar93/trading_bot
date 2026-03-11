"""
Main Streamlit application for Trading Bot
No async/await — uses st.session_state + st.rerun() polling pattern.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from deployment.web_interface.utils.state import init_session_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PAGES = [
    "Training Dashboard",
    "Backtest Results",
    "Ensemble Monitor",
    "Paper Trading",
    "Config Editor",
]


def main() -> None:
    """Synchronous main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Trading Bot UI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    st.sidebar.title("Trading Bot")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", PAGES)

    st.sidebar.markdown("---")
    st.sidebar.caption("v0.11.0")

    logger.info("Rendering page: %s", page)

    if page == "Training Dashboard":
        from deployment.web_interface.pages.training_dashboard import (
            render_training_dashboard,
        )
        render_training_dashboard()

    elif page == "Backtest Results":
        from deployment.web_interface.pages.Backtest_Results import (
            backtest_results_page,
        )
        backtest_results_page()

    elif page == "Ensemble Monitor":
        from deployment.web_interface.pages.ensemble_monitor import (
            render_ensemble_monitor,
        )
        render_ensemble_monitor()

    elif page == "Paper Trading":
        from deployment.web_interface.pages.paper_trading import render_paper_trading
        render_paper_trading()

    elif page == "Config Editor":
        from deployment.web_interface.pages.config_editor import render_config_editor
        render_config_editor()


if __name__ == "__main__":
    main()
