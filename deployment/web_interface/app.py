"""Main application entry point for Trading Bot UI"""

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.insert(0, project_root)

import streamlit as st
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler(os.path.join("logs", "app.log")),  # File handler
    ],
)

# Import relative to the current package
from deployment.web_interface.utils.config_manager import load_config
from deployment.web_interface.utils.state import init_session_state
from deployment.web_interface.components.data_management import (
    show_data_management,
)
from deployment.web_interface.components.model_settings import (
    show_model_settings,
)
from deployment.web_interface.components.training import show_training

logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Configure app
        st.set_page_config(
            page_title="Trading Bot",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Load config and initialize state
        config = load_config(project_root)
        init_session_state()

        # Navigation sidebar
        st.sidebar.title("Navigation")

        # Store previous page to detect page changes
        if "previous_page" not in st.session_state:
            st.session_state["previous_page"] = None

        page = st.sidebar.selectbox(
            "Select Page",
            ["Data Management", "Model Settings", "Training"],
            key="navigation",
        )

        # Detect page changes and preserve state
        if st.session_state["previous_page"] != page:
            # Save important state before page change
            if "data_state" in st.session_state:
                st.session_state["preserved_data_state"] = st.session_state[
                    "data_state"
                ].copy()
            st.session_state["previous_page"] = page

        # Restore state after page change
        if (
            "preserved_data_state" in st.session_state
            and "data_state" in st.session_state
        ):
            st.session_state["data_state"].update(
                st.session_state["preserved_data_state"]
            )

        # Header
        st.title("Trading Bot Control Panel")

        # Show selected page
        if page == "Data Management":
            show_data_management()
        elif page == "Model Settings":
            show_model_settings(config, project_root)
        elif page == "Training":
            show_training()

    except Exception as e:
        logger.error("Application error", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
