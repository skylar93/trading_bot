"""Main application entry point for Trading Bot UI"""

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import streamlit as st
import logging

# Import relative to the current package
from deployment.web_interface.utils.config_manager import load_config
from deployment.web_interface.utils.state import init_session_state
from deployment.web_interface.components.data_management import show_data_management
from deployment.web_interface.components.model_settings import show_model_settings
from deployment.web_interface.components.training import show_training

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBotUI:
    """Main UI class for Trading Bot"""
    
    def __init__(self):
        st.set_page_config(page_title="Trading Bot", layout="wide")
        self.config = load_config(project_root)
        init_session_state()
    
    def run(self):
        """Run the Streamlit app"""
        st.title("Trading Bot Control Panel")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Data Management", "Model Settings", "Training"]
        )
        
        # Show selected page
        if page == "Data Management":
            show_data_management(st.session_state['feature_generator'])
        elif page == "Model Settings":
            show_model_settings(self.config, project_root)
        elif page == "Training":
            show_training()

if __name__ == "__main__":
    app = TradingBotUI()
    app.run()