"""Session state management utility"""

import streamlit as st
from utils.feature_generator import BackgroundFeatureGenerator

def init_session_state():
    """Initialize session state variables"""
    if 'feature_generator' not in st.session_state:
        st.session_state['feature_generator'] = BackgroundFeatureGenerator()
    if 'progress' not in st.session_state:
        st.session_state['progress'] = 0.0
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []