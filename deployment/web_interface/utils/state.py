"""Session state management utility"""

import streamlit as st


def init_session_state():
    """Initialize session state variables"""
    if "page" not in st.session_state:
        st.session_state["page"] = "Data Management"
    if "progress" not in st.session_state:
        st.session_state["progress"] = 0.0
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
