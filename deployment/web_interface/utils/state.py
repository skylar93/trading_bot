"""
State management utilities for the Trading Bot UI
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any

def init_session_state():
    """Initialize or update session state with defaults"""
    defaults = {
        "debug_mode": False,
        "test_mode": True,  # Default to test mode for safety
        "last_update": datetime.now(),
        "error_log": [],
        "portfolio_history": [],
        "trading_enabled": False,
        "selected_symbol": "BTC/USDT",
        "timeframe": "1m",
        "initial_balance": 10000.0,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def log_error(error: Exception, context: str = "") -> None:
    """Log error with context and add to session state"""
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
        
    error_entry = {
        "timestamp": datetime.now(),
        "error": str(error),
        "context": context
    }
    st.session_state.error_log.append(error_entry)

def get_debug_info() -> Dict[str, Any]:
    """Get current debug information"""
    return {
        "session_state": {k: v for k, v in st.session_state.items()},
        "last_update": st.session_state.get("last_update", datetime.now()),
        "error_log": st.session_state.get("error_log", [])
    }

def update_portfolio_history(portfolio_value: float) -> None:
    """Update portfolio history in session state"""
    if "portfolio_history" not in st.session_state:
        st.session_state.portfolio_history = []
    
    st.session_state.portfolio_history.append({
        "timestamp": datetime.now(),
        "value": portfolio_value
    })
