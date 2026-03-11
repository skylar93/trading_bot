"""
Live Trading page — simplified stub (no async/await).
Full paper-trading simulation lives in paper_trading.py.
"""

import logging

import streamlit as st

from deployment.web_interface.utils.state import init_session_state

logger = logging.getLogger(__name__)


def render_live_trading() -> None:
    """Render live trading page (synchronous — no async/await)."""
    st.title("Live Trading")

    init_session_state()

    st.sidebar.header("Live Trading Settings")
    is_trading = st.sidebar.toggle("Enable Live Trading", value=False)

    if is_trading:
        st.warning("Live Trading Mode — real trades would be executed.")
    else:
        st.info(
            "Trading is disabled. Use the **Paper Trading** page for simulation. "
            "Connect a broker API here for real-money execution (not yet implemented)."
        )


if __name__ == "__main__":
    render_live_trading()
