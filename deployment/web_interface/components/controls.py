"""
Control components for the Trading Bot UI
"""

import streamlit as st
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def trading_controls() -> Tuple[bool, Dict[str, Any]]:
    """Trading control panel"""
    try:
        st.sidebar.subheader("Trading Controls")

        # Trading pair selection
        symbol = st.sidebar.selectbox(
            "Trading Pair",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            key="selected_symbol"
        )

        # Timeframe selection
        timeframe = st.sidebar.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            key="timeframe"
        )

        # Initial balance
        initial_balance = st.sidebar.number_input(
            "Initial Balance (USDT)",
            min_value=10.0,
            value=10000.0,
            step=100.0,
            key="initial_balance"
        )

        # Risk management
        st.sidebar.subheader("Risk Management")
        
        max_position_size = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=100,
            value=50,
            key="max_position_size"
        )

        stop_loss = st.sidebar.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=20,
            value=2,
            key="stop_loss"
        )

        # Trading controls
        st.sidebar.subheader("Trading Status")
        
        is_trading = st.sidebar.checkbox(
            "Enable Trading",
            value=False,
            key="trading_enabled"
        )

        if is_trading:
            st.sidebar.warning("⚠️ Live Trading Enabled")
        else:
            st.sidebar.info("📊 Paper Trading Mode")

        # Return settings
        settings = {
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_balance": initial_balance,
            "max_position_size": max_position_size / 100,  # Convert to decimal
            "stop_loss": stop_loss / 100,  # Convert to decimal
        }

        return is_trading, settings

    except Exception as e:
        logger.error(f"Error in trading controls: {str(e)}", exc_info=True)
        st.error("Failed to load trading controls")
        return False, {}

def debug_controls() -> Tuple[bool, bool]:
    """Debug and test mode controls"""
    try:
        st.sidebar.subheader("Debug Controls")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            debug_mode = st.checkbox(
                "Debug Mode",
                value=False,
                key="debug_mode"
            )
        
        with col2:
            test_mode = st.checkbox(
                "Test Mode",
                value=True,
                key="test_mode"
            )

        if debug_mode:
            st.sidebar.info("🔍 Debug Mode Active")
        
        if test_mode:
            st.sidebar.info("🧪 Test Mode Active")
        else:
            st.sidebar.warning("⚠️ Production Mode")

        return debug_mode, test_mode

    except Exception as e:
        logger.error(f"Error in debug controls: {str(e)}", exc_info=True)
        st.error("Failed to load debug controls")
        return False, True  # Default to safe mode

def indicator_controls() -> Dict[str, bool]:
    """Technical indicator controls"""
    try:
        st.sidebar.subheader("Technical Indicators")

        indicators = {}
        
        col1, col2 = st.sidebar.columns(2)

        with col1:
            indicators["sma"] = st.checkbox("SMA", value=True)
            indicators["ema"] = st.checkbox("EMA", value=True)
            indicators["rsi"] = st.checkbox("RSI", value=False)

        with col2:
            indicators["macd"] = st.checkbox("MACD", value=False)
            indicators["bbands"] = st.checkbox("Bollinger", value=False)
            indicators["volume"] = st.checkbox("Volume", value=True)

        return indicators

    except Exception as e:
        logger.error(f"Error in indicator controls: {str(e)}", exc_info=True)
        st.error("Failed to load indicator controls")
        return {"volume": True}  # Default to just volume
