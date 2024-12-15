"""Data Management Component with Real-time Support"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from data.utils.data_loader import DataLoader
from deployment.web_interface.utils.websocket_manager import WebSocketManager
from deployment.web_interface.components.charts.ohlcv_chart import OHLCVChart
from deployment.web_interface.utils.feature_generation.async_generator import AsyncFeatureGenerator

logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state variables"""
    if 'feature_generator' not in st.session_state:
        st.session_state['feature_generator'] = AsyncFeatureGenerator()
    if 'ws_manager' not in st.session_state:
        st.session_state['ws_manager'] = WebSocketManager()
    if 'chart' not in st.session_state:
        st.session_state['chart'] = OHLCVChart()
    if 'data_mode' not in st.session_state:
        st.session_state['data_mode'] = 'historical'

def show_data_controls():
    """Show data control panel"""
    # Data mode selection
    st.subheader("Data Mode")
    data_mode = st.radio(
        "Select Mode",
        ["Historical", "Real-time"],
        key="data_mode"
    )
    
    st.markdown("---")
    
    # Basic settings
    st.subheader("Basic Settings")
    exchange = st.selectbox(
        "Exchange",
        ["Binance", "Coinbase"],
        key="exchange"
    )
    
    symbol = st.selectbox(
        "Trading Pair",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        key="symbol"
    )
    
    if data_mode == "Historical":
        st.markdown("---")
        st.subheader("Historical Data Settings")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            key="timeframe"
        )
        
        # Date range selection
        st.markdown("#### Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=7),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="end_date"
            )
    else:
        timeframe = "1m"  # Real-time mode always uses 1m timeframe
        start_date = datetime.now()
        end_date = datetime.now()
        
    return {
        'mode': data_mode.lower(),
        'exchange': exchange.lower(),
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': start_date if data_mode == "Historical" else None,
        'end_date': end_date if data_mode == "Historical" else None
    }

def fetch_historical_data(params: dict) -> pd.DataFrame:
    """Fetch historical data"""
    try:
        data_loader = DataLoader(
            exchange_id=params['exchange'],
            symbol=params['symbol'],
            timeframe=params['timeframe']
        )
        
        df = data_loader.fetch_data(
            start_date=params['start_date'].strftime("%Y-%m-%d"),
            end_date=params['end_date'].strftime("%Y-%m-%d")
        )
        return df
        
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

def show_data_view(data: pd.DataFrame = None):
    """Show data visualization"""
    chart = st.session_state['chart']
    
    if st.session_state['data_mode'] == 'historical':
        if data is not None:
            st.plotly_chart(chart.update(data), use_container_width=True)
            
            with st.expander("Raw Data"):
                st.dataframe(data)
    else:
        placeholder = st.empty()
        ws_manager = st.session_state['ws_manager']
        
        # Real-time update loop
        while ws_manager.is_running:
            data = ws_manager.get_current_ohlcv()
            latest = ws_manager.get_latest_data()
            
            if data is not None:
                with placeholder.container():
                    st.plotly_chart(
                        chart.update(data, latest),
                        use_container_width=True
                    )
            
            time.sleep(1)

def show_data_management():
    """Main data management interface"""
    init_session_state()
    
    st.title("Data Management")
    
    # Split into two columns
    left_col, right_col = st.columns([1, 2])
    
    # Data controls in left column
    with left_col:
        params = show_data_controls()
        
        st.markdown("---")
        
        # Action buttons
        if params['mode'] == 'historical':
            if st.session_state['ws_manager'].is_running:
                st.session_state['ws_manager'].stop()
            
            fetch_col1, fetch_col2 = st.columns([3, 1])
            with fetch_col1:
                fetch_button = st.button("Fetch Historical Data", use_container_width=True)
            
            if fetch_button:
                with st.spinner("Fetching historical data..."):
                    data = fetch_historical_data(params)
                    
                    if len(data) > 0:
                        st.session_state['raw_data'] = data
                        st.success("Data fetched successfully!")
                        
                        # Show in right column
                        with right_col:
                            show_data_view(data)
                    
        else:  # Real-time mode
            stream_col1, stream_col2 = st.columns([3, 1])
            with stream_col1:
                if not st.session_state['ws_manager'].is_running:
                    if st.button("Start Real-time Stream", use_container_width=True):
                        st.session_state['ws_manager'].start(params['symbol'])
                        # Show in right column
                        with right_col:
                            show_data_view()
                else:
                    if st.button("Stop Real-time Stream", use_container_width=True):
                        st.session_state['ws_manager'].stop()
    
    # Feature generation section in left column when data is available
    if 'raw_data' in st.session_state:
        with left_col:
            st.markdown("---")
            st.header("Feature Generation")
        
            generate_col1, generate_col2 = st.columns([3, 1])
            with generate_col1:
                if st.button("Generate Features", use_container_width=True):
                    feature_generator = st.session_state['feature_generator']
                    feature_generator.start(st.session_state['raw_data'])
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        progress = feature_generator.get_progress()
                        if progress:
                            info = progress
                            progress_bar.progress(info['progress'])
                            status_text.text(info['message'])
                    
                result = feature_generator.get_result()
                if result:
                    status, data = result
                    if status == 'success':
                        st.session_state['features'] = data
                        st.success("Feature generation complete!")
                        st.write("Features Preview:")
                        st.dataframe(data.head())
                        break
                    else:
                        st.error(f"Error generating features: {data}")
                        break
                        
                        break
                else:
                    st.error(f"Error generating features: {data}")
                    break
                    
            time.sleep(0.1)

def show_data_management():
    """Main data management interface"""
    ensure_session_state()
    
    # Create two-column layout
    controls_col, view_col = st.columns([1, 2])
    
    with controls_col:
        st.markdown("### Data Controls")
        
        # Mode selection
        st.session_state['data_tab'] = st.radio(
            "Data Mode",
            ["Historical", "Real-time"],
            key='mode'
        ).lower()
        
        st.markdown("---")
        
        # Get base parameters
        params = get_data_params()
        
        # Show mode-specific controls
        if st.session_state['data_tab'] == 'historical':
            start_date, end_date = show_historical_controls()
            if st.button("Fetch Data", use_container_width=True):
                fetch_historical_data(params, start_date, end_date)
        else:
            show_realtime_controls()
            
        # Feature generation
        show_feature_generation()
    
    # Data visualization
    show_data_view(view_col)
    
    # Cleanup on page change
    if st.session_state.get('previous_page', 'Data Management') != 'Data Management':
        if 'ws_manager' in st.session_state and st.session_state['ws_manager'].is_running:
            st.session_state['ws_manager'].stop()
        if 'feature_generator' in st.session_state:
            st.session_state['feature_generator'].stop()
    
    st.session_state['previous_page'] = 'Data Management'
    
    # Cleanup on page change
    if st.session_state.get('previous_page', 'Data Management') != 'Data Management':
        if st.session_state['ws_manager'].is_running:
            st.session_state['ws_manager'].stop()
        if 'feature_generator' in st.session_state:
            st.session_state['feature_generator'].stop()
    
    st.session_state['previous_page'] = 'Data Management'