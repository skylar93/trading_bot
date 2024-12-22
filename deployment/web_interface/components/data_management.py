"""Data Management Component with Real-time Support"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from data.utils.data_loader import DataLoader
from deployment.web_interface.utils.websocket_manager import WebSocketManager
from deployment.web_interface.components.charts.ohlcv_chart import OHLCVChart
from deployment.web_interface.utils.feature_generation.async_generator import (
    AsyncFeatureGenerator,
)

logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize session state variables"""
    if "data_state" not in st.session_state:
        st.session_state["data_state"] = {
            "mode": "historical",
            "raw_data": None,
            "features": None,
            "is_processing": False,
        }

    if "generated_features" not in st.session_state:
        st.session_state["generated_features"] = None

    if "feature_generator" not in st.session_state:
        st.session_state["feature_generator"] = AsyncFeatureGenerator()
    if "ws_manager" not in st.session_state:
        st.session_state["ws_manager"] = WebSocketManager()
    if "chart" not in st.session_state:
        st.session_state["chart"] = OHLCVChart()


def show_data_controls():
    """Show data control panel"""
    # Data mode selection
    st.subheader("Data Mode")
    data_mode = st.radio(
        "Select Mode",
        options=["Historical", "Real-time"],
        index=0,
        key="data_mode_radio",
    )

    # Update data state
    st.session_state["data_state"]["mode"] = data_mode.lower()

    st.markdown("---")

    # Basic settings
    st.subheader("Basic Settings")
    exchange = st.selectbox(
        "Exchange",
        options=["Binance", "Coinbase"],
        index=0,
        key="exchange_select",
    )

    symbol = st.selectbox(
        "Trading Pair",
        options=["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        index=0,
        key="symbol_select",
    )

    if data_mode == "Historical":
        st.markdown("---")
        st.subheader("Historical Data Settings")

        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            index=0,
            key="timeframe_select",
        )

        # Date range selection
        st.markdown("#### Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=7),
                key="start_date_input",
            )
        with col2:
            end_date = st.date_input(
                "End Date", value=datetime.now(), key="end_date_input"
            )
    else:
        timeframe = "1m"  # Real-time mode always uses 1m timeframe
        start_date = datetime.now()
        end_date = datetime.now()

    return {
        "mode": data_mode.lower(),
        "exchange": exchange.lower(),
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date if data_mode == "Historical" else None,
        "end_date": end_date if data_mode == "Historical" else None,
    }


def fetch_historical_data(params: dict) -> pd.DataFrame:
    """Fetch historical data"""
    try:
        data_loader = DataLoader(
            exchange_id=params["exchange"],
            symbol=params["symbol"],
            timeframe=params["timeframe"],
        )

        df = data_loader.fetch_data(
            start_date=params["start_date"].strftime("%Y-%m-%d"),
            end_date=params["end_date"].strftime("%Y-%m-%d"),
        )
        return df

    except Exception as e:
        logger.error(
            f"Error fetching historical data: {str(e)}", exc_info=True
        )
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()


def show_data_view(data: pd.DataFrame = None):
    """Show data visualization"""
    chart = st.session_state["chart"]
    data_state = st.session_state["data_state"]

    if data_state["mode"] == "historical":
        if data is not None:
            with st.spinner("Updating chart..."):
                st.plotly_chart(
                    chart.update(data),
                    use_container_width=True,
                    key="historical_chart",
                )

            with st.expander("Raw Data"):
                st.dataframe(data)
    else:
        placeholder = st.empty()
        ws_manager = st.session_state["ws_manager"]
        error_placeholder = st.empty()

        try:
            # Real-time update loop
            while ws_manager.is_running:
                try:
                    data = ws_manager.get_current_ohlcv()
                    latest = ws_manager.get_latest_data()

                    if data is not None:
                        with placeholder.container():
                            current_time = int(time.time() * 1000)
                            st.plotly_chart(
                                chart.update(data, latest),
                                use_container_width=True,
                                key=f"realtime_chart_{current_time}",
                            )
                            error_placeholder.empty()  # Clear any previous errors

                    time.sleep(0.1)
                except Exception as e:
                    error_placeholder.error(f"Error updating chart: {str(e)}")
                    time.sleep(1)  # Wait longer on error

        except Exception as e:
            logger.error(f"Error in real-time view: {str(e)}", exc_info=True)
            error_placeholder.error(
                "Real-time stream encountered an error. Please stop and restart the stream."
            )


def show_data_management():
    """Main data management interface"""
    init_session_state()
    data_state = st.session_state["data_state"]

    st.title("Data Management")

    # Split into two columns
    left_col, right_col = st.columns([1, 2])

    # Data controls in left column
    with left_col:
        params = show_data_controls()

        st.markdown("---")

        # Action buttons
        if params["mode"] == "historical":
            if st.session_state["ws_manager"].is_running:
                st.session_state["ws_manager"].stop()

            fetch_col1, fetch_col2 = st.columns([3, 1])
            with fetch_col1:
                fetch_button = st.button(
                    "Fetch Historical Data",
                    use_container_width=True,
                    disabled=data_state["is_processing"],
                )

            if fetch_button:
                with st.spinner("Fetching historical data..."):
                    data_state["is_processing"] = True
                    data = fetch_historical_data(params)

                    if len(data) > 0:
                        data_state["raw_data"] = data
                        st.success("Data fetched successfully!")

                        # Show in right column
                        with right_col:
                            show_data_view(data)

                    data_state["is_processing"] = False

        else:  # Real-time mode
            stream_col1, stream_col2 = st.columns([3, 1])
            with stream_col1:
                if not st.session_state["ws_manager"].is_running:
                    if st.button(
                        "Start Real-time Stream",
                        use_container_width=True,
                        disabled=data_state["is_processing"],
                    ):
                        data_state["is_processing"] = True
                        st.session_state["ws_manager"].start(params["symbol"])
                        # Show in right column
                        with right_col:
                            show_data_view()
                        data_state["is_processing"] = False
                else:
                    if st.button(
                        "Stop Real-time Stream",
                        use_container_width=True,
                        disabled=data_state["is_processing"],
                    ):
                        st.session_state["ws_manager"].stop()

    # Feature generation section in left column when data is available
    if data_state["raw_data"] is not None:
        with left_col:
            st.markdown("---")
            st.header("Feature Generation")

            generate_col1, generate_col2 = st.columns([3, 1])
            with generate_col1:
                if st.button(
                    "Generate Features",
                    use_container_width=True,
                    disabled=data_state["is_processing"],
                ):
                    data_state["is_processing"] = True
                    feature_generator = st.session_state["feature_generator"]

                    try:
                        logger.info("Starting feature generation...")
                        with st.spinner("Generating features..."):
                            # Start feature generation
                            feature_generator.start(data_state["raw_data"])

                            progress_placeholder = st.empty()
                            status_placeholder = st.empty()

                            # Monitor progress
                            while True:
                                progress = feature_generator.get_progress()
                                logger.debug(f"Progress update: {progress}")

                                if progress:
                                    # Create a new progress bar each time to ensure update
                                    with progress_placeholder:
                                        st.progress(progress["progress"])
                                    with status_placeholder:
                                        st.text(progress["message"])

                                result = feature_generator.get_result()
                                if result:
                                    status, data = result
                                    if status == "success":
                                        data_state["features"] = data
                                        # Store features in a separate session state variable for persistence
                                        st.session_state[
                                            "generated_features"
                                        ] = data
                                        st.success(
                                            "Feature generation complete!"
                                        )
                                        st.write("Features Preview:")
                                        st.dataframe(data.head())
                                        break
                                    elif status == "error":
                                        st.error(
                                            f"Error generating features: {data}"
                                        )
                                        break

                                time.sleep(0.1)

                    except Exception as e:
                        logger.error(
                            f"Error in feature generation: {str(e)}",
                            exc_info=True,
                        )
                        st.error(
                            f"An error occurred during feature generation: {str(e)}"
                        )

                    finally:
                        data_state["is_processing"] = False
