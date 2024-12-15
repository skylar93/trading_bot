"""Data Management Component"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import time
from data.utils.data_loader import DataLoader

logger = logging.getLogger(__name__)

def show_data_management(feature_generator):
    """Show data management interface"""
    st.header("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Collection")
        exchange = st.selectbox("Exchange", ["Binance", "Coinbase"])
        symbol = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT"])
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"])
        
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching data..."):
                try:
                    data_loader = DataLoader(
                        exchange_id=exchange.lower(),
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    df = data_loader.fetch_data(
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                    st.session_state['raw_data'] = df
                    st.success("Data fetched successfully!")
                except Exception as e:
                    logger.error("Data fetch error", exc_info=True)
                    st.error(f"Error fetching data: {str(e)}")
    
    with col2:
        st.subheader("Feature Generation")
        if 'raw_data' in st.session_state:
            df = st.session_state['raw_data']
            st.write("Raw Data Preview:")
            st.dataframe(df.head())
            
            if st.button("Generate Features"):
                feature_generator.start(df)
                progress_bar = st.progress(0)
                status_text = st.empty()
                logs = st.empty()
                
                while True:
                    # Check progress
                    progress_update = feature_generator.get_progress()
                    if progress_update:
                        progress, message = progress_update
                        progress_bar.progress(progress)
                        status_text.text(f"Progress: {progress*100:.0f}%")
                        if 'logs' not in st.session_state:
                            st.session_state['logs'] = []
                        st.session_state['logs'].append(message)
                        logs.text("\n".join(st.session_state['logs'][-5:]))
                    
                    # Check completion
                    result = feature_generator.get_result()
                    if result:
                        status, data = result
                        if status == 'success':
                            st.session_state['features'] = data
                            
                            st.success("Feature generation complete!")
                            st.write("Features Preview:")
                            st.dataframe(data.head())
                            
                            # Correlation heatmap
                            st.subheader("Feature Correlations")
                            fig = go.Figure(data=go.Heatmap(
                                z=data.corr(),
                                x=data.columns,
                                y=data.columns,
                                colorscale='RdBu'
                            ))
                            fig.update_layout(height=600)
                            st.plotly_chart(fig)
                            break
                        else:
                            st.error(f"Error generating features: {data}")
                            break
                    
                    time.sleep(0.1)
            
            # Stop generation on page change
            def on_change():
                if 'feature_generator' in st.session_state:
                    feature_generator.stop()
            
            st.on_change(on_change)
        else:
            st.info("Please fetch data first")