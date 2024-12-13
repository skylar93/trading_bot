import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.utils.data_loader import DataLoader
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Trading Bot Dashboard")

@st.cache_data
def load_data(symbol, limit):
    loader = DataLoader()
    raw_data = loader.fetch_ohlcv(symbol, limit=limit)
    processed_data = loader.fetch_and_process(symbol, limit=limit)
    return raw_data, processed_data

def plot_candlestick(df, title):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['open'] if 'open' in df.columns else df['$open'],
                                        high=df['high'] if 'high' in df.columns else df['$high'],
                                        low=df['low'] if 'low' in df.columns else df['$low'],
                                        close=df['close'] if 'close' in df.columns else df['$close'])])
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=600)
    return fig

def plot_technical_indicators(df):
    # Create subplots
    fig = make_subplots(rows=4, cols=1, 
                       subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD', 'Volume Analysis'),
                       vertical_spacing=0.05,
                       row_heights=[0.4, 0.2, 0.2, 0.2])
    
    # Price and Bollinger Bands
    fig.add_trace(go.Candlestick(x=df['datetime'],
                                open=df['$open'],
                                high=df['$high'],
                                low=df['$low'],
                                close=df['$close'],
                                name='Price'),
                  row=1, col=1)
    
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_upper'],
                                name='BB Upper', line=dict(color='gray', dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_middle'],
                                name='BB Middle', line=dict(color='gray')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_lower'],
                                name='BB Lower', line=dict(color='gray', dash='dash')),
                      row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'],
                                name='RSI', line=dict(color='blue')),
                      row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD'],
                                name='MACD', line=dict(color='blue')),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['Signal'],
                                name='Signal', line=dict(color='orange')),
                      row=3, col=1)
        fig.add_trace(go.Bar(x=df['datetime'], y=df['Histogram'],
                            name='Histogram'),
                      row=3, col=1)
    
    # Volume
    volume_colors = ['red' if close < open else 'green' 
                    for close, open in zip(df['$close'], df['$open'])]
    
    fig.add_trace(go.Bar(x=df['datetime'], y=df['$volume'],
                         name='Volume', marker_color=volume_colors),
                  row=4, col=1)
    
    if 'Volume_SMA' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['Volume_SMA'],
                                name='Volume SMA', line=dict(color='blue')),
                      row=4, col=1)
    
    # Update layout
    fig.update_layout(height=1200, showlegend=True,
                     title_text="Technical Analysis Dashboard")
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def plot_correlation_heatmap(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=800,
        width=800)
    
    return fig

def main():
    st.title("Trading Bot Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    symbol = st.sidebar.selectbox("Select Symbol", ["BTC/USDT"])
    limit = st.sidebar.slider("Number of Days", min_value=10, max_value=100, value=30)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Technical Indicators", "Feature Analysis"])
    
    # Load data
    raw_data, processed_data = load_data(symbol, limit)
    
    with tab1:
        st.subheader("Raw Price Data")
        st.plotly_chart(plot_candlestick(raw_data, "Raw Price Data"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Sample")
            st.dataframe(raw_data.head())
        with col2:
            st.subheader("Basic Statistics")
            st.dataframe(raw_data.describe())
    
    with tab2:
        if not processed_data.empty:
            st.plotly_chart(plot_technical_indicators(processed_data), use_container_width=True)
        else:
            st.error("No processed data available")
    
    with tab3:
        if not processed_data.empty:
            st.subheader("Feature Correlation Analysis")
            st.plotly_chart(plot_correlation_heatmap(processed_data))
            
            st.subheader("Feature List")
            feature_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
            st.write(", ".join(feature_cols))
            
            # Feature histogram
            selected_feature = st.selectbox("Select Feature for Distribution", feature_cols)
            fig = px.histogram(processed_data, x=selected_feature, 
                             title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig)
            
            st.subheader("Processed Data Sample")
            st.dataframe(processed_data.head())
        else:
            st.error("No processed data available")

if __name__ == "__main__":
    main()