"""
Backtest results page for the Trading Bot UI
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from utils.backtest import BacktestManager
from components.charts import create_price_chart, create_portfolio_chart
from components.metrics import display_trading_metrics

logger = logging.getLogger(__name__)

def main():
    """Main function for the backtest results page"""
    st.title("Backtest Results")
    
    try:
        # Sidebar settings
        with st.sidebar:
            st.header("Backtest Settings")
            
            # Date range selection
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30)
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
            # Trading pair selection
            trading_pair = st.selectbox(
                "Trading Pair",
                ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            )
            
            # Strategy selection
            strategy = st.selectbox(
                "Strategy",
                ["Random Strategy"]  # Only DummyAgent for now
            )
            
            # Initial balance
            initial_balance = st.number_input(
                "Initial Balance (USDT)",
                min_value=100.0,
                value=10000.0,
                step=100.0
            )
            
            # Risk parameters
            st.header("Risk Parameters")
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=100,
                value=50
            )
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=20,
                value=10
            )
            
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=1,
                max_value=50,
                value=30
            )
            
            # Run backtest button
            run_backtest = st.button("Run Backtest")
        
        if run_backtest:
            # Create settings dictionary
            settings = {
                "start_date": start_date,
                "end_date": end_date,
                "trading_pair": trading_pair,
                "strategy": strategy,
                "initial_balance": initial_balance,
                "max_position_size": max_position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            
            # Initialize backtest manager
            manager = BacktestManager(settings)
            
            # Load market data
            data = manager.load_market_data()
            if data is None:
                st.error("Failed to load market data")
                return
            
            # Run backtest
            results = manager.run_backtest(data)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portfolio Performance")
                portfolio_chart = create_portfolio_chart(results.get("portfolio_values", []))
                if portfolio_chart:
                    st.plotly_chart(portfolio_chart, use_container_width=True)
                else:
                    st.warning("No portfolio data available")
            
            with col2:
                st.subheader("Trading Metrics")
                display_trading_metrics(results.get("metrics", {}))
            
            # Display price chart with trades
            st.subheader("Trade History")
            price_chart = create_price_chart(data)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            
            # Display trade list
            trades = results.get("trades", [])
            if trades:
                st.dataframe(pd.DataFrame(trades))
            else:
                st.info("No trades to display")
                
    except Exception as e:
        logger.error(f"Error in backtest page: {str(e)}", exc_info=True)
        st.error("An error occurred in the backtest page. Check the logs for details.")

if __name__ == "__main__":
    main() 