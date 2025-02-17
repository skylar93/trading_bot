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
from training.backtesting.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)

def display_scenario_metrics(scenario_type: str, metrics: dict):
    """Display scenario-specific metrics in a formatted way"""
    st.subheader(f"{scenario_type} Scenario Metrics")
    
    if scenario_type == "flash_crash":
        cols = st.columns(2)
        with cols[0]:
            st.metric("Recovery Speed (periods)", f"{metrics['recovery_speed']:.0f}")
            st.metric("Drawdown Depth", f"{metrics['drawdown_depth']:.1f}%")
        with cols[1]:
            st.metric("Recovery Percentage", f"{metrics['recovery_percentage']:.1f}%")
            st.metric("Trade Efficacy", f"{metrics['crash_trade_efficacy']:.2f}")
            
    elif scenario_type == "low_liquidity":
        cols = st.columns(2)
        with cols[0]:
            st.metric("Fill Rate", f"{metrics['fill_rate']:.1f}%")
            st.metric("Avg Trade Cost", f"{metrics['avg_trade_cost']:.4f}")
        with cols[1]:
            st.metric("Avg Spread", f"{metrics['avg_spread']:.2f}%")
            st.metric("Execution Delay", f"{metrics['execution_delay']:.1f} periods")

def collect_scenario_params(scenario_type: str) -> dict:
    """Collect scenario parameters from the sidebar based on scenario type."""
    params = {}
    
    if scenario_type == "flash_crash":
        params["crash_size"] = st.sidebar.slider(
            "Crash Size (%)",
            min_value=10,
            max_value=50,
            value=30
        )
        params["crash_at"] = st.sidebar.slider(
            "Crash Position (%)",
            min_value=20,
            max_value=80,
            value=50
        )
        params["crash_duration"] = st.sidebar.slider(
            "Crash Duration (periods)",
            min_value=3,
            max_value=20,
            value=5
        )
        params["recovery_duration"] = st.sidebar.slider(
            "Recovery Duration (periods)",
            min_value=5,
            max_value=40,
            value=10
        )
        
    elif scenario_type == "low_liquidity":
        params["volume_reduction"] = st.sidebar.slider(
            "Volume Reduction (%)",
            min_value=50,
            max_value=95,
            value=80
        )
        params["low_liq_start"] = st.sidebar.slider(
            "Start Position (%)",
            min_value=20,
            max_value=80,
            value=30
        )
        params["low_liq_length"] = st.sidebar.slider(
            "Duration (periods)",
            min_value=50,
            max_value=200,
            value=100
        )
        
    return params

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
            
            # Agent (Strategy) selection
            agent_name = st.selectbox(
                "Select Agent (Strategy)",
                ["Dummy", "MeanReversion", "Momentum", "PPO"]
            )
            
            # Scenario selection - using internal strings directly
            scenario_type = st.selectbox(
                "Scenario Type",
                ["none", "flash_crash", "low_liquidity"],
                format_func=lambda x: {
                    "none": "None",
                    "flash_crash": "Flash Crash",
                    "low_liquidity": "Low Liquidity"
                }[x]  # This formats display while keeping internal values
            )
            
            # Collect scenario parameters
            scenario_params = collect_scenario_params(scenario_type)
            
            # Risk and balance settings
            initial_balance = st.number_input(
                "Initial Balance (USDT)",
                min_value=100.0,
                value=10000.0,
                step=100.0
            )
            
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
                "agent_name": agent_name,
                "initial_balance": initial_balance,
                "max_position_size": max_position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            
            # Initialize managers
            backtest_manager = BacktestManager(settings)
            scenario_manager = ScenarioManager()
            
            # Stage 1: Load raw market data
            raw_data = backtest_manager.load_market_data()
            if raw_data is None:
                st.error("Failed to load market data")
                return
                
            # Stage 2: Apply scenario transformation
            try:
                modified_data = scenario_manager.apply_scenario(
                    raw_data=raw_data,
                    scenario_type=scenario_type,  # Now using internal string directly
                    params=scenario_params
                )
            except ValueError as e:
                st.error(f"Error applying scenario: {str(e)}")
                return
                
            # Stage 3: Run backtest
            results = backtest_manager.run_backtest(modified_data)
            
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
            price_chart = create_price_chart(modified_data)  # Use modified data for chart
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