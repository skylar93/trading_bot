"""
Backtest results page for the Trading Bot UI

This page is responsible for UI presentation only. All backtest logic is handled by BacktestPresenter.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import os

from components.charts import create_price_chart, create_portfolio_chart
from components.metrics import display_trading_metrics
from deployment.web_interface.backtest_presenter import BacktestPresenter

logger = logging.getLogger(__name__)

def display_scenario_metrics(scenario_type: str, metrics: dict):
    """Display scenario-specific metrics in a formatted way"""
    st.subheader(f"{scenario_type.title()} Scenario Metrics")
    
    if scenario_type == "flash_crash":
        cols = st.columns(2)
        with cols[0]:
            st.metric("Recovery Speed (periods)", f"{metrics.get('recovery_speed', 0):.0f}")
            st.metric("Drawdown Depth", f"{metrics.get('drawdown_depth', 0):.1f}%")
        with cols[1]:
            st.metric("Recovery Percentage", f"{metrics.get('recovery_percentage', 0):.1f}%")
            st.metric("Trade Efficacy", f"{metrics.get('crash_trade_efficacy', 0):.2f}")
            
    elif scenario_type == "low_liquidity":
        cols = st.columns(2)
        with cols[0]:
            st.metric("Fill Rate", f"{metrics.get('fill_rate', 100):.1f}%")
            st.metric("Avg Trade Cost", f"{metrics.get('avg_trade_cost', 0):.4f}")
        with cols[1]:
            st.metric("Avg Spread", f"{metrics.get('avg_spread', 0):.2f}%")
            st.metric("Execution Delay", f"{metrics.get('execution_delay', 0):.1f} periods")

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

def backtest_sidebar() -> Dict[str, Any]:
    """Handle backtest sidebar settings and return a settings dictionary"""
    st.sidebar.header("Backtest Settings")
    
    # Date range selection
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=30)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now()
    )
    
    # Symbol and timeframe selection
    symbol = st.sidebar.selectbox(
        "Symbol",
        options=["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"],
        index=0
    )
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3
    )
    
    # Agent selection
    st.sidebar.subheader("Agent Settings")
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        options=["PPO", "Momentum", "MeanReversion", "LSTM", "Random"],
        index=0
    )
    
    # Model checkpoint selection
    use_trained_model = st.sidebar.checkbox("Use trained model", value=False)
    model_path = None
    
    if use_trained_model:
        # Find available model checkpoints based on agent type
        checkpoint_options = []
        
        if agent_type == "Momentum":
            # Check for momentum models
            momentum_models = [
                "models/multi_agent/momentum_trader_episode_2.pt",
                "models/multi_agent/momentum_trader_episode_4.pt",
                "checkpoints/momentum/best_model.pt" if os.path.exists("checkpoints/momentum/best_model.pt") else None,
                "checkpoints/final_agent.pt" if os.path.exists("checkpoints/final_agent.pt") else None
            ]
            checkpoint_options = [path for path in momentum_models if path is not None]
            
        elif agent_type == "MeanReversion":
            # Check for mean reversion models
            meanrev_models = [
                "models/multi_agent/mean_reversion_trader_episode_2.pt",
                "models/multi_agent/mean_reversion_trader_episode_4.pt",
                "checkpoints/meanrev/best_model.pt" if os.path.exists("checkpoints/meanrev/best_model.pt") else None,
                "checkpoints/final_agent.pt" if os.path.exists("checkpoints/final_agent.pt") else None
            ]
            checkpoint_options = [path for path in meanrev_models if path is not None]
            
        elif agent_type == "PPO":
            # Check for PPO models
            ppo_models = [
                "models/best_model.pt" if os.path.exists("models/best_model.pt") else None,
                "checkpoints/final_agent.pt" if os.path.exists("checkpoints/final_agent.pt") else None
            ]
            checkpoint_options = [path for path in ppo_models if path is not None]
            
        # Add a 'none' option
        if not checkpoint_options:
            st.sidebar.warning(f"No trained models found for {agent_type}.")
            checkpoint_options = ["None"]
        
        # Let user select a model
        selected_model = st.sidebar.selectbox(
            "Select trained model",
            options=checkpoint_options,
            index=0
        )
        
        if selected_model != "None":
            model_path = selected_model
            st.sidebar.success(f"Using trained model: {model_path}")

    # Initial balance and portfolio settings
    initial_balance = st.sidebar.number_input(
        "Initial Balance (USDT)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0
    )
    
    # Risk parameters
    st.sidebar.subheader("Risk Settings")
    max_position_size = st.sidebar.slider(
        "Max Position Size (%)",
        min_value=10,
        max_value=100,
        value=50
    )
    stop_loss = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=0,
        max_value=20,
        value=5
    )
    min_trade_size = st.sidebar.slider(
        "Min Trade Size (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Minimum position change required to execute a trade (as % of portfolio)"
    )
    
    # Advanced scenario testing
    st.sidebar.subheader("Scenario Testing")
    use_scenario = st.sidebar.checkbox("Apply Market Scenario", value=False)
    
    scenario_type = None
    scenario_params = {}
    
    if use_scenario:
        scenario_type = st.sidebar.selectbox(
            "Scenario Type",
            options=["flash_crash", "low_liquidity"],
            index=0
        )
        scenario_params = collect_scenario_params(scenario_type)
    
    # Run backtest button
    run_backtest = st.sidebar.button("Run Backtest")
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "symbol": symbol,
        "timeframe": timeframe,
        "agent_type": agent_type,
        "initial_balance": initial_balance,
        "risk_params": {
            "max_position_size": max_position_size / 100.0,  # Convert to decimal
            "stop_loss": stop_loss / 100.0,  # Convert to decimal
            "min_trade_size": min_trade_size / 100.0  # Convert to decimal
        },
        "use_scenario": use_scenario,
        "scenario_type": scenario_type,
        "scenario_params": scenario_params,
        "run_backtest": run_backtest,
        "model_path": model_path
    }

def display_backtest_results(results: Dict[str, Any]):
    """Display backtest results in the main content area"""
    if not results.get("success", False):
        st.warning(results.get("message", "No backtest results available"))
        return
    
    # Get results data
    portfolio_data = results.get("portfolio_data")
    trade_list = results.get("trade_list")
    metrics = results.get("metrics", {})
    scenario_type = results.get("scenario_type")
    scenario_metrics = results.get("scenario_metrics", {})
    
    # Display metrics
    st.subheader("Performance Metrics")
    display_trading_metrics(metrics)
    
    # Display scenario-specific metrics if applicable
    if scenario_type and scenario_metrics:
        display_scenario_metrics(scenario_type, scenario_metrics)
    
    # Display portfolio chart
    st.subheader("Portfolio Performance")
    if portfolio_data is not None and not portfolio_data.empty:
        portfolio_chart = create_portfolio_chart(portfolio_data)
        if portfolio_chart:
            st.plotly_chart(portfolio_chart, use_container_width=True)
    
    # Display trade list
    if trade_list:
        st.subheader("Trade History")
        
        # Convert trade list to DataFrame for display
        trade_df = pd.DataFrame(trade_list)
        
        # Format dates and numbers
        if not trade_df.empty and "timestamp" in trade_df.columns:
            trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"])
            trade_df = trade_df.sort_values("timestamp", ascending=False)
            
            if "pnl" in trade_df.columns:
                trade_df["pnl"] = trade_df["pnl"].map("${:.2f}".format)
                
            if "price" in trade_df.columns:
                trade_df["price"] = trade_df["price"].map("${:.2f}".format)
        
        st.dataframe(trade_df, use_container_width=True)

def backtest_results_page():
    """Main function for the backtest results page"""
    st.title("Backtest Results")
    
    try:
        # Initialize presenter if not exists
        if "backtest_presenter" not in st.session_state:
            st.session_state.backtest_presenter = BacktestPresenter()
        
        presenter = st.session_state.backtest_presenter
        
        # Get settings from sidebar
        settings = backtest_sidebar()
        
        # Run backtest if requested
        if settings["run_backtest"]:
            with st.spinner("Running backtest..."):
                # Step 1: Load market data
                success = presenter.load_market_data(
                    symbol=settings["symbol"],
                    timeframe=settings["timeframe"],
                    start_date=settings["start_date"],
                    end_date=settings["end_date"]
                )
                
                if not success:
                    st.error("Failed to load market data")
                    return
                
                # Step 2: Apply scenario if selected
                if settings["use_scenario"] and settings["scenario_type"]:
                    success = presenter.apply_scenario(
                        scenario_type=settings["scenario_type"],
                        scenario_params=settings["scenario_params"]
                    )
                    
                    if not success:
                        st.error(f"Failed to apply {settings['scenario_type']} scenario")
                        return
                
                # Step 3: Run backtest
                success = presenter.run_backtest(
                    agent_type=settings["agent_type"],
                    risk_params=settings["risk_params"],
                    initial_balance=settings["initial_balance"],
                    model_path=settings.get("model_path")
                )
                
                if not success:
                    st.error("Failed to run backtest")
                    return
                
                st.success("Backtest completed successfully!")
                
                # Get results
                results = presenter.get_results()
                
                # Display detailed results
                st.subheader("Trading Metrics")
                if settings["use_scenario"] and settings["scenario_type"]:
                    # If scenario metrics exist, use two columns
                    col1, col2 = st.columns(2)
                    with col1:
                        display_trading_metrics(results.get("metrics", {}))
                    with col2:
                        display_scenario_metrics(settings["scenario_type"], results.get("scenario_metrics", {}))
                else:
                    # If no scenario metrics, use full width for trading metrics
                    display_trading_metrics(results.get("metrics", {}))
                
                # Display portfolio value time series
                st.subheader("Portfolio Value Time Series")
                portfolio_values = results.get("portfolio_values", [])
                timestamps = results.get("timestamps", [])
                
                if portfolio_values and timestamps:
                    # Create DataFrame with portfolio values
                    portfolio_df = pd.DataFrame({
                        "timestamp": timestamps,
                        "portfolio_value": portfolio_values
                    })
                    
                    # Display both table and chart
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.dataframe(portfolio_df)
                    
                    with col4:
                        fig = px.line(
                            portfolio_df, 
                            x="timestamp", 
                            y="portfolio_value",
                            title="Portfolio Value Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        logger.info("Successfully created portfolio value time series visualization")
                else:
                    st.warning("No portfolio history data available")
                
                # Display price chart with trades
                st.subheader("Market Data and Trades")
                # Get the appropriate data for the chart
                chart_data = presenter.modified_data if presenter.modified_data is not None else presenter.data
                
                if chart_data is not None and not chart_data.empty:
                    price_chart = create_price_chart(chart_data)  # Use modified data for chart
                    if price_chart:
                        st.plotly_chart(price_chart, use_container_width=True)
                        logger.info("Successfully created price chart")
                    else:
                        st.warning("Failed to create price chart")
                else:
                    st.warning("No market data available for charting")
                
                # Display trade list
                st.subheader("Trade History")
                trades = results.get("trades", [])
                if trades:
                    trades_df = pd.DataFrame(trades)
                    
                    # Define exact column order to match the trade dictionary structure
                    display_columns = [
                        "timestamp",          # Basic trade info
                        "symbol",
                        "type",              # buy/sell
                        "amount",
                        "price",
                        
                        # Transaction details (in order of calculation)
                        "fee",               # Transaction fee
                        "cost",              # For sells: cost basis portion
                        "revenue",           # For sells: revenue from sale
                        "profit",            # For sells: revenue - cost - fee
                        
                        # Portfolio impact
                        "portfolio_value_before",
                        "portfolio_value_after",
                        "cumulative_pnl",
                        
                        # Position details
                        "position_units",
                        "position_value",
                        "cash_after",
                        
                        # Status
                        "success",
                        "reason"
                    ]
                    
                    # Only use columns that exist in the DataFrame
                    display_columns = [col for col in display_columns if col in trades_df.columns]
                    trades_df = trades_df[display_columns]
                    
                    # Format numeric columns with clear labels
                    format_cols = {
                        "price": lambda x: f"${x:,.2f}",
                        "fee": lambda x: f"Fee: ${x:,.2f}",
                        "cost": lambda x: f"Cost: ${x:,.2f}",
                        "revenue": lambda x: f"Rev: ${x:,.2f}",
                        "profit": lambda x: f"P/L: ${x:,.2f}",
                        "cumulative_pnl": lambda x: f"Cum.P/L: ${x:,.2f}",
                        "portfolio_value_before": lambda x: f"${x:,.2f}",
                        "portfolio_value_after": lambda x: f"${x:,.2f}",
                        "position_value": lambda x: f"${x:,.2f}",
                        "cash_after": lambda x: f"${x:,.2f}",
                        "position_units": lambda x: f"{x:,.6f}"
                    }
                    
                    # Apply formatting only to columns that exist
                    for col, format_func in format_cols.items():
                        if col in trades_df.columns:
                            trades_df[col] = trades_df[col].map(format_func)
                    
                    # Display with Streamlit's column configuration
                    st.dataframe(
                        trades_df,
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn(
                                "Time",
                                format="DD/MM/YY HH:mm:ss"
                            ),
                            "type": st.column_config.TextColumn(
                                "Type",
                                width="small"
                            ),
                            "symbol": st.column_config.TextColumn(
                                "Symbol",
                                width="small"
                            ),
                            "fee": st.column_config.TextColumn(
                                "Fee",
                                width="medium",
                                help="Transaction fee"
                            ),
                            "cost": st.column_config.TextColumn(
                                "Cost Basis",
                                width="medium",
                                help="Portion of original purchase cost"
                            ),
                            "revenue": st.column_config.TextColumn(
                                "Revenue",
                                width="medium",
                                help="Amount received from sale"
                            ),
                            "profit": st.column_config.TextColumn(
                                "Profit/Loss",
                                width="medium",
                                help="Revenue - Cost - Fee"
                            ),
                            "success": st.column_config.CheckboxColumn(
                                "Success",
                                width="small",
                                help="Whether the trade was successful"
                            ),
                            "reason": st.column_config.TextColumn(
                                "Reason",
                                width="medium"
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    logger.info("Displayed %d trades with corrected column order", len(trades))
                else:
                    st.info("No trades to display")
        else:
            # If no backtest requested, show existing results if available
            results = presenter.get_results()
            if results:
                display_backtest_results(results)
            else:
                st.info("Click 'Run Backtest' to start a new backtest")
        
    except Exception as e:
        logger.error(f"Error in backtest results page: {str(e)}", exc_info=True)
        st.error("An error occurred in the backtest results page. Check the logs for details.")

if __name__ == "__main__":
    backtest_results_page() 