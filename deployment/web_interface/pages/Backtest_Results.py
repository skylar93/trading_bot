"""
Backtest results page for the Trading Bot UI
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from utils.backtest import BacktestManager, logger  # Import the shared logger
from components.charts import create_price_chart, create_portfolio_chart
from components.metrics import display_trading_metrics
from training.backtesting.scenario_manager import ScenarioManager

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
            # Debug log selected settings
            logger.info("Selected agent: %s", agent_name)
            logger.info("Selected scenario: %s", scenario_type)
            logger.info("Scenario params: %s", scenario_params)
            
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
            logger.info("Loading market data...")
            raw_data = backtest_manager.load_market_data()
            if raw_data is None:
                st.error("Failed to load market data")
                return
            logger.info("Loaded market data shape: %s", raw_data.shape)
                
            # Stage 2: Apply scenario transformation
            try:
                logger.info("Applying scenario transformation...")
                modified_data = scenario_manager.apply_scenario(
                    raw_data=raw_data,
                    scenario_type=scenario_type,  # Now using internal string directly
                    params=scenario_params
                )
                logger.info("Modified data shape: %s", modified_data.shape)
            except ValueError as e:
                st.error(f"Error applying scenario: {str(e)}")
                return
                
            # Stage 3: Run backtest
            logger.info("Running backtest with agent: %s", agent_name)
            results = backtest_manager.run_backtest(modified_data)
            
            # Debug log results
            if "portfolio_values" in results:
                final_value = results["portfolio_values"][-1] if results["portfolio_values"] else None
                logger.info("Final portfolio value: %.2f", final_value)
                logger.info("Total trades: %d", len(results.get("trades", [])))
                logger.info("Portfolio history length: %d", len(results.get("portfolio_values", [])))
            
            # Display results
            st.subheader("Trading Metrics")
            if scenario_type != "none":
                # If scenario metrics exist, use two columns
                col1, col2 = st.columns(2)
                with col1:
                    display_trading_metrics(results.get("metrics", {}))
                with col2:
                    display_scenario_metrics(scenario_type, results.get("scenario_metrics", {}))
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
                    import plotly.express as px
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
            st.subheader("Trade History")
            price_chart = create_price_chart(modified_data)  # Use modified data for chart
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
                logger.info("Successfully created price chart")
            
            # Display trade list
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
                
                # Add debug logging for profit calculation verification
                for trade in trades:
                    if trade["type"] == "sell" and trade.get("profit", 0) > 0:
                        logger.info(
                            "Profitable sell trade found: fee=%.2f, cost=%.2f, revenue=%.2f, profit=%.2f",
                            trade.get("fee", 0),
                            trade.get("cost", 0),
                            trade.get("revenue", 0),
                            trade.get("profit", 0)
                        )
                
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
                logger.warning("No trades in results")
                
    except Exception as e:
        logger.error(f"Error in backtest page: {str(e)}", exc_info=True)
        st.error("An error occurred in the backtest page. Check the logs for details.")

if __name__ == "__main__":
    main() 