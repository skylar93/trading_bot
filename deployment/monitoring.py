import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow


def main():
    st.set_page_config(
        page_title="Trading Bot Monitor", page_icon="📈", layout="wide"
    )
    st.title("Trading Bot Monitoring Dashboard")

    # Sidebar for controls
    st.sidebar.title("Control Panel")
    experiment_name = st.sidebar.text_input(
        "MLflow Experiment Name", "trading_bot"
    )
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)", 5, 60, 10
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Portfolio Performance")
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxis=True,
            subplot_titles=("Portfolio Value", "Trading Actions"),
        )

        # Get latest MLflow run
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
            )

            if runs:
                latest_run = runs[0]

                # Plot portfolio value
                portfolio_values = [
                    m.value
                    for m in client.get_metric_history(
                        latest_run.info.run_id, "portfolio_value"
                    )
                ]
                fig.add_trace(
                    go.Scatter(y=portfolio_values, name="Portfolio Value"),
                    row=1,
                    col=1,
                )

                # Plot trading actions
                actions = [
                    m.value
                    for m in client.get_metric_history(
                        latest_run.info.run_id, "action"
                    )
                ]
                fig.add_trace(
                    go.Scatter(y=actions, name="Trading Actions"), row=2, col=1
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display current metrics
                st.subheader("Current Metrics")
                metrics = latest_run.data.metrics
                metrics_df = pd.DataFrame(
                    {"Metric": metrics.keys(), "Value": metrics.values()}
                )
                st.dataframe(metrics_df)

    with col2:
        st.subheader("Latest Trades")
        trades = [
            m.value
            for m in client.get_metric_history(
                latest_run.info.run_id, "trades"
            )
        ]
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df)

        st.subheader("Learning Progress")
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Episode", len(portfolio_values), "Current")
        with col2_2:
            if len(portfolio_values) > 1:
                pct_change = (
                    (portfolio_values[-1] - portfolio_values[0])
                    / portfolio_values[0]
                    * 100
                )
                st.metric(
                    "Return",
                    f"{pct_change:.2f}%",
                    f"{pct_change - portfolio_values[-2]:.2f}%",
                )

    # Auto refresh
    st.empty()
    st.button("Refresh")


if __name__ == "__main__":
    main()
