"""
MLflow Results Visualization Tool.

This script provides visualization and analysis of MLflow experiment results
to help understand and compare training performance across different models,
agents, and configurations.

Features:
- Visualizes training metrics over time
- Compares performance across different models/configurations
- Analyzes agent interaction in multi-agent setups
- Provides statistical summaries of training results
- Generates performance reports with key insights

Implementation Notes:
- Uses MLflow API to access run data
- Generates interactive Plotly visualizations
- Includes statistical analysis of results
- Supports filtering and grouping of experiments
- HTML report generation for easy sharing
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlflow_visualizer")

def setup_mlflow(tracking_uri: Optional[str] = None):
    """Set up MLflow client with tracking URI."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Default to local mlruns directory
        default_uri = os.path.join(project_root, "mlruns")
        if os.path.exists(default_uri):
            mlflow.set_tracking_uri(f"file://{os.path.abspath(default_uri)}")
    
    client = MlflowClient(mlflow.get_tracking_uri())
    return client

def get_experiments(client: MlflowClient, experiment_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get experiments, optionally filtered by name."""
    experiments = client.search_experiments()
    
    if experiment_filter:
        experiments = [e for e in experiments if experiment_filter.lower() in e.name.lower()]
    
    return experiments

def get_runs_df(client: MlflowClient, experiment_id: str) -> pd.DataFrame:
    """Get runs for an experiment as a DataFrame with metrics and params."""
    runs = client.search_runs(experiment_id)
    
    # Convert to DataFrame for easier analysis
    run_data = []
    for run in runs:
        data = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time/1000),
            "end_time": datetime.fromtimestamp(run.info.end_time/1000) if run.info.end_time else None,
            "artifact_uri": run.info.artifact_uri,
            "lifecycle_stage": run.info.lifecycle_stage,
            "duration": (run.info.end_time - run.info.start_time)/1000 if run.info.end_time else None,
        }
        
        # Add all metrics
        for key, value in run.data.metrics.items():
            data[f"metric.{key}"] = value
        
        # Add all params
        for key, value in run.data.params.items():
            data[f"param.{key}"] = value
        
        # Add all tags
        for key, value in run.data.tags.items():
            data[f"tag.{key}"] = value
        
        run_data.append(data)
    
    if not run_data:
        return pd.DataFrame()
    
    return pd.DataFrame(run_data)

def get_metric_history(client: MlflowClient, run_id: str, metric_key: str) -> pd.DataFrame:
    """Get history of a specific metric for a run."""
    history = client.get_metric_history(run_id, metric_key)
    
    if not history:
        return pd.DataFrame()
    
    return pd.DataFrame([
        {"step": h.step, "timestamp": datetime.fromtimestamp(h.timestamp/1000), "value": h.value}
        for h in history
    ])

def create_metric_comparison_plot(runs_df: pd.DataFrame, metric_name: str, 
                                 group_by: Optional[str] = None) -> go.Figure:
    """Create comparison plot for a specific metric across runs."""
    if f"metric.{metric_name}" not in runs_df.columns:
        logger.warning(f"Metric '{metric_name}' not found in runs")
        return go.Figure()
    
    if group_by and f"param.{group_by}" in runs_df.columns:
        fig = px.bar(
            runs_df, 
            x="run_id", 
            y=f"metric.{metric_name}",
            color=f"param.{group_by}",
            title=f"{metric_name} by {group_by}",
            labels={f"metric.{metric_name}": metric_name, "run_id": "Run ID"},
            hover_data=["start_time", "duration"]
        )
    else:
        fig = px.bar(
            runs_df, 
            x="run_id", 
            y=f"metric.{metric_name}",
            title=f"{metric_name} comparison",
            labels={f"metric.{metric_name}": metric_name, "run_id": "Run ID"},
            hover_data=["start_time", "duration"]
        )
    
    return fig

def create_metric_history_plot(client: MlflowClient, runs_df: pd.DataFrame, 
                              metric_name: str) -> go.Figure:
    """Create plot showing metric history over time for multiple runs."""
    fig = go.Figure()
    
    for _, run in runs_df.iterrows():
        run_id = run["run_id"]
        history_df = get_metric_history(client, run_id, metric_name)
        
        if not history_df.empty:
            run_name = run.get("tag.mlflow.runName", run_id[:8])
            
            fig.add_trace(go.Scatter(
                x=history_df["step"],
                y=history_df["value"],
                mode='lines',
                name=f"{run_name}"
            ))
    
    fig.update_layout(
        title=f"{metric_name} over training steps",
        xaxis_title="Step",
        yaxis_title=metric_name,
        legend_title="Run"
    )
    
    return fig

def create_multi_agent_performance_plot(runs_df: pd.DataFrame) -> Optional[go.Figure]:
    """Create plot comparing agent performance in multi-agent runs."""
    # Find all metrics that look like they belong to agents (contain agent names)
    agent_metrics = [col for col in runs_df.columns if col.startswith("metric.") and "/" in col]
    
    if not agent_metrics:
        return None
    
    # Group metrics by agent (assuming format like "metric.best_eval_reward/agent_1")
    agent_data = {}
    for metric in agent_metrics:
        # Extract agent name and metric type
        parts = metric.split("/")
        if len(parts) != 2:
            continue
            
        metric_type = parts[0].replace("metric.", "")
        agent_name = parts[1]
        
        if agent_name not in agent_data:
            agent_data[agent_name] = {}
        
        agent_data[agent_name][metric_type] = runs_df[metric]
    
    if not agent_data:
        return None
    
    # Create comparison visualization
    fig = make_subplots(rows=1, cols=1)
    
    for agent_name, metrics in agent_data.items():
        if "best_eval_reward" in metrics:
            fig.add_trace(
                go.Bar(
                    x=[agent_name],
                    y=[metrics["best_eval_reward"].mean()],
                    name=agent_name,
                    error_y=dict(
                        type='data',
                        array=[metrics["best_eval_reward"].std()],
                        visible=True
                    )
                )
            )
    
    fig.update_layout(
        title="Agent Performance Comparison",
        yaxis_title="Average Best Evaluation Reward",
        showlegend=True
    )
    
    return fig

def analyze_runs(runs_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze runs data to extract key insights."""
    if runs_df.empty:
        return {}
    
    analysis = {
        "total_runs": len(runs_df),
        "completed_runs": len(runs_df[runs_df["status"] == "FINISHED"]),
        "failed_runs": len(runs_df[runs_df["status"] == "FAILED"]),
        "average_duration": runs_df["duration"].mean() if "duration" in runs_df else None,
        "metrics_summary": {},
        "best_runs": {}
    }
    
    # Analyze metrics
    metric_cols = [col for col in runs_df.columns if col.startswith("metric.")]
    for col in metric_cols:
        metric_name = col.replace("metric.", "")
        if "/" in metric_name:
            # Skip complex metrics for summary
            continue
            
        metric_data = runs_df[col].dropna()
        if not metric_data.empty:
            analysis["metrics_summary"][metric_name] = {
                "mean": metric_data.mean(),
                "median": metric_data.median(),
                "min": metric_data.min(),
                "max": metric_data.max(),
                "std": metric_data.std()
            }
            
            # Find best run for this metric
            if "best_eval_reward" in metric_name or "return" in metric_name:
                best_idx = metric_data.idxmax()
                analysis["best_runs"][metric_name] = {
                    "run_id": runs_df.loc[best_idx, "run_id"],
                    "value": metric_data.max(),
                    "params": {
                        k.replace("param.", ""): v 
                        for k, v in runs_df.loc[best_idx].items() 
                        if k.startswith("param.")
                    }
                }
    
    return analysis

def create_html_report(
    experiment_name: str,
    runs_df: pd.DataFrame,
    analysis: Dict[str, Any],
    figures: List[go.Figure]
) -> str:
    """Create an HTML report with analysis and visualizations."""
    # Convert plotly figures to HTML
    plot_html = ""
    for i, fig in enumerate(figures):
        if fig is not None:
            plot_html += f'<div class="plot-container">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>'
    
    # Format analysis as HTML
    analysis_html = "<h2>Analysis Summary</h2>"
    analysis_html += f"<p>Total Runs: {analysis.get('total_runs', 0)}</p>"
    analysis_html += f"<p>Completed Runs: {analysis.get('completed_runs', 0)}</p>"
    analysis_html += f"<p>Failed Runs: {analysis.get('failed_runs', 0)}</p>"
    
    if analysis.get("average_duration"):
        analysis_html += f"<p>Average Duration: {analysis['average_duration']:.2f} seconds</p>"
    
    # Add metrics summary
    if analysis.get("metrics_summary"):
        analysis_html += "<h3>Metrics Summary</h3>"
        analysis_html += "<table class='metrics-table'>"
        analysis_html += "<tr><th>Metric</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Std</th></tr>"
        
        for metric, stats in analysis["metrics_summary"].items():
            analysis_html += f"<tr><td>{metric}</td>"
            analysis_html += f"<td>{stats['mean']:.4f}</td>"
            analysis_html += f"<td>{stats['median']:.4f}</td>"
            analysis_html += f"<td>{stats['min']:.4f}</td>"
            analysis_html += f"<td>{stats['max']:.4f}</td>"
            analysis_html += f"<td>{stats['std']:.4f}</td></tr>"
            
        analysis_html += "</table>"
    
    # Add best runs
    if analysis.get("best_runs"):
        analysis_html += "<h3>Best Runs</h3>"
        
        for metric, run_info in analysis["best_runs"].items():
            analysis_html += f"<h4>Best Run for {metric}</h4>"
            analysis_html += f"<p>Run ID: {run_info['run_id']}</p>"
            analysis_html += f"<p>Value: {run_info['value']:.4f}</p>"
            
            if run_info.get("params"):
                analysis_html += "<h5>Parameters</h5>"
                analysis_html += "<table class='params-table'>"
                analysis_html += "<tr><th>Parameter</th><th>Value</th></tr>"
                
                for param, value in run_info["params"].items():
                    analysis_html += f"<tr><td>{param}</td><td>{value}</td></tr>"
                    
                analysis_html += "</table>"
    
    # Create full HTML document
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLflow Analysis: {experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3, h4, h5 {{ color: #333; }}
            .plot-container {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics-table {{ width: 100%; }}
            .params-table {{ width: 100%; }}
        </style>
    </head>
    <body>
        <h1>MLflow Experiment Analysis: {experiment_name}</h1>
        <p>Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        {analysis_html}
        
        <h2>Visualizations</h2>
        {plot_html}
    </body>
    </html>
    """
    
    return html

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize MLflow experiment results")
    
    parser.add_argument(
        "--tracking-uri", 
        type=str, 
        help="MLflow tracking URI. If not specified, uses local mlruns directory"
    )
    
    parser.add_argument(
        "--experiment-filter", 
        type=str, 
        help="Filter experiments by name (case-insensitive)"
    )
    
    parser.add_argument(
        "--experiment-id", 
        type=str, 
        help="Specific experiment ID to analyze"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="mlflow_reports",
        help="Directory for output reports"
    )
    
    parser.add_argument(
        "--metrics", 
        type=str, 
        nargs="+", 
        default=["best_eval_reward"],
        help="Metrics to include in analysis and visualization"
    )
    
    parser.add_argument(
        "--group-by", 
        type=str, 
        help="Parameter to group results by"
    )
    
    parser.add_argument(
        "--show-agent-comparison", 
        action="store_true",
        help="Show agent comparison for multi-agent runs"
    )
    
    return parser.parse_args()

def main():
    """Main function to generate MLflow visualizations and reports."""
    args = parse_args()
    
    # Set up MLflow
    client = setup_mlflow(args.tracking_uri)
    logger.info(f"Connected to MLflow tracking at {mlflow.get_tracking_uri()}")
    
    # Get experiments
    if args.experiment_id:
        try:
            experiment = client.get_experiment(args.experiment_id)
            experiments = [experiment]
        except Exception as e:
            logger.error(f"Failed to get experiment {args.experiment_id}: {e}")
            return 1
    else:
        experiments = get_experiments(client, args.experiment_filter)
    
    if not experiments:
        logger.error("No experiments found matching criteria")
        return 1
    
    logger.info(f"Found {len(experiments)} experiments")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each experiment
    for experiment in experiments:
        experiment_name = experiment.name
        logger.info(f"Processing experiment: {experiment_name}")
        
        # Get runs for this experiment
        runs_df = get_runs_df(client, experiment.experiment_id)
        
        if runs_df.empty:
            logger.warning(f"No runs found for experiment {experiment_name}")
            continue
        
        logger.info(f"Found {len(runs_df)} runs for experiment {experiment_name}")
        
        # Analyze runs
        analysis = analyze_runs(runs_df)
        
        # Create visualizations
        figures = []
        
        # Create metric comparison plots
        for metric in args.metrics:
            comparison_fig = create_metric_comparison_plot(runs_df, metric, args.group_by)
            if comparison_fig:
                figures.append(comparison_fig)
            
            history_fig = create_metric_history_plot(client, runs_df[:5], metric)  # Limit to 5 runs for readability
            if history_fig:
                figures.append(history_fig)
        
        # Create multi-agent comparison if requested
        if args.show_agent_comparison:
            agent_fig = create_multi_agent_performance_plot(runs_df)
            if agent_fig:
                figures.append(agent_fig)
        
        # Create HTML report
        html_report = create_html_report(
            experiment_name=experiment_name,
            runs_df=runs_df,
            analysis=analysis,
            figures=figures
        )
        
        # Save report
        report_filename = f"{experiment_name.replace(' ', '_')}_report.html"
        report_path = os.path.join(args.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Saved report to {report_path}")
        
        # Save analysis as JSON
        analysis_filename = f"{experiment_name.replace(' ', '_')}_analysis.json"
        analysis_path = os.path.join(args.output_dir, analysis_filename)
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Saved analysis to {analysis_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 