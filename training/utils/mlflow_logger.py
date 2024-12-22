import mlflow
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


class MLflowLogger:
    """MLflow experiment tracking for trading bot"""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow logger

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local)
            artifact_location: Location to store artifacts (default: ./mlruns)
        """
        self.experiment_name = experiment_name

        # Set up MLflow tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Get or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                if artifact_location:
                    experiment_id = mlflow.create_experiment(
                        experiment_name, artifact_location=artifact_location
                    )
                else:
                    experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)

            logger.info(
                f"Using experiment '{experiment_name}' "
                f"(experiment_id: {self.experiment.experiment_id})"
            )

        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {str(e)}")
            raise

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run"""
        try:
            mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=run_name
                or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            logger.info(
                f"Started MLflow run: {mlflow.active_run().info.run_id}"
            )
        except Exception as e:
            logger.error(f"Error starting MLflow run: {str(e)}")
            raise

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        try:
            # Convert complex types to strings
            params = {
                k: str(v) if isinstance(v, (dict, list)) else v
                for k, v in params.items()
            }
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log PyTorch model to MLflow"""
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise

    def log_figure(self, figure: Any, artifact_path: str) -> None:
        """Log matplotlib figure to MLflow"""
        try:
            mlflow.log_figure(figure, artifact_path)
        except Exception as e:
            logger.error(f"Error logging figure: {str(e)}")
            raise

    def log_dict(self, dictionary: Dict, artifact_path: str) -> None:
        """Log dictionary as JSON artifact"""
        try:
            # Convert to JSON string
            json_str = json.dumps(dictionary, indent=2, default=str)

            # Save as artifact
            with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
                mlflow.log_text(json_str, artifact_path)
        except Exception as e:
            logger.error(f"Error logging dictionary: {str(e)}")
            raise

    def log_backtest_results(
        self, results: Dict[str, Any], artifact_path: str = "backtest_results"
    ) -> None:
        """Log backtest results to MLflow"""
        try:
            # Log performance metrics
            if "metrics" in results:
                self.log_metrics(results["metrics"])

            # Save trades history
            if "trades" in results and results["trades"]:
                trades_df = pd.DataFrame(results["trades"])
                trades_path = f"{artifact_path}/trades.csv"
                mlflow.log_table(
                    data=trades_df.to_dict(orient="records"),
                    artifact_file=trades_path,
                )

            # Save portfolio values
            if "portfolio_values_with_time" in results:
                portfolio_df = pd.DataFrame(
                    results["portfolio_values_with_time"],
                    columns=["timestamp", "portfolio_value"],
                )
                portfolio_path = f"{artifact_path}/portfolio_values.csv"
                mlflow.log_table(
                    data=portfolio_df.to_dict(orient="records"),
                    artifact_file=portfolio_path,
                )

            # Log full results as JSON
            self.log_dict(results, f"{artifact_path}/full_results.json")

        except Exception as e:
            logger.error(f"Error logging backtest results: {str(e)}")
            raise

    def log_agent_info(
        self,
        agent: Any,
        artifact_path: str = "agent_info",
        include_weights: bool = False,
    ) -> None:
        """Log trading agent information"""
        try:
            # Get agent configuration
            agent_info = {
                "type": agent.__class__.__name__,
                "architecture": str(agent),
            }

            if hasattr(agent, "get_config"):
                agent_info["config"] = agent.get_config()

            # Save agent info
            self.log_dict(agent_info, f"{artifact_path}/agent_info.json")

            # Save model weights if requested
            if include_weights and hasattr(agent, "state_dict"):
                self.log_model(agent, f"{artifact_path}/model")

        except Exception as e:
            logger.error(f"Error logging agent info: {str(e)}")
            raise

    @staticmethod
    def end_run() -> None:
        """End current MLflow run"""
        try:
            mlflow.end_run()
        except Exception as e:
            logger.error(f"Error ending MLflow run: {str(e)}")
            raise

    def get_best_run(
        self, metric_name: str, mode: str = "max"
    ) -> Optional[Dict]:
        """Get best run based on metric"""
        try:
            order_by = (
                f"metrics.{metric_name} {'DESC' if mode == 'max' else 'ASC'}"
            )
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[order_by],
                max_results=1,
            )

            if len(runs) == 0:
                return None

            best_run = runs.iloc[0]
            return {
                "run_id": best_run.run_id,
                "metrics": {
                    name: best_run[f"metrics.{name}"]
                    for name in best_run.filter(
                        regex="^metrics\."
                    ).index.str.replace("metrics.", "")
                },
                "params": {
                    name: best_run[f"params.{name}"]
                    for name in best_run.filter(
                        regex="^params\."
                    ).index.str.replace("params.", "")
                },
            }

        except Exception as e:
            logger.error(f"Error getting best run: {str(e)}")
            raise

    def load_model(self, run_id: str, artifact_path: str) -> Any:
        """Load model from MLflow"""
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
