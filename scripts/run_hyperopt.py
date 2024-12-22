"""Run hyperparameter optimization"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from training.hyperopt.hyperopt_tuner import MinimalTuner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Download recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = yf.download("BTC-USD", start=start_date, end=end_date, interval="5m")

    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    tuner = MinimalTuner(df=df, mlflow_experiment="trading-bot-opt")

    best_config = tuner.run_optimization(
        num_samples=10  # Adjust based on available time/resources
    )

    logger.info(f"Best configuration found: {best_config}")

    # Evaluate best config
    metrics = tuner.evaluate_config(best_config, episodes=3)
    logger.info(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
