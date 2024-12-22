import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.train import TrainingPipeline
from data.utils.data_loader import DataLoader


def main():
    # Load configuration
    config_path = os.path.join(project_root, "config", "default_config.yaml")

    # Initialize pipeline
    pipeline = TrainingPipeline(config_path)

    # Load data
    data_loader = DataLoader(
        exchange_id=pipeline.config["data"]["exchange"],
        symbol=pipeline.config["data"]["symbols"][0],
        timeframe=pipeline.config["data"]["timeframe"],
    )

    # Fetch and prepare data
    data = data_loader.fetch_data(
        start_date=pipeline.config["data"]["start_date"]
    )

    # Split data into train/val
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    # Train the agent
    agent = pipeline.train(train_data, val_data)

    # Evaluate on test data
    test_metrics = pipeline.evaluate(agent, test_data)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
