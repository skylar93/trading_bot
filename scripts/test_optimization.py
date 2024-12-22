"""Test optimization and monitoring system"""

import os
import sys
import logging
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import ray
import numpy as np
import torch
from typing import Dict, Any

from training.monitoring.optimizer import TrainingOptimizer, OptimizationConfig
from training.monitoring.metrics_collector import MetricsCollector
from training.monitoring.worker_manager import WorkerManager
from training.monitoring.performance_analyzer import PerformanceAnalyzer
from data.utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DummyModel:
    """Dummy model for testing"""

    def __init__(self, input_size: int = 10):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.MSELoss()

    def train_batch(self, batch_size: int) -> Dict[str, float]:
        """Process a training batch"""
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 1)

        start_time = datetime.now()

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        end_time = datetime.now()

        return {
            "loss": loss.item(),
            "batch_size": batch_size,
            "start_time": start_time.timestamp(),
            "end_time": end_time.timestamp(),
            "memory_used": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            ),
        }


def run_test(config: OptimizationConfig):
    """Run optimization test"""

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=config.max_workers,
            num_gpus=1 if torch.cuda.is_available() else None,
        )

    try:
        # Create components
        metrics_collector = MetricsCollector.remote()
        worker_manager = WorkerManager.remote(
            min_workers=config.min_workers, max_workers=config.max_workers
        )

        optimizer = TrainingOptimizer.remote(
            config, metrics_collector, worker_manager
        )

        # Start monitoring and optimization
        ray.get(
            [
                metrics_collector.start_monitoring.remote(),
                worker_manager.start.remote(),
                optimizer.start.remote(),
            ]
        )

        # Create test model
        model = DummyModel()

        logger.info("Starting test training...")

        # Run test iterations
        for i in range(100):
            # Get current batch size
            current_config = ray.get(optimizer.get_current_config.remote())
            batch_size = current_config["batch_size"]

            # Process batch
            metrics = model.train_batch(batch_size)

            # Record metrics
            ray.get(metrics_collector.record_batch.remote(metrics))

            # Log progress
            if i % 10 == 0:
                stats = ray.get(worker_manager.get_worker_stats.remote())
                logger.info(
                    f"Iteration {i}: batch_size={batch_size}, "
                    f"workers={stats['active_workers']}, "
                    f"loss={metrics['loss']:.4f}"
                )

        # Get final analysis
        analyzer = PerformanceAnalyzer()
        final_metrics = ray.get(metrics_collector.get_recent_metrics.remote())
        if final_metrics["batch_metrics"]:
            analysis = analyzer.analyze_batch(
                final_metrics["batch_metrics"][-1],
                final_metrics["resource_metrics"][-1],
            )

            suggestions = analyzer.get_optimization_suggestions()

            logger.info("Final Performance Analysis:")
            logger.info(f"Throughput: {analysis.throughput:.2f} samples/sec")
            logger.info(f"Memory Efficiency: {analysis.memory_efficiency:.2%}")
            if analysis.gpu_efficiency:
                logger.info(f"GPU Efficiency: {analysis.gpu_efficiency:.2%}")
            logger.info(f"Cost Efficiency: {analysis.cost_efficiency:.2f}")

            logger.info("\nOptimization Suggestions:")
            for suggestion in suggestions:
                logger.info(f"- {suggestion}")

    finally:
        # Cleanup
        if "optimizer" in locals():
            ray.get(optimizer.stop.remote())
        if "worker_manager" in locals():
            ray.get(worker_manager.stop.remote())
        if "metrics_collector" in locals():
            ray.get(metrics_collector.stop.remote())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test optimization system")

    parser.add_argument(
        "--min-workers", type=int, default=2, help="Minimum number of workers"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of workers"
    )
    parser.add_argument(
        "--min-batch", type=int, default=32, help="Minimum batch size"
    )
    parser.add_argument(
        "--max-batch", type=int, default=256, help="Maximum batch size"
    )

    args = parser.parse_args()

    config = OptimizationConfig(
        min_batch_size=args.min_batch,
        max_batch_size=args.max_batch,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
    )

    run_test(config)
