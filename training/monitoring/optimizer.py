"""Automated training optimization"""

import ray
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
import time

from .performance_analyzer import PerformanceAnalyzer
from .metrics_collector import MetricsCollector
from .worker_manager import WorkerManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""

    min_batch_size: int = 32
    max_batch_size: int = 512
    min_workers: int = 2
    max_workers: int = 8
    target_batch_time: float = 0.5
    optimization_interval: float = 60.0
    performance_window: int = 100


@ray.remote
class TrainingOptimizer:
    """Automated training optimization"""

    def __init__(
        self,
        config: OptimizationConfig,
        metrics_collector: MetricsCollector,
        worker_manager: WorkerManager,
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.worker_manager = worker_manager

        self.performance_analyzer = PerformanceAnalyzer()
        self._running = False

        # Optimization state
        self._current_batch_size = config.min_batch_size
        self._optimization_history = []

    def start(self):
        """Start optimization"""
        if self._running:
            return

        self._running = True
        self._optimize_thread = threading.Thread(
            target=self._optimize_loop, daemon=True
        )
        self._optimize_thread.start()

    def stop(self):
        """Stop optimization"""
        self._running = False
        if hasattr(self, "_optimize_thread"):
            self._optimize_thread.join()

    def get_current_config(self) -> Dict[str, Any]:
        """Get current training configuration

        Returns:
            Current configuration dictionary
        """
        return {
            "batch_size": self._current_batch_size,
            "num_workers": len(self.worker_manager.get_active_workers()),
            "optimization_history": self._optimization_history,
        }

    def _optimize_loop(self):
        """Main optimization loop"""
        while self._running:
            try:
                # Collect recent metrics
                metrics = self.metrics_collector.get_recent_metrics(
                    self.config.performance_window
                )

                # Analyze performance
                analysis = self.performance_analyzer.analyze_batch(
                    metrics["batch_metrics"][-1],
                    metrics["resource_metrics"][-1],
                )

                # Get optimization suggestions
                suggestions = (
                    self.performance_analyzer.get_optimization_suggestions()
                )

                # Apply optimizations
                if suggestions:
                    self._apply_optimizations(analysis, suggestions)

            except Exception as e:
                logger.error(
                    f"Error in optimization loop: {str(e)}", exc_info=True
                )

            time.sleep(self.config.optimization_interval)

    def _apply_optimizations(self, analysis: Any, suggestions: List[str]):
        """Apply optimization suggestions

        Args:
            analysis: Performance analysis
            suggestions: Optimization suggestions
        """
        changes_made = []

        # Optimize batch size
        if "batch size" in " ".join(suggestions).lower():
            new_batch_size = self._optimize_batch_size(analysis)
            if new_batch_size != self._current_batch_size:
                self._current_batch_size = new_batch_size
                changes_made.append(f"Adjusted batch size to {new_batch_size}")

        # Optimize worker count
        if any("resource" in s.lower() for s in suggestions):
            worker_stats = self.worker_manager.get_worker_stats()
            if worker_stats["avg_batch_time"] > self.config.target_batch_time:
                # Need more workers
                current = worker_stats["active_workers"]
                if current < self.config.max_workers:
                    self.worker_manager._scale_to(current + 1)
                    changes_made.append(f"Increased workers to {current + 1}")
            elif (
                worker_stats["avg_batch_time"]
                < self.config.target_batch_time * 0.5
            ):
                # Can reduce workers
                current = worker_stats["active_workers"]
                if current > self.config.min_workers:
                    self.worker_manager._scale_to(current - 1)
                    changes_made.append(f"Decreased workers to {current - 1}")

        # Record optimization
        if changes_made:
            self._optimization_history.append(
                {
                    "timestamp": time.time(),
                    "changes": changes_made,
                    "metrics": {
                        "batch_time": analysis.batch_time,
                        "throughput": analysis.throughput,
                        "memory_efficiency": analysis.memory_efficiency,
                        "cost_efficiency": analysis.cost_efficiency,
                    },
                }
            )

    def _optimize_batch_size(self, analysis: Any) -> int:
        """Optimize batch size

        Args:
            analysis: Performance analysis

        Returns:
            Optimized batch size
        """
        current = self._current_batch_size

        # Check memory efficiency
        if analysis.memory_efficiency > 0.9:
            # Memory pressure - reduce batch size
            return max(self.config.min_batch_size, int(current * 0.8))
        elif analysis.memory_efficiency < 0.5:
            # Can increase batch size
            return min(self.config.max_batch_size, int(current * 1.2))

        # Check GPU efficiency if available
        if analysis.gpu_efficiency:
            if analysis.gpu_efficiency < 0.7:
                # GPU underutilized - increase batch size
                return min(self.config.max_batch_size, int(current * 1.2))
            elif analysis.gpu_efficiency > 0.95:
                # GPU overutilized - reduce batch size
                return max(self.config.min_batch_size, int(current * 0.8))

        # Consider throughput trend
        if len(self._optimization_history) >= 2:
            prev_throughput = self._optimization_history[-2]["metrics"][
                "throughput"
            ]
            if analysis.throughput < prev_throughput * 0.9:
                # Significant throughput drop - revert changes
                return self._optimization_history[-2].get(
                    "batch_size", current
                )

        return current  # No change needed
