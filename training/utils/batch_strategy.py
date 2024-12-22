"""Advanced batch processing strategies"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    """Statistics for a processed batch"""

    batch_size: int
    processing_time: float
    loss: float
    metrics: Dict[str, float]


class BatchStrategy:
    """Dynamic batch processing strategy"""

    def __init__(
        self,
        min_batch_size: int = 32,
        max_batch_size: int = 256,
        target_time: float = 0.1,  # Target processing time in seconds
        adjustment_rate: float = 0.1,  # How quickly to adjust batch size
    ):
        """Initialize strategy

        Args:
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_time: Target processing time per batch
            adjustment_rate: Rate at which to adjust batch size
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_time = target_time
        self.adjustment_rate = adjustment_rate

        self.current_batch_size = min_batch_size
        self.history: List[BatchStats] = []

    def get_next_batch_size(self) -> int:
        """Get batch size for next processing round

        Returns:
            Recommended batch size
        """
        if not self.history:
            return self.current_batch_size

        # Calculate average processing time
        recent_times = [stat.processing_time for stat in self.history[-5:]]
        avg_time = np.mean(recent_times)

        # Adjust batch size based on processing time
        if avg_time > self.target_time:
            # Processing too slow, decrease batch size
            adjustment = -self.adjustment_rate * self.current_batch_size
        else:
            # Can try larger batches
            adjustment = self.adjustment_rate * self.current_batch_size

        # Apply adjustment with bounds
        new_size = int(self.current_batch_size + adjustment)
        new_size = max(self.min_batch_size, min(self.max_batch_size, new_size))

        self.current_batch_size = new_size
        return new_size

    def record_batch(self, stats: BatchStats) -> None:
        """Record statistics for a processed batch

        Args:
            stats: Batch processing statistics
        """
        self.history.append(stats)

        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def get_stats(self) -> Dict[str, float]:
        """Get strategy statistics

        Returns:
            Dictionary of statistics
        """
        if not self.history:
            return {}

        recent_stats = self.history[-100:]

        return {
            "avg_batch_size": np.mean([s.batch_size for s in recent_stats]),
            "avg_processing_time": np.mean(
                [s.processing_time for s in recent_stats]
            ),
            "avg_loss": np.mean([s.loss for s in recent_stats]),
        }


class TimeSensitiveBatchStrategy(BatchStrategy):
    """Batch strategy that adapts to time-sensitive data"""

    def __init__(
        self,
        *args,
        time_weight: float = 0.5,
        **kwargs  # Weight for recency in data
    ):
        """Initialize strategy

        Args:
            time_weight: Weight given to recent data (0-1)
            *args: Arguments for parent class
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.time_weight = time_weight

    def get_next_batch_size(self) -> int:
        """Get next batch size with time sensitivity

        Returns:
            Recommended batch size
        """
        base_size = super().get_next_batch_size()

        if not self.history:
            return base_size

        # Calculate time-weighted performance
        times = np.array([s.processing_time for s in self.history])
        weights = np.exp(-self.time_weight * np.arange(len(times)))
        weighted_time = np.average(times, weights=weights)

        # Adjust based on weighted performance
        if weighted_time > self.target_time:
            adjustment = -0.1 * base_size
        else:
            adjustment = 0.05 * base_size

        new_size = int(base_size + adjustment)
        return max(self.min_batch_size, min(self.max_batch_size, new_size))


class AdaptiveBatchStrategy(BatchStrategy):
    """Strategy that adapts to data characteristics"""

    def analyze_data(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze data characteristics

        Args:
            data: Input data

        Returns:
            Dictionary of data characteristics
        """
        return {
            "volatility": float(data.std().mean()),
            "missing_ratio": float(data.isna().mean().mean()),
            "unique_ratio": float(
                np.mean(
                    [
                        len(data[col].unique()) / len(data)
                        for col in data.columns
                    ]
                )
            ),
        }

    def adjust_for_data(self, data: pd.DataFrame, base_size: int) -> int:
        """Adjust batch size based on data characteristics

        Args:
            data: Input data
            base_size: Base batch size

        Returns:
            Adjusted batch size
        """
        stats = self.analyze_data(data)

        # Adjust based on data characteristics
        adjustments = [
            -0.1 * stats["volatility"],  # Reduce size for volatile data
            -0.2 * stats["missing_ratio"],  # Reduce more for missing data
            0.1 * stats["unique_ratio"],  # Increase for diverse data
        ]

        total_adjustment = sum(adjustments)
        new_size = int(base_size * (1 + total_adjustment))

        return max(self.min_batch_size, min(self.max_batch_size, new_size))
