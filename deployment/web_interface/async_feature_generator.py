"""Asynchronous Feature Generator"""

import asyncio
from typing import Callable, Any, Dict, Optional, Tuple
import pandas as pd
import logging
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from data.utils.feature_generator import FeatureGenerator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create log file with timestamp
log_file = os.path.join(
    log_dir,
    f"async_feature_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# Add file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console shows INFO and above
console_formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info(f"Logging to file: {log_file}")


class AsyncFeatureGenerator:
    """Asynchronous wrapper for feature generation"""

    def __init__(self):
        self._generator = FeatureGenerator()
        self._running = False
        self._progress = {"progress": 0.0, "message": ""}
        self._result: Optional[Tuple[str, pd.DataFrame]] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._start_time = None
        logger.info("AsyncFeatureGenerator initialized")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress"""
        progress_data = self._progress.copy()
        if self._start_time and self._running:
            elapsed = time.time() - self._start_time
            progress = progress_data["progress"]
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            progress_data["elapsed"] = elapsed
            progress_data["eta"] = eta
            logger.debug(
                f"Progress: {progress*100:.1f}% - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s"
            )
        return progress_data

    def get_result(self) -> Optional[Tuple[str, pd.DataFrame]]:
        """Get generation result"""
        return self._result

    def _progress_callback(self, progress: float, message: str):
        """Internal progress callback"""
        self._progress = {"progress": progress, "message": message}
        elapsed = time.time() - self._start_time
        eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        logger.info(f"Feature generation - {progress*100:.1f}% - {message}")
        logger.debug(f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    def start(self, data: pd.DataFrame):
        """Start feature generation in background"""
        if self._running:
            logger.warning("Feature generation already running")
            return

        if data is None or data.empty:
            logger.error("Cannot start feature generation with empty data")
            self._result = ("error", "No data provided")
            return

        logger.info(
            f"Starting feature generation for {len(data)} rows of data"
        )
        logger.debug(f"Data columns: {data.columns.tolist()}")
        logger.debug(f"Data sample:\n{data.head()}")

        self._running = True
        self._progress = {
            "progress": 0.0,
            "message": "Starting feature generation...",
        }
        self._result = None
        self._start_time = time.time()

        def _run_generation():
            try:
                logger.info(
                    "Starting feature generation process in background thread"
                )
                start_time = time.time()

                # Add memory usage logging
                import psutil

                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(
                    f"Memory usage before generation: {mem_before:.1f} MB"
                )

                result = self._generator.generate_features(
                    data, progress_callback=self._progress_callback
                )

                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(
                    f"Memory usage after generation: {mem_after:.1f} MB"
                )

                elapsed = time.time() - start_time
                logger.info(
                    f"Feature generation completed successfully in {elapsed:.2f} seconds"
                )
                logger.debug(f"Generated features: {result.columns.tolist()}")
                self._result = ("success", result)

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Error in feature generation after {elapsed:.2f} seconds: {str(e)}",
                    exc_info=True,
                )
                self._result = ("error", str(e))
            finally:
                self._running = False
                logger.info("Feature generation thread completed")

        logger.info("Submitting feature generation task to executor")
        self._executor.submit(_run_generation)

    def stop(self):
        """Stop feature generation"""
        if not self._running:
            return

        logger.info("Stopping feature generation")
        self._running = False
        self._executor.shutdown(wait=False)
        self._executor = ThreadPoolExecutor(max_workers=1)

        if self._start_time:
            elapsed = time.time() - self._start_time
            logger.info(
                f"Feature generation stopped after {elapsed:.2f} seconds"
            )
