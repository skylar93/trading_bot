"""Background feature generation utility"""

import logging
import queue
import threading
import pandas as pd
from data.utils.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)


class BackgroundFeatureGenerator:
    """Background worker for feature generation"""

    def __init__(self):
        self._generator = FeatureGenerator()
        self._running = False
        self._progress_queue = queue.Queue()
        self._result_queue = queue.Queue()

    def generate_features(self, df: pd.DataFrame) -> None:
        """Generate features in background thread"""
        self._running = True

        def progress_callback(progress: float, message: str) -> None:
            """Handle progress updates"""
            if self._running:
                self._progress_queue.put((progress, message))

        try:
            # Run feature generation
            result = self._generator.generate_features(df, progress_callback)
            self._result_queue.put(("success", result))

        except Exception as e:
            logger.error("Feature generation error", exc_info=True)
            self._result_queue.put(("error", str(e)))

        finally:
            self._running = False

    def start(self, df: pd.DataFrame) -> None:
        """Start feature generation in background"""
        threading.Thread(
            target=self.generate_features, args=(df,), daemon=True
        ).start()

    def get_progress(self) -> tuple:
        """Get latest progress update"""
        try:
            return self._progress_queue.get_nowait()
        except queue.Empty:
            return None

    def get_result(self) -> tuple:
        """Get generation result if complete"""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop feature generation"""
        self._running = False
