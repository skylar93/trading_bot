"""Asynchronous Feature Generator with Queue-based Progress Tracking"""

import threading
import queue
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging
import time

from data.utils.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)

class AsyncFeatureGenerator:
    """Asynchronous wrapper for feature generation"""
    
    def __init__(self):
        self._base_generator = FeatureGenerator()
        self._running = False
        self._progress_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        
    def start(self, df: pd.DataFrame):
        """Start feature generation in background"""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        def _progress_callback(progress: float, message: str):
            """Handle progress updates"""
            if not self._stop_event.is_set():
                self._progress_queue.put({
                    'progress': progress,
                    'message': message,
                    'time': time.time()
                })
                # Add small delay to prevent queue overflow
                time.sleep(0.1)
        def _generate():
            """Run feature generation"""
            try:
                # Initial progress
                _progress_callback(0.0, "Starting feature generation")
                
                # Run generation
                result = self._base_generator.generate_features(df, _progress_callback)
                
                # Put result in queue if not stopped
                if not self._stop_event.is_set():
                    self._result_queue.put(('success', result))
                    
            except Exception as e:
                logger.error("Feature generation error", exc_info=True)
                if not self._stop_event.is_set():
                    error_msg = str(e)
                    self._update_progress(1.0, f"Error: {error_msg}")
                    self._result_queue.put(('error', error_msg))
            finally:
                self._running = False
                
        # Start generation thread
        self._thread = threading.Thread(target=_generate, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop feature generation"""
        if self._running:
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=1.0)
            self._running = False
            
            # Clear queues
            for queue_obj in [self._progress_queue, self._result_queue]:
                while not queue_obj.empty():
                    try:
                        queue_obj.get_nowait()
                    except queue.Empty:
                        break
                        
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get latest progress information if available"""
        try:
            return self._progress_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_result(self) -> Optional[Tuple[str, pd.DataFrame]]:
        """Get final result if available"""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None
            
    def is_running(self) -> bool:
        """Check if generation is still running"""
        return self._running