import asyncio
from typing import Callable, Any, Dict
import pandas as pd
import logging
from data.utils.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)

class AsyncFeatureGenerator:
    """Asynchronous wrapper for feature generation"""
    
    def __init__(self):
        self._generator = FeatureGenerator()
        self._running = False
        self._progress = 0.0
        self._messages = []
        
    @property
    def progress(self) -> float:
        return self._progress
        
    @property
    def messages(self) -> list:
        return self._messages
        
    async def generate_features(self, df: pd.DataFrame, 
                              progress_callback: Callable[[float, str], None] = None) -> pd.DataFrame:
        """Generate features asynchronously"""
        try:
            self._running = True
            self._progress = 0.0
            self._messages = []
            
            def _progress_handler(progress: float, message: str):
                self._progress = progress
                self._messages.append(message)
                if progress_callback:
                    progress_callback(progress, message)
            
            # Run feature generation in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._generator.generate_features(df, _progress_handler)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async feature generation: {str(e)}", exc_info=True)
            raise
            
        finally:
            self._running = False