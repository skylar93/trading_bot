"""WebSocket Manager for real-time data handling"""

import asyncio
import queue
import threading
from typing import Optional, Dict, Any, Callable
import pandas as pd
import logging
from data.utils.websocket_loader import WebSocketLoader

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for Streamlit interface"""
    
    def __init__(self):
        self._ws_loader = None
        self._data_queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    async def _data_callback(self, data: Dict[str, Any]):
        """Callback for WebSocket data"""
        try:
            if not self._running:
                return
                
            self._data_queue.put(data)
        except Exception as e:
            logger.error(f"Callback error: {str(e)}")
            
    def start(self, symbol: str = "BTC/USDT"):
        """Start WebSocket connection"""
        if self._running:
            return
            
        self._running = True
        self._ws_loader = WebSocketLoader(symbol=symbol)
        self._ws_loader.add_callback(self._data_callback)
        
        def _run_websocket():
            """Run WebSocket in background thread"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self._ws_loader.start())
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
            finally:
                loop.close()
                
        self._thread = threading.Thread(target=_run_websocket, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop WebSocket connection"""
        if not self._running:
            return
            
        self._running = False
        if self._ws_loader:
            asyncio.run(self._ws_loader.stop())
            
        if self._thread:
            self._thread.join(timeout=1.0)
            
        # Clear queue
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break
                
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest data from queue"""
        try:
            return self._data_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_current_ohlcv(self) -> Optional[pd.DataFrame]:
        """Get current OHLCV data"""
        if self._ws_loader:
            return self._ws_loader.get_current_data()
        return None
        
    @property
    def is_running(self) -> bool:
        return self._running