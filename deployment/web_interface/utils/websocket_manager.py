"""WebSocket Manager for real-time data streaming"""

import asyncio
import queue
import threading
from typing import Optional, Dict, Any, Callable
import pandas as pd
import logging
from data.utils.websocket_loader import WebSocketLoader

logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket Manager for real-time data streaming"""
    
    def __init__(self):
        self._ws_loader = WebSocketLoader()
        self._running = False
        self._loop = None
        self._task = None
        self._thread = None
    
    @property
    def is_running(self):
        return self._running
    
    def start(self, symbol: str):
        """Start WebSocket connection"""
        if self._running:
            return
            
        self._running = True
        
        def run_websocket():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._ws_loader = WebSocketLoader(symbol=symbol)
                self._task = self._loop.create_task(self._ws_loader.start())
                self._loop.run_forever()
            except Exception as e:
                logger.error(f"Error in WebSocket thread: {str(e)}", exc_info=True)
            finally:
                if self._loop and self._loop.is_running():
                    self._loop.stop()
        
        # Start WebSocket in a separate thread
        self._thread = threading.Thread(target=run_websocket, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop WebSocket connection"""
        if not self._running:
            return
            
        try:
            self._running = False
            
            if self._loop is not None:
                async def cleanup():
                    try:
                        if self._ws_loader:
                            await self._ws_loader.stop()
                        
                        # Cancel all running tasks
                        tasks = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()]
                        for task in tasks:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                    except Exception as e:
                        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
                    finally:
                        self._loop.stop()
                
                if self._loop.is_running():
                    self._loop.create_task(cleanup())
                else:
                    self._loop.run_until_complete(cleanup())
                
                # Wait for thread to finish
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=2.0)
                
                self._loop.close()
        except Exception as e:
            logger.error(f"Error stopping WebSocket: {str(e)}", exc_info=True)
        finally:
            self._loop = None
            self._task = None
            self._thread = None
            self._ws_loader = None
    
    def get_current_ohlcv(self) -> Optional[pd.DataFrame]:
        """Get current OHLCV data"""
        if not self._running or not self._ws_loader:
            return None
        try:
            return self._ws_loader.get_current_data()
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {str(e)}")
            return None
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get latest ticker data"""
        if not self._running or not self._ws_loader:
            return None
        try:
            return self._ws_loader._latest_ticker
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return None