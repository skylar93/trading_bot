"""WebSocket Manager for real-time data streaming"""

import asyncio
import queue
import threading
from typing import Optional, Dict, Any, Callable
import pandas as pd
import logging
from data.utils.websocket_loader import WebSocketLoader

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WebSocketManager:
    """WebSocket Manager for real-time data streaming"""

    def __init__(self):
        self._ws_loader = None
        self._running = False
        self._loop = None
        self._task = None
        self._thread = None
        self._cleanup_event = threading.Event()
        logger.info("WebSocketManager initialized")

    @property
    def is_running(self):
        return self._running

    def start(self, symbol: str):
        """Start WebSocket connection"""
        if self._running:
            logger.info("WebSocket already running, ignoring start request")
            return

        logger.info(f"Starting WebSocket connection for symbol: {symbol}")
        self._running = True
        self._cleanup_event.clear()

        def run_websocket():
            try:
                logger.debug("Creating new event loop")
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                logger.debug("Initializing WebSocket loader")
                self._ws_loader = WebSocketLoader(symbol=symbol)

                logger.debug("Creating WebSocket task")
                self._task = self._loop.create_task(self._ws_loader.start())

                try:
                    logger.info("Starting event loop")
                    self._loop.run_forever()
                except Exception as e:
                    logger.error(
                        f"Error in event loop: {str(e)}", exc_info=True
                    )
                finally:
                    try:
                        logger.debug("Cleaning up event loop")
                        # Cancel all tasks
                        pending = asyncio.all_tasks(self._loop)
                        logger.debug(
                            f"Cancelling {len(pending)} pending tasks"
                        )
                        for task in pending:
                            task.cancel()

                        # Wait for task cancellation
                        if pending:
                            logger.debug("Waiting for tasks to cancel")
                            self._loop.run_until_complete(
                                asyncio.gather(
                                    *pending, return_exceptions=True
                                )
                            )

                        logger.debug("Shutting down async generators")
                        self._loop.run_until_complete(
                            self._loop.shutdown_asyncgens()
                        )

                        logger.debug("Closing event loop")
                        self._loop.close()
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up event loop: {str(e)}",
                            exc_info=True,
                        )

            except Exception as e:
                logger.error(
                    f"Error in WebSocket thread: {str(e)}", exc_info=True
                )
            finally:
                logger.info("WebSocket thread cleanup complete")
                self._cleanup_event.set()

        # Start WebSocket in a separate thread
        logger.debug("Starting WebSocket thread")
        self._thread = threading.Thread(target=run_websocket, daemon=True)
        self._thread.start()
        logger.info("WebSocket thread started")

    def stop(self):
        """Stop WebSocket connection"""
        if not self._running:
            logger.info("WebSocket not running, ignoring stop request")
            return

        logger.info("Stopping WebSocket connection")
        try:
            self._running = False

            if self._loop is not None and self._ws_loader is not None:

                async def cleanup():
                    try:
                        logger.debug("Stopping WebSocket loader")
                        await self._ws_loader.stop()
                    except Exception as e:
                        logger.error(
                            f"Error stopping WebSocket loader: {str(e)}",
                            exc_info=True,
                        )

                if self._loop.is_running():
                    logger.debug("Scheduling cleanup tasks")
                    self._loop.call_soon_threadsafe(
                        lambda: self._loop.create_task(cleanup())
                    )
                    self._loop.call_soon_threadsafe(self._loop.stop)

                # Wait for cleanup to complete
                logger.debug("Waiting for cleanup to complete")
                cleanup_completed = self._cleanup_event.wait(timeout=5.0)
                if not cleanup_completed:
                    logger.warning("Cleanup timeout reached")

                # Wait for thread to finish
                if self._thread and self._thread.is_alive():
                    logger.debug("Waiting for thread to finish")
                    self._thread.join(timeout=2.0)
                    if self._thread.is_alive():
                        logger.warning("Thread join timeout reached")

        except Exception as e:
            logger.error(f"Error stopping WebSocket: {str(e)}", exc_info=True)
        finally:
            logger.debug("Resetting WebSocket manager state")
            self._ws_loader = None
            self._loop = None
            self._task = None
            self._thread = None
            self._cleanup_event.clear()
            logger.info("WebSocket cleanup complete")

    def get_current_ohlcv(self) -> Optional[pd.DataFrame]:
        """Get current OHLCV data"""
        if not self._running or not self._ws_loader:
            return None
        try:
            data = self._ws_loader.get_current_data()
            if data is not None:
                logger.debug(f"Retrieved OHLCV data: {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {str(e)}")
            return None

    def get_latest_data(self) -> Optional[Dict]:
        """Get latest ticker data"""
        if not self._running or not self._ws_loader:
            return None
        try:
            data = self._ws_loader._latest_ticker
            if data is not None:
                logger.debug("Retrieved latest ticker data")
            return data
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return None
