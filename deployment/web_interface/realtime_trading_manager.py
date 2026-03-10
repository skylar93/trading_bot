"""
Real-time trading manager for web interface.

This module separates core trading logic from UI presentation in the Streamlit interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
from typing import Dict, List, Optional, Tuple

from envs.live_trading_env import LiveTradingEnvironment
from agents.strategies.single.ppo_agent import PPOAgent
from deployment.web_interface.utils.data_stream import DataStream
from deployment.web_interface.utils.state import update_portfolio_history

logger = logging.getLogger(__name__)

class RealTimeTradingManager:
    """
    Manager for real-time trading that coordinates:
    1. Environment - LiveTradingEnvironment (RL interface, state, orders)
    2. Agent - Trained RL model that computes actions
    3. Data - Market data streaming and indicators
    
    This class is responsible for:
    - Loading and initializing the trading environment
    - Loading trained agent models
    - Running the main trading loop
    - Providing data for UI visualization
    - Managing trading state (start/stop)
    
    Features:
    - Asynchronous trading loop
    - Portfolio tracking
    - Trade history recording
    - Indicator calculation
    
    Implementation Notes:
    - Clear separation between core logic and UI presentation
    - Manages asyncio tasks for the trading loop
    - Properly delegates data streaming to DataStream
    
    Recent Changes:
    - Created as part of UI/logic separation refactoring
    - Based on the original RealTimeTrading class
    """
    
    def __init__(self):
        """Initialize the trading manager with default settings"""
        self.is_running = False
        self.env = None
        self.agent = None
        self.data_stream = None
        self.trading_task = None
        self.current_step = 0
        self.portfolio_history = []
        self.trade_history = []
        self.metrics = {
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_trade": 0.0
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("RealTimeTradingManager initialized")

    def configure(self, settings: Dict):
        """Configure the trading environment with user settings"""
        # Initialize data stream if not exists
        if self.data_stream is None:
            self.data_stream = DataStream(
                symbol=settings.get("symbol", "BTC/USDT"),
                timeframe=settings.get("timeframe", "1m")
            )
        else:
            # Update existing data stream settings
            self.data_stream.symbol = settings.get("symbol", "BTC/USDT")
            self.data_stream.timeframe = settings.get("timeframe", "1m")
        
        # Update settings for environment/agent as needed
        self.settings = settings
        self.logger.info(f"Trading manager configured with settings: {settings}")

    async def start(self):
        """Start the trading loop"""
        if self.is_running:
            self.logger.warning("Trading is already running")
            return

        self.logger.info("Starting real-time trading")
        self.is_running = True
        
        # Start data stream if not running
        if not self.data_stream.is_running:
            await self.data_stream.start()
        
        # TODO: Initialize environment and agent here
        # self.env = LiveTradingEnvironment(...)
        # self.agent = PPOAgent(...)
        
        # Start trading loop
        # self.trading_task = asyncio.create_task(self.trading_loop(...))
        
        # For now, we'll just simulate trading by updating metrics
        asyncio.create_task(self._simulate_trading())

    async def stop(self):
        """Stop the trading loop"""
        if not self.is_running:
            self.logger.warning("Trading is not running")
            return

        self.logger.info("Stopping real-time trading")
        self.is_running = False
        
        # Cancel trading task if it exists
        if self.trading_task:
            self.trading_task.cancel()
            self.trading_task = None
            
        # Stop data stream
        if self.data_stream:
            self.data_stream.stop()
            
        # Clean up environment and agent
        self.env = None
        self.agent = None

    async def _simulate_trading(self):
        """Temporary method to simulate trading activity for UI development"""
        while self.is_running:
            # Update portfolio value (simulated)
            current_price = self.data_stream.get_latest_price()
            if current_price:
                # Simulate some randomness in portfolio value
                noise = np.random.normal(0, 0.001)
                direction = 1 if np.random.random() > 0.3 else -1
                change = direction * (noise + 0.0005)
                
                portfolio_value = self.settings.get("initial_balance", 10000) * (1 + change)
                
                # Add to history
                self.portfolio_history.append({
                    "timestamp": datetime.now(),
                    "value": portfolio_value
                })
                
                # Limit history length
                if len(self.portfolio_history) > 1000:
                    self.portfolio_history = self.portfolio_history[-1000:]
                    
                # Update session state portfolio history
                update_portfolio_history(portfolio_value)
                
                # Occasionally add a trade
                if np.random.random() < 0.05:  # 5% chance per iteration
                    trade_type = "buy" if np.random.random() > 0.5 else "sell"
                    trade_size = np.random.uniform(0.1, 0.5)
                    self.trade_history.append({
                        "timestamp": datetime.now(),
                        "type": trade_type,
                        "price": current_price,
                        "size": trade_size,
                        "value": trade_size * current_price
                    })
                    
                    # Limit trade history length
                    if len(self.trade_history) > 100:
                        self.trade_history = self.trade_history[-100:]
                        
                # Update metrics
                self._update_simulated_metrics()
            
            # Sleep for a while
            await asyncio.sleep(1)
    
    def _update_simulated_metrics(self):
        """Update simulated trading metrics"""
        # In a real implementation, these would be calculated from actual trading data
        self.metrics = {
            "sharpe_ratio": np.random.uniform(1.2, 2.0),
            "win_rate": np.random.uniform(50.0, 70.0),
            "max_drawdown": -np.random.uniform(3.0, 8.0),
            "profit_factor": np.random.uniform(1.2, 2.5),
            "total_trades": len(self.trade_history),
            "avg_trade": np.mean([trade["value"] for trade in self.trade_history]) if self.trade_history else 0
        }
    
    def update_ui(self, selected_indicators, debug_mode):
        """Provide data for UI updates, but don't directly manipulate the UI"""
        # Return data needed for UI updates as a dictionary
        return {
            "price_data": self.data_stream.get_current_data(),
            "indicators": self._calculate_indicators(selected_indicators),
            "portfolio_history": self.portfolio_history,
            "trade_history": self.trade_history,
            "metrics": self.metrics,
            "last_update": self.data_stream.last_update,
            "data_buffer_size": len(self.data_stream.data_buffer) if self.data_stream else 0,
            "latest_price": self.data_stream.get_latest_price() if self.data_stream else None
        }
    
    def _calculate_indicators(self, selected_indicators):
        """Calculate selected technical indicators"""
        if not self.data_stream:
            return {}
            
        data = self.data_stream.get_current_data()
        if data.empty or not any(selected_indicators.values()):
            return {}
            
        indicators = self.data_stream.calculate_indicators(data)
        
        # Filter selected indicators
        return {
            k: v for k, v in indicators.items()
            if k.split("_")[0].lower() in selected_indicators
            and selected_indicators[k.split("_")[0].lower()]
        } 