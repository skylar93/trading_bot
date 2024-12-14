"""
Real-time trading environment that extends base trading environment.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from .trading_env import TradingEnvironment
from data.utils.realtime_data import TradingDataStream
import logging

logger = logging.getLogger(__name__)

class RealtimeTradingEnvironment(TradingEnvironment):
    """Real-time trading environment"""
    
    def __init__(self,
                 symbol: str = 'BTC/USDT