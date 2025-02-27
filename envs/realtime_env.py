"""
Real-time trading environment that extends base trading environment.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from .single_asset_rl_env import SingleAssetRLTradingEnv
from data.utils.realtime_data import TradingDataStream
import logging

logger = logging.getLogger(__name__)

class RealtimeTradingEnvironment(SingleAssetRLTradingEnv):
    """Real-time trading environment"""
    
    def __init__(self,
                 symbol: str = 'BTC/USDT