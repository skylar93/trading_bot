import warnings
from typing import Optional

import pandas as pd

from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.risk_manager import RiskConfig, RiskManager

class RiskAwareBacktester(BaseBacktester):
    """
    Deprecated wrapper for BaseBacktester with RiskManager.
    This class is maintained for backward compatibility and will be removed in a future version.
    
    Use BaseBacktester with risk_config argument instead:
    ```python
    risk_config = RiskConfig(max_position_size=0.1)
    backtester = BaseBacktester(data=data, risk_config=risk_config)
    ```
    """
    
    def __init__(
        self,
        data: pd.DataFrame = None,
        risk_config: Optional[RiskConfig] = None,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        max_position: float = 1.0,
    ):
        """
        Initialize risk-aware backtester.
        
        Args:
            data (pd.DataFrame): OHLCV data
            risk_config (RiskConfig): Risk management configuration
            initial_capital (float): Initial capital
            trading_fee (float): Trading fee as fraction
            max_position (float): Maximum position size as fraction
        """
        warnings.warn(
            "RiskAwareBacktester is deprecated and will be removed in a future version. "
            "Use BaseBacktester with risk_config argument instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if risk_config is None:
            risk_config = RiskConfig()
            
        super().__init__(
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            max_position=max_position,
            data=data,
            risk_config=risk_config
        ) 