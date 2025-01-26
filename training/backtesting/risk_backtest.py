"""
Deprecated Risk-Aware Backtesting System
======================================

This module is deprecated. Use BaseBacktester with risk_config argument instead.

Example:
--------
```python
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.risk_manager import RiskConfig

risk_config = RiskConfig(
    max_position_size=0.1,
    stop_loss_pct=0.02,
    max_drawdown_pct=0.15
)
backtester = BaseBacktester(data=data, risk_config=risk_config)
```
"""

import warnings
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.risk_manager import RiskManager, RiskConfig


class RiskAwareBacktester(BaseBacktester):
    """
    Deprecated: Use BaseBacktester with risk_config argument instead.
    
    This class is maintained only for backward compatibility.
    It will be removed in a future version.
    
    Example:
    --------
    ```python
    from training.backtesting.base_backtester import BaseBacktester
    from training.backtesting.risk_manager import RiskConfig
    
    risk_config = RiskConfig(
        max_position_size=0.1,
        stop_loss_pct=0.02,
        max_drawdown_pct=0.15
    )
    backtester = BaseBacktester(data=data, risk_config=risk_config)
    ```
    """
    
    def __init__(self, data, risk_config=None, **kwargs):
        """Initialize with deprecation warning"""
        warnings.warn(
            "RiskAwareBacktester is deprecated. Use BaseBacktester with risk_config argument.",
            DeprecationWarning,
            stacklevel=2
        )
        if risk_config is None:
            risk_config = RiskConfig()
            
        super().__init__(data=data, risk_config=risk_config, **kwargs) 