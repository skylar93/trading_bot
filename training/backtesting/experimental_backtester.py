"""
Deprecated Experimental Backtester
================================

This module is deprecated. Use BaseBacktester with ExperimentalMixin instead.

Example:
--------
```python
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.experimental_features import ExperimentalMixin

class MyBacktester(ExperimentalMixin, BaseBacktester):
    pass

backtester = MyBacktester(data=data)
```
"""

import warnings
from typing import Optional
import pandas as pd

from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.experimental_features import ExperimentalMixin

class ExperimentalBacktester(ExperimentalMixin, BaseBacktester):
    """
    Deprecated: Use BaseBacktester with ExperimentalMixin instead.
    
    This class is maintained only for backward compatibility.
    It will be removed in a future version.
    
    Example:
    --------
    ```python
    from training.backtesting.base_backtester import BaseBacktester
    from training.backtesting.experimental_features import ExperimentalMixin
    
    class MyBacktester(ExperimentalMixin, BaseBacktester):
        pass
        
    backtester = MyBacktester(data=data)
    ```
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, **kwargs):
        warnings.warn(
            "ExperimentalBacktester is deprecated. Use BaseBacktester with ExperimentalMixin instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(data=data, **kwargs) 