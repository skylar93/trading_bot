## SpecStory

## Streamlit Metrics Transformation Discussion (2/4/2025, 9:39:05 AM)

_**User**_

지금은 streamlit을 실행해도.. metrics가 계산이 안 돼. 그걸 고치려고 base_backtester.py의 metric calculation을 추가한 거거든. 이제 그걸 고칠 거야. 

네, **`base_backtester.py`를 되도록 건드리지 않고**, 이미 계산된 결과 딕셔너리를 **UI에서 원하는 형태로 후처리**하는 방식을 쓰시면 됩니다.  

즉, **`BaseBacktester`가 돌려주는 `results["metrics"]`**를 `metrics.py`에서 **한 번 더 가공**해주면 되죠.  
이를 위해, 예를 들어 `metrics.py` 안에 **중간 변환 함수**를 하나 두고, UI가 필요한 형태로 매핑해주시면 좋습니다.

---

## 1) `base_backtester.py`가 반환하는 딕셔너리 예시

`BaseBacktester`의 `_calculate_metrics()`에서 대략 다음 키들을 반환합니다:

```python
{
  'total_return': float,  # e.g. 0.25 => +25%
  'sharpe_ratio': float,
  'sortino_ratio': float,
  'max_drawdown': float,  # +값일 때 0.2 => 20% 낙폭
  'total_trades': int,
  'win_rate': float,      # 0.3 => 30% 승률
  'final_balance': float,
  'final_portfolio_value': float
}
```

여기서 **UI가 기대하는** `win_rate`는 실질적으로 30처럼 퍼센트를 원할 수도 있고,  
`max_drawdown`도 20처럼 “%” 단위로 표시되길 원할 수 있습니다(현재는 0.2로만 들어옴).  
또, UI에서는 `profit_factor`, `avg_trade` 같은 필드를 표시하지만, 백테스터에는 없을 수도 있습니다.

---

## 2) 변환(매핑) 함수 추가하기

예를 들어 `metrics.py` 내에,  
```python
def convert_backtester_metrics_to_ui(backtester_metrics: Dict) -> Dict:
    """
    Transform the raw metrics dict from BaseBacktester into the format
    that display_trading_metrics() expects.
    """
    # 1) 원본에서 키 꺼내기
    raw_win_rate = backtester_metrics.get("win_rate", 0.0)      # e.g. 0.3
    raw_max_dd = backtester_metrics.get("max_drawdown", 0.0)    # e.g. 0.2
    raw_total_trades = backtester_metrics.get("total_trades", 0)
    raw_sharpe = backtester_metrics.get("sharpe_ratio", 0.0)
    
    # 2) 변환/후처리
    #    - win_rate 0.3 -> 30.0
    #    - max_drawdown 0.2 -> 20.0
    #    - profit_factor, avg_trade가 없으면 0 or None
    win_rate_percent = raw_win_rate * 100.0
    max_dd_percent = raw_max_dd * 100.0

    # profit_factor, avg_trade 등은 백테스터에서 계산하지 않으므로 0 처리(또는 None)
    profit_factor = 0.0
    avg_trade = 0.0
    
    # 3) 최종 UI용 딕셔너리 구성
    ui_metrics = {
        "sharpe_ratio": raw_sharpe,
        "win_rate": win_rate_percent,            # UI는 %.1f % 로 뿌리기
        "max_drawdown": max_dd_percent,          # UI는 %.1f % 로 뿌리기
        "profit_factor": profit_factor,
        "total_trades": raw_total_trades,
        "avg_trade": avg_trade
    }
    
    return ui_metrics
```

- 여기서 **0.3(30%) → 30.0** 같은 식으로 변환  
- `profit_factor`, `avg_trade`는 백테스터 로직에 없다면 **0.0** 등으로 대체  

---

## 3) `display_trading_metrics()`에서 변환 함수 사용하기

`Backtest_Results.py`에서 **백테스트를 돌린 후** `results["metrics"]`를 얻게 되면,  
바로 `convert_backtester_metrics_to_ui()` 함수에 넘겨서 변환된 딕셔너리를 `display_trading_metrics()`에 주면 됩니다.  

예시:

```python
# Backtest 후 얻은 metrics
backtest_metrics = results["metrics"]  # base_backtester가 계산한 원본

# UI 변환
from components.metrics import convert_backtester_metrics_to_ui
ui_metrics = convert_backtester_metrics_to_ui(backtest_metrics)

# 그다음 표시
display_trading_metrics(ui_metrics)
```

그럼 `display_trading_metrics()`에서는:

```python
def display_trading_metrics(metrics: Dict):
    try:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")

        with col2:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1f}%")
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

        with col3:
            st.metric("Total Trades", metrics.get('total_trades', 0))
            st.metric("Average Trade", f"${metrics.get('avg_trade', 0):.2f}")

    except Exception as e:
        ...
```

여기서 `win_rate=30.0`이 들어가면 **"30.0%"**로 표시되고, `max_drawdown=20.0`이 들어가면 **"20.0%"**로 표시되는 식입니다.

---

## 4) 정리

- **`BaseBacktester`를 그대로 두고 싶다면**, “UI에서 표시하기 전에 중간 변환 로직을 넣는다.”  
- 변환 로직에서,  
  - **비율(0.3)을 퍼센트(30.0)로 바꾸거나**,  
  - 없는 지표(`profit_factor`, `avg_trade`)를 0으로 채우거나,  
  - 원하는 대로 가공하면 됩니다.  
- 이렇게 하면 **원본 백테스터의 구조**나 **기존 테스트**도 깨지지 않고, UI에서 원하는 지표 형식도 충족될 것입니다. 

이걸 따라하면 될 것 같아.

---

_**Assistant**_

네, 이해했습니다. 단계별로 진행해보겠습니다.

먼저 현재 코드를 확인해보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



이제 `metrics.py`에 변환 함수를 추가하겠습니다:

---

_**Assistant**_

```
"""
Metrics components for the Trading Bot UI
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def convert_backtester_metrics_to_ui(backtester_metrics: Dict) -> Dict:
    """
    Transform the raw metrics dict from BaseBacktester into the format
    that display_trading_metrics() expects.

    Parameters
    ----------
    backtester_metrics : Dict
        Raw metrics dictionary from BaseBacktester containing values like:
        - win_rate as decimal (e.g. 0.3 for 30%)
        - max_drawdown as decimal (e.g. 0.2 for 20%)
        - sharpe_ratio, total_trades etc.

    Returns
    -------
    Dict
        Processed metrics dictionary with values formatted for UI display:
        - win_rate as percentage (e.g. 30.0)
        - max_drawdown as percentage (e.g. 20.0)
        - Additional derived metrics like profit_factor and avg_trade
    """
    try:
        # Extract raw values with defaults
        raw_win_rate = backtester_metrics.get("win_rate", 0.0)
        raw_max_dd = backtester_metrics.get("max_drawdown", 0.0)
        raw_total_trades = backtester_metrics.get("total_trades", 0)
        raw_sharpe = backtester_metrics.get("sharpe_ratio", 0.0)
        raw_sortino = backtester_metrics.get("sortino_ratio", 0.0)
        
        # Convert decimals to percentages
        win_rate_percent = raw_win_rate * 100.0
        max_dd_percent = raw_max_dd * 100.0
        
        # Calculate or default additional metrics
        # Note: These could be enhanced with actual calculations if data available
        profit_factor = backtester_metrics.get("profit_factor", 0.0)
        avg_trade = backtester_metrics.get("avg_trade", 0.0)
        
        # Construct UI metrics dictionary
        ui_metrics = {
            "sharpe_ratio": raw_sharpe,
            "sortino_ratio": raw_sortino,
            "win_rate": win_rate_percent,
            "max_drawdown": max_dd_percent,
            "profit_factor": profit_factor,
            "total_trades": raw_total_trades,
            "avg_trade": avg_trade
        }
        
        return ui_metrics
        
    except Exception as e:
        logger.error(f"Error converting backtester metrics: {str(e)}", exc_info=True)
        # Return safe defaults if conversion fails
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_trade": 0.0
        }

// ... existing code ...
```

---

_**Assistant**_



이제 Backtest Results 페이지에서 이 변환 함수를 사용하도록 수정하겠습니다:

---

_**Assistant**_



---

_**Assistant**_

```
"""
Backtest results page for the Trading Bot UI
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from utils.backtest import BacktestManager
from components.charts import create_price_chart, create_portfolio_chart
from components.metrics import display_trading_metrics, convert_backtester_metrics_to_ui
from training.backtesting.scenario_manager import ScenarioManager

// ... existing code ...

            with col2:
                st.subheader("Trading Metrics")
                # Convert backtester metrics to UI format before display
                raw_metrics = results.get("metrics", {})
                ui_metrics = convert_backtester_metrics_to_ui(raw_metrics)
                display_trading_metrics(ui_metrics)

// ... existing code ...
```

---

_**Assistant**_



이제 변경사항을 정리하면:

1. `metrics.py`에 새로운 `convert_backtester_metrics_to_ui()` 함수를 추가했습니다:
   - 백테스터의 원본 메트릭을 UI에 맞는 형식으로 변환
   - win_rate와 max_drawdown을 퍼센트 값으로 변환 (0.3 → 30.0)
   - 없는 메트릭들(profit_factor, avg_trade)에 대한 기본값 처리
   - 에러 처리와 로깅 추가

2. `Backtest_Results.py`를 수정했습니다:
   - 새로운 변환 함수를 import
   - 메트릭 표시 전에 변환 함수를 통해 데이터 가공

이렇게 하면:
- `BaseBacktester`의 코드는 그대로 유지

---



---

