## SpecStory

## 사이드바 UI 에이전트 선택 추가 (1/27/2025, 10:45:01 AM)

_**User**_

2) 사이드바 UI에 에이전트 선택 드롭다운 추가

(A) UI 코드 (예: deployment/web_interface/pages/Backtest_Results.py)에서 사이드바에 “Strategy” 항목을 추가:
import streamlit as st
from deployment.web_interface.utils.backtest import BacktestManager
from training.agents.agent_factory import create_agent
# ... (다른 import들)
# (시나리오 import는 1번에서 이미 했다고 가정)

def main():
    st.title("Backtest Results")

    with st.sidebar:
        st.header("Backtest Settings")

        # Strategy(Agent) 선택

	포인트
		•	st.selectbox([...])로 “Dummy / MeanReversion / Momentum / PPO”를 선택하도록 함.
	•	선택된 값 agent_name을 settings dict에 저장 → BacktestManager 생성할 때 settings["agent_name"]로 넘김.
        agent_name = st.selectbox(
            "Select Agent (Strategy)",
            ["Dummy", "MeanReversion", "Momentum", "PPO"]
        )

        # 예: 기존: StartDate, EndDate 등등
        # scenario = st.selectbox("Scenario", ["Normal", "FlashCrash", ...])

        run_backtest = st.button("Run Backtest")

    if run_backtest:
        settings = {
            # 여기에 start_date, end_date, trading_pair 등 기존꺼 넣고
            "agent_name": agent_name,  # <- 에이전트 이름
            # ... other settings ...
        }

        manager = BacktestManager(settings)
        data = manager.load_market_data()  # 실제 시세 데이터 로드
        if data is None:
            st.error("No data")
            return

        # 시나리오 적용 로직(1번에서 추가)도 들어갈 수 있음
        # if scenario == "FlashCrash": data = generate_flash_crash_data(data, ...)

        results = manager.run_backtest(data)
        # ... 결과 표시 ...

---

_**Assistant**_

I'll help you add the agent selection dropdown to the sidebar UI. First, let me check if the file exists and then make the necessary modifications.

---

_**Assistant**_



---

_**Assistant**_



Let me read the current content of the file to see how to best integrate the new agent selection dropdown.

---

_**Assistant**_



---

_**Assistant**_



I see that the file already has a strategy selection dropdown, but we need to update it to include the specific agent types you mentioned. I'll modify the file to add the agent selection dropdown with the requested options.

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
from components.metrics import display_trading_metrics
from training.backtesting.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)
// ... existing code ...

def main():
    """Main function for the backtest results page"""
    st.title("Backtest Results")
    
    try:
        # Sidebar settings
        with st.sidebar:
            st.header("Backtest Settings")
            
            # Date range selection
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30)
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
            # Trading pair selection
            trading_pair = st.selectbox(
                "Trading Pair",
                ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            )
            
            # Agent (Strategy) selection
            agent_name = st.selectbox(
                "Select Agent (Strategy)",
                ["Dummy", "MeanReversion", "Momentum", "PPO"]
            )
            
            # Scenario selection - using internal strings directly
// ... existing code ...

            if run_backtest:
                # Create settings dictionary
                settings = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "trading_pair": trading_pair,
                    "agent_name": agent_name,  # Add agent name to settings
                    "initial_balance": initial_balance,
                    "max_position_size": max_position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
// ... existing code ...
```

---

_**Assistant**_



I've updated the Backtest Results page to include the agent selection dropdown as requested. The changes include:

1. Replaced the old strategy selection with a new agent selection dropdown
2. Added the four specified agent types: "Dummy", "MeanReversion", "Momentum", and "PPO"
3. Updated the settings dictionary to use "agent_name" instead of "strategy"

The agent selection is now integrated into the sidebar UI and will be passed to the BacktestManager through the settings dictionary. The selected agent name will be available as `settings["agent_name"]` for use in creating the appropriate agent.

Would you like me to make any additional changes to the UI or help with implementing the agent creation logic in the BacktestManager?

---



---

