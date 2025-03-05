## SpecStory

## Refactoring BaseBacktester to Use Update Method (2/21/2025, 5:32:41 PM)

_**User**_

아래는 **기존 `BaseBacktester.run()` 메서드를** `update()` 메서드를 활용하도록 리팩토링하는 예시입니다.  
이렇게 수정하면 **모든 Bar(캔들)마다** 포트폴리오 가치가 기록되고, “트레이드가 없어도 시세 변동으로 인한 가치 변동”을 알 수 있게 됩니다.  
(= 마치 OpenAI Gym 스타일로, 매 timestep마다 `env.step(action)`을 호출하여 state‧reward를 업데이트하는 방식)

---

## 1. 커서(Copilot 등)에게 지시할 주요 단계

1. **`run()` 메서드에서 `execute_trade()` 대신 `update()`를 호출**하게 변경.  
2. `update()`에 넘길 **prices 딕셔너리**와 **actions 딕셔너리**를 구성.  
   - 단일 심볼이라면:  
     ```python
     prices = {"default": current_data["$close"]}
     actions = {"default": action_in_0_to_1}
     ```
3. `update()` 내부에서 이미
   - `execute_trade(...)`를 호출 (단일 심볼이면 1번, 멀티 심볼이면 여러 번)
   - `self.portfolio_history.append(new_portfolio_value)` 등 **포트폴리오 변화를 매 Bar마다** 기록
4. 이렇게 하면, **매 Bar**가 끝날 때마다 `update()` 결과가 리턴되어, “trade가 없었더라도” 그 시점의 포트폴리오 가치가 기록된다.  
5. (옵션) 기존에 `execute_trade()`로부터 직접 리턴받아 처리하던 로직은 지우거나 주석 처리하여 **중복 호출**이 없도록 한다.

---

## 2. 예시 코드: `run()` 메서드 수정안

아래는 **단일 심볼**(예: `"default"`) 기준으로, `run()` 메서드를 수정하는 방법 예시입니다.

```python
def run(
    self,
    strategy: Any,
    window_size: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reworked run method using update() for single-asset strategies.
    Now each bar calls update() once, so we track portfolio changes every step.
    """
    if self.data is None:
        raise ValueError("No data provided for backtesting")
        
    try:
        for i in range(window_size, len(self.data)):
            window_data = self.data.iloc[i - window_size : i].copy()
            current_data = self.data.iloc[i].copy()
            timestamp = current_data.name

            # 1) get fraction in [0,1] from the strategy
            try:
                raw_action = strategy.get_action(window_data)
                action = float(np.clip(raw_action, 0.0, 1.0))
            except Exception as e:
                self.logger.error(f"Error getting action from strategy: {str(e)}")
                continue

            # 2) Prepare 'prices' and 'actions' dict for update()
            prices = {"default": current_data["$close"]}
            actions = {"default": action}

            # 3) Call self.update() once per bar
            update_result = self.update(
                timestamp=timestamp,
                prices=prices,
                actions=actions
            )
            # update_result 안에는 portfolio_value, trades, 등등이 들어있음.

            # (옵션) verbose 모드에서 진행 상황 로깅
            if verbose and i % 100 == 0:
                self.logger.info(f"Progress: {i}/{len(self.data)} bars processed.")

    except Exception as e:
        self.logger.error(f"Error during backtest: {str(e)}")
        raise

    # 4) Calculate final metrics
    metrics = self._calculate_metrics()

    # 5) Adjust length of self.portfolio_history if needed
    expected_length = len(self.data) - window_size + 1
    if len(self.portfolio_history) < expected_length:
        last_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_capital
        self.portfolio_history.extend([last_value] * (expected_length - len(self.portfolio_history)))
    elif len(self.portfolio_history) > expected_length:
        self.portfolio_history = self.portfolio_history[:expected_length]

    return {
        "metrics": metrics,
        "trades": self.trades,  # still populated by execute_trade() calls inside update()
        "portfolio_values": self.portfolio_history,
        "timestamps": self.data.index[window_size - 1 :].tolist(),
    }
```

**핵심 포인트**:  
- `update()`가 “trade가 없을 때도” **현 시점의 포트폴리오 가치**를 `self.portfolio_history.append(...)`에 저장하기 때문에,  
  - Bar가 지날 때마다 시가가 변동 → 내가 fraction holding 중이라면 “이미 들고 있던 포지션의 평가가치”도 변동.  
  - 그 변동이 매번 기록됨.  
- `update()` 내부에서 각 Bar마다 수행한 트레이드(또는 trade_size_too_small일 경우) 정보를 `self.trades`에 누적.

---

## 3. `update()` 내부 흐름 (이미 존재하지만)

현재 `update()`는 대체로 이런 식입니다:

```python
def update(self, timestamp, prices, actions):
    # 1) record initial portfolio_value
    initial_portfolio_value = self.get_portfolio_value(prices)

    total_fees = 0.0
    results = {}

    # 2) for each symbol in actions: execute_trade()
    for symbol, fraction in actions.items():
        if symbol not in prices:
            continue
        trade_result = self.execute_trade(timestamp, fraction, prices, symbol)
        results[symbol] = trade_result
        if trade_result["success"]:
            total_fees += trade_result["fee"]

    # 3) after trading, record new portfolio_value
    new_portfolio_value = self.get_portfolio_value(prices)
    self.portfolio_history.append(new_portfolio_value)
    self.cash_history.append(self.cash)

    # 4) Return a summary
    return {
        "timestamp": timestamp,
        "trades": results,
        "portfolio_value": new_portfolio_value,
        "cash": self.cash,
        "positions": self.positions.copy(),
        "returns": (
            (new_portfolio_value - initial_portfolio_value) / initial_portfolio_value
            if initial_portfolio_value > 1e-12 else 0.0
        ),
        "total_fees": total_fees
    }
```

**주요 변경점**:  
- 원래는 `update()`가 “외부에서 안 쓰이고”, `execute_trade()`만 직접 불렀던 걸, 이제 **매 Bar마다 `update()`를 호출**하게 바꿨다.  
- 단일 심볼이라면 `actions = {"default": action}`, 멀티 심볼이라면 `{"BTC": 0.3, "ETH": 0.2, ...}` 식으로 호출.

---

## 4. 커서(Copilot)에게 말할 문구 예시

아래는 ChatGPT나 Copilot(Cursor) 등에 **“어떻게 말하면”** 위 작업을 좀 더 쉽게 수행시킬 수 있는지 예시입니다:

> **"Cursor, please modify the `BaseBacktester.run()` method to use `update()` instead of calling `execute_trade()` directly. For each bar:**
> 1. **Compute the action** by calling `strategy.get_action(window_data)`, clamp it to `[0,1]`.  
> 2. Create a `prices = {"default": current_data["$close"]}` dictionary.  
> 3. Create an `actions = {"default": action}` dictionary.  
> 4. **Call `self.update(timestamp, prices, actions)`**.  
> 5. Remove or comment out any direct calls to `execute_trade()` from within `run()`.  
> 6. Make sure the returned results from `update()` are stored or logged if needed.  
> 7. Finally, compute metrics as before.  
> 8. Return the same structure with metrics, trades, portfolio_values, etc.”  
>
> **"This way, every bar will append the new portfolio value to `self.portfolio_history`, even if there's no actual trade."**

이렇게 구체적으로 지시하면, 커서(Copilot 등)도 기존 코드를 분석해 **적절히 리팩토링**할 것입니다.

---

### 정리

- **목표**: “모든 Bar마다 포트폴리오 가치 변화를 기록”하고 싶다면, `run()`에서 각 Bar마다 `update()`를 호출하도록 변경.  
- 그 결과, **트레이드 여부와 상관없이** 매번 `portfolio_history`에 기록이 쌓이며, 차트나 표로 “계좌 추이”를 직관적으로 볼 수 있음.  
- **커서**에게는 “`execute_trade()` 대신 `update()`를 활용”하라고 명확히 지시하고, 구체적 변환 단계를 알려주면 된다.  

이렇게 리팩토링하면, **fraction holding이든, 다중 심볼이든** 매 timestep마다 포트폴리오 변화를 쉽게 추적할 수 있게 됩니다.

---

_**Assistant**_

I'll help you modify the `BaseBacktester.run()` method according to the requirements. First, let me search for the relevant files.

---

_****_

<details>
            <summary>Searched codebase "class BaseBacktester run method execute_trade" • **25** results</summary>
        
| File | Lines |
|------|-------|
| `training/backtesting/base_backtester.py` | L156-213 |
| `.specstory/history/creating-base-backtester-in-python.md` | L228-271 |
| `.specstory/history/creating-base-backtester-in-python.md` | L5204-5260 |
| `.specstory/history/creating-base-backtester-in-python.md` | L4180-4237 |
| `.specstory/history/creating-base-backtester-in-python.md` | L685-734 |
| `.specstory/history/creating-base-backtester-in-python.md` | L488-537 |
| `.specstory/history/creating-base-backtester-in-python.md` | L949-991 |
| `.specstory/history/creating-base-backtester-in-python.md` | L5413-5468 |
| `.specstory/history/creating-base-backtester-in-python.md` | L1439-1488 |
| `.specstory/history/creating-base-backtester-in-python.md` | L1216-1265 |
| `training/backtesting/base_backtester.py` | L100-156 |
| `.specstory/history/creating-base-backtester-in-python.md` | L1684-1734 |
| `.specstory/history/backtester-trade-metrics-test-implementation.md` | L467-513 |
| `training/backtesting/base_backtester.py` | L436-491 |
| `tests/test_backtest_execution.py` | L73-89 |
| `.specstory/history/creating-base-backtester-in-python.md` | L4613-4647 |
| `.specstory/history/trading-backtester-enhancements.md` | L3764-3814 |
| `.specstory/history/trading-backtester-enhancements.md` | L2490-2538 |
| `.specstory/history/trading-backtester-enhancements.md` | L3111-3160 |
| `.specstory/history/creating-base-backtester-in-python.md` | L907-949 |
| `.specstory/history/fixing-backtester-metrics-calculation.md` | L20-81 |
| `training/backtesting/base_backtester.py` | L491-511 |
| `.specstory/history/fixing-backtester-metrics-calculation.md` | L344-405 |
| `training/backtesting/base_backtester.py` | L279-313 |
| `training/backtesting/base_backtester.py` | L355-386 |

</details>

---

_****_



Let me analyze the code and make the necessary changes to the `run()` method to use `update()` instead of directly calling `execute_trade()`. I'll also ensure that portfolio values are tracked for every bar.

---

_****_

```diff
          Run backtest for single-asset strategy (Fractional Holding).
          strategy.get_action() => fraction in [0,1].
+         
+         Now uses update() for each bar to track portfolio changes every step,
+         even when no trade occurs.
          """
          if self.data is None:
                      continue
  
-                 # 2) do trade
-                 price_data = {"default": current_data["$close"]}
-                 trade_result = self.execute_trade(
+                 # 2) Prepare prices and actions dict for update()
+                 prices = {"default": current_data["$close"]}
+                 actions = {"default": action}
+ 
+                 # 3) Call update() once per bar
+                 update_result = self.update(
                      timestamp=timestamp,
-                     action=action,
-                     price_data=price_data,
-                     asset="default"
-                 )
- 
-                 # 3) if we didn't store portfolio_value in trade_result, update manually
-                 if "portfolio_value" not in trade_result:
-                     pv = self.get_portfolio_value(price_data)
-                     self.portfolio_history.append(pv)
-                     self.peak_value = max(self.peak_value, pv)
+                     prices=prices,
+                     actions=actions
+                 )
  
                  if verbose and i % 100 == 0:
-                     self.logger.info(f"Progress: {i}/{len(self.data)}")
+                     self.logger.info(f"Progress: {i}/{len(self.data)} bars processed")
  
          except Exception as e:
```

---

_****_



I've modified the `run()` method to use `update()` instead of directly calling `execute_trade()`. Here are the key changes:

1. Updated the docstring to reflect that we now track portfolio changes every step using `update()`

2. Changed the trade execution logic:
   - Instead of calling `execute_trade()` directly, we now prepare `prices` and `actions` dictionaries
   - Call `update()` with these dictionaries, which internally handles:
     - Executing trades via `execute_trade()`
     - Recording portfolio values in `portfolio_history`
     - Tracking cash and position changes

3. Removed the manual portfolio value update since `update()` already handles this

4. Kept the existing metrics calculation and length alignment logic

The main benefits of this change are:
- Portfolio value is now tracked for every bar, even when no trade occurs
- More consistent with OpenAI Gym style where each timestep updates state
- Better handling of position value changes due to price movements
- Cleaner separation of concerns between trade execution and state updates

The code will now:
1. Get the strategy's action for each bar
2. Convert it to a dictionary format for `update()`
3. Call `update()` which handles both trade execution and state tracking
4. Record all portfolio value changes, whether from trades or just price movements

This should give you a more complete picture of how your portfolio value changes over time, including periods where you're just holding positions.

Would you like me to make any additional changes or clarifications to the implementation?

---

_**User**_

매 Bar(혹은 각 시간단위)별로 Portfolio Value를 한눈에 보고 싶다면,

trade 이력이 아닌 results["portfolio_values"]와 results["timestamps"]로 별도 표/그래프를 하나 더 표시합니다.
예:
python
Copy
portfolio_df = pd.DataFrame({
    "timestamp": results["timestamps"],
    "portfolio_value": results["portfolio_values"]
})
st.dataframe(portfolio_df)
그러면 트레이드 실패 여부와 관계없이, 시계열 전체의 계좌 변동을 알 수 있습니다.
**“실패 거래도 table에 나오게 할거지만, portfolio_value_after도 0이 아닌 실제 값으로 기록하고 싶다”**면,

execute_trade()의 else: (trade["success"]가 False) 분기에서
python
Copy
trade["portfolio_value_after"] = self.get_portfolio_value(price_data)
trade["cumulative_pnl"] = trade["portfolio_value_after"] - self.initial_capital
등을 넣어서 0이 아닌 현재 포트폴리오 가치를 기록해두면 됩니다. (원하시면 reason=... 이나 success=False 상태는 그대로 두더라도)

---

_**Assistant**_

I'll help you modify the `execute_trade()` method to record actual portfolio values even for unsuccessful trades, while still maintaining the trade failure status. This will give us a more accurate view of portfolio value changes over time.

---

_****_

```diff
                  trade["position_value"] = 0.0
          else:
-             # For unsuccessful trades, use before values
-             trade["portfolio_value_after"] = trade["portfolio_value_before"]
-             trade["cumulative_pnl"] = trade["portfolio_value_before"] - self.initial_capital
+             # For unsuccessful trades, still record actual portfolio value
+             updated_portfolio_value = self.get_portfolio_value(price_data)
+             trade["portfolio_value_after"] = updated_portfolio_value
+             trade["cumulative_pnl"] = updated_portfolio_value - self.initial_capital
              trade["cash_after"] = self.cash
              trade["position_units"] = self.positions[asset]["units"] if asset in self.positions else 0.0
```

---

_****_



I've modified the `execute_trade()` method to record actual portfolio values even when trades are unsuccessful. Here are the key changes:

1. In the `else` block (when `trade["success"] = False`), instead of using the portfolio value from before the trade attempt:
   - We now calculate the actual current portfolio value using `self.get_portfolio_value(price_data)`
   - We store this actual value in `trade["portfolio_value_after"]`
   - We calculate the actual cumulative PnL based on this value
   - Position units and values are still recorded accurately

This change means that:
1. Failed trades will still be marked as unsuccessful (`success=False`) with their failure reason
2. But their portfolio value records will reflect the actual state of the account
3. The trade history will show a more accurate picture of portfolio value changes over time

Now when you create a portfolio value visualization like:
```python
portfolio_df = pd.DataFrame({
    "timestamp": results["timestamps"],
    "portfolio_value": results["portfolio_values"]
})
```

You'll see:
- A continuous record of portfolio values at each timestep
- Accurate values even when trades fail
- The true impact of market price changes on your portfolio, regardless of trade execution status

This gives you a more complete picture of your portfolio's performance over time, while still maintaining accurate records of which trades succeeded or failed.

Would you like me to make any additional changes to improve the portfolio value tracking?

---

_**User**_

아주 간단히 요약하자면, **`Backtest_Results.py`**에서 이미 “Trade History” 표를 표시하는 부분과 유사한 위치(즉, 백테스트 결과를 받은 뒤 시각화하고 표를 표시하는 구간)에, **한 단계 더** “포트폴리오 가치 시계열” 표를 추가해주시면 됩니다.  

---

## 1. 실제 예시 코드

아래는 `Backtest_Results.py` 내 **`if run_backtest:`** 이후 쪽에, 트레이드 히스토리를 보여주는 곳 다음(또는 이전)에 **포트폴리오 이력** 표(또는 차트)를 추가하는 예시입니다:

```python
# 이미 backtest 실행 후 나온 results
results = backtest_manager.run_backtest(modified_data)

# 1) 일반 성능, 차트, 트레이드 히스토리 표시 (기존 코드)
...

# 2) 포트폴리오 가치 시계열 테이블 추가
portfolio_values = results.get("portfolio_values", [])
timestamps = results.get("timestamps", [])

if portfolio_values and timestamps:
    portfolio_df = pd.DataFrame({
        "timestamp": timestamps,
        "portfolio_value": portfolio_values
    })
    
    st.subheader("Portfolio Value Over Time")
    st.dataframe(portfolio_df)

    # (옵션) Plotly 등을 이용해 라인차트로도 표시 가능
    #   import plotly.express as px
    #   fig = px.line(portfolio_df, x="timestamp", y="portfolio_value", title="Portfolio Value")
    #   st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No portfolio history data available.")
```

**이것만 추가**하면,  
- 백테스트 결과인 `results["portfolio_values"]`와 `results["timestamps"]`를 이용해,  
- “**매 Bar(또는 매 시점)** 계좌 가치”를 **표**나 **차트**로 확인할 수 있게 됩니다.

---

## 2. 코드 삽입 위치

지금 `Backtest_Results.py`에서 **트레이드 테이블**을 그리는 부분이 아래와 유사합니다:

```python
# Display trade list
trades = results.get("trades", [])
if trades:
    trades_df = pd.DataFrame(trades)
    ...
    st.dataframe(trades_df)
else:
    st.info("No trades to display")
```

위 코드 바로 **전에** 혹은 **후에**(원하시는 위치)에, 위에서 보여드린 **포트폴리오 테이블 생성 코드**를 추가해주시면 됩니다.  
예를 들면:

```python
...
# Stage 3: Run backtest
results = backtest_manager.run_backtest(modified_data)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Portfolio Performance")
    portfolio_chart = create_portfolio_chart(results.get("portfolio_values", []))
    ...

with col2:
    st.subheader("Trading Metrics")
    display_trading_metrics(results.get("metrics", {}))

# 이제 여기서 "Portfolio Value Table"을 추가로 보여주고 싶다면
portfolio_values = results.get("portfolio_values", [])
timestamps = results.get("timestamps", [])
if portfolio_values and timestamps:
    portfolio_df = pd.DataFrame({
        "timestamp": timestamps,
        "portfolio_value": portfolio_values
    })
    st.subheader("Portfolio Value (Table)")
    st.dataframe(portfolio_df)

# 그리고 나서 trade history
st.subheader("Trade History")
...
```

---

## 3. 차트로 그려주는 방법 (선택사항)

Streamlit에서 **시계열로 시각화**하고 싶다면, Plotly나 Altair 등을 활용 가능합니다. 예를 들어 Plotly Express:

```python
import plotly.express as px

portfolio_df = pd.DataFrame({
    "timestamp": timestamps,
    "portfolio_value": portfolio_values
})

fig = px.line(portfolio_df, x="timestamp", y="portfolio_value", title="Portfolio Value Over Time")
st.plotly_chart(fig, use_container_width=True)
```

그럼 “시간축 vs 포트폴리오 가치” 라인차트를 쉽게 표시할 수 있어요.

---

## 4. 정리

- **넣는 위치**: Backtest 결과(`results`)가 반환된 직후, 이미 **Trade History**나 **Portfolio Performance** 차트를 뿌리는 로직 근처에서 적절히 추가하면 됩니다.  
- **데이터**: `results["portfolio_values"]`와 `results["timestamps"]`가 “매 바마다의 계좌 가치 시계열”이므로, 그대로 DataFrame 만들고 `st.dataframe()` 호출.  
- **마찬가지로** 별도의 차트 라이브러리를 이용해 그래프를 그려주면, “시간에 따른 포트폴리오 가치 추이”를 직관적으로 볼 수 있습니다.

---

_**Assistant**_

I'll help you add the portfolio value time series visualization to the Backtest_Results.py file. First, let me search for the relevant file.

---

_****_



---

_****_



Let me read the current content of the file to see where to add the portfolio value visualization.

---

_****_

Read file: deployment/web_interface/pages/Backtest_Results.py

---

_****_



Let me read more of the file to see the trade history display section.

---

_****_

Read file: deployment/web_interface/pages/Backtest_Results.py

---

_****_



Now I'll add the portfolio value time series visualization. I'll edit the file to add it just before the trade history section.

---

_****_

```diff
                  display_trading_metrics(results.get("metrics", {}))
              
+             # Display portfolio value time series
+             st.subheader("Portfolio Value Time Series")
+             portfolio_values = results.get("portfolio_values", [])
+             timestamps = results.get("timestamps", [])
+             
+             if portfolio_values and timestamps:
+                 # Create DataFrame with portfolio values
+                 portfolio_df = pd.DataFrame({
+                     "timestamp": timestamps,
+                     "portfolio_value": portfolio_values
+                 })
+                 
+                 # Display both table and chart
+                 col3, col4 = st.columns(2)
+                 
+                 with col3:
+                     st.dataframe(portfolio_df)
+                 
+                 with col4:
+                     import plotly.express as px
+                     fig = px.line(
+                         portfolio_df, 
+                         x="timestamp", 
+                         y="portfolio_value",
+                         title="Portfolio Value Over Time"
+                     )
+                     st.plotly_chart(fig, use_container_width=True)
+                     logger.info("Successfully created portfolio value time series visualization")
+             else:
+                 st.warning("No portfolio history data available")
+             
              # Display price chart with trades
              st.subheader("Trade History")
```

---

_****_



I've added the portfolio value time series visualization to the Backtest_Results.py file. Here are the key additions:

1. Added a new section titled "Portfolio Value Time Series" after the Trading Metrics section and before the Trade History section

2. Created a two-column layout to show both table and chart:
   ```python
   col3, col4 = st.columns(2)
   ```

3. In the left column (`col3`):
   - Displays a DataFrame table with timestamps and portfolio values
   - Uses `st.dataframe()` for interactive table view

4. In the right column (`col4`):
   - Creates a line chart using Plotly Express
   - Shows portfolio value over time
   - Uses `st.plotly_chart()` with `use_container_width=True` for responsive sizing

5. Added proper error handling:
   - Checks if both `portfolio_values` and `timestamps` exist
   - Shows warning message if data is not available
   - Logs successful visualization creation

The visualization will show:
- A table with exact values at each timestamp
- An interactive line chart showing the portfolio value trend
- Both successful and unsuccessful trades' impact on portfolio value

This gives users a complete view of how their portfolio value changed over time, regardless of whether trades were executed or not.

Would you like me to make any adjustments to the visualization, such as:
1. Adding more chart customization (colors, hover data, etc.)?
2. Changing the layout (e.g., stacking table and chart vertically)?
3. Adding additional metrics or statistics to the display?

---

_**User**_

이제 Backtest_Results.py 페이지에서 가장 상단에 있던 Portfolio Performance 그래프를 없애줄 수 있어? 왜냐면 밑의 portfolio value over time 하고 내용이 중복돼. 가장 위에는 trading metric만 보이면 좋을 것 같다.

---

_**Assistant**_

네, Portfolio Performance 그래프를 제거하고 Trading Metrics만 표시하도록 수정하겠습니다.

---

_****_

```diff
              
              with col1:
-                 st.subheader("Portfolio Performance")
-                 portfolio_chart = create_portfolio_chart(results.get("portfolio_values", []))
-                 if portfolio_chart:
-                     st.plotly_chart(portfolio_chart, use_container_width=True)
-                     logger.info("Successfully created portfolio chart")
-                 else:
-                     st.warning("No portfolio data available")
-                     logger.warning("Failed to create portfolio chart - no data")
-             
-             with col2:
                  st.subheader("Trading Metrics")
                  display_trading_metrics(results.get("metrics", {}))
              
+             with col2:
+                 # Display scenario-specific metrics if scenario is active
+                 if scenario_type != "none":
+                     display_scenario_metrics(scenario_type, results.get("scenario_metrics", {}))
+             
              # Display portfolio value time series
              st.subheader("Portfolio Value Time Series")
```

---

_****_



변경사항:
1. 상단의 Portfolio Performance 차트와 관련된 코드를 제거했습니다
2. Trading Metrics를 왼쪽 컬럼으로 이동했습니다
3. 오른쪽 컬럼에는 시나리오가 활성화된 경우에만 시나리오 관련 메트릭을 표시하도록 했습니다
4. Portfolio Value Time Series 섹션은 그대로 유지했습니다

이제 페이지 구조가 다음과 같이 깔끔하게 정리되었습니다:
1. 상단: Trading Metrics + Scenario Metrics (시나리오 사용 시)
2. 중간: Portfolio Value Time Series (표 + 차트)
3. 하단: Trade History (차트 + 거래 목록)

이렇게 하면 중복되는 포트폴리오 차트가 제거되고

---



---

