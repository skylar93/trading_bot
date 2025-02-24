## SpecStory

## Understanding 0% Win Rate in Trading Logic (2/23/2025, 10:05:29 PM)

_**User**_

2. **Win Rate가 0%**로 찍히는 이유
코드상 “Win Rate = (profit > 0 인 트레이드 수 / 전체 트레이드 수)” 로 계산합니다:

python
Copy
profitable_trades = sum(
    1
    for t in trades
    if t["success"] and t["profit"] > 0
)
win_rate = profitable_trades / total_trades
즉, trade["profit"]이 0보다 커야 “승리(Win)”로 카운팅합니다.

그런데 Fractional Holding 로직에서, execute_trade()는 “포지션 일부를 팔 때”만 profit이 실현됩니다. (BUY 시 profit=0, SELL 시 revenue - cost_portion - fee)

만약 실제 fill(체결) 로직에서 코스트 계산이 항상 손해로만 잡히거나,
자주 “trade_size_too_small”나 “insufficient_funds”로 인해 매수/매도가 정상 체결되지 않거나,
부분 매도할 때도 수수료나 약간의 가격 하락 때문에 이익이 실현되지 않고(= profit ≤ 0) 계속 손실만 본다면,
→ 결과적으로 profit>0이 한 번도 없어 Win Rate=0%가 됩니다.
또 한 가지:

Fractional Holding에서 “이미 50% 보유 중인데 → Action=0.52 → 0.02 * portfolio만큼만 매수”를 시도하면, 아주 작은 차이에 수수료도 포함되어 실현익이 안 나오고 오히려 약간 손실로 찍히거나,
“action=0.3 → 기존 0.31 보유” 같은 식이면 거의 체결되지 않거나, 체결되어도 실현익이 0에 가깝게 처리될 수 있음.
결과:

표면상은 727번의 트레이드가 기록되었더라도, 정작 profit 칼럼이 전부 0 이하가 되어 Win Rate=0%로 보이는 현상이 발생할 수 있습니다.

---

_**Assistant**_

문제를 이해했습니다. Win Rate 계산 로직에서 몇 가지 개선이 필요해 보입니다. 현재 구현은 너무 엄격하게 profit > 0 만을 기준으로 하고 있어서, Fractional Holding 전략의 특성을 제대로 반영하지 못하고 있습니다.

먼저 코드를 살펴보고 수정하겠습니다:

---

_****_

Read file: training/backtesting/base_backtester.py

---

_****_

Read file: training/backtesting/base_backtester.py

---

_****_



Fractional Holding 전략의 특성을 고려하여 Win Rate 계산 로직을 개선하겠습니다. 현재 구현의 문제점은:

1. 매수 거래는 항상 profit=0으로 기록되어 승리로 카운트되지 않음
2. 매도 시에도 수수료와 작은 가격 변동으로 인해 profit이 0보다 작아질 수 있음
3. 매우 작은 거래는 "trade_size_too_small"로 처리되어 성공하지 못한 거래로 기록됨

이를 개선하기 위해 다음과 같이 수정하겠습니다:

---

_****_

```diff
                  max_dd = max(max_dd, dd)
              
-             profitable_trades = sum(
-                 1
-                 for t in self.trades
-                 if t["success"] and t["profit"] > 0
-             )
-             total_trades = len(self.trades)
-             win_rate = (profitable_trades / total_trades) if total_trades>0 else 0.0
+             # Improved win rate calculation for Fractional Holding
+             successful_trades = [t for t in self.trades if t["success"]]
+             if not successful_trades:
+                 win_rate = 0.0
+             else:
+                 # For buys: count as win if portfolio value increased after the trade
+                 # For sells: count as win if profit >= -fee (i.e., broke even after fees)
+                 profitable_trades = sum(
+                     1 for t in successful_trades
+                     if (t["type"] == "buy" and t["portfolio_value_after"] > t["portfolio_value_before"]) or
+                        (t["type"] == "sell" and t["profit"] >= -t["fee"])
+                 )
+                 win_rate = profitable_trades / len(successful_trades)
+             
+             # Count actual executed trades (excluding too small/failed)
+             total_trades = len(successful_trades)
              
              return {
                  'win_rate': win_rate,
                  'final_balance': self.cash,
-                 'final_portfolio_value': values[-1]
+                 'final_portfolio_value': values[-1],
+                 'successful_trades': total_trades,
+                 'total_trade_attempts': len(self.trades)
              }
          except Exception as e:
```

---

_****_



주요 개선 사항을 설명드리겠습니다:

1. **성공한 거래만 고려**
   - `successful_trades = [t for t in self.trades if t["success"]]`
   - "trade_size_too_small"이나 "insufficient_funds" 등으로 실패한 거래는 제외

2. **매수/매도 별 다른 승리 조건 적용**
   - 매수(buy): 포트폴리오 가치가 증가했다면 승리
     ```python
     t["type"] == "buy" and t["portfolio_value_after"] > t["portfolio_value_before"]
     ```
   - 매도(sell): 수수료를 고려한 손익분기점 이상이면 승리
     ```python
     t["type"] == "sell" and t["profit"] >= -t["fee"]
     ```

3. **거래 통계 추가**
   - `successful_trades`: 실제로 실행된 거래 수
   - `total_trade_attempts`: 시도된 총 거래 수 (실패 포함)

이러한 변경으로 인한 효과:

1. 매수 거래도 포트폴리오 가치 증가를 통해 승리로 인정될 수 있음
2. 매도 시 수수료를 고려한 현실적인 손익분기점 적용
3. 너무 작은 거래나 실패한 거래는 승률 계산에서 제외
4. 실제 거래 수와 시도된 거래 수를 구분하여 볼 수 있음

이렇게 하면 Fractional Holding 전략의 특성을 더 잘 반영하는 Win Rate가 계산될 것입니다. 매수 후 포트폴리오 가치가 증가하는 경우와 매도 시 수수료를 고려한 실질적인 손익을 기준으로 승률을 평가하게 됩니다.

---

_**User**_

1. **_calculate_metrics()**에서 Win Rate 로직을 수정했는지 확인
우선 base_backtester.py 내 **_calculate_metrics()**가 “백테스트 종료 후 최종적으로 metrics 딕셔너리”를 만드는 함수입니다.
여기서 Win Rate 계산 로직을 개선하셨다면, 반드시 반환되는 metrics 딕셔너리에 win_rate가 반영되어 있어야 합니다.

예: (가장 아래 return 부분)

python
Copy
def _calculate_metrics(self) -> Dict[str, float]:
    ...
    # 누락되지 않도록 아래 처럼 포함
    return {
        'total_return': total_return,     # (예: -8.64 => -8.64%)
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_dd,
        'total_trades': total_trades,    # or successful trades
        'win_rate': win_rate,            # 개선된 로직
        'final_balance': self.cash,
        'final_portfolio_value': values[-1],
        # 필요한 항목 더 추가 가능
    }
주의:

“성공한 트레이드만” 세서 Win Rate를 구하시려면, Win Rate 계산 후 metrics["win_rate"] = some_value 식으로 넣어두셔야 합니다.
또한 “총 트레이드 수”를 실패‧성공 전체(len(self.trades))로 할지, “성공한 거래 수만”으로 할지는 결정하시고, total_trades 필드에도 반영해야 합니다.
아래는 예시:

python
Copy
successful_trades = [t for t in self.trades if t["success"]]
if successful_trades:
    profitable_trades = 0
    for t in successful_trades:
        if t["type"] == "buy":
            # BUY 승리 조건: portfolio_value_after > before
            if t["portfolio_value_after"] > t["portfolio_value_before"]:
                profitable_trades += 1
        elif t["type"] == "sell":
            # SELL 승리 조건: profit >= -fee
            if t["profit"] >= -t["fee"]:
                profitable_trades += 1
    win_rate = (profitable_trades / len(successful_trades)) * 100.0
else:
    win_rate = 0.0

total_trades = len(successful_trades)  # 혹은 len(self.trades)

metrics = {
    'total_return': total_return * 100,  # 예: -0.0864 => -8.64
    'win_rate': win_rate,               # ex) 34.5 => 34.5%
    ...
}
return metrics
포인트:

“승률”을 퍼센트로 넣을지(예: 34.5) 혹은 0~1 범위로 넣을지는 결정 후, UI 쪽에서 어떤 형식으로 표시할지 맞춰야 합니다.

---

_**Assistant**_

네, 현재 `_calculate_metrics()` 함수에서 Win Rate 계산 로직이 이미 수정되어 있고, metrics 딕셔너리에도 올바르게 포함되어 있습니다. 하지만 몇 가지 개선할 점이 있습니다:

1. 퍼센트 표시 통일
2. 예외 처리 시의 반환값도 동일한 형식 유지
3. 메트릭 설명 docstring 추가

수정하겠습니다:

---

_****_

```diff
          """
          Calculate trading performance metrics.
+         
+         Returns:
+             Dict[str, float] containing:
+             - total_return: Total return in percentage (e.g., -8.64 means -8.64%)
+             - sharpe_ratio: Annualized Sharpe ratio
+             - sortino_ratio: Annualized Sortino ratio
+             - max_drawdown: Maximum drawdown in percentage (positive)
+             - total_trades: Number of successful trades (excluding failed attempts)
+             - win_rate: Win rate in percentage (e.g., 34.5 means 34.5%)
+             - final_balance: Final cash balance
+             - final_portfolio_value: Final total portfolio value
+             - successful_trades: Number of successfully executed trades
+             - total_trade_attempts: Total number of trade attempts (including failed)
          """
          try:
              values = np.array(self.portfolio_history)
              if len(values) < 2:
                  return {
-                     'total_return': 0,
-                     'sharpe_ratio': 0,
-                     'sortino_ratio': 0,
-                     'max_drawdown': 0,
-                     'total_trades': len(self.trades),
-                     'win_rate': 0,
+                     'total_return': 0.0,
+                     'sharpe_ratio': 0.0,
+                     'sortino_ratio': 0.0,
+                     'max_drawdown': 0.0,
+                     'total_trades': 0,
+                     'win_rate': 0.0,
                      'final_balance': self.cash,
                      'final_portfolio_value': self.cash,
+                     'successful_trades': 0,
+                     'total_trade_attempts': len(self.trades)
                  }
  
              returns = np.diff(values) / values[:-1]
              
-             total_return = (values[-1] - values[0]) / values[0]
+             # Total return in percentage
+             total_return = (values[-1] - values[0]) / values[0] * 100.0
              
              if len(returns) > 1 and np.std(returns)>0:
                  sortino_ratio = 0.0
              
-             # max drawdown
+             # Max drawdown in percentage
              peak = values[0]
              max_dd = 0.0
                  dd = (peak-val)/peak
                  max_dd = max(max_dd, dd)
+             max_dd *= 100.0  # Convert to percentage
              
              # Improved win rate calculation for Fractional Holding
                         (t["type"] == "sell" and t["profit"] >= -t["fee"])
                  )
-                 win_rate = profitable_trades / len(successful_trades)
+                 win_rate = (profitable_trades / len(successful_trades)) * 100.0  # Convert to percentage
              
              # Count actual executed trades (excluding too small/failed)
              total_trades = len(successful_trades)
              
              return {
-                 'total_return': total_return,
+                 'total_return': total_return,          # In percentage (e.g., -8.64)
                  'sharpe_ratio': sharpe_ratio,
                  'sortino_ratio': sortino_ratio,
-                 'max_drawdown': max_dd,  # positive
-                 'total_trades': total_trades,
-                 'win_rate': win_rate,
+                 'max_drawdown': max_dd,                # In percentage (positive)
+                 'total_trades': total_trades,          # Successful trades only
+                 'win_rate': win_rate,                  # In percentage (e.g., 34.5)
                  'final_balance': self.cash,
                  'final_portfolio_value': values[-1],
              self.logger.error(f"Error calculating metrics: {str(e)}")
              return {
-                 'total_return': 0,
-                 'sharpe_ratio': 0,
-                 'sortino_ratio': 0,
-                 'max_drawdown': 0,
-                 'total_trades': len(self.trades),
-                 'win_rate': 0,
+                 'total_return': 0.0,
+                 'sharpe_ratio': 0.0,
+                 'sortino_ratio': 0.0,
+                 'max_drawdown': 0.0,
+                 'total_trades': 0,
+                 'win_rate': 0.0,
                  'final_balance': self.cash,
                  'final_portfolio_value': self.cash,
+                 'successful_trades': 0,
+                 'total_trade_attempts': len(self.trades)
              }
  
```

---

_****_



주요 변경사항을 설명드리겠습니다:

1. **퍼센트 표시 통일**
   - `total_return`: 소수점 → 퍼센트 (`* 100.0`)
   - `max_drawdown`: 소수점 → 퍼센트 (`* 100.0`)
   - `win_rate`: 소수점 → 퍼센트 (`* 100.0`)

2. **메트릭 설명 추가**
   ```python
   Returns:
       Dict[str, float] containing:
       - total_return: Total return in percentage (e.g., -8.64 means -8.64%)
       - win_rate: Win rate in percentage (e.g., 34.5 means 34.5%)
       ...
   ```

3. **거래 수 구분**
   - `total_trades`: 성공한 거래만 포함
   - `successful_trades`: total_trades와 동일 (성공한 거래)
   - `total_trade_attempts`: 모든 시도된 거래 (실패 포함)

4. **예외 처리 통일**
   - 에러 발생 시에도 동일한 형식의 딕셔너리 반환
   - 모든 숫자형 값을 `0.0`으로 통일
   - 실패한 거래 수는 `len(self.trades)`로 유지

5. **Win Rate 계산 로직 (기존대로 유지)**
   ```python
   if (t["type"] == "buy" and t["portfolio_value_after"] > t["portfolio_value_before"]) or
      (t["type"] == "sell" and t["profit"] >= -t["fee"])
   ```
   - 매수: 포트폴리오 가치 증가 시 승리
   - 매도: 수수료 고려한 손익분기점 이상이면 승리

이제 모든 퍼센트 값이 통일되어 있고 (`34.5`는 `34.5%`를 의미), 거래 수도 명확하게 구분되어 있습니다. UI에서는 이 값들을 그대로 사용하여 퍼센트 기호(%)를 붙여 표시하면 됩니다.

---

_**User**_

2. **UI(display_trading_metrics)**에서 Win Rate, Total Return을 어떻게 뿌리는지 확인
현재 metrics.py의 display_trading_metrics(metrics)는 아래처럼 몇 가지 필드를 가져와 st.metric으로 표시합니다:

python
Copy
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
    ...
(A) “win_rate” 표시 확인
현재 "Win Rate" 라벨에서 metrics.get('win_rate', 0)를 :.1f}%로 찍고 있습니다.

**만약 _calculate_metrics()에서 win_rate가 “0~100 범위”**로 되어 있다면, 여기서는 1.0f%로 표시가 적절합니다.

예: win_rate = 35.2 → "35.2%"
만약 _calculate_metrics()에서 “0.0 ~ 1.0 범위”(ex. 0.352)로 리턴한다면, 여기서 다시 100을 곱해주거나 formatting을 바꿔야 합니다.

(B) “Total Return”을 표시하고 싶다면?
**display_trading_metrics**에는 현재 “Total Return” 표시가 없습니다.
원하시면, col1 / col2 / col3 어느 칼럼이든 배치해서, 예:
python
Copy
with col1:
    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")

with col2:
    ...
이렇게 metrics.get('total_return', 0) 값을 %로 표시하게 만들면 UI에 나옵니다.

---

_**Assistant**_

네, UI와 백엔드의 메트릭 포맷을 일치시키고 Total Return을 추가하겠습니다.

1. 우선 `_calculate_metrics()`에서는 이미 모든 퍼센트 값이 0~100 범위로 통일되어 있습니다:
```python
total_return = (values[-1] - values[0]) / values[0] * 100.0  # 예: 35.2 means 35.2%
win_rate = (profitable_trades / len(successful_trades)) * 100.0  # 예: 35.2 means 35.2%
max_dd *= 100.0  # 예: 35.2 means 35.2%
```

2. UI의 `display_trading_metrics()`를 수정하겠습니다:

---

_****_

```diff
  
  def display_trading_metrics(metrics: Dict):
-     """Display trading performance metrics"""
+     """
+     Display trading performance metrics
+     
+     Args:
+         metrics: Dictionary containing:
+             - total_return: Total return in percentage (e.g., 35.2 means 35.2%)
+             - sharpe_ratio: Annualized Sharpe ratio
+             - sortino_ratio: Annualized Sortino ratio
+             - max_drawdown: Maximum drawdown in percentage (positive)
+             - total_trades: Number of successful trades
+             - win_rate: Win rate in percentage (e.g., 35.2 means 35.2%)
+             - successful_trades: Number of successfully executed trades
+             - total_trade_attempts: Total number of trade attempts
+     """
      try:
          # Create three columns
          with col1:
              st.metric(
-                 "Sharpe Ratio",
-                 f"{metrics.get('sharpe_ratio', 0):.2f}"
+                 "Total Return",
+                 f"{metrics.get('total_return', 0):.1f}%",
+                 delta_color="normal"
              )
              st.metric(
                  "Win Rate",
-                 f"{metrics.get('win_rate', 0):.1f}%"
+                 f"{metrics.get('win_rate', 0):.1f}%"  # Already in 0-100 range
              )
  
          with col2:
              st.metric(
                  "Max Drawdown",
-                 f"{metrics.get('max_drawdown', 0):.1f}%"
+                 f"{metrics.get('max_drawdown', 0):.1f}%"  # Already in 0-100 range
              )
              st.metric(
-                 "Profit Factor",
-                 f"{metrics.get('profit_factor', 0):.2f}"
+                 "Sharpe Ratio",
+                 f"{metrics.get('sharpe_ratio', 0):.2f}"
              )
  
          with col3:
              st.metric(
                  "Total Trades",
-                 metrics.get('total_trades', 0)
+                 f"{metrics.get('successful_trades', 0)} / {metrics.get('total_trade_attempts', 0)}"
              )
              st.metric(
-                 "Average Trade",
-                 f"${metrics.get('avg_trade', 0):.2f}"
+                 "Sortino Ratio",
+                 f"{metrics.get('sortino_ratio', 0):.2f}"
              )
  
```

---

_****_



주요 변경사항을 설명드리겠습니다:

1. **메트릭 표시 개선**
   - Total Return 추가 (col1 첫 번째 항목)
   - Win Rate 포맷 유지 (이미 0-100 범위)
   - Max Drawdown 포맷 유지 (이미 0-100 범위)

2. **지표 재구성**
   - Total Return, Win Rate (col1)
   - Max Drawdown, Sharpe Ratio (col2)
   - Total Trades, Sortino Ratio (col3)

3. **거래 수 표시 개선**
   ```python
   f"{metrics.get('successful_trades', 0)} / {metrics.get('total_trade_attempts', 0)}"
   ```
   - 성공한 거래 수 / 전체 시도 수 형태로 표시
   - 예: "150 / 200" (150개 성공, 200번 시도)

4. **불필요한 지표 제거**
   - Profit Factor 제거 (현재 계산되지 않음)
   - Average Trade 제거 (현재 계산되지 않음)

5. **문서화 개선**
   ```python
   """
   Display trading performance metrics
   
   Args:
       metrics: Dictionary containing:
           - total_return: Total return in percentage (e.g., 35.2 means 35.2%)
           - win_rate: Win rate in percentage (e.g., 35.2 means 35.2%)
           ...
   """
   ```

이제 UI는 백엔드(`_calculate_metrics()`)에서 제공하는 메트릭과 완벽하게 일치하며, 모든 퍼센트 값이 0-100 범위로 일관되게 표시됩니다. 또한 거래 수도 성공/시도 비율로 더 자세히 보여줍니다.

---

_**User**_

3. “Total Return” 계산 시 중복 곱하기/표시 문제 주의
예: _calculate_metrics()에서

python
Copy
total_return = (values[-1] / values[0]) - 1  # -0.0864 => -8.64%
total_return *= 100.0  # => -8.64
그렇다면 UI에서 f"{metrics.get('total_return', 0):.2f}%" → “-8.64%”로 표시 가능.

주의:

만약 _calculate_metrics가 “-0.0864”을 그대로 담고, UI에서 :.2f}%를 붙이면서 100배를 안 해준다면, UI에 “-0.09%”로 오해될 수 있음.
“Max Drawdown”도 같은 이슈가 있으니, 둘 중 한 곳에서만 “x 100”을 수행하시고, 일관성을 지켜야 합니다.

---

_**Assistant**_

네, 중요한 지적입니다. 현재 코드를 보면 Total Return과 Max Drawdown에서 백엔드와 UI 사이에 퍼센트 변환이 중복되어 있지 않은지 확인하겠습니다.

1. **백엔드 (`_calculate_metrics()`)에서:**
```python
# Total return in percentage
total_return = (values[-1] - values[0]) / values[0] * 100.0  # 이미 퍼센트로 변환 (예: -8.64)

# Max drawdown in percentage
max_dd = max(max_dd, dd)
max_dd *= 100.0  # 이미 퍼센트로 변환 (예: 15.2)
```

2. **UI (`display_trading_metrics()`)에서:**
```python
st.metric(
    "Total Return",
    f"{metrics.get('total_return', 0):.1f}%",  # 이미 퍼센트 값을 받아서 % 기호만 붙임
)
...
st.metric(
    "Max Drawdown",
    f"{metrics.get('max_drawdown', 0):.1f}%",  # 이미 퍼센트 값을 받아서 % 기호만 붙임
)
```

현재 구현은 다행히 중복 곱하기 문제가 없습니다. 하지만 더 명확하게 하기 위해 코드를 개선하겠습니다:

---

_****_

```diff
              Dict[str, float] containing:
              - total_return: Total return in percentage (e.g., -8.64 means -8.64%)
-             - sharpe_ratio: Annualized Sharpe ratio
-             - sortino_ratio: Annualized Sortino ratio
-             - max_drawdown: Maximum drawdown in percentage (positive)
-             - total_trades: Number of successful trades (excluding failed attempts)
+             - sharpe_ratio: Annualized Sharpe ratio (not percentage)
+             - sortino_ratio: Annualized Sortino ratio (not percentage)
+             - max_drawdown: Maximum drawdown in percentage (e.g., 15.2 means 15.2%)
+             - total_trades: Number of successful trades (count)
              - win_rate: Win rate in percentage (e.g., 34.5 means 34.5%)
-             - final_balance: Final cash balance
-             - final_portfolio_value: Final total portfolio value
-             - successful_trades: Number of successfully executed trades
-             - total_trade_attempts: Total number of trade attempts (including failed)
+             - final_balance: Final cash balance (absolute value)
+             - final_portfolio_value: Final total portfolio value (absolute value)
+             - successful_trades: Number of successfully executed trades (count)
+             - total_trade_attempts: Total number of trade attempts (count)
          """
          try:
              values = np.array(self.portfolio_history)
              if len(values) < 2:
                  return {
-                     'total_return': 0.0,
-                     'sharpe_ratio': 0.0,
-                     'sortino_ratio': 0.0,
-                     'max_drawdown': 0.0,
-                     'total_trades': 0,
-                     'win_rate': 0.0,
-                     'final_balance': self.cash,
-                     'final_portfolio_value': self.cash,
-                     'successful_trades': 0,
-                     'total_trade_attempts': len(self.trades)
+                     'total_return': 0.0,  # percentage
+                     'sharpe_ratio': 0.0,  # ratio
+                     'sortino_ratio': 0.0,  # ratio
+                     'max_drawdown': 0.0,  # percentage
+                     'total_trades': 0,    # count
+                     'win_rate': 0.0,      # percentage
+                     'final_balance': self.cash,  # absolute value
+                     'final_portfolio_value': self.cash,  # absolute value
+                     'successful_trades': 0,  # count
+                     'total_trade_attempts': len(self.trades)  # count
                  }
  
-             returns = np.diff(values) / values[:-1]
+             returns = np.diff(values) / values[:-1]  # returns as decimals
              
-             # Total return in percentage
-             total_return = (values[-1] - values[0]) / values[0] * 100.0
-             
+             # Calculate total return percentage
+             total_return_decimal = (values[-1] / values[0]) - 1  # decimal form (e.g., -0.0864)
+             total_return = total_return_decimal * 100.0  # percentage form (e.g., -8.64)
+             
+             # Calculate Sharpe ratio (this is a ratio, not a percentage)
              if len(returns) > 1 and np.std(returns)>0:
                  sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
              else:
                  sharpe_ratio = 0.0
              
+             # Calculate Sortino ratio (this is a ratio, not a percentage)
              downside_returns = returns[returns < 0]
              if len(downside_returns) > 0 and np.std(downside_returns)>0:
-                 sortino_ratio = (
-                     np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
-                 )
+                 sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
              else:
                  sortino_ratio = 0.0
              
-             # Max drawdown in percentage
+             # Calculate max drawdown percentage
              peak = values[0]
-             max_dd = 0.0
+             max_dd_decimal = 0.0  # decimal form (e.g., 0.152)
              for val in values[1:]:
                  if val>peak:
                      peak=val
                  dd = (peak-val)/peak
-                 max_dd = max(max_dd, dd)
-             max_dd *= 100.0  # Convert to percentage
+                 max_dd_decimal = max(max_dd_decimal, dd)
+             max_dd = max_dd_decimal * 100.0  # percentage form (e.g., 15.2)
              
-             # Improved win rate calculation for Fractional Holding
+             # Calculate win rate percentage
              successful_trades = [t for t in self.trades if t["success"]]
              if not successful_trades:
-                 win_rate = 0.0
+                 win_rate = 0.0  # percentage form
              else:
-                 # For buys: count as win if portfolio value increased after the trade
-                 # For sells: count as win if profit >= -fee (i.e., broke even after fees)
                  profitable_trades = sum(
                      1 for t in successful_trades
                      if (t["type"] == "buy" and t["portfolio_value_after"] > t["portfolio_value_before"]) or
                         (t["type"] == "sell" and t["profit"] >= -t["fee"])
                  )
-                 win_rate = (profitable_trades / len(successful_trades)) * 100.0  # Convert to percentage
+                 win_rate_decimal = profitable_trades / len(successful_trades)  # decimal form (e.g., 0.345)
+                 win_rate = win_rate_decimal * 100.0  # percentage form (e.g., 34.5)
              
-             # Count actual executed trades (excluding too small/failed)
              total_trades = len(successful_trades)
              
              return {
-                 'total_return': total_return,          # In percentage (e.g., -8.64)
-                 'sharpe_ratio': sharpe_ratio,
-                 'sortino_ratio': sortino_ratio,
-                 'max_drawdown': max_dd,                # In percentage (positive)
-                 'total_trades': total_trades,          # Successful trades only
-                 'win_rate': win_rate,                  # In percentage (e.g., 34.5)
-                 'final_balance': self.cash,
-                 'final_portfolio_value': values[-1],
-                 'successful_trades': total_trades,
-                 'total_trade_attempts': len(self.trades)
+                 'total_return': total_return,          # percentage (e.g., -8.64 means -8.64%)
+                 'sharpe_ratio': sharpe_ratio,          # ratio (not percentage)
+                 'sortino_ratio': sortino_ratio,        # ratio (not percentage)
+                 'max_drawdown': max_dd,                # percentage (e.g., 15.2 means 15.2%)
+                 'total_trades': total_trades,          # count
+                 'win_rate': win_rate,                  # percentage (e.g., 34.5 means 34.5%)
+                 'final_balance': self.cash,            # absolute value
+                 'final_portfolio_value': values[-1],   # absolute value
+                 'successful_trades': total_trades,     # count
+                 'total_trade_attempts': len(self.trades)  # count
              }
          except Exception as e:
```

---

_****_

```diff
      Args:
          metrics: Dictionary containing:
-             - total_return: Total return in percentage (e.g., 35.2 means 35.2%)
-             - sharpe_ratio: Annualized Sharpe ratio
-             - sortino_ratio: Annualized Sortino ratio
-             - max_drawdown: Maximum drawdown in percentage (positive)
-             - total_trades: Number of successful trades
-             - win_rate: Win rate in percentage (e.g., 35.2 means 35.2%)
-             - successful_trades: Number of successfully executed trades
-             - total_trade_attempts: Total number of trade attempts
+             - total_return: Total return in percentage (e.g., -8.64 means -8.64%)
+             - sharpe_ratio: Annualized Sharpe ratio (not percentage)
+             - sortino_ratio: Annualized Sortino ratio (not percentage)
+             - max_drawdown: Maximum drawdown in percentage (e.g., 15.2 means 15.2%)
+             - total_trades: Number of successful trades (count)
+             - win_rate: Win rate in percentage (e.g., 34.5 means 34.5%)
+             - successful_trades: Number of successfully executed trades (count)
+             - total_trade_attempts: Total number of trade attempts (count)
+             
+     Note:
+         Percentage values are already in percentage form (e.g., 34.5 means 34.5%).
+         We just need to append the '%' symbol for display.
      """
      try:
              st.metric(
                  "Total Return",
-                 f"{metrics.get('total_return', 0):.1f}%",
+                 f"{metrics.get('total_return', 0):.1f}%",  # Already in percentage form
                  delta_color="normal"
              )
              st.metric(
                  "Win Rate",
-                 f"{metrics.get('win_rate', 0):.1f}%"  # Already in 0-100 range
+                 f"{metrics.get('win_rate', 0):.1f}%"  # Already in percentage form
              )
  
          with col2:
              st.metric(
                  "Max Drawdown",
-                 f"{metrics.get('max_drawdown', 0):.1f}%"  # Already in 0-100 range
+                 f"{metrics.get('max_drawdown', 0):.1f}%"  # Already in percentage form
              )
              st.metric(
                  "Sharpe Ratio",
-                 f"{metrics.get('sharpe_ratio', 0):.2f}"
+                 f"{metrics.get('sharpe_ratio', 0):.2f}"  # Not a percentage
              )
  
          with col3:
              st.metric(
                  "Total Trades",
-                 f"{metrics.get('successful_trades', 0)} / {metrics.get('total_trade_attempts', 0)}"
+                 f"{metrics.get('successful_trades', 0)} / {metrics.get('total_trade_attempts', 0)}"  # Counts
              )
              st.metric(
                  "Sortino Ratio",
-                 f"{metrics.get('sortino_ratio', 0):.2f}"
+                 f"{metrics.get('sortino_ratio', 0):.2f}"  # Not a percentage
              )
  
```

---

_****_



주요 개선사항을 설명드리겠습니다:

1. **백엔드 (`_calculate_metrics()`)에서:**
   - 모든 퍼센트 계산을 명확하게 2단계로 분리
     ```python
     total_return_decimal = (values[-1] / values[0]) - 1  # decimal form (e.g., -0.0864)
     total_return = total_return_decimal * 100.0  # percentage form (e.g., -8.64)
     ```
   - 각 변수의 형식을 명확하게 주석으로 표시
   - 반환값의 형식도 명확하게 문서화

2. **UI (`display_trading_metrics()`)에서:**
   - 입력 메트릭의 형식을 명확하게 문서화
   - 각 메트릭의 성격 명시 (percentage vs ratio vs count)
   - 퍼센트 처리에 대한 명확한 주석 추가
     ```python
     # Already in percentage form (e.g., 34.5 means 34.5%)
     # We just need to append the '%' symbol for display
     ```

이제 백엔드와 UI 사이의 퍼센트 처리가 명확해졌습니다:
- 백엔드에서 모든 퍼센트 값을 0-100 범위로 변환 (예: 34.5는 34.5%를 의미)
- UI에서는 단순히 % 기호만 붙여서 표시
- 비율(ratio)과 개수(count)는 그대로 표시

이렇게 하면 중복 곱하기나 잘못된 표시 문제를 방지할 수 있습니다.

---

_**User**_

맨 위에 trading metrics가 조금 더 넓은 칸을 가졌으면 좋겠어... 지금은 가로로 따지면 반만 쓰고 있는 것 같은데, 어차피 옆에 아무것도 없어서. 나중에 metrics 추가할 수도 있으니까.

---

_**Assistant**_

네, `Backtest_Results.py`에서 trading metrics를 표시하는 부분의 레이아웃을 수정하겠습니다. 현재는 `col1, col2 = st.columns(2)`로 화면을 반으로 나누고 있는데, trading metrics가 전체 너비를 사용하도록 변경하겠습니다.

---

_****_

```diff
              
              # Display results
-             col1, col2 = st.columns(2)
-             
-             with col1:
-                 st.subheader("Trading Metrics")
-                 display_trading_metrics(results.get("metrics", {}))
-             
-             with col2:
-                 # Display scenario-specific metrics if scenario is active
-                 if scenario_type != "none":
+             st.subheader("Trading Metrics")
+             if scenario_type != "none":
+                 # If scenario metrics exist, use two columns
+                 col1, col2 = st.columns(2)
+                 with col1:
+                     display_trading_metrics(results.get("metrics", {}))
+                 with col2:
                      display_scenario_metrics(scenario_type, results.get("scenario_metrics", {}))
+             else:
+                 # If no scenario metrics, use full width for trading metrics
+                 display_trading_metrics(results.get("metrics", {}))
              
              # Display portfolio value time series
```

---

_****_



변경 사항을 설명드리겠습니다:

1. **레이아웃 조건부 처리**
   - 시나리오가 없을 때 (`scenario_type == "none"`):
     ```python
     st.subheader("Trading Metrics")
     display_trading_metrics(results.get("metrics", {}))  # 전체 너비 사용
     ```
   - 시나리오가 있을 때:
     ```python
     st.subheader("Trading Metrics")
     col1, col2 = st.columns(2)  # 화면을 반으로 나눔
     with col1:
         display_trading_metrics(results.get("metrics", {}))
     with col2:
         display_scenario_metrics(scenario_type, results.get("scenario_metrics", {}))
     ```

2. **개선된 점**
   - 시나리오가 없을 때는 trading metrics가 전체 화면 너비를 사용
   - 시나리오가 있을 때만 2열 레이아웃 사용
   - 향후 metrics 추가 시 더 넓은 공간 활용 가능

이렇게 하면 trading metrics가 더 넓은 공간을 활용할 수 있고, 시나리오 유무에 따라 레이아웃이 자동으로 조정됩니다.

---

_**User**_

우선 **지금 표에서 “profit” 컬럼에 표시되는 값이 실제 코드상의 `trade["profit"]`과 동일한 값인지**부터 다시 확인하셔야 합니다.  
로그를 보면, 예:  
```
2025-01-25 12:00:00 default sell -0.018376...  $104,753.07  $3.38  $-10.91  $9,989.09 ...
```
이 행을 “타임스탬프, 심볼, 타입, amount, price, profit, cumulative_pnl, ...” 순으로 읽으면,  
- **여섯 번째 칼럼**이 **$3.38** → 현재는 이것이 “profit” 컬럼이라고 UI에서 표시 중  
- **일곱 번째 칼럼**이 **$-10.91** → `cumulative_pnl`  
- 그럼 “$3.38면 분명 이익 아닌가?” 하겠지만,  
실제 `BaseBacktester.execute_trade()`를 보면,
```python
trade = {
    "timestamp": ...,
    "symbol": ...,
    "amount": 0.0,
    "price": 0.0,
    "fee": 0.0,
    "cost": 0.0,
    "revenue": 0.0,
    "profit": 0.0,
    "success": False,
    ...
}
```
**즉** 딕셔너리에는 `fee`가 `profit`보다 먼저 들어가 있고, 또 `cost`, `revenue`, `profit` 순서대로 들어 있습니다.  

정리하면, **테이블에 나타나는 5번째/6번째/7번째 열**이 진짜로 `profit`인지, `fee`인지, `cost`인지가 뒤섞일 가능성이 매우 큽니다.

---

## 1. 컬럼 순서가 달라진 이유

- 파이썬 딕셔너리는 삽입 순서를 기억하기 때문에, `trade`를 DataFrame으로 만들면 **키를 추가한 순서**대로 열이 만들어집니다.  
- 게다가 Streamlit에서 `display_columns = [...]`로 원하는 순서를 지정하긴 했지만, 그 순서가 **실제 code에서 딕셔너리 키를 추가한 순서**와 달라질 수 있습니다.

현재 `display_columns`를 보시면:

```python
display_columns = [
    "timestamp", "symbol", "type", "amount", "price",
    "profit", "cumulative_pnl", "portfolio_value_after",
    "position_units", "position_value", "cash_after",
    "fee", "success", "reason"
]
```

하지만 `execute_trade()`에서는 아래 순으로 키를 생성합니다:

1. `"timestamp"`  
2. `"symbol"`  
3. `"amount"`  
4. `"price"`  
5. `"fee"`  
6. `"cost"`  
7. `"revenue"`  
8. `"profit"`  
9. `"success"`  
10. `"reason"`  
11. `"action"`  
12. `"type"`  
13. `"value"`  
14. `"portfolio_value_before"`  
15. `"portfolio_value_after"`  
16. `"cumulative_pnl"`  
17. `"cash_after"`  
18. `"position_units"`  
19. `"position_value"`  

이 차이점 때문에, **테이블의 5번째~7번째 열이** 실제로는 전혀 다른 키(`fee`, `cost`, `revenue` 등)를 가리킬 수 있습니다.

---

## 2. 어떻게 교정해야 할까?

### (A) 컬럼 순서를 확실히 맞추기

가장 깔끔한 방법은, **정확히 어떤 순서로 어떤 키를 표시할지를** 명시적으로 정한 뒤, 그 순서대로 DataFrame을 만들면 됩니다.  
예: 

```python
# trades_df = pd.DataFrame(trades) ...

# 우리가 실제로 보고 싶은 순서 (예시):
correct_order = [
  "timestamp", 
  "symbol", 
  "type", 
  "amount",
  "price",
  "fee",          # 5) 명시적으로 fee를 여기에 둠
  "cost",         # 6) cost 
  "revenue",      # 7) revenue
  "profit",       # 8) profit
  "cumulative_pnl",
  "portfolio_value_after",
  "position_units",
  "position_value",
  "cash_after",
  "success",
  "reason",
]
# columns 중 실제로 존재하는 키만 걸러서 사용
correct_order = [c for c in correct_order if c in trades_df.columns]

trades_df = trades_df[correct_order]
```

이렇게 하면,  
1) 5번째 칼럼은 무조건 `fee`  
2) 8번째 칼럼은 무조건 `profit`  
등으로 확실해집니다.

> **주의**  
> “table에서 5번째 열이 ‘profit’처럼 보이는데 사실 `fee`다” 같은 혼동을 없애려면 꼭 이렇게 수동으로 **정렬**해야 합니다.

### (B) 표에 찍히는 값이 정말 `profit`인지 재확인

표로 만든 뒤, “$3.38”이 **어느 컬럼**에 들어가 있는지 체크해보세요.  
- 만약 `fee` 칼럼에 “$3.38”이 뜬다면, 그건 수수료로 해석됨 → win rate 계산에 반영되지 않음.  
- 만약 `profit` 칼럼에 “$3.38”이 떴다면, code 상 **trade["profit"]** = 3.38인지 확인해서, “정말 code에서 realized_profit = 3.38이냐?”를 `logger.info()`로 찍어볼 수도 있습니다.

---

## 3. “profit이 양수로 나오는데도 win rate=0인” 이유

(1) 혹은 **실제로**는 `profit = 3.38`이 아니라, “`fee = 3.38`, `profit`은 -10.91” 인 식으로 code가 기록했을 수 있음.  
   - 그럼 Win Rate 로직(`profit >= -fee`)에서 **실제로 -10.91 >= -3.38 ? => 아니므로** 미실현 or 손실이 됨.  

(2) Win Rate 계산 조건:

```python
if (t["type"] == "buy" and t["portfolio_value_after"] > t["portfolio_value_before"])
    or (t["type"] == "sell" and t["profit"] >= -t["fee"])
```
- SELL인 경우 → `t["profit"] >= -t["fee"]` → 예) profit= -10.91 vs fee=3.38 → -10.91 >= -3.38? => false  
- 즉 손실로 계산

즉, 표에서 `$3.38`이 “profit”이라고 믿고 있으나, 실제 code는 “profit=-10.91, fee=3.38”으로 기록했을 가능성이 높습니다.

---

## 4. 표기법 바꾸는 법

만약 “매도 시 ‘profit’ = revenue - cost - fee” 구조가 헷갈린다면,  
- 한눈에 수수료, 원가, 실현 이익을 구분해서 보이도록 **각 항목에 레이블**을 붙이는 방식을 쓰거나,  
- **BUY 시 profit=0** 대신 “미실현 profit”을 표시하지 않고, SELL 시만 profit을 남기는 등등, user가 혼동하지 않도록 정리해야 합니다.

또한 Win Rate를 “(매도 시 profit > 0) vs (매수 후 포트폴리오 가치가 증가)” 등으로 계산하는 것이 Fractional Holding 환경에 부합하는지 재검토해볼 수도 있습니다.

---

## 5. 정리

1. **현재 테이블 열 순서**가 코드의 딕셔너리 순서와 달라서, “$3.38가 profit이 맞느냐”가 뒤섞였을 확률이 큽니다.
2. Win Rate가 0인 것은, 실제 `trade["profit"]`가 음수/0이거나, Win Rate 조건을 만족 못 해서입니다.  
3. **해결**:  
   - (A) **컬럼 순서를 명시적으로 지정**하여, `fee, cost, revenue, profit` 순을 정확히 보고,  
   - (B) 정말 `profit`이 +인 기록이 있는지 확인.  
   - (C) Win Rate 계산 조건도 잘 맞는지(“fee=3.38, profit=-10.91”이면 수수료보다 큰 손실이므로 fail) 체크.  

이렇게 하면 “표에선 profit이 +3.38로 보이는데 code는 왜 Win Rate=0이냐” 같은 문제를 해결할 수 있을 것입니다.  

---

_**Assistant**_

네, 매우 상세한 분석 감사합니다. 테이블의 컬럼 순서와 실제 값 표시에 대한 문제를 해결하겠습니다.

---

_****_

```diff
                  trades_df = pd.DataFrame(trades)
                  
-                 # Define column order and formatting
+                 # Define exact column order to match the trade dictionary structure
                  display_columns = [
-                     "timestamp", "symbol", "type", "amount", "price",
-                     "profit", "cumulative_pnl", "portfolio_value_after",
-                     "position_units", "position_value", "cash_after",
-                     "fee", "success", "reason"
+                     "timestamp",          # Basic trade info
+                     "symbol",
+                     "type",              # buy/sell
+                     "amount",
+                     "price",
+                     
+                     # Transaction details (in order of calculation)
+                     "fee",               # Transaction fee
+                     "cost",              # For sells: cost basis portion
+                     "revenue",           # For sells: revenue from sale
+                     "profit",            # For sells: revenue - cost - fee
+                     
+                     # Portfolio impact
+                     "portfolio_value_before",
+                     "portfolio_value_after",
+                     "cumulative_pnl",
+                     
+                     # Position details
+                     "position_units",
+                     "position_value",
+                     "cash_after",
+                     
+                     # Status
+                     "success",
+                     "reason"
                  ]
                  
                  trades_df = trades_df[display_columns]
                  
-                 # Format numeric columns
-                 if "price" in trades_df.columns:
-                     trades_df["price"] = trades_df["price"].map("${:,.2f}".format)
-                 if "profit" in trades_df.columns:
-                     trades_df["profit"] = trades_df["profit"].map("${:,.2f}".format)
-                 if "cumulative_pnl" in trades_df.columns:
-                     trades_df["cumulative_pnl"] = trades_df["cumulative_pnl"].map("${:,.2f}".format)
-                 if "portfolio_value_after" in trades_df.columns:
-                     trades_df["portfolio_value_after"] = trades_df["portfolio_value_after"].map("${:,.2f}".format)
-                 if "position_value" in trades_df.columns:
-                     trades_df["position_value"] = trades_df["position_value"].map("${:,.2f}".format)
-                 if "cash_after" in trades_df.columns:
-                     trades_df["cash_after"] = trades_df["cash_after"].map("${:,.2f}".format)
-                 if "fee" in trades_df.columns:
-                     trades_df["fee"] = trades_df["fee"].map("${:,.4f}".format)
-                 if "position_units" in trades_df.columns:
-                     trades_df["position_units"] = trades_df["position_units"].map("{:,.6f}".format)
+                 # Add debug logging for profit calculation verification
+                 for trade in trades:
+                     if trade["type"] == "sell" and trade.get("profit", 0) > 0:
+                         logger.info(
+                             "Profitable sell trade found: fee=%.2f, cost=%.2f, revenue=%.2f, profit=%.2f",
+                             trade.get("fee", 0),
+                             trade.get("cost", 0),
+                             trade.get("revenue", 0),
+                             trade.get("profit", 0)
+                         )
+                 
+                 # Format numeric columns with clear labels
+                 format_cols = {
+                     "price": lambda x: f"${x:,.2f}",
+                     "fee": lambda x: f"Fee: ${x:,.2f}",
+                     "cost": lambda x: f"Cost: ${x:,.2f}",
+                     "revenue": lambda x: f"Rev: ${x:,.2f}",
+                     "profit": lambda x: f"P/L: ${x:,.2f}",
+                     "cumulative_pnl": lambda x: f"Cum.P/L: ${x:,.2f}",
+                     "portfolio_value_before": lambda x: f"${x:,.2f}",
+                     "portfolio_value_after": lambda x: f"${x:,.2f}",
+                     "position_value": lambda x: f"${x:,.2f}",
+                     "cash_after": lambda x: f"${x:,.2f}",
+                     "position_units": lambda x: f"{x:,.6f}"
+                 }
+                 
+                 # Apply formatting only to columns that exist
+                 for col, format_func in format_cols.items():
+                     if col in trades_df.columns:
+                         trades_df[col] = trades_df[col].map(format_func)
                  
                  # Display with Streamlit's column configuration
                              width="small"
                          ),
+                         "fee": st.column_config.TextColumn(
+                             "Fee",
+                             width="medium",
+                             help="Transaction fee"
+                         ),
+                         "cost": st.column_config.TextColumn(
+                             "Cost Basis",
+                             width="medium",
+                             help="Portion of original purchase cost"
+                         ),
+                         "revenue": st.column_config.TextColumn(
+                             "Revenue",
+                             width="medium",
+                             help="Amount received from sale"
+                         ),
+                         "profit": st.column_config.TextColumn(
+                             "Profit/Loss",
+                             width="medium",
+                             help="Revenue - Cost - Fee"
+                         ),
                          "success": st.column_config.CheckboxColumn(
                              "Success",
                      use_container_width=True
                  )
-                 logger.info("Displayed %d trades", len(trades))
+                 logger.info("Displayed %d trades with corrected column order", len(trades))
              else:
                  st.info("No trades to display")
```

---

_****_



주요 개선사항을 설명드리겠습니다:

1. **컬럼 순서 명확화**
   ```python
   display_columns = [
       # Basic trade info
       "timestamp", "symbol", "type", "amount", "price",
       
       # Transaction details (in order of calculation)
       "fee",               # Transaction fee
       "cost",              # For sells: cost basis portion
       "revenue",           # For sells: revenue from sale
       "profit",            # For sells: revenue - cost - fee
       ...
   ]
   ```

2. **값 표시 개선**
   ```python
   format_cols = {
       "fee": lambda x: f"Fee: ${x:,.2f}",
       "cost": lambda x: f"Cost: ${x:,.2f}",
       "revenue": lambda x: f"Rev: ${x:,.2f}",
       "profit": lambda x: f"P/L: ${x:,.2f}",
       ...
   }
   ```

3. **컬럼 설명 추가**
   ```python
   column_config={
       "fee": st.column_config.TextColumn(
           "Fee",
           width="medium",
           help="Transaction fee"
       ),
       "profit": st.column_config.TextColumn(
           "Profit/Loss",
           width="medium",
           help="Revenue - Cost - Fee"
       ),
       ...
   }
   ```

4. **디버그 로깅 추가**
   ```python
   # 이익이 난 매도 거래 검증
   for trade in trades:
       if trade["type"] == "sell" and trade.get("profit", 0) > 0:
           logger.info(
               "Profitable sell trade found: fee=%.2f, cost=%.2f, revenue=%.2f, profit=%.2f",
               trade.get("fee", 0),
               trade.get("cost", 0),
               trade.get("revenue", 0),
               trade.get("profit", 0)
           )
   ```

이러한 변경으로:
1. 컬럼이 정확한 순서로 표시됨
2. 각 값에 레이블이 붙어서 혼동 방지 (예: "Fee: $3.38", "P/L: $-10.91")
3. 컬럼에 마우스를 올리면 설명이 표시됨
4. 이익이 난 매도 거래가 있으면 로그에 상세 내역이 기록됨

이제 Win Rate가 0%로 나오는 이유를 더 쉽게 파악할 수 있을 것입니다.

---



---

