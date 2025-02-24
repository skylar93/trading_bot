## SpecStory

## Backtester Trade Metrics Test Implementation (1/31/2025, 9:34:47 PM)

_**User**_

Below is an example test function you can add to your test_base_backtester.py that specifically checks (a) whether each trade has cost, revenue, and profit recorded, and (b) whether _calculate_metrics() produces non‐zero values (like win rate) when there are both winning and losing trades.

	Note: This test assumes you have already modified your BaseBacktester (especially execute_trade) to compute and store trade["cost"], trade["revenue"], and trade["profit"]. If that logic is not implemented, the test will fail.

def test_metrics_calculation():
    """
    Test that the backtester calculates cost/revenue/profit for each trade,
    and that overall metrics (win rate, final balance, etc.) are non-zero
    when there are winning and losing trades.
    """
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,  # 0.1% fee
        max_position=1.0
    )

    # We'll manually create a scenario of a few trades to ensure
    # some are profitable and some are losing.

    # Day 1: Buy half position at price=100
    t1 = pd.Timestamp("2025-01-01")
    backtester.update(
        timestamp=t1,
        prices={"default": 100.0},
        actions={"default": 0.5}  # 50% position
    )

    # Day 2: Price rises to 120, buy more (making cost higher),
    # so that partial sells later can realize different PnLs
    t2 = pd.Timestamp("2025-01-02")
    backtester.update(
        timestamp=t2,
        prices={"default": 120.0},
        actions={"default": 0.3}  # further +30% => bigger position
    )

    # Day 3: Price at 110, let's do a partial sell (some profit realized, but price is down from 120)
    t3 = pd.Timestamp("2025-01-03")
    backtester.update(
        timestamp=t3,
        prices={"default": 110.0},
        actions={"default": -0.2}  # reduce position by 20% => partial close
    )

    # Day 4: Price rises to 130, sell everything => big profit
    t4 = pd.Timestamp("2025-01-04")
    backtester.update(
        timestamp=t4,
        prices={"default": 130.0},
        actions={"default": 0.0}  # fully close position
    )

    # Day 5: Price is 90, intentionally buy 80% => big position
    # Next day price falls => losing trade
    t5 = pd.Timestamp("2025-01-05")
    backtester.update(
        timestamp=t5,
        prices={"default": 90.0},
        actions={"default": 0.8}  # 80% position
    )

    # Day 6: Price drops to 70 => sell all => losing trade
    t6 = pd.Timestamp("2025-01-06")
    backtester.update(
        timestamp=t6,
        prices={"default": 70.0},
        actions={"default": 0.0}  # close (loss)
    )

    # Now we expect several trades:
    # - Some partial closes with positive profit
    # - One bigger losing trade

    # 1) Check each recorded trade for cost/revenue/profit
    for i, trade in enumerate(backtester.trades, start=1):
        assert "cost" in trade, f"Trade #{i} missing 'cost'"
        assert "revenue" in trade, f"Trade #{i} missing 'revenue'"
        assert "profit" in trade, f"Trade #{i} missing 'profit'"

    # 2) Check the final metrics
    metrics = backtester._calculate_metrics()

    # We expect:
    # - total_trades > 0
    # - some trades were winning, some losing => non-zero 'win_rate' if at least one winner
    # - final_balance different from initial
    assert metrics["total_trades"] > 0, "No trades recorded"
    assert metrics["win_rate"] > 0, "Expected some winning trades => win_rate should be > 0"
    # We also expect the final balance could be > or < initial depending on net PnL
    assert metrics["final_balance"] != 10000.0, "Final balance unchanged => no realized PnL?"

    # Optionally check that final_portfolio_value is also not the same as initial
    assert metrics["final_portfolio_value"] != 10000.0, "No net change in portfolio value?"

    # 3) If you plan to track Sharpe/Sortino, we expect them possibly non-zero
    # (But if the data is short or if there's not enough variation, they might still be near 0.)
    # So either do a looser check or skip if unimportant:
    assert metrics["sharpe_ratio"] != 0, "Sharpe ratio unexpectedly zero"
    # etc.

해설
	1.	Trades scenario
	•	We create 6 consecutive days (t1~t6) with known price changes.
	•	Some trades will be partial closes at a profit, some at a loss.
	•	This ensures we end up with multiple trades, some positive profit, some negative.
	2.	Cost/Revenue/Profit Checks
	•	In the loop for trade in backtester.trades:, we assert each trade has 'cost', 'revenue', and 'profit'.
	•	If your code does not fill these keys, the test fails, prompting you to implement them.
	3.	Metric Checks
	•	We call metrics = backtester._calculate_metrics() at the end.
	•	We check that total_trades > 0, win_rate > 0 (there should be at least one winning trade).
	•	We also check final_balance is not the same as the initial capital.
	•	We do a mild check for sharpe_ratio != 0, just as a sentinel that variance in returns was non‐trivial.

	This approach ensures that once you add real PnL calculation in your BaseBacktester and store it in each trade dict, the test will confirm that the metrics reflect that real PnL.

Next Steps
	•	Implement the partial‐close logic (e.g. average entry price, realized PnL) in execute_trade.
	•	Make sure each trade has trade["cost"], trade["revenue"], trade["profit"].
	•	Possibly store trade["unrealized_pnl"] if you want more advanced metrics.
	•	Then run pytest to confirm that test_metrics_calculation passes (along with your other tests).

This should give you a complete “end-to-end” test verifying that (a) you do store cost/revenue/profit in each trade and (b) your _calculate_metrics() has non-zero results when there are real winning & losing trades.

일단 이 테스트를 똑바로 쓰는 것까지만 해 줘.

---

_**Assistant**_

Let me help you add this test function to your test file. First, I'll check the current content of the test file.

---

_**Assistant**_



---

_**Assistant**_



Now I'll add the new test function to the file. I'll make sure to follow the existing test style and include all necessary imports.

---

_**Assistant**_

```
// ... existing code ...

def test_metrics_calculation():
    """
    Test that the backtester calculates cost/revenue/profit for each trade,
    and that overall metrics (win rate, final balance, etc.) are non-zero
    when there are winning and losing trades.
    """
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,  # 0.1% fee
        max_position=1.0
    )

    # Day 1: Buy half position at price=100
    t1 = pd.Timestamp("2025-01-01")
    backtester.update(
        timestamp=t1,
        prices={"default": 100.0},
        actions={"default": 0.5}  # 50% position
    )

    # Day 2: Price rises to 120, buy more (making cost higher),
    # so that partial sells later can realize different PnLs
    t2 = pd.Timestamp("2025-01-02")
    backtester.update(
        timestamp=t2,
        prices={"default": 120.0},
        actions={"default": 0.3}  # further +30% => bigger position
    )

    # Day 3: Price at 110, let's do a partial sell (some profit realized, but price is down from 120)
    t3 = pd.Timestamp("2025-01-03")
    backtester.update(
        timestamp=t3,
        prices={"default": 110.0},
        actions={"default": -0.2}  # reduce position by 20% => partial close
    )

    # Day 4: Price rises to 130, sell everything => big profit
    t4 = pd.Timestamp("2025-01-04")
    backtester.update(
        timestamp=t4,
        prices={"default": 130.0},
        actions={"default": 0.0}  # fully close position
    )

    # Day 5: Price is 90, intentionally buy 80% => big position
    # Next day price falls => losing trade
    t5 = pd.Timestamp("2025-01-05")
    backtester.update(
        timestamp=t5,
        prices={"default": 90.0},
        actions={"default": 0.8}  # 80% position
    )

    # Day 6: Price drops to 70 => sell all => losing trade
    t6 = pd.Timestamp("2025-01-06")
    backtester.update(
        timestamp=t6,
        prices={"default": 70.0},
        actions={"default": 0.0}  # close (loss)
    )

    # 1) Check each recorded trade for cost/revenue/profit
    for i, trade in enumerate(backtester.trades, start=1):
        assert "cost" in trade, f"Trade #{i} missing 'cost'"
        assert "revenue" in trade, f"Trade #{i} missing 'revenue'"
        assert "profit" in trade, f"Trade #{i} missing 'profit'"

    # 2) Check the final metrics
    metrics = backtester._calculate_metrics()

    # We expect:
    # - total_trades > 0
    # - some trades were winning, some losing => non-zero 'win_rate' if at least one winner
    # - final_balance different from initial
    assert metrics["total_trades"] > 0, "No trades recorded"
    assert metrics["win_rate"] > 0, "Expected some winning trades => win_rate should be > 0"
    # We also expect the final balance could be > or < initial depending on net PnL
    assert metrics["final_balance"] != 10000.0, "Final balance unchanged => no realized PnL?"

    # Optionally check that final_portfolio_value is also not the same as initial
    assert metrics["final_portfolio_value"] != 10000.0, "No net change in portfolio value?"

    # 3) Check Sharpe ratio is non-zero (we have both winning and losing trades)
    assert metrics["sharpe_ratio"] != 0, "Sharpe ratio unexpectedly zero"
```

---

_**Assistant**_



I've successfully added the new test function `test_metrics_calculation()` to your test file. The test follows the same style as your existing tests and maintains consistency with your codebase. Here's what the test does:

1. Creates a backtester instance with initial capital of 10,000 and 0.1% trading fee
2. Executes a sequence of 6 trades over different days with varying prices to ensure both winning and losing trades
3. Verifies that each trade record contains cost, revenue, and profit information
4. Checks that the metrics calculation produces meaningful non-zero values for key metrics

The test is designed to fail if:
- Trades don't have cost/revenue/profit information
- No trades are recorded
- Win rate is 0 (which shouldn't happen with our mix of winning/losing trades)
- Final balance equals initial balance (which would indicate no PnL was realized)
- Portfolio value hasn't changed
- Sharpe ratio is exactly 0 (which shouldn't happen with our mix of returns)

Note that for this test to pass, you'll need to ensure your `BaseBacktester` class:
1. Records cost, revenue, and profit for each trade
2. Properly handles partial position closes
3. Correctly calculates metrics including win rate and Sharpe ratio

Would you like me to check anything specific about the test implementation or help with implementing the required functionality in the `BaseBacktester` class?

---

_**User**_

아래는 Backtester에 “수익/손실(PnL) 계산” 및 “매트릭(metric) 계산”을 추가하는 과정을 단계별로 요약한 예시입니다. 이미 BaseBacktester에 execute_trade와 _calculate_metrics가 존재하므로, (1) 트레이드마다 cost/revenue/profit을 어떻게 기록할지, (2) 전체 매트릭에서 이 profit 값을 어떻게 반영할지를 중심으로 살펴봅니다.

1. 각 트레이드에서 cost/revenue/profit 계산하기

(1) 개념적 개요
	•	cost: 체결된 포지션을 매수할 때 쓰인 금액 (또는 매수 포지션 확장 시 추가로 들어간 금액)
	•	revenue: 체결된 포지션을 매도(감축)할 때 회수된 금액
	•	profit: 부분 청산(감축) 시점에서의 realized PnL, 즉 revenue - cost (또는 수수료(fee)를 감안하면 revenue - cost - total_fees)
	•	여러 번에 걸쳐 매수와 매도가 이루어질 경우, 가중평균 단가(average price)로 코스트를 추적할 수 있음.
	•	이때 “부분 청산” 시점에서, “상대적으로 몇 개를 얼마에 샀었는지”를 파악한 후, 그 비율만큼 cost를 배분하여, 그 차이(= profit)를 기록.

(2) 구현 흐름
	1.	포지션별 Tracking 구조
	•	예: positions["default"]에 단순히 “현재 몇 개(코인 or 주식 등) 보유 중인지”만 저장해둠 → 이외에 “평단가(average_price)”나 “누적 cost”도 필요할 수 있음.
	•	예로, position_info = { "units": float, "avg_price": float, "cost_basis": float } 형태로 관리.
	•	만약 현재는 “단순 백테스터” 수준이라, 부분 청산 시 profit을 크게 신경 안 쓰고 “전부 동일 평단가”로 계산해도 된다면, 아래 3번에서 “부분 청산 PnL 계산 로직” 간단히 구현.
	2.	execute_trade 안에서 cost/revenue/profit 기록
	•	만약 trade_amount가 양수(매수)라면:
	•	실제 지불금액 = trade_value + fee
	•	cost: trade["cost"] = trade_value + fee
	•	revenue: trade["revenue"] = 0 (매수에서 수익은 없으므로)
	•	profit: trade["profit"] = 0 (매수 시점에 실현된 이익 없음)
	•	만약 trade_amount가 음수(매도)라면:
	•	revenue = abs(trade_amount) * current_price (또는 trade_value)
	•	cost = 이때 “평단가 × 청산한 units”만큼이 cost
	•	profit = revenue - cost - fee
	•	fee는 trade_fee를 더해주거나 빼주거나 (프로젝트 정책에 따라 수수료 반영이 달라질 수 있음)
	•	“음수 포지션” (공매도)까지 포함하려면 로직이 더 복잡해질 수 있으나, 단순 long만 가정하면 이 정도로 충분
	3.	부분 청산 시 PnL 배분
	•	매수 시:
	•	position_info["units"] += trade_amount
	•	“새로운 평균 단가” = (기존 평단가×기존 보유수 + 새로 매수한 금액) / (기존 보유수 + 새로 매수한 수량)
	•	“cost_basis”도 accordingly 갱신
	•	매도 시:
	•	“realized profit” = (매도 단가 - 기존 평균 단가) × 매도 수량 ( - fee )
	•	그리고 cost/revenue/profit을 trade dict에 기록.
	•	“cost_basis”도 매도분만큼 감소
	4.	trade dict에 cost/revenue/profit 추가

# 예: execute_trade 내, 매수 로직
trade["cost"] = trade_value + trade_fee
trade["revenue"] = 0.0
trade["profit"] = 0.0

# 예: execute_trade 내, 매도 로직 (부분 청산)
realized_cost = avg_price * abs(trade_amount)
trade["cost"] = realized_cost
trade["revenue"] = trade_value  # = abs(trade_amount)*current_price
trade["profit"] = trade_value - realized_cost - trade_fee

	•	이런 식으로 기록한 뒤, self.trades.append(trade)에 포함시키면 trade 기록에서 바로 보고 가능.

2. 최종 매트릭에 profit 반영하기

(1) profit 정보를 _calculate_metrics에서 활용
	•	현재 _calculate_metrics는 “revenue와 cost를 통해 어떠한 계산”을 하진 않고, 단순히 “포트폴리오 변동” 기반으로 Sharpe, Sortino, drawdown 등을 내고 있음.
	•	만약 “Profit Factor”(= 총이익 / 총손실)나 “avg_trade” 같은 항목을 표시하고 싶다면, 다음과 같은 계산이 필요합니다.

total_profit = 0.0  # 총 이익
total_loss = 0.0    # 총 손실
num_win_trades = 0
num_loss_trades = 0
sum_pnl = 0.0

for trade in self.trades:
    trade_pnl = trade.get("profit", 0.0)
    if trade_pnl > 0:
        total_profit += trade_pnl
        num_win_trades += 1
    elif trade_pnl < 0:
        total_loss += abs(trade_pnl)
        num_loss_trades += 1
    
    sum_pnl += trade_pnl

all_trades_count = len(self.trades)

profit_factor = total_profit / total_loss if total_loss > 1e-8 else 0
avg_trade = sum_pnl / all_trades_count if all_trades_count else 0
win_rate = 100.0 * num_win_trades / all_trades_count if all_trades_count else 0

	•	그런 뒤 _calculate_metrics 리턴 dict에 이 값들을 추가:

return {
    # ... 기존 값들 ...
    "profit_factor": profit_factor,
    "avg_trade": avg_trade,
    "win_rate": win_rate,
    # ...
}

(2) UI와 연동
	•	display_trading_metrics (metrics.py) 안에 st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
st.metric("Average Trade", f"${metrics.get('avg_trade', 0):.2f}")
이런 식으로 추가해주면 UI에 표시됨.

3. 통합 테스트 흐름
	1.	중간 단계
	•	execute_trade에서 cost/revenue/profit을 계산 & 저장 (또는 추가 함수 update_position_and_pnl(...) 만들어 호출).
	•	초기에는 간단히 “매도 시점에 (current_price - avg_price) × 수량 - fee”만 계산해도 됨.
	2.	테스트 코드
	•	아까 답변에서 예시로 제시한 test_metrics_calculation() 같은 “시나리오형 테스트” 작성:
	•	여러 번 매수/매도 → 몇 번은 이익, 몇 번은 손실
	•	마지막에 _calculate_metrics() 호출 → profit_factor, avg_trade, win_rate 등 검증
	3.	UI에서 확인
	•	streamlit UI에서 backtest를 돌리면, logs & final metrics에 profit이 제대로 나타남.

4. 순서 요약
	1.	(가장 먼저) execute_trade 수정
	•	포지션을 “units”와 “avg_price” 형태로 관리할지, 혹은 “cost_basis”를 직접 관리할지 결정.
	•	매수/매도 시점에서 trade["cost"], trade["revenue"], trade["profit"]를 계산하여 기록.
	•	주의: 부분 청산 시, cost basis를 일정 비율로만 감소시키는 로직이 필요.
	2.	그 다음 _calculate_metrics에 추가
	•	전체 self.trades를 순회 → profit 값을 더해 “win_rate, profit_factor, avg_trade” 등 계산.
	•	이미 있는 Sharpe/Sortino는 “포트폴리오 가치 변동”을 활용하고, 부분 PnL 통계치는 trade["profit"] 이용.
	3.	마지막
	•	test_base_backtester.py(혹은 다른 테스트)에서 “여러 승패가 뒤섞인 시나리오”를 작성.
	•	(예: 3~5번 트레이드 중 일부만 수익, 일부 손실)
	•	assert metrics["win_rate"] > 0 등으로 체크.

이렇게 단계별로 구현하면, 개별 트레이드 수익/손실(Trade-Level PnL)과 전체 매트릭(Portfolio-Level Metric)을 모두 검증할 수 있게 됩니다.

	Tip: 나중에 “멀티자산 / 공매도 / 레버리지” 같은 확장이 필요하다면, 지금의 “units & avg_price” 구조를 더 일반화하는 게 좋습니다(예: positions[symbol]을 dict로 두어 units, avg_price, cost_basis를 모두 관리).

---

_**Assistant**_

네, 이해했습니다. BaseBacktester를 수정하여 PnL 계산과 메트릭을 구현해보겠습니다. 먼저 현재 BaseBacktester의 구현을 확인해보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



BaseBacktester를 수정하여 PnL 계산과 메트릭을 구현하겠습니다. 먼저 포지션 정보를 확장하고, execute_trade 메서드를 수정하여 cost/revenue/profit을 계산하도록 하겠습니다.

---

_**Assistant**_

```
// ... existing code ...
    def reset(self):
        """
        Reset portfolio state to initial conditions.
        Clears all positions, trades, and history.
        
        For single-asset mode:
        - Initializes positions with {'default': {'units': 0.0, 'avg_price': 0.0, 'cost_basis': 0.0}}
        
        For multi-asset mode:
        - Initializes empty positions dictionary with the same structure per asset
        """
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}  # {symbol: {'units': float, 'avg_price': float, 'cost_basis': float}}
        self.trades: List[Dict] = []
        self.portfolio_history: List[float] = [self.initial_capital]
        self.cash_history: List[float] = [self.initial_capital]  # Track cash balance history
        self.peak_value = self.initial_capital  # For drawdown calculation
        self.current_timestamp = None  # Track current timestamp
        
        # Reset risk manager if exists
        if self.risk_manager:
            self.risk_manager.reset()
// ... existing code ...
```

---

_**Assistant**_



이제 execute_trade 메서드를 수정하여 cost/revenue/profit을 계산하고 기록하도록 하겠습니다.

---

_**Assistant**_

```
// ... existing code ...
    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        action: float,
        price_data: Dict[str, float],
        asset: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute a trade based on the given action and current price data.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            action (float): Desired position size (-1 to 1)
            price_data (Dict[str, float]): Current price data
            asset (str): Asset symbol (default: "default" for single-asset mode)
            
        Returns:
            Dict[str, Any]: Trade execution results including:
                - success (bool): Whether trade was executed
                - reason (str): If failed, reason for failure
                - type (str): 'buy' or 'sell'
                - amount (float): Trade size in units
                - price (float): Execution price
                - value (float): Trade value
                - fee (float): Trading fee
                - cost (float): Total cost basis for this trade
                - revenue (float): Revenue from this trade (for sells)
                - profit (float): Realized profit/loss from this trade
        """
        current_price = price_data.get(asset)
        if current_price is None:
            return {
                'success': False,
                'reason': 'price_not_available',
                'timestamp': timestamp,
                'asset': asset
            }
        
        # Current position info
        current_position = self.positions.get(asset, {'units': 0.0, 'avg_price': 0.0, 'cost_basis': 0.0})
        current_units = current_position['units']
        current_avg_price = current_position['avg_price']
        current_cost_basis = current_position['cost_basis']
        
        # For buys, check if we have enough cash
        if action > 0:
            portfolio_value = self.get_portfolio_value(price_data)
            max_trade_value = self.cash / (1 + self.trading_fee)
            max_position_value = max_trade_value * self.max_position
            target_position = (action * max_position_value) / current_price
        # For sells, calculate target position directly
        else:
            target_position = current_units * (1 + action)  # action is negative
        
        # Calculate trade size
        trade_amount = target_position - current_units
        trade_value = abs(trade_amount * current_price)
        trade_fee = trade_value * self.trading_fee
        
        # Skip tiny trades
        if abs(trade_amount) < 1e-6:
            return {
                'success': True,
                'reason': 'trade_size_too_small',
                'timestamp': timestamp,
                'asset': asset,
                'amount': 0,
                'fee': 0
            }
        
        # For buys, verify we have enough cash
        if trade_amount > 0 and trade_value + trade_fee > self.cash:
            return {
                'success': False,
                'reason': 'insufficient_funds',
                'timestamp': timestamp,
                'asset': asset,
                'amount': 0,
                'fee': 0
            }
        
        # Initialize trade record
        trade = {
            'timestamp': timestamp,
            'symbol': asset,
            'price': current_price,
            'amount': trade_amount,
            'value': trade_value,
            'fee': trade_fee,
            'success': True
        }
        
        # Handle buy vs sell logic
        if trade_amount > 0:  # Buy
            trade['type'] = 'buy'
            trade['cost'] = trade_value + trade_fee
            trade['revenue'] = 0.0
            trade['profit'] = 0.0
            
            # Update position info
            new_units = current_units + trade_amount
            new_cost = current_cost_basis + trade_value + trade_fee
            new_avg_price = new_cost / new_units if new_units > 0 else 0.0
            
            self.positions[asset] = {
                'units': new_units,
                'avg_price': new_avg_price,
                'cost_basis': new_cost
            }
            
        else:  # Sell
            trade['type'] = 'sell'
            
            # Calculate realized cost and profit
            sell_ratio = abs(trade_amount) / current_units
            realized_cost = current_cost_basis * sell_ratio
            realized_revenue = trade_value
            realized_profit = realized_revenue - realized_cost - trade_fee
            
            trade['cost'] = realized_cost
            trade['revenue'] = realized_revenue
            trade['profit'] = realized_profit
            
            # Update position info
            new_units = current_units + trade_amount
            new_cost = current_cost_basis * (1 - sell_ratio)
            new_avg_price = current_avg_price  # Keep same average price for remaining position
            
            if abs(new_units) < 1e-6:  # Position closed
                if asset in self.positions:
                    del self.positions[asset]
            else:
                self.positions[asset] = {
                    'units': new_units,
                    'avg_price': new_avg_price,
                    'cost_basis': new_cost
                }
        
        # Update cash
        self.cash -= (trade_value + trade_fee) if trade_amount > 0 else -(trade_value - trade_fee)
        
        # Record trade
        self.trades.append(trade)
        
        return trade
// ... existing code ...
```

---

_**Assistant**_







이제 _calculate_metrics 메서드를 수정하여 profit 정보를 활용한 메트릭을 계산하도록 하겠습니다.

---

_**Assistant**_

```
// ... existing code ...
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics based on portfolio history and trades.
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - total_trades: Total number of trades executed
                - win_rate: Percentage of profitable trades
                - profit_factor: Total profit / Total loss
                - avg_trade: Average profit per trade
                - sharpe_ratio: Risk-adjusted return metric
                - sortino_ratio: Downside risk-adjusted return
                - max_drawdown: Maximum peak to trough decline
                - final_balance: Final portfolio value
                - final_portfolio_value: Final total value including positions
                - total_fees: Total trading fees paid
                - total_profit: Sum of all realized profits
                - total_loss: Sum of all realized losses
        """
        if len(self.portfolio_history) < 2:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_balance': self.initial_capital,
                'final_portfolio_value': self.initial_capital,
                'total_fees': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0
            }
        
        # Calculate trade-based metrics
        total_profit = 0.0
        total_loss = 0.0
        total_fees = 0.0
        winning_trades = 0
        
        for trade in self.trades:
            profit = trade.get('profit', 0.0)
            fee = trade.get('fee', 0.0)
            
            total_fees += fee
            
            if profit > 0:
                total_profit += profit
                winning_trades += 1
            elif profit < 0:
                total_loss += abs(profit)
        
        total_trades = len(self.trades)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 1e-8 else 0.0
        avg_trade = (total_profit - total_loss) / total_trades if total_trades > 0 else 0.0
        
        # Calculate returns-based metrics
        returns = pd.Series(self.portfolio_history).pct_change().dropna()
        
        if len(returns) > 0:
            excess_returns = returns - 0.0  # Assuming 0% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0
            
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = abs(drawdowns.min()) * 100  # Convert to percentage
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'final_balance': self.cash,
            'final_portfolio_value': self.portfolio_history[-1],
            'total_fees': total_fees,
            'total_profit': total_profit,
            'total_loss': total_loss
        }
// ... existing code ...
```

---

_**Assistant**_



수정이 완료되었습니다. 주요 변경사항을 요약하면 다음과 같습니다:

1. 포지션 정보 확장
- 각 포지션을 `{'units': float, 'avg_price': float, 'cost_basis': float}` 형태의 딕셔너리로 관리
- `reset()` 메서드에서 포지션 초기화 로직 수정

2. `execute_trade` 메서드 개선
- 매수/매도 시 cost/revenue/profit 계산 및 기록
- 부분 청산 시 평균단가와 cost basis 관리
- 매수 시:
  ```python
  trade['cost'] = trade_value + trade_fee
  trade['revenue'] = 0.0
  trade['profit'] = 0.0
  ```
- 매도 시:
  ```python
  sell_ratio = abs(trade_amount) / current_units
  realized_cost = current_cost_basis * sell_ratio
  realized_revenue = trade_value
  realized_profit = realized_revenue - realized_cost - trade_fee
  ```

3. `_calculate_metrics` 메서드 개선
- 트레이드 기반 메트릭 추가:
  - total_profit: 모든 이익 거래의 합
  - total_loss: 모든 손실 거래의 합
  - win_rate: 승률
  - profit_factor: 이익/손실 비율
  - avg_trade: 평균 거래당 수익
- 수익률 기반 메트릭 개선:
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown

이제 테스트 코드를 실행하면 다음을 검증할 수 있습니다:
1. 각 거래에 cost/revenue/profit이 기록되는지
2. 부분 청산 시 profit이 올바르게 계산되는지
3. 최종 메트릭이 거래 이력을 정확히 반영하는지

추가로 필요한 작업이 있다면 말씀해 주세요.

---



---

