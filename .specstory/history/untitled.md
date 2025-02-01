## SpecStory

## Untitled (1/31/2025, 9:23:05 PM)

_**User**_

(venv) (base) MacBook-Air:trading_bot skylar$ streamlit run deployment/web_interface/app.py
이거 돌리니까 이렇게 뜨는데 저 0.00 부분들 좀 고쳐서 제대로 계산이 반영되게 해 줄 수 있니? 이건 backtest_results.py에 속한 걸 거야 (under the deployment) 

---

_**Assistant**_

I'll help you fix the metrics calculations in the backtest results page. Let me first check the relevant code.

---

_**Assistant**_



---

_**Assistant**_



Let me also check the backtest utility code to see how the metrics are calculated.

---

_**Assistant**_



---

_**Assistant**_



Let me also check how the metrics are displayed.

---

_**Assistant**_



---

_**Assistant**_



I see the issue. The metrics are being calculated correctly in the `BacktestManager` class, but there are a few problems in how they're being processed and displayed. Let me fix this by updating the relevant files.

First, I'll update the metrics display component to properly format the values:

---

_**Assistant**_

```
def display_trading_metrics(metrics: Dict):
    """Display trading performance metrics"""
    try:
        # Create three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0.0):.2f}"
            )
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0.0):.1f}%"
            )

        with col2:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0.0) * 100:.1f}%"  # Convert to percentage
            )
            st.metric(
                "Profit Factor",
                f"{metrics.get('profit_factor', 0.0):.2f}"
            )

        with col3:
            st.metric(
                "Total Trades",
                f"{metrics.get('total_trades', 0)}"
            )
            st.metric(
                "Average Trade",
                f"${metrics.get('avg_trade', 0.0):.2f}"
            )

    except Exception as e:
        logger.error(f"Error displaying trading metrics: {str(e)}", exc_info=True)
        st.error("Failed to display trading metrics")
// ... existing code ...
```

---

_**Assistant**_



Now, let me update the BacktestManager to ensure metrics are properly calculated:

---

_**Assistant**_

```
    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process backtest results and calculate metrics"""
        metrics = {}
        
        portfolio_values = results.get("portfolio_values", [])
        initial_balance = self.settings.get("initial_balance", 10000.0)
        
        if "trades" in results and results["trades"]:
            trades = results["trades"]
            pnls = [t.get("pnl", 0.0) for t in trades]
            
            # Calculate basic metrics
            total_pnl = sum(pnls)
            profitable_trades = sum(1 for pnl in pnls if pnl > 0)
            total_trades = len(trades)
            
            # Calculate profit factor
            winning_pnls = [pnl for pnl in pnls if pnl > 0]
            losing_pnls = [abs(pnl) for pnl in pnls if pnl < 0]
            total_profits = sum(winning_pnls) if winning_pnls else 0
            total_losses = sum(losing_pnls) if losing_pnls else 0
            profit_factor = total_profits / total_losses if total_losses > 0 else float('inf') if total_profits > 0 else 0.0
            
            # Calculate average trade
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0.0
            
            # Calculate win rate
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # Calculate returns for Sharpe ratio
            if len(portfolio_values) > 1:
                returns = [(v2 - v1) / v1 for v1, v2 in zip(portfolio_values[:-1], portfolio_values[1:])]
            else:
                returns = []
            
            metrics.update({
                "total_return": (portfolio_values[-1] - initial_balance) / initial_balance if portfolio_values else 0.0,
                "sharpe_ratio": self._calculate_sharpe_ratio(returns) if returns else 0.0,
                "max_drawdown": self._calculate_max_drawdown(portfolio_values),
                "win_rate": win_rate,
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "total_pnl": total_pnl,
                "profit_factor": min(profit_factor, 999.99),  # Cap profit factor for display
                "avg_trade": avg_trade
            })
        else:
            # Initialize with default values if no trades
            metrics.update({
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "profitable_trades": 0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0
            })
        
        return metrics
        
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns"""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_series = pd.Series(returns)
        std = returns_series.std()
        if std == 0:
            return 0.0
            
        # Assuming daily returns, annualize with sqrt(252)
        return (returns_series.mean() / std) * np.sqrt(252)
        
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
            
        peak = float('-inf')
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            if peak > 0:  # Avoid division by zero
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
        return max_dd
// ... existing code ...
```

---

_**Assistant**_







The changes I've made include:

1. In the metrics display component:
- Added proper default values (0.0 instead of 0)
- Fixed the max drawdown display to show as percentage
- Improved formatting for all metrics
- Added proper string formatting for Total

---



---

