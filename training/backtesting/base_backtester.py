import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from .risk_manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)

class BaseBacktester:
    """
    Unified backtester that can handle both single-asset and multi-asset logic.
    
    Features:
    - Supports both single-asset and multi-asset backtesting
    - Tracks portfolio value, positions, and trade history
    - Handles trading fees
    - Position size limits
    - Basic risk management
    - Performance metrics calculation
    - Cost basis tracking and profit realization
    
    Implementation Notes:
    - For single-asset mode, uses 'default' as the asset key
    - For multi-asset mode, uses asset symbols as keys
    - All prices and position data stored in dictionaries for consistency
    - Handles transaction fees for accurate PnL calculation
    - Implements peak value tracking for drawdown calculation
    - Tracks cost basis per position for accurate profit calculation
    
    Now configured for FRACTIONAL HOLDING:
    - action in [0,1] => fraction of total portfolio to hold in the asset
    - No short selling (units never go below 0)
    """

    REQUIRED_COLUMNS = {"$open", "$high", "$low", "$close", "$volume"}
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,  # 0.1% trading fee
        max_position: float = 1.0,
        data: pd.DataFrame = None,
        risk_config: Optional[RiskConfig] = None,
    ):
        """
        Initialize the backtester with common parameters for both single and multi-asset testing.
        
        Args:
            initial_capital (float): Starting capital for the portfolio
            trading_fee (float): Fee per trade as a fraction of trade value (default: 0.1%)
            max_position (float): (Optional) Maximum fraction to allow holding. 
                                  For example, 1.0 means up to 100% of portfolio in coin.
            data (pd.DataFrame, optional): OHLCV data with columns: $open, $high, $low, $close, $volume
            risk_config (RiskConfig, optional): Risk management configuration
            
        Notes:
        - Data columns must be prefixed with '$' (e.g., '$close')
        - For multi-asset data, use asset_$column format (e.g., 'BTC_$close')
        """
        if data is not None:
            # Validate required columns for single-asset mode
            if not any("_$" in col for col in data.columns):  # Single-asset mode
                missing_columns = self.REQUIRED_COLUMNS - set(data.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {list(missing_columns)}")
            
            self.data = data.copy()
            
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.max_position = max_position
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize risk manager if config provided
        self.risk_manager = RiskManager(risk_config) if risk_config else None
        
        self.reset()

    def reset(self):
        """
        Reset portfolio state to initial conditions.
        Clears all positions, trades, and history.
        
        For single-asset mode:
        - Initializes positions with {'default': {'units': 0.0, 'avg_price': 0.0, 'cost_basis': 0.0}}
        
        For multi-asset mode:
        - Initializes empty positions dictionary
        """
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}
        self.trades: List[Dict] = []
        self.portfolio_history: List[float] = [self.initial_capital]
        self.cash_history: List[float] = [self.initial_capital]
        self.peak_value = self.initial_capital
        self.current_timestamp = None
        
        if self.risk_manager:
            self.risk_manager.reset()

    def update(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        actions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Update portfolio state based on current prices and desired fractional actions in [0,1].
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            prices (Dict[str, float]): Current prices for each asset
            actions (Dict[str, float]): Desired fraction [0,1] for each asset 
            
        Returns:
            Dict[str, Any]: Dictionary containing execution results
            
        Notes:
        - For single-asset mode, use {'default': price} and {'default': action}
        - For multi-asset mode, use e.g. {'BTC': fraction1, 'ETH': fraction2}
        - This is FRACTIONAL HOLDING. 
          e.g. action=0.3 => want 30% of total portfolio in 'asset'
        """
        self.current_timestamp = timestamp
        initial_portfolio_value = self.get_portfolio_value(prices)
        total_fees = 0.0
        results = {}
        
        # Execute trades
        for symbol, fraction in actions.items():
            if symbol not in prices:
                continue
                
            # Ensure position record
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "units": 0.0,
                    "avg_price": 0.0,
                    "cost_basis": 0.0
                }
            
            price_data = {symbol: prices[symbol]}
            trade_result = self.execute_trade(
                timestamp=timestamp,
                action=fraction,  # in [0,1]
                price_data=price_data,
                asset=symbol
            )
            
            results[symbol] = trade_result
            if trade_result['success']:
                total_fees += trade_result['fee']
        
        # Update history
        new_portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_history.append(new_portfolio_value)
        self.cash_history.append(self.cash)
        
        return {
            'timestamp': timestamp,
            'trades': results,
            'portfolio_value': new_portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'returns': (
                (new_portfolio_value - initial_portfolio_value) / initial_portfolio_value
                if initial_portfolio_value > 1e-12 else 0.0
            ),
            'total_fees': total_fees
        }
    
    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        action: float,
        price_data: Dict[str, float],
        asset: str = "default"
    ) -> Dict[str, Any]:
        """
        Fractional Holding version:
        action in [0,1] => fraction of total portfolio to hold in this asset.
        """
        trade = {
            "timestamp": timestamp,
            "symbol": asset,
            "amount": 0.0,
            "price": 0.0,
            "fee": 0.0,
            "cost": 0.0,
            "revenue": 0.0,
            "profit": 0.0,
            "success": False,
            "reason": "",
            "action": action,
            "type": "none",
            "value": 0.0,
            "portfolio_value_before": self.get_portfolio_value(price_data),
            "portfolio_value_after": 0.0,
            "cumulative_pnl": 0.0,  # Track cumulative PnL
            "cash_after": 0.0,  # Track remaining cash
            "position_units": 0.0,  # Track position size
            "position_value": 0.0,  # Track position value
        }

        if asset not in price_data:
            trade["reason"] = "price_not_available"
            self.trades.append(trade)
            self.logger.debug("[TRADE_SKIP] %s: price_not_available", asset)
            return trade

        current_price = price_data[asset]
        trade["price"] = current_price
        
        # clamp action in [0,1]
        if action < 0.0:
            action = 0.0
        elif action > 1.0:
            action = 1.0
        
        if asset not in self.positions:
            self.positions[asset] = {
                "units": 0.0,
                "avg_price": 0.0,
                "cost_basis": 0.0
            }

        pos_dict = self.positions[asset]
        old_units = pos_dict["units"]
        old_cost_basis = pos_dict["cost_basis"]

        # current coin value
        current_coin_value = old_units * current_price
        # total portfolio
        portfolio_value = self.cash + current_coin_value
        
        # desired coin value
        target_coin_value = action * portfolio_value
        
        diff_value = target_coin_value - current_coin_value
        if abs(diff_value) < 1e-12:
            trade["reason"] = "trade_size_too_small"
            self.trades.append(trade)
            return trade

        trade_amount = diff_value / current_price if abs(current_price) > 1e-12 else 0.0
        if abs(trade_amount) < 1e-12:
            trade["reason"] = "trade_size_too_small"
            self.trades.append(trade)
            return trade

        trade_value = abs(diff_value)
        fee = trade_value * self.trading_fee

        # Risk Manager check
        if self.risk_manager and abs(trade_amount) > 1e-12:
            risk_check = self.risk_manager.check_trade(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                trade_size=trade_amount,
                price=current_price,
                positions=self.positions,
                asset=asset
            )
            if not risk_check['allowed']:
                trade["success"] = False
                trade["reason"] = risk_check['reason']
                self.trades.append(trade)
                return trade

            # if size adjusted
            if abs(risk_check['adjusted_size'] - trade_amount) > 1e-12:
                trade_amount = risk_check['adjusted_size']
                diff_value = trade_amount * current_price
                trade_value = abs(diff_value)
                fee = trade_value*self.trading_fee

        # Only log significant trades (value > 1% of portfolio)
        is_significant = trade_value > (portfolio_value * 0.01)

        # BUY
        if diff_value > 0:
            trade["type"] = "buy"
            total_cost = trade_value + fee
            if total_cost > self.cash + 1e-12:
                trade["reason"] = "insufficient_funds"
                self.trades.append(trade)
                return trade
            
            new_units = old_units + trade_amount
            new_cost_basis = old_cost_basis + total_cost
            avg_price = new_cost_basis/new_units if new_units>1e-12 else 0.0
            
            self.positions[asset] = {
                "units": new_units,
                "avg_price": avg_price,
                "cost_basis": new_cost_basis,
            }
            self.cash -= total_cost

            trade["amount"] = trade_amount
            trade["value"] = trade_value
            trade["fee"] = fee
            trade["cost"] = total_cost
            trade["revenue"] = 0.0
            trade["profit"] = 0.0
            trade["success"] = True

            if is_significant:
                self.logger.info(
                    "[BUY] %s: Amount=%.6f, Price=$%.2f, Total=$%.2f, Fee=$%.2f",
                    asset, trade_amount, current_price, total_cost, fee
                )

        else:
            # SELL
            trade["type"] = "sell"
            sell_amount = abs(trade_amount)
            if sell_amount > old_units+1e-12:
                sell_amount = old_units
                diff_value = sell_amount*current_price
                trade_value = abs(diff_value)
                fee = trade_value*self.trading_fee

            fraction = sell_amount/old_units if old_units>1e-12 else 1.0
            cost_portion = old_cost_basis*fraction
            revenue = trade_value
            realized_profit = revenue - cost_portion - fee

            new_units = old_units - sell_amount
            if new_units>1e-12:
                new_cost_basis = old_cost_basis - cost_portion
                self.positions[asset] = {
                    "units": new_units,
                    "avg_price": pos_dict["avg_price"],
                    "cost_basis": new_cost_basis,
                }
            else:
                del self.positions[asset]

            self.cash += (revenue - fee)

            trade["amount"] = -sell_amount
            trade["value"] = revenue
            trade["fee"] = fee
            trade["cost"] = cost_portion
            trade["revenue"] = revenue
            trade["profit"] = realized_profit
            trade["success"] = True

            if is_significant:
                self.logger.info(
                    "[SELL] %s: Amount=%.6f, Price=$%.2f, Revenue=$%.2f, Profit=$%.2f",
                    asset, sell_amount, current_price, revenue, realized_profit
                )
        
        # If trade was successful, update portfolio value after and related metrics
        if trade["success"]:
            updated_portfolio_value = self.get_portfolio_value(price_data)
            trade["portfolio_value_after"] = updated_portfolio_value
            trade["cumulative_pnl"] = updated_portfolio_value - self.initial_capital
            trade["cash_after"] = self.cash
            
            # Track position details
            if asset in self.positions:
                pos = self.positions[asset]
                trade["position_units"] = pos["units"]
                trade["position_value"] = pos["units"] * current_price
            else:
                trade["position_units"] = 0.0
                trade["position_value"] = 0.0
        else:
            # For unsuccessful trades, still record actual portfolio value
            updated_portfolio_value = self.get_portfolio_value(price_data)
            trade["portfolio_value_after"] = updated_portfolio_value
            trade["cumulative_pnl"] = updated_portfolio_value - self.initial_capital
            trade["cash_after"] = self.cash
            trade["position_units"] = self.positions[asset]["units"] if asset in self.positions else 0.0
            trade["position_value"] = (self.positions[asset]["units"] * current_price) if asset in self.positions else 0.0

        self.trades.append(trade)
        
        # risk manager post-update
        if self.risk_manager and trade["success"]:
            self.risk_manager.update_after_trade(timestamp)

        return trade
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and all positions.
        """
        position_value = sum(
            self.positions[asset]["units"] * price
            for asset, price in prices.items()
            if asset in self.positions
        )
        portfolio_value = self.cash + position_value
        
        if self.risk_manager:
            current_drawdown = self.risk_manager.update_drawdown(portfolio_value)
            
            # Initialize update counter if not exists
            if not hasattr(self, '_update_counter'):
                self._update_counter = 0
                self._last_logged_value = portfolio_value
            
            self._update_counter += 1
            
            # Log on significant events:
            # 1. Significant drawdown (>5%)
            # 2. Significant value change (>2%)
            # 3. Every 100 updates
            should_log = (
                current_drawdown > 0.05 or  # Significant drawdown
                abs(portfolio_value - self._last_logged_value) / self._last_logged_value > 0.02 or  # Significant value change
                self._update_counter % 100 == 0  # Periodic update
            )
            
            if should_log:
                if current_drawdown > 0.05:
                    self.logger.warning(
                        "High drawdown: %.2f%% (Portfolio: $%.2f, Peak: $%.2f)",
                        current_drawdown * 100,
                        portfolio_value,
                        self.risk_manager.peak_value
                    )
                else:
                    self.logger.info(
                        "Portfolio Update - Value: $%.2f (%.2f%% change), Drawdown: %.2f%%",
                        portfolio_value,
                        ((portfolio_value - self._last_logged_value) / self._last_logged_value) * 100,
                        current_drawdown * 100
                    )
                self._last_logged_value = portfolio_value
                
        return portfolio_value

    def run(
        self,
        strategy: Any,
        window_size: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run backtest for single-asset strategy (Fractional Holding).
        strategy.get_action() => fraction in [0,1].
        
        Now uses update() for each bar to track portfolio changes every step,
        even when no trade occurs.
        """
        if self.data is None:
            raise ValueError("No data provided for backtesting")
            
        try:
            for i in range(window_size, len(self.data)):
                window_data = self.data.iloc[i - window_size : i].copy()
                current_data = self.data.iloc[i].copy()
                timestamp = current_data.name

                # 1) get fraction in [0,1]
                try:
                    raw_action = strategy.get_action(window_data)
                    # clamp
                    action = float(np.clip(raw_action, 0.0, 1.0))
                except Exception as e:
                    self.logger.error(
                        f"Error getting action from strategy: {str(e)}"
                    )
                    continue

                # 2) Prepare prices and actions dict for update()
                prices = {"default": current_data["$close"]}
                actions = {"default": action}

                # 3) Call update() once per bar
                update_result = self.update(
                    timestamp=timestamp,
                    prices=prices,
                    actions=actions
                )

                if verbose and i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(self.data)} bars processed")

        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise

        # final metrics
        metrics = self._calculate_metrics()

        # align length
        expected_length = len(self.data) - window_size + 1
        if len(self.portfolio_history) < expected_length:
            last_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_capital
            self.portfolio_history.extend([last_value]*(expected_length - len(self.portfolio_history)))
        elif len(self.portfolio_history) > expected_length:
            self.portfolio_history = self.portfolio_history[:expected_length]

        return {
            "metrics": metrics,
            "trades": self.trades,
            "portfolio_values": self.portfolio_history,
            "timestamps": self.data.index[window_size - 1 :].tolist(),
        }

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        """
        try:
            values = np.array(self.portfolio_history)
            if len(values) < 2:
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'max_drawdown': 0,
                    'total_trades': len(self.trades),
                    'win_rate': 0,
                    'final_balance': self.cash,
                    'final_portfolio_value': self.cash,
                }

            returns = np.diff(values) / values[:-1]
            
            total_return = (values[-1] - values[0]) / values[0]
            
            if len(returns) > 1 and np.std(returns)>0:
                sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
            else:
                sharpe_ratio = 0.0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns)>0:
                sortino_ratio = (
                    np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
                )
            else:
                sortino_ratio = 0.0
            
            # max drawdown
            peak = values[0]
            max_dd = 0.0
            for val in values[1:]:
                if val>peak:
                    peak=val
                dd = (peak-val)/peak
                max_dd = max(max_dd, dd)
            
            profitable_trades = sum(
                1
                for t in self.trades
                if t["success"] and t["profit"] > 0
            )
            total_trades = len(self.trades)
            win_rate = (profitable_trades / total_trades) if total_trades>0 else 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_dd,  # positive
                'total_trades': total_trades,
                'win_rate': win_rate,
                'final_balance': self.cash,
                'final_portfolio_value': values[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'total_trades': len(self.trades),
                'win_rate': 0,
                'final_balance': self.cash,
                'final_portfolio_value': self.cash,
            }

    def get_returns(self) -> pd.Series:
        """
        Calculate returns series.
        """
        returns = pd.Series(self.portfolio_history).pct_change()
        if self.current_timestamp is not None:
            returns.index = pd.date_range(
                start=self.current_timestamp - pd.Timedelta(days=len(returns) - 1),
                end=self.current_timestamp,
                periods=len(returns),
            )
        return returns
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Get position value history for each asset.
        """
        position_values = []
        if self.current_timestamp is not None:
            timestamps = pd.date_range(
                start=self.current_timestamp - pd.Timedelta(days=len(self.portfolio_history) - 1),
                end=self.current_timestamp,
                periods=len(self.portfolio_history),
            )
        else:
            timestamps = range(len(self.portfolio_history))
        
        for timestamp, portfolio_value in zip(timestamps, self.portfolio_history):
            row = {'timestamp': timestamp, 'total': portfolio_value}
            row.update(self.positions)
            position_values.append(row)
            
        return pd.DataFrame(position_values).set_index('timestamp')

    def run_scenario(
        self,
        strategy: Any,
        scenario_type: str,
        window_size: int = 20,
        verbose: bool = True,
        **scenario_params
    ) -> Dict[str, Any]:
        """
        Run backtest with a specific scenario (flash_crash, low_liquidity).
        """
        from .scenario import (
            generate_flash_crash_data,
            generate_low_liquidity_data,
            calculate_flash_crash_metrics,
            calculate_low_liquidity_metrics
        )
        
        # Generate scenario data
        if scenario_type == "flash_crash":
            self.data = generate_flash_crash_data(**scenario_params)
            metric_fn = calculate_flash_crash_metrics
        elif scenario_type == "low_liquidity":
            self.data = generate_low_liquidity_data(**scenario_params)
            metric_fn = calculate_low_liquidity_metrics
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        results = self.run(strategy, window_size, verbose)
        
        results["scenario_metrics"] = metric_fn(results)
        results["scenario_type"] = scenario_type
        return results

    def plot_scenario_results(
        self,
        results: Dict,
        save_path: str = None
    ):
        """Plot results with scenario-specific annotations."""
        from .scenario import plot_scenario_results
        plot_scenario_results(
            results=results,
            scenario_type=results["scenario_type"],
            save_path=save_path
        )

    def save_scenario_results(
        self,
        results: Dict,
        save_dir: str
    ):
        """Save scenario-specific results and plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save standard results
        self.save_results(results, save_dir)

        # Save scenario-specific metrics
        scenario_df = pd.DataFrame([results["scenario_metrics"]])
        scenario_df.to_csv(
            save_dir / f"{results['scenario_type']}_metrics.csv",
            index=False
        )

        # Save enhanced plots
        self.plot_scenario_results(
            results,
            save_path=str(save_dir / f"{results['scenario_type']}_plot.png")
        )

        logger.info(f"Scenario results saved to {save_dir}")

    def save_results(self, results: Dict, save_dir: Path):
        """Helper to save backtest results (trades, portfolio, etc.) to CSV."""
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results["trades"]).to_csv(save_dir / "trades.csv", index=False)
        pd.DataFrame({"portfolio_values": results["portfolio_values"]}).to_csv(save_dir / "portfolio.csv", index=False)
        pd.DataFrame({"timestamps": results["timestamps"]}).to_csv(save_dir / "timestamps.csv", index=False)
        pd.DataFrame([results["metrics"]]).to_csv(save_dir / "metrics.csv", index=False)
        logger.info(f"Backtest results saved to {save_dir}")
