import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Optional
from datetime import datetime
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
    
    Implementation Notes:
    - For single-asset mode, uses 'default' as the asset key
    - For multi-asset mode, uses asset symbols as keys
    - All prices and position data stored in dictionaries for consistency
    - Handles transaction fees for accurate PnL calculation
    - Implements peak value tracking for drawdown calculation
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
            max_position (float): Maximum allowed position size as a fraction of portfolio value
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
        - Initializes positions with {'default': 0.0}
        
        For multi-asset mode:
        - Initializes empty positions dictionary
        """
        self.cash = self.initial_capital
        self.positions: Dict[str, float] = {}  # Empty dict for both modes
        self.trades: List[Dict] = []
        self.portfolio_history: List[float] = [self.initial_capital]
        self.cash_history: List[float] = [self.initial_capital]  # Track cash balance history
        self.peak_value = self.initial_capital  # For drawdown calculation
        self.current_timestamp = None  # Track current timestamp
        
        # Reset risk manager if exists
        if self.risk_manager:
            self.risk_manager.reset()

    def update(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        actions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Update portfolio state based on current prices and desired actions.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            prices (Dict[str, float]): Current prices for each asset
            actions (Dict[str, float]): Desired position sizes for each asset (-1 to 1)
            
        Returns:
            Dict[str, Any]: Dictionary containing execution results
            
        Notes:
        - For single-asset mode, use {'default': price} and {'default': action}
        - For multi-asset mode, use {'BTC': price1, 'ETH': price2} format
        """
        self.current_timestamp = timestamp
        initial_portfolio_value = self.get_portfolio_value(prices)
        total_fees = 0.0
        results = {}
        
        # Execute trades
        for symbol, action in actions.items():
            if symbol not in prices:
                continue
                
            # Calculate target position
            current_price = prices[symbol]
            portfolio_value = self.get_portfolio_value(prices)
            
            # For action == 0, fully close the position
            if abs(action) < 1e-6:
                target_position = 0
            else:
                # Calculate maximum trade value considering fees
                max_trade_value = self.cash / (1 + self.trading_fee) if action > 0 else portfolio_value
                max_position_value = max_trade_value * self.max_position
                target_position_value = action * max_position_value
                target_position = target_position_value / current_price
            
            # Current position
            current_position = self.positions.get(symbol, 0.0)
            
            # Calculate trade size
            trade_amount = target_position - current_position
            trade_value = abs(trade_amount * current_price)
            trade_fee = trade_value * self.trading_fee
            
            # Check with risk manager if available
            if self.risk_manager and abs(trade_amount) > 1e-6:
                risk_check = self.risk_manager.check_trade(
                    timestamp=timestamp,
                    portfolio_value=portfolio_value,
                    trade_size=trade_amount,
                    price=current_price,
                    positions=self.positions,
                    asset=symbol
                )
                
                if not risk_check['allowed']:
                    results[symbol] = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': current_price,
                        'amount': 0,
                        'fee': 0,
                        'success': False,
                        'reason': risk_check['reason']
                    }
                    continue
                    
                # Adjust trade size if needed
                if risk_check['adjusted_size'] != trade_amount:
                    trade_amount = risk_check['adjusted_size']
                    trade_value = abs(trade_amount * current_price)
                    trade_fee = trade_value * self.trading_fee
            
            if abs(trade_amount) > 1e-6:  # Minimum trade threshold
                # Check if we have enough cash for buying
                if trade_amount > 0 and trade_value + trade_fee > self.cash:
                    # Insufficient funds
                    results[symbol] = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': current_price,
                        'amount': 0,
                        'fee': 0,
                        'success': False,
                        'reason': 'insufficient_funds'
                    }
                    continue
                
                # Determine trade type based on position change
                if target_position > current_position:
                    trade_type = 'buy'  # Increasing position
                else:
                    trade_type = 'sell'  # Decreasing position
                
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': current_price,
                    'amount': trade_amount,
                    'value': trade_value,
                    'type': trade_type,
                    'fee': trade_fee,
                    'success': True
                }
                self.trades.append(trade)
                
                # Update position and cash
                self.positions[symbol] = current_position + trade_amount
                self.cash -= (trade_value + trade_fee) if trade_amount > 0 else -(trade_value - trade_fee)
                
                # Remove position if close to zero
                if abs(self.positions[symbol]) < 1e-6:
                    del self.positions[symbol]
                    
                # Update risk manager if available
                if self.risk_manager:
                    self.risk_manager.update_after_trade(timestamp)
                    
                results[symbol] = trade
                total_fees += trade_fee
                
            else:
                # Skip trade
                results[symbol] = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': current_price,
                    'amount': 0,
                    'fee': 0,
                    'success': True,
                    'reason': 'trade size too small'
                }
        
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
            'returns': (new_portfolio_value - initial_portfolio_value) / initial_portfolio_value,
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
        Execute a single trade for one asset.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            action (float): Desired position size (-1 to 1)
            price_data (Dict[str, float]): Price data for the asset
            asset (str): Asset identifier (default for single-asset)
            
        Returns:
            Dict[str, Any]: Trade execution results
            
        Notes:
            - For single-asset mode, use asset='default'
            - For multi-asset mode, specify the asset symbol
            - price_data should contain the asset key
        """
        if asset not in price_data:
            return {
                'timestamp': timestamp,
                'symbol': asset,
                'amount': 0,
                'price': 0,
                'fee': 0,
                'success': False,
                'reason': 'price not available'
            }
            
        current_price = price_data[asset]
        
        # Update portfolio value first to get current drawdown
        portfolio_value = self.get_portfolio_value(price_data)
        
        # Skip very small actions
        if abs(action) < 1e-6:
            return {
                'timestamp': timestamp,
                'symbol': asset,
                'price': current_price,
                'amount': 0,
                'fee': 0,
                'success': True,
                'reason': 'trade size too small'
            }
            
        # Current position
        current_position = self.positions.get(asset, 0.0)
        
        # For buys, check if we have enough cash
        if action > 0:
            # Calculate maximum affordable position
            max_trade_value = self.cash / (1 + self.trading_fee)
            if max_trade_value < 1e-6:  # Not enough cash for meaningful trade
                return {
                    'timestamp': timestamp,
                    'symbol': asset,
                    'price': current_price,
                    'amount': 0,
                    'fee': 0,
                    'success': True,
                    'reason': 'insufficient_funds'
                }
                
            # Calculate target position
            max_position_value = max_trade_value * self.max_position
            target_position_value = action * max_position_value
            target_position = target_position_value / current_price
            
        # For sells, calculate target position directly
        else:
            target_position = current_position * (1 + action)  # action is negative
            
        # Calculate trade size
        trade_amount = target_position - current_position
        trade_value = abs(trade_amount * current_price)
        trade_fee = trade_value * self.trading_fee
        
        # Check with risk manager if available
        if self.risk_manager and abs(trade_amount) > 1e-6:
            risk_check = self.risk_manager.check_trade(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                trade_size=trade_amount,
                price=current_price,
                positions=self.positions,
                asset=asset
            )
            
            if not risk_check['allowed']:
                return {
                    'timestamp': timestamp,
                    'symbol': asset,
                    'price': current_price,
                    'amount': 0,
                    'fee': 0,
                    'success': False,
                    'reason': risk_check['reason']
                }
                
            # Adjust trade size if needed
            if risk_check['adjusted_size'] != trade_amount:
                trade_amount = risk_check['adjusted_size']
                trade_value = abs(trade_amount * current_price)
                trade_fee = trade_value * self.trading_fee
        
        # Check minimum trade size (0.1% of portfolio or $1, whichever is larger)
        min_trade_value = max(portfolio_value * 0.001, 1.0)
        if trade_value < min_trade_value:
            return {
                'timestamp': timestamp,
                'symbol': asset,
                'price': current_price,
                'amount': 0,
                'fee': 0,
                'success': True,
                'reason': 'trade size too small'
            }
            
        # Final check for buying power
        if trade_amount > 0 and trade_value + trade_fee > self.cash:
            return {
                'timestamp': timestamp,
                'symbol': asset,
                'price': current_price,
                'amount': 0,
                'fee': 0,
                'success': True,
                'reason': 'insufficient_funds'
            }
        
        # Determine trade type
        trade_type = 'buy' if trade_amount > 0 else 'sell'
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': asset,
            'price': current_price,
            'amount': trade_amount,
            'value': trade_value,
            'type': trade_type,
            'fee': trade_fee,
            'success': True
        }
        self.trades.append(trade)
        
        # Update position and cash
        self.positions[asset] = current_position + trade_amount
        self.cash -= (trade_value + trade_fee) if trade_amount > 0 else -(trade_value - trade_fee)
        
        # Remove position if close to zero
        if abs(self.positions[asset]) < 1e-6:
            del self.positions[asset]
            
        # Update risk manager if available
        if self.risk_manager:
            self.risk_manager.update_after_trade(timestamp)
        
        return trade
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and all positions.
        
        Args:
            prices (Dict[str, float]): Current prices for each asset
            
        Returns:
            float: Total portfolio value
        """
        position_value = sum(
            self.positions.get(asset, 0) * price
            for asset, price in prices.items()
        )
        portfolio_value = self.cash + position_value
        
        # Update risk manager's peak value tracking
        if self.risk_manager:
            current_drawdown = self.risk_manager.update_drawdown(portfolio_value)
            self.logger.debug(
                f"Portfolio value: {portfolio_value:.2f}, "
                f"Peak value: {self.risk_manager.peak_value:.2f}, "
                f"Drawdown: {current_drawdown:.2%}"
            )
            
        return portfolio_value

    def run(
        self,
        strategy: Any,
        window_size: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run backtest for single-asset strategy.
        
        Args:
            strategy: Strategy object with get_action method
            window_size (int): Size of the data window for strategy
            verbose (bool): Whether to print progress
            
        Returns:
            Dict containing:
            - metrics: Performance metrics
            - trades: List of all trades
            - portfolio_values: History of portfolio values
            - timestamps: List of timestamps
        """
        if self.data is None:
            raise ValueError("No data provided for backtesting")
            
        try:
            for i in range(window_size, len(self.data)):
                window_data = self.data.iloc[i - window_size : i].copy()
                current_data = self.data.iloc[i].copy()
                timestamp = current_data.name

                # Get strategy action
                try:
                    action = strategy.get_action(window_data)
                    if not isinstance(action, (int, float, np.ndarray)):
                        self.logger.warning(
                            f"Invalid action type: {type(action)}, expected float"
                        )
                        continue
                    action = float(action)  # Ensure action is float
                except Exception as e:
                    self.logger.error(
                        f"Error getting action from strategy: {str(e)}"
                    )
                    continue

                # Execute trade
                price_data = {
                    'default': current_data['$close']  # For single-asset mode
                }
                trade_result = self.execute_trade(
                    timestamp=timestamp,
                    action=action,
                    price_data=price_data
                )

                # Update portfolio value even if trade was skipped
                if 'portfolio_value' not in trade_result:
                    portfolio_value = self.get_portfolio_value(price_data)
                    self.portfolio_history.append(portfolio_value)
                    self.peak_value = max(self.peak_value, portfolio_value)

                if verbose and i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(self.data)}")

        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Ensure portfolio values match the data length
        expected_length = len(self.data) - window_size + 1
        if len(self.portfolio_history) < expected_length:
            last_value = (
                self.portfolio_history[-1]
                if self.portfolio_history
                else self.initial_capital
            )
            self.portfolio_history.extend(
                [last_value] * (expected_length - len(self.portfolio_history))
            )
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
        
        Returns:
            Dict containing:
            - total_return: Total return percentage
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Annualized Sortino ratio
            - max_drawdown: Maximum drawdown percentage
            - total_trades: Number of trades
            - win_rate: Percentage of profitable trades
            - final_balance: Final cash balance
            - final_portfolio_value: Final portfolio value
        """
        try:
            values = np.array(self.portfolio_history)
            returns = np.diff(values) / values[:-1]
            
            # Total return
            total_return = (values[-1] - values[0]) / values[0]
            
            # Annualized Sharpe Ratio (assuming daily data)
            excess_returns = returns  # Assuming 0 risk-free rate
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if len(returns) > 1 else 0
            
            # Sortino Ratio (using downside deviation)
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 1 else 0
            
            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = min(max_drawdown, -drawdown)  # Negative drawdown
            
            # Win rate
            profitable_trades = sum(
                1
                for trade in self.trades
                if (
                    'revenue' in trade
                    and trade['revenue'] > trade.get('cost', 0)
                )
                or (
                    'cost' in trade
                    and trade['cost'] < trade.get('revenue', float('inf'))
                )
            )
            total_trades = len(self.trades)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': -max_drawdown,  # Return positive drawdown
                'total_trades': total_trades,
                'win_rate': win_rate,
                'final_balance': self.cash,
                'final_portfolio_value': values[-1] if len(values) > 0 else self.cash,
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
        
        Returns:
            pd.Series: Historical returns with timestamps as index
        """
        returns = pd.Series(self.portfolio_history)
        returns = returns.pct_change()
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
        
        Returns:
            pd.DataFrame: All trades with details
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Get position value history for each asset.
        
        Returns:
            pd.DataFrame: Position values over time with timestamps as index
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
            values = {'timestamp': timestamp, 'total': portfolio_value}
            values.update(self.positions)
            position_values.append(values)
            
        return pd.DataFrame(position_values).set_index('timestamp') 

    def run_scenario(
        self,
        strategy: Any,
        scenario_type: str,
        window_size: int = 20,
        verbose: bool = True,
        **scenario_params
    ) -> Dict[str, Any]:
        """Run backtest with a specific scenario
        
        Args:
            strategy: Trading strategy to test
            scenario_type (str): Type of scenario ('flash_crash' or 'low_liquidity')
            window_size (int): Lookback window size for strategy
            verbose (bool): Whether to print progress
            **scenario_params: Parameters for scenario generation
            
        Returns:
            Dict[str, Any]: Results including scenario-specific metrics
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
        
        # Run standard backtest
        results = self.run(strategy, window_size, verbose)
        
        # Add scenario-specific metrics
        results["scenario_metrics"] = metric_fn(results)
        results["scenario_type"] = scenario_type
        
        return results

    def plot_scenario_results(
        self,
        results: Dict,
        save_path: str = None
    ):
        """Plot results with scenario-specific annotations
        
        Args:
            results (Dict): Results from run_scenario
            save_path (str, optional): Path to save plot
        """
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
        """Save scenario-specific results and plots
        
        Args:
            results (Dict): Results from run_scenario
            save_dir (str): Directory to save results
        """
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