"""Enhanced backtester with realistic market simulation features."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

from .base_backtester import BaseBacktester
from .risk_manager import RiskManager, RiskConfig
from .market_simulator import MarketSimulator
from data.utils.enhanced_data_loader import EnhancedDataLoader

logger = logging.getLogger(__name__)


class EnhancedBacktester(BaseBacktester):
    """
    Enhanced backtester with realistic market simulation including slippage and partial fills.
    
    Features:
    - All features from BaseBacktester
    - Multi-asset data support
    - Realistic slippage modeling
    - Partial fill simulation
    - Order book-based execution
    - Dynamic fee structure
    - Streaming data capability
    - Real-time and historical data modes
    
    Implementation Notes:
    - Extends BaseBacktester to maintain compatibility
    - Uses EnhancedDataLoader for data acquisition
    - Uses MarketSimulator for trade execution
    - Supports both historical and real-time backtesting
    - Adds realistic market impact modeling
    - Handles order book data when available
    
    Recent Changes:
    - Integrated MarketSimulator for realistic execution
    - Added multi-asset correlation support
    - Implemented dynamic fee structure based on liquidity
    - Added partial fill logic based on order book depth
    
    Example:
        >>> loader = EnhancedDataLoader(
        ...     symbols=["BTC/USDT", "ETH/USDT"],
        ...     include_orderbook=True
        ... )
        >>> data = loader.fetch_multi_asset_data("2023-01-01", "2023-01-31")
        >>> backtester = EnhancedBacktester(
        ...     data=data,
        ...     initial_capital=10000.0,
        ...     trading_fee=0.001,
        ...     slippage_model="orderbook",
        ...     partial_fill=True
        ... )
        >>> results = backtester.run(strategy)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        max_position: float = 1.0,
        data: Optional[pd.DataFrame] = None,
        risk_config: Optional[RiskConfig] = None,
        slippage_model: str = "volume",
        partial_fill: bool = True,
        orderbook_enabled: bool = False,
        market_impact_factor: float = 0.1,
        data_loader: Optional[EnhancedDataLoader] = None,
    ):
        """
        Initialize the enhanced backtester.
        
        Args:
            initial_capital: Starting capital for the portfolio
            trading_fee: Base fee per trade as a fraction of trade value (default: 0.1%)
            max_position: Maximum fraction to allow holding
            data: OHLCV data (optional)
            risk_config: Risk management configuration
            slippage_model: Slippage model to use ('volume', 'fixed', 'orderbook')
            partial_fill: Whether to simulate partial fills
            orderbook_enabled: Whether order book data is available
            market_impact_factor: Factor for market impact calculation
            data_loader: Optional EnhancedDataLoader instance
        """
        # Initialize base backtester
        super().__init__(
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            max_position=max_position,
            data=data,
            risk_config=risk_config,
        )
        
        # Initialize market simulator
        self.market_simulator = MarketSimulator(
            base_fee=trading_fee,
            slippage_model=slippage_model,
            partial_fill=partial_fill,
            market_impact_factor=market_impact_factor,
            orderbook_enabled=orderbook_enabled,
        )
        
        # Initialize data loader if provided
        self.data_loader = data_loader
        
        # Add enhanced tracking metrics
        self.slippage_history = []
        self.fill_rate_history = []
        self.execution_delay_history = []
        self.market_impact_history = []
        
        # Multi-asset specific data
        self.symbols = []
        if data is not None:
            self.extract_symbols_from_data()
            
        # Real-time backtest settings
        self.is_realtime = False
        self.streaming_callback = None
        
        logger.info(
            f"Initialized EnhancedBacktester with slippage_model={slippage_model}, "
            f"partial_fill={partial_fill}, orderbook_enabled={orderbook_enabled}"
        )

    def extract_symbols_from_data(self) -> None:
        """
        Extract unique symbols from the data DataFrame.
        Handles both single-asset and multi-asset data formats.
        """
        if self.data is None:
            return
            
        # Check if this is multi-asset data by looking for symbol prefixes
        has_prefixes = any("_$" in col for col in self.data.columns)
        
        if has_prefixes:
            # Multi-asset data format (e.g., "BTC/USDT_$close")
            self.symbols = []
            for col in self.data.columns:
                if "_$" in col:
                    symbol = col.split("_$")[0]
                    if symbol not in self.symbols:
                        self.symbols.append(symbol)
        else:
            # Single-asset data format
            self.symbols = ["default"]
            
        logger.info(f"Detected {len(self.symbols)} symbols in data: {self.symbols}")

    def reset(self):
        """
        Reset portfolio state to initial conditions.
        Extends BaseBacktester.reset() with enhanced metrics tracking.
        """
        super().reset()
        
        # Reset enhanced tracking metrics
        self.slippage_history = []
        self.fill_rate_history = []
        self.execution_delay_history = []
        self.market_impact_history = []

    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        action: float,
        price_data: Dict[str, float],
        asset: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute a trade with realistic market conditions.
        
        This overrides the BaseBacktester.execute_trade() method to use
        the MarketSimulator for more realistic execution.
        
        Args:
            timestamp: Current timestamp
            action: Desired fraction [0,1] of portfolio to hold in asset
            price_data: Current prices for each asset
            asset: Asset symbol to trade
            
        Returns:
            Dict with execution results
        """
        # Get basic trade info like in base class, but don't execute yet
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
            "action": float(action),
            "type": "none",
            "value": 0.0,
            "portfolio_value_before": self.get_portfolio_value(price_data),
            "portfolio_value_after": 0.0,
            "cumulative_pnl": 0.0,
            "cash_after": 0.0,
            "position_units": 0.0,
            "position_value": 0.0,
        }

        # Validate price data
        if asset not in price_data:
            trade["reason"] = "price_not_available"
            trade["success"] = False
            self.trades.append(trade)
            self.logger.debug("[TRADE_SKIP] %s: price_not_available", asset)
            return trade

        current_price = float(price_data[asset])
        trade["price"] = current_price
        
        # Clamp action to [0,1]
        if action < 0.0:
            action = 0.0
        elif action > 1.0:
            action = 1.0
        
        # Set up position if needed
        if asset not in self.positions:
            self.positions[asset] = {
                "units": 0.0,
                "avg_price": 0.0,
                "cost_basis": 0.0
            }

        pos_dict = self.positions[asset]
        old_units = float(pos_dict["units"])
        old_cost_basis = float(pos_dict["cost_basis"])

        # Calculate current holding value
        current_coin_value = old_units * current_price
        portfolio_value = self.cash + current_coin_value
        
        # Calculate target holding
        target_coin_value = action * portfolio_value
        
        # Calculate difference
        diff_value = target_coin_value - current_coin_value
        if abs(diff_value) < 1e-12:
            trade["reason"] = "trade_size_too_small"
            trade["success"] = False
            self.trades.append(trade)
            return trade

        # Calculate trade amount
        trade_amount = diff_value / current_price if abs(current_price) > 1e-12 else 0.0
        if abs(trade_amount) < 1e-12:
            trade["reason"] = "trade_size_too_small"
            trade["success"] = False
            self.trades.append(trade)
            return trade

        # Get volume data if available
        volume = None
        if self.data is not None:
            # Try to get volume from data at current timestamp
            if asset == "default" and "$volume" in self.data.columns:
                try:
                    volume = self.data.loc[timestamp, "$volume"]
                except (KeyError, TypeError):
                    pass
            elif f"{asset}_$volume" in self.data.columns:
                try:
                    volume = self.data.loc[timestamp, f"{asset}_$volume"]
                except (KeyError, TypeError):
                    pass
        
        # Get order book if available
        orderbook = None
        if self.data_loader is not None and hasattr(self.data_loader, "get_orderbook_snapshot"):
            try:
                symbol_for_orderbook = asset if asset != "default" else self.data_loader.symbols[0]
                orderbook = self.data_loader.get_orderbook_snapshot(symbol_for_orderbook)
            except Exception as e:
                logger.debug(f"Error getting orderbook: {e}")
        
        # Check with risk manager
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

            # Apply size adjustment if needed
            if abs(risk_check['adjusted_size'] - trade_amount) > 1e-12:
                trade_amount = float(risk_check['adjusted_size'])
                diff_value = trade_amount * current_price
        
        # Determine order side and size
        side = "buy" if diff_value > 0 else "sell"
        amount = abs(trade_amount)
        
        # Execute order through market simulator
        symbol_for_simulator = asset if asset != "default" else (self.symbols[0] if self.symbols else "BTC/USDT")
        execution_result = self.market_simulator.execute_order(
            symbol=symbol_for_simulator,
            order_type="market",
            side=side,
            amount=amount,
            current_price=current_price,
            volume=volume,
            orderbook=orderbook
        )
        
        # Track execution metrics
        self.slippage_history.append({
            "timestamp": timestamp,
            "symbol": asset,
            "slippage_percentage": execution_result['slippage_percentage']
        })
        
        self.fill_rate_history.append({
            "timestamp": timestamp,
            "symbol": asset,
            "fill_percentage": execution_result['fill_percentage']
        })
        
        # Process execution result
        if execution_result['success']:
            # For buys
            if side == "buy":
                executed_amount = execution_result['executed_amount']
                fee_amount = execution_result['fee_amount']
                total_cost = execution_result['total_cost']
                
                # Check if we have enough cash
                if total_cost > self.cash + 1e-12:
                    trade["reason"] = "insufficient_funds"
                    trade["success"] = False
                    self.trades.append(trade)
                    return trade
                
                # Update position
                new_units = old_units + executed_amount
                new_cost_basis = old_cost_basis + total_cost
                avg_price = new_cost_basis/new_units if new_units > 1e-12 else 0.0
                
                self.positions[asset] = {
                    "units": float(new_units),
                    "avg_price": float(avg_price),
                    "cost_basis": float(new_cost_basis),
                }
                self.cash -= total_cost
                
                # Update trade record
                trade["type"] = "buy"
                trade["amount"] = float(executed_amount)
                trade["value"] = float(executed_amount * execution_result['average_price'])
                trade["fee"] = float(fee_amount)
                trade["cost"] = float(total_cost)
                trade["revenue"] = 0.0
                trade["profit"] = 0.0
                trade["success"] = True
                
            # For sells
            else:
                executed_amount = execution_result['executed_amount']
                fee_amount = execution_result['fee_amount']
                
                # Check if we have enough units
                if executed_amount > old_units + 1e-12:
                    executed_amount = old_units
                    # Recalculate fee
                    fee_amount = executed_amount * execution_result['average_price'] * execution_result['fee_percentage']
                
                # Calculate fraction of position being sold
                fraction = executed_amount / old_units if old_units > 1e-12 else 1.0
                cost_portion = old_cost_basis * fraction
                revenue = executed_amount * execution_result['average_price']
                realized_profit = revenue - cost_portion - fee_amount
                
                # Update position
                new_units = old_units - executed_amount
                if new_units > 1e-12:
                    new_cost_basis = old_cost_basis - cost_portion
                    self.positions[asset] = {
                        "units": float(new_units),
                        "avg_price": float(pos_dict["avg_price"]),
                        "cost_basis": float(new_cost_basis),
                    }
                else:
                    del self.positions[asset]
                
                self.cash += (revenue - fee_amount)
                
                # Update trade record
                trade["type"] = "sell"
                trade["amount"] = float(-executed_amount)
                trade["value"] = float(revenue)
                trade["fee"] = float(fee_amount)
                trade["cost"] = float(cost_portion)
                trade["revenue"] = float(revenue)
                trade["profit"] = float(realized_profit)
                trade["success"] = True
            
            # Update common fields for successful trades
            updated_portfolio_value = self.get_portfolio_value(price_data)
            trade["portfolio_value_after"] = float(updated_portfolio_value)
            trade["cumulative_pnl"] = float(updated_portfolio_value - self.initial_capital)
            trade["cash_after"] = float(self.cash)
            
            # Track position details
            if asset in self.positions:
                pos = self.positions[asset]
                trade["position_units"] = float(pos["units"])
                trade["position_value"] = float(pos["units"] * current_price)
            else:
                trade["position_units"] = 0.0
                trade["position_value"] = 0.0
                
            # Risk manager post-update
            if self.risk_manager:
                self.risk_manager.update_after_trade(timestamp)
                
        else:
            # For unsuccessful trades
            trade["success"] = False
            trade["reason"] = execution_result.get('message', 'execution_failed')
            trade["portfolio_value_after"] = float(self.get_portfolio_value(price_data))
            trade["cumulative_pnl"] = float(trade["portfolio_value_after"] - self.initial_capital)
            trade["cash_after"] = float(self.cash)
            
        # Add trade to history
        self.trades.append(trade)
        return trade

    def start_realtime_backtest(
        self,
        strategy: Any,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1m",
        include_orderbook: bool = False,
        update_interval: int = 5,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Start a real-time backtest using streaming data.
        
        Args:
            strategy: Trading strategy to use
            symbols: List of symbols to trade (optional)
            timeframe: Timeframe for data
            include_orderbook: Whether to include order book data
            update_interval: Seconds between updates
            callback: Optional callback for real-time updates
        """
        # Initialize data loader if not already done
        if self.data_loader is None:
            if symbols is None:
                symbols = ["BTC/USDT"]  # Default
                
            self.data_loader = EnhancedDataLoader(
                symbols=symbols,
                timeframe=timeframe,
                include_orderbook=include_orderbook
            )
            
        self.symbols = symbols if symbols else self.data_loader.symbols
        self.is_realtime = True
        self.streaming_callback = callback
        
        # Set up data streaming callback
        def on_data_update(data):
            # Process new data point
            symbol = data['symbol']
            current_price = data['last']
            timestamp = data['timestamp']
            
            # Update price data
            price_data = {symbol: current_price}
            
            # Get strategy action
            # Simplified: pass the latest price data to the strategy
            action = strategy.get_action(symbol, current_price, timestamp)
            
            # Execute trade
            trade_result = self.execute_trade(
                timestamp=timestamp,
                action=action,
                price_data=price_data,
                asset=symbol
            )
            
            # Update portfolio value history
            portfolio_value = self.get_portfolio_value(price_data)
            self.portfolio_history.append(portfolio_value)
            self.cash_history.append(self.cash)
            
            # Call user callback if provided
            if self.streaming_callback:
                self.streaming_callback({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': current_price,
                    'action': action,
                    'trade_result': trade_result,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'positions': self.positions
                })
                
        # Start data streaming
        self.data_loader.start_streaming(
            callback=on_data_update,
            interval_seconds=update_interval
        )
        
        logger.info(f"Started real-time backtest for {self.symbols}")

    def stop_realtime_backtest(self) -> Dict[str, Any]:
        """
        Stop the real-time backtest and return results.
        
        Returns:
            Dict with backtest results
        """
        if not self.is_realtime:
            return {}
            
        # Stop data streaming
        if self.data_loader:
            self.data_loader.stop_streaming()
            
        self.is_realtime = False
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            "metrics": metrics,
            "trades": self.trades,
            "portfolio_values": self.portfolio_history,
            "cash_history": self.cash_history,
            "slippage_history": self.slippage_history,
            "fill_rate_history": self.fill_rate_history,
        }

    def run_multi_asset(
        self,
        strategy: Any,
        window_size: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a multi-asset backtest using the provided strategy.
        
        Args:
            strategy: Trading strategy that can handle multiple assets
            window_size: Window size for historical data
            verbose: Whether to print progress
            
        Returns:
            Dict with backtest results
        """
        if self.data is None:
            raise ValueError("No data provided for backtesting")
            
        if not self.symbols or len(self.symbols) <= 1:
            logger.warning("Running multi-asset backtest with only one symbol")
            
        # Reset state
        self.reset()
        
        # Verify we have enough data
        if len(self.data) <= window_size:
            raise ValueError(f"Not enough data points. Have {len(self.data)}, need > {window_size}")
            
        # Track progress
        total_steps = len(self.data) - window_size
        progress_interval = max(1, total_steps // 20) if verbose else 0
        
        # Iterate through data
        for i in range(window_size, len(self.data)):
            current_idx = i
            timestamp = self.data.index[current_idx]
            
            # Get current OHLCV data for all symbols
            prices = {}
            volumes = {}
            
            for symbol in self.symbols:
                if symbol == "default":
                    # Single-asset format
                    if "$close" in self.data.columns:
                        prices[symbol] = float(self.data.iloc[current_idx]["$close"])
                    if "$volume" in self.data.columns:
                        volumes[symbol] = float(self.data.iloc[current_idx]["$volume"])
                else:
                    # Multi-asset format
                    col = f"{symbol}_$close"
                    if col in self.data.columns:
                        prices[symbol] = float(self.data.iloc[current_idx][col])
                    
                    vol_col = f"{symbol}_$volume"
                    if vol_col in self.data.columns:
                        volumes[symbol] = float(self.data.iloc[current_idx][vol_col])
            
            # Skip if no prices available
            if not prices:
                continue
                
            # Get strategy actions for all symbols
            actions = {}
            
            # Pass the window of data to the strategy
            window_data = self.data.iloc[current_idx - window_size + 1:current_idx + 1]
            strategy_actions = strategy.get_actions(window_data, self.symbols)
            
            # Merge strategy actions with our symbols
            for symbol in self.symbols:
                if symbol in strategy_actions:
                    actions[symbol] = strategy_actions[symbol]
                else:
                    # Default to holding current position
                    current_units = self.positions.get(symbol, {}).get("units", 0.0)
                    current_value = current_units * prices.get(symbol, 0.0)
                    current_portfolio = self.get_portfolio_value(prices)
                    current_fraction = current_value / current_portfolio if current_portfolio > 0 else 0.0
                    actions[symbol] = current_fraction
            
            # Update portfolio with all actions
            portfolio_update = self.update(timestamp, prices, actions)
            
            # Print progress
            if verbose and progress_interval > 0 and (i - window_size) % progress_interval == 0:
                progress = (i - window_size) / total_steps * 100
                portfolio_value = portfolio_update["portfolio_value"]
                logger.info(
                    f"Progress: {progress:.1f}% - Step {i-window_size}/{total_steps} - "
                    f"Portfolio: ${portfolio_value:.2f}"
                )
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            "metrics": metrics,
            "trades": self.trades,
            "portfolio_values": self.portfolio_history,
            "cash_history": self.cash_history,
            "timestamps": self.data.index[window_size:].tolist(),
            "slippage_history": self.slippage_history,
            "fill_rate_history": self.fill_rate_history,
        }

    def _calculate_advanced_metrics(self) -> Dict[str, float]:
        """
        Calculate advanced trading metrics including slippage impact.
        
        Returns:
            Dict with advanced metrics
        """
        # Get basic metrics
        metrics = super()._calculate_metrics()
        
        # Calculate additional metrics if we have enough data
        if self.slippage_history:
            # Average slippage
            slippage_values = [entry['slippage_percentage'] for entry in self.slippage_history]
            metrics['avg_slippage'] = float(np.mean(slippage_values))
            metrics['max_slippage'] = float(np.max(slippage_values))
            
            # Total slippage cost estimate
            trade_values = [trade['value'] for trade in self.trades if trade['success']]
            if trade_values:
                avg_trade_value = np.mean(trade_values)
                total_slippage_cost = avg_trade_value * np.sum(slippage_values)
                metrics['estimated_slippage_cost'] = float(total_slippage_cost)
                
        # Average fill rate
        if self.fill_rate_history:
            fill_rates = [entry['fill_percentage'] for entry in self.fill_rate_history]
            metrics['avg_fill_rate'] = float(np.mean(fill_rates))
            metrics['min_fill_rate'] = float(np.min(fill_rates))
            
        return metrics

    def plot_enhanced_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot enhanced backtest metrics including slippage and fill rates.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.portfolio_history:
            logger.warning("No backtest data to plot")
            return
            
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot portfolio value
        axes[0].plot(self.portfolio_history)
        axes[0].set_title("Portfolio Value")
        axes[0].set_ylabel("Value ($)")
        axes[0].grid(True)
        
        # Plot slippage if available
        if self.slippage_history:
            # Convert to DataFrame for easier plotting
            slippage_df = pd.DataFrame(self.slippage_history)
            slippage_df.set_index('timestamp', inplace=True)
            
            # Plot slippage percentage
            slippage_df['slippage_percentage'].plot(ax=axes[1], kind='line')
            axes[1].set_title("Trade Slippage")
            axes[1].set_ylabel("Slippage (%)")
            axes[1].grid(True)
            
        # Plot fill rates if available
        if self.fill_rate_history:
            # Convert to DataFrame
            fill_df = pd.DataFrame(self.fill_rate_history)
            fill_df.set_index('timestamp', inplace=True)
            
            # Plot fill percentage
            fill_df['fill_percentage'].plot(ax=axes[2], kind='line')
            axes[2].set_title("Order Fill Rate")
            axes[2].set_ylabel("Fill Rate (%)")
            axes[2].grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved enhanced metrics plot to {save_path}")
        else:
            plt.show()

    def save_enhanced_results(self, results: Dict, save_dir: Path) -> None:
        """
        Save enhanced backtest results to files.
        
        Args:
            results: Results dictionary
            save_dir: Directory to save results
        """
        # Create directory if it doesn't exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results.get('metrics', {}), f, indent=2)
            
        # Save trades
        trades_path = save_dir / "trades.csv"
        trades_df = pd.DataFrame(results.get('trades', []))
        if not trades_df.empty:
            trades_df.to_csv(trades_path, index=False)
            
        # Save portfolio history
        portfolio_path = save_dir / "portfolio.csv"
        portfolio_df = pd.DataFrame({
            'timestamp': results.get('timestamps', range(len(results.get('portfolio_values', [])))),
            'portfolio_value': results.get('portfolio_values', []),
            'cash': results.get('cash_history', [])
        })
        portfolio_df.to_csv(portfolio_path, index=False)
        
        # Save enhanced metrics
        if 'slippage_history' in results:
            slippage_path = save_dir / "slippage.csv"
            pd.DataFrame(results['slippage_history']).to_csv(slippage_path, index=False)
            
        if 'fill_rate_history' in results:
            fill_path = save_dir / "fill_rates.csv"
            pd.DataFrame(results['fill_rate_history']).to_csv(fill_path, index=False)
            
        # Save plots
        plot_path = save_dir / "portfolio_plot.png"
        self.plot_enhanced_metrics(save_path=str(plot_path))
        
        logger.info(f"Saved enhanced backtest results to {save_dir}")

    def run_scenario_with_slippage(
        self,
        strategy: Any,
        scenario_type: str,
        window_size: int = 20,
        slippage_factor: float = 1.0,
        fill_rate_min: float = 0.8,
        **scenario_params
    ) -> Dict[str, Any]:
        """
        Run a backtest scenario with customized slippage and fill rates.
        
        Args:
            strategy: Trading strategy
            scenario_type: Type of scenario to run
            window_size: Size of observation window
            slippage_factor: Factor to multiply slippage by (1.0 = normal)
            fill_rate_min: Minimum fill rate (0.8 = 80% fill minimum)
            **scenario_params: Additional scenario parameters
            
        Returns:
            Dict with scenario results
        """
        # Store original settings
        original_slippage_model = self.market_simulator.slippage_model
        original_market_impact = self.market_simulator.market_impact_factor
        original_min_fill = self.market_simulator.min_fill_rate
        original_data = self.data.copy() if self.data is not None else None
        
        # Apply scenario-specific settings
        if scenario_type == "high_volatility":
            # Higher slippage in volatile markets
            self.market_simulator.slippage_model = "volume"
            self.market_simulator.market_impact_factor = 0.5 * slippage_factor  # Increased from 0.2 to 0.5
            self.market_simulator.min_fill_rate = max(0.3, fill_rate_min - 0.3)  # Reduced from 0.5 to 0.3
            
            # Modify data to simulate higher volatility
            if self.data is not None:
                # Add extreme volatility by scaling price movements
                for symbol in self.symbols or [""]:
                    prefix = f"{symbol}_$" if self.symbols else "$"
                    
                    # Get price columns
                    high_col = f"{prefix}high"
                    low_col = f"{prefix}low"
                    close_col = f"{prefix}close"
                    open_col = f"{prefix}open"
                    
                    if high_col in self.data.columns and low_col in self.data.columns:
                        # Increase volatility by 80%
                        mid_price = (self.data[high_col] + self.data[low_col]) / 2
                        high_low_range = (self.data[high_col] - self.data[low_col]) * 1.8
                        
                        # Apply enhanced volatility
                        self.data[high_col] = mid_price + high_low_range / 2
                        self.data[low_col] = mid_price - high_low_range / 2
                        
                        # Adjust open/close if needed
                        if close_col in self.data.columns:
                            # Ensure close price respects high/low bounds
                            self.data[close_col] = np.clip(
                                self.data[close_col], 
                                self.data[low_col], 
                                self.data[high_col]
                            )
                        
                        if open_col in self.data.columns:
                            # Ensure open price respects high/low bounds
                            self.data[open_col] = np.clip(
                                self.data[open_col], 
                                self.data[low_col], 
                                self.data[high_col]
                            )
                
            logger.info("Running high volatility scenario with increased slippage and price volatility")
            
        elif scenario_type == "low_liquidity":
            # Lower fill rates in illiquid markets
            self.market_simulator.slippage_model = "volume"
            self.market_simulator.market_impact_factor = 0.8 * slippage_factor  # Increased from 0.3 to 0.8
            self.market_simulator.min_fill_rate = max(0.2, fill_rate_min - 0.5)  # Reduced from 0.3 to 0.2
            
            # Modify data to simulate lower liquidity
            if self.data is not None:
                for symbol in self.symbols or [""]:
                    prefix = f"{symbol}_$" if self.symbols else "$"
                    volume_col = f"{prefix}volume"
                    
                    if volume_col in self.data.columns:
                        # Reduce volume to 30% of original
                        self.data[volume_col] = self.data[volume_col] * 0.3
                        
                        # Add more variance to volume
                        random_factors = np.random.uniform(0.2, 1.0, size=len(self.data))
                        self.data[volume_col] = self.data[volume_col] * random_factors
                        
            logger.info("Running low liquidity scenario with reduced volume and extreme partial fills")
            
        elif scenario_type == "flash_crash":
            # Extreme slippage and low fills during crash
            self.market_simulator.slippage_model = "volume"
            self.market_simulator.market_impact_factor = 1.0 * slippage_factor  # Increased from 0.5 to 1.0
            self.market_simulator.min_fill_rate = max(0.1, fill_rate_min - 0.6)  # Reduced from 0.2 to 0.1
            
            # Modify data to simulate flash crash
            if self.data is not None:
                # Identify a point for the crash (25-75% through the dataset)
                crash_point = int(len(self.data) * np.random.uniform(0.25, 0.75))
                crash_duration = int(len(self.data) * 0.1)  # 10% of the dataset
                recovery_duration = int(len(self.data) * 0.2)  # 20% of the dataset
                
                for symbol in self.symbols or [""]:
                    prefix = f"{symbol}_$" if self.symbols else "$"
                    
                    # Get price columns
                    high_col = f"{prefix}high"
                    low_col = f"{prefix}low"
                    close_col = f"{prefix}close"
                    open_col = f"{prefix}open"
                    volume_col = f"{prefix}volume"
                    
                    if all(col in self.data.columns for col in [high_col, low_col, close_col, open_col]):
                        # Calculate crash magnitude (30-60% drop)
                        crash_magnitude = np.random.uniform(0.3, 0.6)
                        
                        # Apply flash crash to prices
                        for i in range(crash_point, min(crash_point + crash_duration, len(self.data))):
                            progress = (i - crash_point) / crash_duration
                            crash_factor = 1.0 - (crash_magnitude * progress)
                            
                            self.data.loc[i, high_col] *= crash_factor
                            self.data.loc[i, low_col] *= crash_factor
                            self.data.loc[i, close_col] *= crash_factor
                            self.data.loc[i, open_col] *= crash_factor
                            
                            # Increase volume during crash
                            if volume_col in self.data.columns:
                                self.data.loc[i, volume_col] *= (2.0 + 3.0 * progress)
                                
                        # Recovery phase
                        for i in range(crash_point + crash_duration, 
                                      min(crash_point + crash_duration + recovery_duration, len(self.data))):
                            progress = (i - (crash_point + crash_duration)) / recovery_duration
                            recovery_factor = (1.0 - crash_magnitude) + (crash_magnitude * 0.7 * progress)
                            
                            self.data.loc[i, high_col] /= (1.0 - crash_magnitude) / recovery_factor
                            self.data.loc[i, low_col] /= (1.0 - crash_magnitude) / recovery_factor
                            self.data.loc[i, close_col] /= (1.0 - crash_magnitude) / recovery_factor
                            self.data.loc[i, open_col] /= (1.0 - crash_magnitude) / recovery_factor
                            
                            # Elevated volume during recovery
                            if volume_col in self.data.columns:
                                self.data.loc[i, volume_col] *= (1.5 + 0.5 * (1.0 - progress))
                
            logger.info("Running flash crash scenario with extreme price movement and slippage")
            
        elif scenario_type == "perfect_execution":
            # Perfect execution (no slippage, full fills)
            self.market_simulator.slippage_model = "fixed"
            self.market_simulator.market_impact_factor = 0.0
            self.market_simulator.min_fill_rate = 1.0
            logger.info("Running perfect execution scenario with no slippage or partial fills")
            
        else:
            # Normal scenario: moderate parameters
            self.market_simulator.slippage_model = "volume"
            self.market_simulator.market_impact_factor = 0.1 * slippage_factor
            self.market_simulator.min_fill_rate = fill_rate_min
            logger.info(f"Running normal scenario with standard settings")
            
        try:
            # Run backtest with scenario settings
            if len(self.symbols) > 1:
                results = self.run_multi_asset(strategy, window_size, verbose=True)
            else:
                results = self.run(strategy, window_size, verbose=True)
                
            # Add scenario-specific information to results
            results['scenario_type'] = scenario_type
            results['scenario_settings'] = {
                'slippage_model': self.market_simulator.slippage_model,
                'market_impact_factor': self.market_simulator.market_impact_factor,
                'min_fill_rate': self.market_simulator.min_fill_rate,
            }
            
            return results
            
        finally:
            # Restore original settings
            self.market_simulator.slippage_model = original_slippage_model
            self.market_simulator.market_impact_factor = original_market_impact
            self.market_simulator.min_fill_rate = original_min_fill
            
            # Restore original data if modified
            if original_data is not None:
                self.data = original_data.copy() 