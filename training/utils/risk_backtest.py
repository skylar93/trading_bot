"""
Risk-aware backtesting system that integrates risk management with trading strategy.
Extends the base backtester with risk management capabilities.
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from training.backtest import Backtester
from risk.risk_manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)


class RiskAwareBacktester(Backtester):
    """Backtester with integrated risk management"""

    def __init__(
        self,
        data: pd.DataFrame,
        risk_config: Optional[RiskConfig] = None,
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
    ):
        """
        Initialize risk-aware backtester

        Args:
            data: DataFrame with OHLCV data (must have $ prefixed columns)
            risk_config: Risk management configuration
            initial_balance: Initial portfolio balance
            trading_fee: Trading fee as decimal
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing RiskAwareBacktester")
        
        # Initialize risk manager first
        self.risk_manager = RiskManager(risk_config or RiskConfig())
        self.trade_counter = 0  # For generating trade IDs
        self.positions = {}  # Dictionary to track positions by asset
        self.entry_prices = {}  # Dictionary to track entry prices by asset
        
        # Convert multi-asset data to single asset format for parent class
        if any("_$" in col for col in data.columns):
            # Store original multi-asset data
            self.full_data = data.copy()
            self.logger.info("Multi-asset data detected with columns: %s", data.columns.tolist())
            
            # Get the first asset's data for parent class
            first_asset = data.columns[0].split("_")[0]
            self.logger.info("Using %s as primary asset", first_asset)
            
            parent_data = pd.DataFrame({
                "$open": data[f"{first_asset}_$open"],
                "$high": data[f"{first_asset}_$high"],
                "$low": data[f"{first_asset}_$low"],
                "$close": data[f"{first_asset}_$close"],
                "$volume": data[f"{first_asset}_$volume"]
            }, index=data.index)
            data = parent_data
        else:
            parent_data = data
            self.full_data = data.copy()
            self.logger.info("Single-asset data detected with columns: %s", data.columns.tolist())
            
        # Then initialize parent class with single asset data
        super().__init__(parent_data, initial_balance, trading_fee)
        self.logger.info("Parent Backtester initialized with balance: %.2f", initial_balance)

    def reset(self):
        """Reset backtester and risk manager state"""
        super().reset()
        self.risk_manager.reset()
        self.trade_counter = 0
        self.positions = {}
        self.entry_prices = {}

    def get_position_value(self, asset: str) -> float:
        """Get current position value for an asset"""
        if asset not in self.positions:
            return 0.0
        
        # Get current price
        price_cols = [col for col in self.full_data.columns if col.startswith(f"{asset}_$close")]
        if not price_cols:
            return 0.0
            
        current_price = self.full_data[price_cols[0]].iloc[-1]
        return self.positions[asset] * current_price

    def calculate_volatility(self, window: int = 20) -> float:
        """Calculate local volatility

        Args:
            window: Rolling window size for volatility calculation

        Returns:
            Current volatility estimate
        """
        if len(self.data) < window:
            return 0.0

        returns = self.data["$close"].pct_change()
        volatility = returns.rolling(window).std().iloc[-1]
        return volatility if not np.isnan(volatility) else 0.0

    def get_current_leverage(self) -> float:
        """Calculate current leverage ratio"""
        total_position_value = 0.0
        for asset in self.positions:
            total_position_value += self.get_position_value(asset)

        portfolio_value = self.balance + total_position_value
        return abs(total_position_value / portfolio_value) if portfolio_value > 0 else 0.0

    def execute_trade(self, timestamp, action, price_data):
        """Execute trade with risk management"""
        # Calculate portfolio metrics
        def get_price_value(v):
            if isinstance(v, pd.Series):
                return float(v.iloc[0])
            return float(v)
            
        current_price = get_price_value(next(v for k, v in price_data.items() if k.endswith("_$close")))
        portfolio_value = self._calculate_portfolio_value(current_price)
        volatility = self.calculate_volatility()
        leverage = self.get_current_leverage()

        self.logger.debug("Portfolio metrics - Value: %.2f, Volatility: %.2f, Leverage: %.2f",
                        portfolio_value, volatility, leverage)

        # Extract asset from price data
        asset = next(k.split("_")[0] for k in price_data.keys() if k.endswith("_$close"))
        self.logger.debug("Processing asset: %s", asset)

        # Update correlation matrix with recent price data
        asset_prices = {}
        for col in self.full_data.columns:
            if col.endswith("_$close"):
                asset_name = col.split("_")[0]
                asset_prices[asset_name] = self.full_data[col]
        self.risk_manager.update_correlation_matrix(asset_prices)

        # Process through risk management
        risk_assessment = self.risk_manager.process_trade_signal(
            signal=pd.Timestamp(timestamp),
            portfolio_value=portfolio_value,
            price=current_price,
            volatility=volatility,
            current_leverage=leverage,
            current_positions=self.positions
        )

        self.logger.info("Risk assessment result: %s", risk_assessment)

        # Check if trade is allowed
        if not risk_assessment["allowed"]:
            self.logger.warning("Trade rejected by risk management: %s", risk_assessment.get("reason", "Unknown"))
            return {
                "timestamp": timestamp,
                "type": "none",
                "price": current_price,
                "size": 0.0,
                "portfolio_value": portfolio_value,
                "balance": self.balance,
                "position": 0.0,
                "pnl": 0.0,
                "action": "error",
                "reason": risk_assessment.get("reason", "Unknown")
            }

        # Adjust position size based on risk limits
        original_size = abs(action)
        risk_adjusted_size = min(
            original_size,
            risk_assessment["position_size"] / portfolio_value
        ) if portfolio_value > 0 else 0.0
        
        # Ensure position size does not exceed max limit
        max_position_size = self.risk_manager.config.max_position_size
        risk_adjusted_size = min(risk_adjusted_size, max_position_size)
        
        self.logger.info("Position sizing - Original: %.2f, Risk-adjusted: %.2f",
                        original_size, risk_adjusted_size)

        # Further adjust based on portfolio VaR
        portfolio_var = self.risk_manager.get_portfolio_var(
            self.positions,
            portfolio_value
        )
        if portfolio_var > self.risk_manager.config.portfolio_var_limit:
            var_scale = self.risk_manager.config.portfolio_var_limit / portfolio_var
            risk_adjusted_size *= var_scale
            self.logger.info("VaR adjustment - Portfolio VaR: %.4f, Scale: %.2f, Final size: %.2f",
                           portfolio_var, var_scale, risk_adjusted_size)

        # Maintain original direction
        risk_adjusted_action = np.sign(action) * risk_adjusted_size
        self.logger.info("Final action: %.2f", risk_adjusted_action)

        # Calculate trade size and costs
        trade_size = risk_adjusted_action * portfolio_value / current_price if current_price > 0 else 0.0
        
        # Ensure trade size does not exceed position limits
        max_trade_size = self.risk_manager.config.max_position_size * portfolio_value / current_price
        if abs(trade_size) > abs(max_trade_size):
            trade_size = np.sign(trade_size) * abs(max_trade_size)
            self.logger.info("Trade size adjusted to respect position limits: %.4f", trade_size)
        
        trading_cost = abs(trade_size * current_price * self.trading_fee)
        
        # Calculate PnL for existing position
        position_pnl = 0.0
        if asset in self.positions and abs(self.positions[asset]) > 1e-6:
            last_position = self.positions[asset]
            entry_price = self.entry_prices.get(asset, current_price)
            if risk_adjusted_action < 0:  # Sell
                # Calculate PnL only for the portion being sold
                sell_size = min(abs(trade_size), abs(last_position))
                position_pnl = sell_size * (current_price - entry_price) - \
                             sell_size * current_price * self.trading_fee
            else:
                position_pnl = last_position * (current_price - entry_price)
        
        # Calculate trade value
        trade_value = abs(trade_size * current_price)

        # Update portfolio value and balance based on trade type
        if risk_adjusted_action > 0:  # Buy
            trading_cost = trade_value * self.trading_fee
            new_balance = self.balance - trade_value - trading_cost
            trade_type = "buy"
            trade_cost = trade_value + trading_cost
            trade_revenue = None
            
            # Update position and entry price
            if asset not in self.positions:
                self.positions[asset] = 0
                self.entry_prices[asset] = current_price
            else:
                # Calculate weighted average entry price
                total_position = self.positions[asset] + trade_size
                if total_position > 0:
                    self.entry_prices[asset] = (
                        self.positions[asset] * self.entry_prices[asset] + trade_size * current_price
                    ) / total_position
            self.positions[asset] += trade_size
            
        elif risk_adjusted_action < 0:  # Sell
            trading_cost = trade_value * self.trading_fee
            trade_revenue = trade_value - trading_cost
            new_balance = self.balance + trade_revenue
            trade_type = "sell"
            trade_cost = None
            
            # Update position and entry price
            if asset not in self.positions:
                self.positions[asset] = 0
                self.entry_prices[asset] = current_price
            else:
                # Calculate weighted average entry price for remaining position
                total_position = self.positions[asset] + trade_size
                if total_position < 0:
                    self.entry_prices[asset] = (
                        self.positions[asset] * self.entry_prices[asset] + trade_size * current_price
                    ) / total_position
                self.positions[asset] += trade_size
            
        else:  # No trade
            trading_cost = 0
            new_balance = self.balance
            trade_type = "none"
            trade_cost = None
            trade_revenue = None

        # Clean up zero positions
        if asset in self.positions and abs(self.positions[asset]) < 1e-6:
            del self.positions[asset]
            del self.entry_prices[asset]

        # Execute trade with adjusted size
        if abs(risk_adjusted_action) > 1e-5:
            self.trade_counter += 1
            trade_id = f"trade_{self.trade_counter}"

            self.risk_manager.update_after_trade(
                trade_id=trade_id,
                timestamp=timestamp,
                entry_price=current_price,
                position_type="long" if action > 0 else "short",
            )
            
            self.logger.info("Updated positions: %s", self.positions)

        # Update portfolio value after trade
        new_portfolio_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(new_portfolio_value)
        self.peak_value = max(self.peak_value, new_portfolio_value)
        
        # Update instance variables
        self.balance = new_balance

        # Create trade result
        trade_result = {
            "timestamp": timestamp,
            "type": trade_type,
            "price": current_price,
            "size": abs(trade_size),  # Use absolute size for consistency
            "portfolio_value": new_portfolio_value,
            "balance": new_balance,
            "position": risk_adjusted_action,
            "pnl": position_pnl - trading_cost,
            "action": "trade"
        }

        # Add cost or revenue
        if trade_cost is not None:
            trade_result["cost"] = trade_cost
        if trade_revenue is not None:
            trade_result["revenue"] = trade_revenue

        # Add risk metrics
        trade_result["risk_metrics"] = {
            "volatility": volatility,
            "leverage": leverage,
            "drawdown": risk_assessment["current_drawdown"],
            "adjusted_size": risk_adjusted_size,
            "original_size": original_size,
            "trading_cost": trading_cost,
            "position_pnl": position_pnl
        }

        # Add to trades list
        self.trades.append(trade_result)

        self.logger.info("Trade execution result: %s", trade_result)
        return trade_result

    def run(
        self, agent: Any, window_size: int = 20, verbose: bool = True
    ) -> Dict:
        """
        Run backtest with risk management

        Args:
            agent: Trading agent
            window_size: Observation window size
            verbose: Whether to print progress

        Returns:
            Dictionary with backtest results including risk metrics
        """
        # Initialize results
        self.trades = []
        self.portfolio_values = []
        self.peak_value = self.initial_balance

        # Run backtest
        for i in range(window_size, len(self.data)):
            window_data = self.data.iloc[i - window_size : i].copy()
            current_data = pd.DataFrame(self.full_data.iloc[i]).T.copy()
            timestamp = current_data.index[0]

            # Get strategy action
            try:
                action = agent.get_action(window_data)
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

            # Determine if we're using single or multi-asset format
            is_multi_asset = any("_$" in col for col in current_data.columns)
            
            if is_multi_asset:
                # Extract asset from column names for multi-asset format
                asset = next(col.split("_")[0] for col in current_data.columns if col.endswith("_$close"))
                price_data = {
                    f"{asset}_$open": current_data[f"{asset}_$open"],
                    f"{asset}_$high": current_data[f"{asset}_$high"],
                    f"{asset}_$low": current_data[f"{asset}_$low"],
                    f"{asset}_$close": current_data[f"{asset}_$close"],
                    f"{asset}_$volume": current_data[f"{asset}_$volume"],
                }
            else:
                # Use simple format for single asset
                asset = "default"
                price_data = {
                    f"{asset}_$open": current_data["$open"],
                    f"{asset}_$high": current_data["$high"],
                    f"{asset}_$low": current_data["$low"],
                    f"{asset}_$close": current_data["$close"],
                    f"{asset}_$volume": current_data["$volume"],
                }

            # Execute trade
            trade_result = self.execute_trade(
                timestamp, action, price_data
            )

            # Store trade result
            if trade_result:
                self.trades.append(trade_result)

            # Update portfolio value
            close_col = f"{asset}_$close" if is_multi_asset else "$close"
            portfolio_value = trade_result.get("portfolio_value", self._calculate_portfolio_value(
                float(current_data[close_col].iloc[0])
            ))
            self.portfolio_values.append(portfolio_value)
            self.peak_value = max(self.peak_value, portfolio_value)

            if verbose and i % 100 == 0:
                self.logger.info(f"Progress: {i}/{len(self.data)}")

        # Calculate final results
        results = {
            "trades": self.trades,
            "portfolio_values": self.portfolio_values,
            "final_balance": self.balance,
            "peak_value": self.peak_value,
            "total_trades": len(self.trades),
            "profitable_trades": sum(1 for t in self.trades if t.get("pnl", 0) > 0),
            "total_pnl": sum(t.get("pnl", 0) for t in self.trades),
            "metrics": {}  # Add empty metrics dict to be filled by BacktestManager
        }

        # Add risk management summary
        risk_summary = {
            "avg_position_size": np.mean(
                [t.get("risk_metrics", {}).get("adjusted_size", 0)
                 for t in self.trades]
            ),
            "avg_volatility": np.mean(
                [t.get("risk_metrics", {}).get("volatility", 0)
                 for t in self.trades]
            ),
            "avg_leverage": np.mean(
                [t.get("risk_metrics", {}).get("leverage", 0)
                 for t in self.trades]
            ),
            "max_drawdown": max(
                [t.get("risk_metrics", {}).get("drawdown", 0)
                 for t in self.trades],
                default=0
            ),
            "portfolio_var": self.risk_manager.get_portfolio_var(
                self.positions,
                results["final_balance"]
            ),
            "avg_correlation": np.mean(
                self.risk_manager._correlation_matrix.values
                if self.risk_manager._correlation_matrix is not None
                else [0]
            )
        }
        results["risk_summary"] = risk_summary

        return results

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value including cash and positions.
        
        Args:
            current_price: Current price of the asset
            
        Returns:
            Total portfolio value
        """
        # Start with cash balance
        total_value = self.balance
        
        # Add value of all positions
        for asset, position in self.positions.items():
            # Get current price for this asset
            if isinstance(current_price, dict):
                asset_price = current_price.get(asset, 0.0)
            else:
                # If single price provided, use it for the current asset
                asset_price = current_price
            
            position_value = position * asset_price
            total_value += position_value
            
            self.logger.debug(
                "Portfolio calculation - Asset: %s, Position: %.4f, Price: %.2f, Value: %.2f",
                asset, position, asset_price, position_value
            )
        
        self.logger.info(
            "Total portfolio value: %.2f (Cash: %.2f, Positions: %s)",
            total_value, self.balance,
            {k: f"{v:.4f}" for k, v in self.positions.items()}
        )
        
        return total_value
