"""
Risk-aware backtesting system that integrates risk management with trading strategy.
Extends the base backtester with risk management capabilities.
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from ..backtest import Backtester
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
        # Initialize risk manager first
        self.risk_manager = RiskManager(risk_config or RiskConfig())
        self.trade_counter = 0  # For generating trade IDs
        self.positions = {}  # Dictionary to track positions by asset
        
        # Convert multi-asset data to single asset format for parent class
        if any("_$" in col for col in data.columns):
            # Store original multi-asset data
            self.full_data = data.copy()
            
            # Get the first asset's data for parent class
            first_asset = data.columns[0].split("_")[0]
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
            
        # Then initialize parent class with single asset data
        super().__init__(parent_data, initial_balance, trading_fee)

    def reset(self):
        """Reset backtester and risk manager state"""
        super().reset()
        self.risk_manager.reset()
        self.trade_counter = 0
        self.positions = {}

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
        if self.position == 0:
            return 0.0

        total_position_value = self.position * self.data["$close"].iloc[-1]
        portfolio_value = self.balance + total_position_value

        return (
            abs(total_position_value / portfolio_value)
            if portfolio_value > 0
            else 0.0
        )

    def execute_trade(
        self, timestamp: datetime, action: float, price_data: Dict[str, float]
    ) -> Dict:
        """
        Execute trade with risk management checks

        Args:
            timestamp: Current timestamp
            action: Trading action (-1 to 1)
            price_data: Dictionary with current prices

        Returns:
            Dictionary with trade results
        """
        # Skip if no action
        if abs(action) < 1e-5:
            return super().execute_trade(timestamp, 0, price_data)

        # Get current state
        portfolio_value = self.balance + sum(self.get_position_value(asset) for asset in self.positions)
        volatility = self.calculate_volatility()
        leverage = self.get_current_leverage()

        # Extract asset from price data
        asset = next(k.split("_")[0] for k in price_data.keys() if k.endswith("_$close"))

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
            price=price_data[f"{asset}_$close"],
            volatility=volatility,
            current_leverage=leverage,
            current_positions=self.positions
        )

        # Check if trade is allowed
        if not risk_assessment["allowed"]:
            logger.info(
                f"Trade rejected by risk management: {risk_assessment['reason']}"
            )
            result = super().execute_trade(timestamp, 0, price_data)
            result["portfolio_value"] = portfolio_value
            return result

        # Adjust position size based on risk limits
        original_size = abs(action)
        risk_adjusted_size = min(
            original_size,
            risk_assessment["position_size"] / portfolio_value
        )

        # Further adjust based on portfolio VaR
        portfolio_var = self.risk_manager.get_portfolio_var(
            self.positions,
            portfolio_value
        )
        if portfolio_var > self.risk_manager.config.portfolio_var_limit:
            var_scale = self.risk_manager.config.portfolio_var_limit / portfolio_var
            risk_adjusted_size *= var_scale

        # Maintain original direction
        risk_adjusted_action = np.sign(action) * risk_adjusted_size

        # Execute trade with adjusted size
        trade_result = super().execute_trade(
            timestamp, risk_adjusted_action, price_data
        )
        trade_result["portfolio_value"] = portfolio_value

        # Update positions if trade was executed
        if abs(risk_adjusted_action) > 1e-5:
            self.trade_counter += 1
            trade_id = f"trade_{self.trade_counter}"
            
            # Update position tracking
            asset = next(k.split("_")[0] for k in price_data.keys() if k.endswith("_$close"))
            if asset not in self.positions:
                self.positions[asset] = 0
            self.positions[asset] += trade_result.get("size", 0)
            
            # Clean up zero positions
            if abs(self.positions[asset]) < 1e-6:
                del self.positions[asset]

            self.risk_manager.update_after_trade(
                trade_id=trade_id,
                timestamp=pd.Timestamp(timestamp),
                entry_price=price_data[f"{asset}_$close"],
                position_type="long" if action > 0 else "short",
            )

            # Add risk metrics to result
            trade_result.update({
                "risk_metrics": {
                    "volatility": volatility,
                    "leverage": leverage,
                    "drawdown": risk_assessment["current_drawdown"],
                    "adjusted_size": risk_adjusted_size,
                    "original_size": original_size,
                    "portfolio_var": portfolio_var
                }
            })

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

            # Extract asset from column names
            asset = next(col.split("_")[0] for col in current_data.columns if col.endswith("_$close"))

            # Execute trade with asset-prefixed price data
            price_data = {
                f"{asset}_$open": current_data[f"{asset}_$open"],
                f"{asset}_$high": current_data[f"{asset}_$high"],
                f"{asset}_$low": current_data[f"{asset}_$low"],
                f"{asset}_$close": current_data[f"{asset}_$close"],
                f"{asset}_$volume": current_data[f"{asset}_$volume"],
            }
            trade_result = self.execute_trade(
                timestamp, action, price_data
            )

            # Store trade result
            if trade_result:
                self.trades.append(trade_result)

            # Update portfolio value
            portfolio_value = trade_result.get("portfolio_value", self._calculate_portfolio_value(
                current_data[f"{asset}_$close"]
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
