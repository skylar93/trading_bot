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
from .risk_management import RiskManager, RiskConfig

logger = logging.getLogger(__name__)

class RiskAwareBacktester(Backtester):
    """Backtester with integrated risk management"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 risk_config: Optional[RiskConfig] = None,
                 initial_balance: float = 10000.0,
                 trading_fee: float = 0.001):
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
        
        # Then initialize parent class
        super().__init__(data, initial_balance, trading_fee)
        
    def reset(self):
        """Reset backtester and risk manager state"""
        super().reset()
        self.risk_manager.reset()
        self.trade_counter = 0
        
    def calculate_volatility(self, window: int = 20) -> float:
        """Calculate local volatility
        
        Args:
            window: Rolling window size for volatility calculation
            
        Returns:
            Current volatility estimate
        """
        if len(self.data) < window:
            return 0.0
            
        returns = self.data['$close'].pct_change()
        volatility = returns.rolling(window).std().iloc[-1]
        return volatility if not np.isnan(volatility) else 0.0
        
    def get_current_leverage(self) -> float:
        """Calculate current leverage ratio"""
        if self.position == 0:
            return 0.0
            
        total_position_value = self.position * self.data['$close'].iloc[-1]
        portfolio_value = self.balance + total_position_value
        
        return abs(total_position_value / portfolio_value) if portfolio_value > 0 else 0.0
        
    def execute_trade(self,
                     timestamp: datetime,
                     action: float,
                     price_data: Dict[str, float]) -> Dict:
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
        portfolio_value = self.balance + (self.position * price_data['$close'])
        volatility = self.calculate_volatility()
        leverage = self.get_current_leverage()
        
        # Process through risk management
        risk_assessment = self.risk_manager.process_trade_signal(
            timestamp=pd.Timestamp(timestamp),
            portfolio_value=portfolio_value,
            price=price_data['$close'],
            volatility=volatility,
            current_leverage=leverage
        )
        
        # Check if trade is allowed
        if not risk_assessment['allowed']:
            logger.info(f"Trade rejected by risk management: {risk_assessment['reason']}")
            result = super().execute_trade(timestamp, 0, price_data)
            result['portfolio_value'] = self.balance + (self.position * price_data['$close'])
            return result
            
        # Adjust position size based on risk limits
        original_size = abs(action)
        risk_adjusted_size = min(
            original_size,
            risk_assessment['position_size'] / portfolio_value
        )
        
        # Maintain original direction
        risk_adjusted_action = np.sign(action) * risk_adjusted_size
        
        # Execute trade with adjusted size
        trade_result = super().execute_trade(
            timestamp, risk_adjusted_action, price_data
        )
        trade_result['portfolio_value'] = self.balance + (self.position * price_data['$close'])
        
        # Update risk management if trade was executed
        if abs(risk_adjusted_action) > 1e-5:
            self.trade_counter += 1
            trade_id = f"trade_{self.trade_counter}"
            
            self.risk_manager.update_after_trade(
                trade_id=trade_id,
                timestamp=pd.Timestamp(timestamp),
                entry_price=price_data['$close'],
                position_type='long' if action > 0 else 'short'
            )
            
            # Add risk metrics to result
            trade_result.update({
                'risk_metrics': {
                    'volatility': volatility,
                    'leverage': leverage,
                    'drawdown': risk_assessment['current_drawdown'],
                    'adjusted_size': risk_adjusted_size,
                    'original_size': original_size
                }
            })
            
        return trade_result
        
    def run(self,
            agent: Any,
            window_size: int = 20,
            verbose: bool = True) -> Dict:
        """
        Run backtest with risk management
        
        Args:
            agent: Trading agent
            window_size: Observation window size
            verbose: Whether to print progress
            
        Returns:
            Dictionary with backtest results including risk metrics
        """
        results = super().run(agent, window_size, verbose)
        
        # Add risk management summary
        risk_summary = {
            'avg_position_size': np.mean([t.get('risk_metrics', {}).get('adjusted_size', 0) 
                                        for t in results['trades']]),
            'avg_volatility': np.mean([t.get('risk_metrics', {}).get('volatility', 0)
                                     for t in results['trades']]),
            'max_leverage': max([t.get('risk_metrics', {}).get('leverage', 0)
                               for t in results['trades']]),
            'max_drawdown': max([t.get('risk_metrics', {}).get('drawdown', 0)
                               for t in results['trades']])
        }
        
        results['risk_summary'] = risk_summary
        return results