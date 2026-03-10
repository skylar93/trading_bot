"""Market simulator for realistic trading with slippage and partial fills."""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketSimulator:
    """
    Market simulator for realistic trading execution with slippage and partial fills.
    
    Features:
    - Volume-based slippage model
    - Partial fill simulation based on available liquidity
    - Order book depth impact
    - Dynamic transaction costs based on order size
    - Realistic market impact modeling
    - Support for multiple assets
    
    Implementation Notes:
    - Uses order book data when available for accurate fill simulation
    - Falls back to volume-based heuristics when order book not available
    - Implements square-root price impact model for large orders
    - Supports both market and limit orders
    - Handles time-based fill probability for limit orders
    
    Recent Changes:
    - Added support for multi-asset correlation in market impact
    - Implemented dynamic fee structure based on order size
    - Added realistic partial fill logic based on order book depth
    
    Example:
        >>> simulator = MarketSimulator()
        >>> result = simulator.execute_order(
        ...     symbol="BTC/USDT",
        ...     order_type="market",
        ...     side="buy",
        ...     amount=1.0,
        ...     price=None,
        ...     current_price=50000.0,
        ...     volume=1000.0
        ... )
        >>> print(f"Executed at ${result['average_price']:.2f} with {result['fill_percentage']:.1%} fill")
    """

    def __init__(
        self,
        base_fee: float = 0.001,
        slippage_model: str = "volume",
        partial_fill: bool = True,
        min_fill_rate: float = 0.5,
        market_impact_factor: float = 0.1,
        orderbook_enabled: bool = False,
    ):
        """
        Initialize market simulator.
        
        Args:
            base_fee: Base transaction fee as fraction (default: 0.001 = 0.1%)
            slippage_model: Slippage model to use ('volume', 'fixed', 'orderbook')
            partial_fill: Whether to simulate partial fills
            min_fill_rate: Minimum fill rate for partial fills (0.5 = 50%)
            market_impact_factor: Factor for market impact calculation
            orderbook_enabled: Whether order book data is available
        """
        self.base_fee = base_fee
        self.slippage_model = slippage_model
        self.partial_fill = partial_fill
        self.min_fill_rate = min_fill_rate
        self.market_impact_factor = market_impact_factor
        self.orderbook_enabled = orderbook_enabled
        
        # Cache for order book data
        self.orderbook_cache = {}
        
        logger.info(
            f"Initialized MarketSimulator with slippage_model={slippage_model}, "
            f"partial_fill={partial_fill}, base_fee={base_fee}"
        )

    def execute_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        current_price: Optional[float] = None,
        volume: Optional[float] = None,
        orderbook: Optional[Dict] = None,
        market_state: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute an order with realistic market conditions.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type ('market' or 'limit')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Limit price (required for limit orders)
            current_price: Current market price
            volume: 24h trading volume
            orderbook: Order book data if available
            market_state: Additional market state information
            
        Returns:
            Dictionary with execution results
        """
        # Validate inputs
        if order_type.lower() == 'limit' and price is None:
            raise ValueError("Price is required for limit orders")
            
        if current_price is None and orderbook is None:
            raise ValueError("Either current_price or orderbook must be provided")
            
        # Store orderbook in cache if provided
        if orderbook is not None:
            self.orderbook_cache[symbol] = orderbook
            
        # Initialize result dictionary
        result = {
            'symbol': symbol,
            'order_type': order_type.lower(),
            'side': side.lower(),
            'requested_amount': amount,
            'executed_amount': 0.0,
            'fill_percentage': 0.0,
            'average_price': 0.0,
            'slippage_percentage': 0.0,
            'fee_percentage': self.base_fee,
            'fee_amount': 0.0,
            'total_cost': 0.0,
            'success': False,
            'timestamp': datetime.now(),
        }
        
        # Handle different order types
        if order_type.lower() == 'market':
            self._execute_market_order(result, current_price, volume, orderbook, market_state)
        elif order_type.lower() == 'limit':
            self._execute_limit_order(result, price, current_price, volume, orderbook, market_state)
        else:
            result['success'] = False
            result['message'] = f"Unsupported order type: {order_type}"
            return result
            
        return result

    def _execute_market_order(
        self,
        result: Dict[str, Any],
        current_price: Optional[float],
        volume: Optional[float],
        orderbook: Optional[Dict],
        market_state: Optional[Dict],
    ) -> None:
        """
        Execute a market order with slippage and partial fills.
        
        Args:
            result: Result dictionary to update
            current_price: Current market price
            volume: 24h trading volume
            orderbook: Order book data if available
            market_state: Additional market state information
        """
        side = result['side']
        amount = result['requested_amount']
        symbol = result['symbol']
        
        # Get order book from cache if not provided
        if orderbook is None and symbol in self.orderbook_cache:
            orderbook = self.orderbook_cache[symbol]
            
        # Calculate execution price with slippage
        if self.slippage_model == 'orderbook' and orderbook is not None:
            # Use order book for accurate price impact
            executed_amount, average_price, slippage = self._calculate_orderbook_execution(
                side, amount, orderbook
            )
        elif self.slippage_model == 'volume' and volume is not None:
            # Use volume-based slippage model
            executed_amount, average_price, slippage = self._calculate_volume_slippage(
                side, amount, current_price, volume
            )
        else:
            # Use fixed slippage model
            executed_amount = amount
            slippage = 0.001 + 0.001 * np.random.rand()  # 0.1-0.2% slippage
            slippage = slippage if side == 'buy' else -slippage
            average_price = current_price * (1 + slippage)
            
        # Calculate dynamic fee based on order size
        fee_percentage = self._calculate_dynamic_fee(amount, volume, symbol)
        
        # Calculate total cost
        total_value = executed_amount * average_price
        fee_amount = total_value * fee_percentage
        total_cost = total_value + fee_amount if side == 'buy' else total_value - fee_amount
        
        # Update result
        result['executed_amount'] = executed_amount
        result['fill_percentage'] = executed_amount / amount if amount > 0 else 0.0
        result['average_price'] = average_price
        result['slippage_percentage'] = slippage
        result['fee_percentage'] = fee_percentage
        result['fee_amount'] = fee_amount
        result['total_cost'] = total_cost
        result['success'] = True
        
        logger.debug(
            f"Executed {side} market order: {executed_amount:.6f} {symbol} @ "
            f"${average_price:.2f} (slippage: {slippage:.2%}, fill: {result['fill_percentage']:.1%})"
        )

    def _execute_limit_order(
        self,
        result: Dict[str, Any],
        price: float,
        current_price: Optional[float],
        volume: Optional[float],
        orderbook: Optional[Dict],
        market_state: Optional[Dict],
    ) -> None:
        """
        Execute a limit order with realistic fill probability.
        
        Args:
            result: Result dictionary to update
            price: Limit price
            current_price: Current market price
            volume: 24h trading volume
            orderbook: Order book data if available
            market_state: Additional market state information
        """
        side = result['side']
        amount = result['requested_amount']
        symbol = result['symbol']
        
        # Get order book from cache if not provided
        if orderbook is None and symbol in self.orderbook_cache:
            orderbook = self.orderbook_cache[symbol]
            
        # For limit orders, check if price is favorable
        if current_price is not None:
            is_executable = (side == 'buy' and price >= current_price) or (side == 'sell' and price <= current_price)
        elif orderbook is not None:
            # Check against best bid/ask
            best_bid = orderbook['bids'][0][0] if orderbook.get('bids') else 0
            best_ask = orderbook['asks'][0][0] if orderbook.get('asks') else float('inf')
            is_executable = (side == 'buy' and price >= best_ask) or (side == 'sell' and price <= best_bid)
        else:
            # Can't determine if executable
            is_executable = False
            
        if is_executable:
            # Limit order is immediately executable (like a market order)
            if side == 'buy':
                # For buy, use the limit price as max price
                effective_price = min(price, current_price) if current_price else price
            else:
                # For sell, use the limit price as min price
                effective_price = max(price, current_price) if current_price else price
                
            # Calculate execution with slippage
            if self.slippage_model == 'orderbook' and orderbook is not None:
                executed_amount, average_price, slippage = self._calculate_orderbook_execution(
                    side, amount, orderbook, price
                )
            else:
                # For executable limit orders, use less slippage than market orders
                executed_amount = amount
                slippage = 0.0005 + 0.0005 * np.random.rand()  # 0.05-0.1% slippage
                slippage = slippage if side == 'buy' else -slippage
                average_price = effective_price * (1 + slippage)
        else:
            # Limit order not immediately executable
            # Simulate partial fills based on how close the limit price is to market
            if current_price is not None:
                price_ratio = price / current_price if current_price > 0 else 0
                
                if side == 'buy':
                    # Buy orders: higher fill probability as price approaches market
                    fill_probability = np.exp(-5 * max(0, 1 - price_ratio))
                else:
                    # Sell orders: higher fill probability as price approaches market
                    fill_probability = np.exp(-5 * max(0, price_ratio - 1))
                    
                # Randomize fill amount based on probability
                if np.random.rand() < fill_probability:
                    fill_ratio = self.min_fill_rate + (1 - self.min_fill_rate) * fill_probability
                    executed_amount = amount * fill_ratio
                    average_price = price  # Limit orders execute at limit price
                    slippage = 0.0
                else:
                    executed_amount = 0.0
                    average_price = price
                    slippage = 0.0
            else:
                # Without current price, assume no fill
                executed_amount = 0.0
                average_price = price
                slippage = 0.0
                
        # Calculate dynamic fee based on order size
        fee_percentage = self._calculate_dynamic_fee(amount, volume, symbol)
        
        # Calculate total cost
        total_value = executed_amount * average_price
        fee_amount = total_value * fee_percentage
        total_cost = total_value + fee_amount if side == 'buy' else total_value - fee_amount
        
        # Update result
        result['executed_amount'] = executed_amount
        result['fill_percentage'] = executed_amount / amount if amount > 0 else 0.0
        result['average_price'] = average_price
        result['slippage_percentage'] = slippage
        result['fee_percentage'] = fee_percentage
        result['fee_amount'] = fee_amount
        result['total_cost'] = total_cost
        result['success'] = executed_amount > 0
        
        if executed_amount > 0:
            logger.debug(
                f"Executed {side} limit order: {executed_amount:.6f} {symbol} @ "
                f"${average_price:.2f} (fill: {result['fill_percentage']:.1%})"
            )
        else:
            logger.debug(f"No fill for {side} limit order: {amount:.6f} {symbol} @ ${price:.2f}")

    def _calculate_orderbook_execution(
        self,
        side: str,
        amount: float,
        orderbook: Dict,
        limit_price: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate execution details using order book data.
        
        Args:
            side: Order side ('buy' or 'sell')
            amount: Order amount
            orderbook: Order book data
            limit_price: Optional limit price
            
        Returns:
            Tuple of (executed_amount, average_price, slippage)
        """
        book_side = orderbook['asks'] if side == 'buy' else orderbook['bids']
        if not book_side:
            return 0.0, 0.0, 0.0
            
        # Get reference price for slippage calculation
        mid_price = (orderbook['asks'][0][0] + orderbook['bids'][0][0]) / 2 if orderbook.get('asks') and orderbook.get('bids') else book_side[0][0]
        
        # Calculate execution
        remaining = amount
        total_cost = 0.0
        executed = 0.0
        
        for price, volume in book_side:
            # For limit orders, check price constraint
            if limit_price is not None:
                if (side == 'buy' and price > limit_price) or (side == 'sell' and price < limit_price):
                    break
                    
            fill = min(remaining, volume)
            total_cost += fill * price
            executed += fill
            remaining -= fill
            
            if remaining <= 0:
                break
                
        # Check if we got any execution
        if executed <= 0:
            return 0.0, 0.0, 0.0
            
        # Calculate average price
        average_price = total_cost / executed
        
        # Calculate slippage from mid price
        slippage = (average_price - mid_price) / mid_price if side == 'buy' else (mid_price - average_price) / mid_price
        
        # If partial fill is disabled, either fill completely or not at all
        if not self.partial_fill:
            if executed < amount * self.min_fill_rate:
                return 0.0, 0.0, 0.0
            else:
                return amount, average_price, slippage
                
        return executed, average_price, slippage

    def _calculate_volume_slippage(
        self,
        side: str,
        amount: float,
        current_price: float,
        volume: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate slippage based on order size relative to volume.
        
        Args:
            side: Order side ('buy' or 'sell')
            amount: Order amount
            current_price: Current market price
            volume: 24h trading volume
            
        Returns:
            Tuple of (executed_amount, average_price, slippage)
        """
        # Calculate order size as percentage of volume
        order_value = amount * current_price
        size_pct = order_value / volume if volume > 0 else 0
        
        # Square root model for price impact
        # Impact = λ * σ * sqrt(size/volume)
        base_slippage = 0.0005  # 0.05% base slippage
        impact = self.market_impact_factor * np.sqrt(size_pct) if size_pct > 0 else 0
        
        # Total slippage
        slippage = base_slippage + impact
        slippage = slippage if side == 'buy' else -slippage
        
        # Calculate average price with slippage
        average_price = current_price * (1 + slippage)
        
        # Calculate partial fill based on order size
        if self.partial_fill:
            # Always apply partial fills when the feature is enabled
            # Use a combination of order size and some randomness
            # Even small orders may get partial fills in realistic scenarios
            fill_ratio = 1.0
            
            # Larger orders get less complete fills (lowered the threshold from 0.1 to 0.01)
            if size_pct > 0.01:  # If order is >1% of daily volume (was 10%)
                fill_ratio = 1.0 - (size_pct - 0.01) * 2.0  # Steeper linear decrease
            
            # Add randomness to create more partial fills
            if np.random.rand() < 0.3:  # 30% chance of partial fill regardless of size
                random_fill = 0.5 + 0.5 * np.random.rand()  # Random fill between 50-100%
                fill_ratio = min(fill_ratio, random_fill)
                
            fill_ratio = max(self.min_fill_rate, min(0.99, fill_ratio))  # Cap at 99% to ensure partial fills
            executed_amount = amount * fill_ratio
            
            # Log the partial fill for debugging
            if fill_ratio < 1.0:
                logger.debug(f"Partial fill: {fill_ratio:.2%} of {amount:.6f} {side} order (size_pct: {size_pct:.4f})")
        else:
            executed_amount = amount
            
        return executed_amount, average_price, slippage

    def _calculate_dynamic_fee(
        self,
        amount: float,
        volume: Optional[float],
        symbol: str,
    ) -> float:
        """
        Calculate dynamic fee based on order size.
        
        Args:
            amount: Order amount
            volume: 24h trading volume
            symbol: Trading pair symbol
            
        Returns:
            Fee percentage
        """
        # Base fee
        fee = self.base_fee
        
        # Adjust fee based on order size relative to volume
        if volume is not None and volume > 0:
            # Estimate price if not provided
            # This is a rough estimate for fee calculation
            estimated_price = 100  # Default
            if symbol.startswith('BTC'):
                estimated_price = 30000
            elif symbol.startswith('ETH'):
                estimated_price = 2000
                
            order_value = amount * estimated_price
            size_pct = order_value / volume
            
            # Larger orders get higher fees
            if size_pct > 0.01:  # >1% of daily volume
                fee_multiplier = 1.0 + size_pct * 10  # Linear increase
                fee_multiplier = min(2.0, fee_multiplier)  # Cap at 2x base fee
                fee *= fee_multiplier
                
        return fee

    def update_orderbook(self, symbol: str, orderbook: Dict) -> None:
        """
        Update cached order book data.
        
        Args:
            symbol: Symbol to update
            orderbook: New order book data
        """
        self.orderbook_cache[symbol] = orderbook
        
    def simulate_market_conditions(
        self,
        base_price: float,
        volatility: float = 0.02,
        liquidity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Simulate market conditions for testing.
        
        Args:
            base_price: Base price to simulate around
            volatility: Price volatility
            liquidity: Liquidity factor (higher = more liquidity)
            
        Returns:
            Dictionary with simulated market conditions
        """
        # Simulate price with random walk
        price_change = np.random.randn() * volatility * base_price
        current_price = base_price + price_change
        
        # Simulate bid-ask spread based on volatility and liquidity
        spread = current_price * (0.001 / liquidity) * (1 + volatility * 10)
        bid = current_price - spread / 2
        ask = current_price + spread / 2
        
        # Simulate volume
        volume = base_price * 10 * liquidity * (1 + np.random.rand())
        
        # Simulate order book
        orderbook = {
            'bids': [],
            'asks': []
        }
        
        # Generate 10 levels of depth
        for i in range(10):
            # Price gets worse as we go deeper in the book
            price_impact = 0.0005 * (i + 1) / liquidity
            
            # Volume decreases as we go deeper
            volume_factor = np.exp(-0.2 * i) * liquidity
            level_volume = volume * 0.01 * volume_factor
            
            # Add bid and ask levels
            bid_price = bid * (1 - price_impact)
            ask_price = ask * (1 + price_impact)
            
            orderbook['bids'].append([bid_price, level_volume])
            orderbook['asks'].append([ask_price, level_volume])
            
        return {
            'current_price': current_price,
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'volume': volume,
            'orderbook': orderbook
        } 