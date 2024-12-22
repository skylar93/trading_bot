import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from envs.paper_trading_env import (
    PaperTradingEnvironment,
    OrderType,
    Order,
    OrderStatus,
)


@pytest.fixture
def paper_env():
    """Create paper trading environment for testing"""
    env = PaperTradingEnvironment(
        symbol="BTC/USDT",
        initial_balance=10000.0,
        trading_fee=0.001,
        window_size=20,
        test_mode=True,
    )
    return env


@pytest.mark.asyncio
async def test_initialization(paper_env):
    """Test environment initialization"""
    await paper_env.initialize()
    try:
        assert paper_env.initial_balance == 10000.0
        assert paper_env.balance == 10000.0
        assert paper_env.position == 0
        assert len(paper_env.trades) == 0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_position_sizing(paper_env):
    """Test position sizing logic"""
    await paper_env.initialize()
    try:
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0}}
        )

        size = await paper_env.calculate_position_size(0.5)
        assert (
            0
            <= size
            <= paper_env.initial_balance * paper_env.max_position_size
        )
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_slippage_simulation(paper_env):
    """Test slippage simulation"""
    await paper_env.initialize()
    try:
        base_price = 50000.0
        slipped_price = paper_env.simulate_slippage(base_price)
        assert abs(slipped_price - base_price) / base_price < 0.01
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_risk_management(paper_env):
    """Test risk management features"""
    await paper_env.initialize()
    try:
        # Set up initial position
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0}}
        )

        # Execute buy order
        result = await paper_env.execute_order(0.5)
        assert result["success"]
        assert paper_env.position > 0
        assert paper_env.stop_loss_price > 0

        # Verify stop loss is below entry price
        assert paper_env.stop_loss_price < paper_env.entry_price

        # Test stop loss trigger
        await paper_env.update_market_data(
            {
                "type": "ticker",
                "data": {
                    "close": paper_env.stop_loss_price * 0.99
                },  # Price below stop loss
            }
        )

        # Position should be closed
        assert paper_env.position == 0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_metrics_calculation(paper_env):
    """Test performance metrics calculation"""
    await paper_env.initialize()
    try:
        for price in [50000.0, 51000.0, 49000.0]:
            await paper_env.update_market_data(
                {"type": "ticker", "data": {"close": price}}
            )

        metrics = await paper_env.get_metrics()
        assert "max_drawdown" in metrics
        assert "portfolio_value" in metrics
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_market_data_handling(paper_env):
    """Test market data processing"""
    await paper_env.initialize()
    try:
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0, "volume": 1.5}}
        )

        assert paper_env.current_price == 50000.0
        assert len(paper_env.price_history) > 0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_trade_execution_limits(paper_env):
    """Test trade execution limits"""
    await paper_env.initialize()
    try:
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0}}
        )

        result = await paper_env.execute_order(2.0)
        assert not result["success"]
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_logging(paper_env):
    """Test logging functionality"""
    await paper_env.initialize()
    try:
        assert paper_env.logger is not None
        await paper_env.log_info("Test log message")
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_stress_conditions(paper_env):
    """Test behavior under stress conditions"""
    await paper_env.initialize()
    try:
        for i in range(100):
            await paper_env.update_market_data(
                {"type": "ticker", "data": {"close": 50000.0 + i * 100}}
            )

        assert paper_env.current_price > 0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_limit_order(paper_env):
    """Test limit order execution"""
    await paper_env.initialize()
    try:
        # Set initial market price
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0}}
        )

        order = Order(
            order_type=OrderType.LIMIT, side="buy", quantity=0.1, price=50000.0
        )
        await paper_env.place_order(order)

        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 49900.0}}
        )

        filled_orders = await paper_env.get_filled_orders()
        assert len(filled_orders) > 0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_stop_limit_order(paper_env):
    """Test stop-limit order execution"""
    await paper_env.initialize()
    try:
        # Set initial market price
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 48000.0}}
        )
        print(f"Initial price: 48000.0")

        # Set initial position
        paper_env.position = 1.0  # Set 1 BTC position
        paper_env.entry_price = 48000.0
        print(
            f"Initial position: {paper_env.position} BTC at {paper_env.entry_price}"
        )

        # Place stop-limit sell order
        order = Order(
            order_type=OrderType.STOP_LIMIT,
            side="sell",
            quantity=0.1,
            price=49000.0,  # Limit price
            stop_price=48500.0,  # Stop trigger price
        )
        result = await paper_env.place_order(order)
        assert result[
            "success"
        ], f"Failed to place stop-limit order: {result.get('message', '')}"
        print(
            f"Placed stop-limit order: stop={order.stop_price}, limit={order.price}"
        )

        # Price moves up to trigger stop price
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 48600.0}}
        )
        print(f"Price moved to trigger: 48600.0")

        # Get active orders after trigger
        active_orders = await paper_env.get_active_orders()
        print(f"Active orders after trigger: {len(active_orders)}")
        for o in active_orders:
            print(
                f"  Order: type={o.order_type.value}, side={o.side}, triggered={o.triggered}"
            )

        # Price moves up to limit price
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 49100.0}}
        )
        print(f"Price moved to limit: 49100.0")

        # Check if order was filled
        filled_orders = await paper_env.get_filled_orders()
        print(f"Filled orders: {len(filled_orders)}")
        for o in filled_orders:
            print(
                f"  Order: type={o.order_type.value}, side={o.side}, status={o.status.value}, price={o.filled_price}"
            )

        assert len(filled_orders) > 0, "Stop-limit order was not filled"
        assert filled_orders[0].status == OrderStatus.FILLED
        assert filled_orders[0].filled_price >= order.price
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_trailing_stop_order(paper_env):
    """Test trailing stop order execution"""
    await paper_env.initialize()
    try:
        # Set initial market price
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0}}
        )

        order = Order(
            order_type=OrderType.TRAILING_STOP,
            side="sell",
            quantity=0.1,
            trailing_pct=0.02,
        )
        await paper_env.place_order(order)

        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 51000.0}}
        )

        active_orders = await paper_env.get_active_orders()
        assert active_orders[0].trailing_price > 50000.0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_iceberg_order(paper_env):
    """Test iceberg order execution"""
    await paper_env.initialize()
    try:
        # Set initial market price
        await paper_env.update_market_data(
            {"type": "ticker", "data": {"close": 50000.0}}
        )

        order = Order(
            order_type=OrderType.ICEBERG,
            side="buy",
            quantity=1.0,
            price=50000.0,
            iceberg_qty=0.1,
        )
        await paper_env.place_order(order)

        for _ in range(10):
            await paper_env.update_market_data(
                {"type": "ticker", "data": {"close": 49900.0}}
            )

        filled_orders = await paper_env.get_filled_orders()
        assert sum(order.filled_qty for order in filled_orders) > 0
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_order_expiration(paper_env):
    """Test order expiration"""
    await paper_env.initialize()
    try:
        # Place an expired order
        order = Order(
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=0.1,
            price=50000.0,
            expire_time=datetime.now() - timedelta(minutes=1),
        )
        result = await paper_env.place_order(order)
        assert result["success"], "Failed to place order"

        # Check initial state
        active_orders = await paper_env.get_active_orders()
        assert (
            len(active_orders) == 1
        ), "Order should be in active orders initially"

        # Check expired orders
        expired_count = await paper_env.check_expired_orders()
        assert expired_count > 0, "Should have expired orders"

        # Verify order lists
        active_orders = await paper_env.get_active_orders()
        expired_orders = await paper_env.get_expired_orders()
        assert len(active_orders) == 0, "No orders should be active"
        assert len(expired_orders) == 1, "Order should be in expired orders"
        assert expired_orders[0].status == OrderStatus.EXPIRED
    finally:
        await paper_env.cleanup()


@pytest.mark.asyncio
async def test_order_cancellation(paper_env):
    """Test order cancellation"""
    await paper_env.initialize()
    try:
        # Place an order
        order = Order(
            order_type=OrderType.LIMIT, side="buy", quantity=0.1, price=50000.0
        )
        result = await paper_env.place_order(order)
        assert result["success"], "Failed to place order"

        # Check initial state
        active_orders = await paper_env.get_active_orders()
        assert (
            len(active_orders) == 1
        ), "Order should be in active orders initially"

        # Cancel the order
        result = await paper_env.cancel_order(order.id)
        assert result["success"], "Order cancellation should succeed"

        # Verify order lists
        active_orders = await paper_env.get_active_orders()
        cancelled_orders = await paper_env.get_cancelled_orders()
        assert len(active_orders) == 0, "No orders should be active"
        assert (
            len(cancelled_orders) == 1
        ), "Order should be in cancelled orders"
        assert cancelled_orders[0].status == OrderStatus.CANCELLED
    finally:
        await paper_env.cleanup()


if __name__ == "__main__":
    pytest.main(["-v", "test_paper_trading.py"])
