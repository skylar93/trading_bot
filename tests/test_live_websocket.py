import pytest
import asyncio
from data.utils.websocket_loader import WebSocketLoader


@pytest.mark.skip(reason="Websocket tests can cause freezing")
@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket data collection"""
    loader = WebSocketLoader(symbol="BTC/USDT")
    received_data = []

    async def test_callback(data):
        received_data.append(data)
        print(f"Received {data['type']}: {data['data']}")

    loader.add_callback(test_callback)

    # Start loader in background task
    task = asyncio.create_task(loader.start())

    try:
        # Wait for some data to arrive
        deadline = asyncio.get_event_loop().time() + 30  # 30 seconds timeout
        while asyncio.get_event_loop().time() < deadline:
            if len(received_data) >= 5:  # At least 5 updates
                break
            await asyncio.sleep(1)

        # Assert we received some data
        assert len(received_data) > 0, "No data received"

        # Check different types of data
        data_types = set(data["type"] for data in received_data)
        assert (
            "ticker" in data_types or "trade" in data_types
        ), "No market data received"

        # Check DataFrame
        df = loader.get_current_data()
        assert not df.empty, "No data in buffer"
        assert all(
            col in df.columns
            for col in ["open", "high", "low", "close", "volume"]
        )

    finally:
        await loader.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(test_websocket_connection())
