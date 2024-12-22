"""Test real-time data streaming"""

import unittest
import asyncio
import pandas as pd
from data.utils.realtime_data import RealtimeDataManager, TradingDataStream


class TestRealtimeData(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.data_manager = RealtimeDataManager(
            exchange_id="binance", symbols=["BTC/USDT"], timeframe="1m"
        )
        self.data_stream = TradingDataStream(
            exchange_id="binance", symbols=["BTC/USDT"], timeframe="1m"
        )

    def test_data_buffer(self):
        """Test data buffer functionality"""
        # Simulate data update
        data = {
            "timestamp": pd.Timestamp.now(),
            "open": 50000,
            "high": 51000,
            "low": 49000,
            "close": 50500,
            "volume": 100,
        }

        symbol = "BTC/USDT"
        self.data_manager.data_buffer[symbol] = data

        # Check data retrieval
        latest = self.data_manager.get_latest_data(symbol)
        self.assertEqual(latest["close"], 50500)

    async def test_data_streaming(self):
        """Test data streaming (mock)"""
        received_data = []

        async def mock_callback(symbol: str, data: dict):
            received_data.append(data)

        self.data_stream.data_manager.add_callback(mock_callback)

        # Simulate a few updates
        test_data = {
            "timestamp": pd.Timestamp.now(),
            "open": 50000,
            "high": 51000,
            "low": 49000,
            "close": 50500,
            "volume": 100,
        }

        await self.data_stream._on_data_update("BTC/USDT", test_data)

        # Check if data was processed
        self.assertTrue(len(received_data) > 0)

    def test_historical_data(self):
        """Test historical data buffer"""
        symbol = "BTC/USDT"

        # Generate some test data
        test_data = []
        base_time = pd.Timestamp.now()

        for i in range(10):
            test_data.append(
                {
                    "timestamp": base_time + pd.Timedelta(minutes=i),
                    "open": 50000 + i,
                    "high": 51000 + i,
                    "low": 49000 + i,
                    "close": 50500 + i,
                    "volume": 100 + i,
                }
            )

        # Add data to stream
        for data in test_data:
            asyncio.run(self.data_stream._on_data_update(symbol, data))

        # Check historical data
        hist_data = self.data_stream.get_historical_data(symbol)
        self.assertEqual(len(hist_data), 10)

        # Check lookback functionality
        lookback_data = self.data_stream.get_historical_data(
            symbol, lookback=5
        )
        self.assertEqual(len(lookback_data), 5)


if __name__ == "__main__":
    unittest.main()
