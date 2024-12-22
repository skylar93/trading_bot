import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import logging

from data.utils.data_loader import DataLoader
from data.utils.feature_generator import FeatureGenerator

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="1h")
    df = pd.DataFrame(
        {
            "$open": np.random.randn(len(dates)) * 100 + 1000,
            "$high": np.random.randn(len(dates)) * 100 + 1000,
            "$low": np.random.randn(len(dates)) * 100 + 1000,
            "$close": np.random.randn(len(dates)) * 100 + 1000,
            "$volume": np.abs(np.random.randn(len(dates)) * 1000 + 5000),
        },
        index=dates,
    )

    # Ensure high is highest and low is lowest
    df["$high"] = df[["$open", "$high", "$low", "$close"]].max(axis=1)
    df["$low"] = df[["$open", "$high", "$low", "$close"]].min(axis=1)
    return df


def test_data_loader():
    """Test DataLoader functionality"""
    loader = DataLoader(
        exchange_id="binance", symbol="BTC/USDT", timeframe="1h"
    )

    # Test data fetching for a small time window
    df = loader.fetch_data("2023-01-01", "2023-01-02")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(
        col in df.columns
        for col in ["$open", "$high", "$low", "$close", "$volume"]
    )
    assert isinstance(df.index, pd.DatetimeIndex)

    # Test data validation
    assert not df.isnull().any().any()
    assert (df["$high"] >= df["$low"]).all()
    assert (df["$volume"] >= 0).all()


def test_feature_generator(sample_data):
    """Test FeatureGenerator functionality"""
    generator = FeatureGenerator()
    logger.info("Starting feature generator test")

    # Test feature generation
    df_features = generator.generate_features(sample_data)

    # Verify all technical indicators
    expected_features = [
        "RSI",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "EMA_5",
        "EMA_10",
        "EMA_20",
        "BB_upper",
        "BB_middle",
        "BB_lower",
        "ATR",
        "volume_ma",
        "PVT",
    ]

    # Check if expected features exist
    missing_features = [
        f for f in expected_features if f not in df_features.columns
    ]
    assert len(missing_features) == 0, f"Missing features: {missing_features}"

    # Check for NaN values
    nan_columns = df_features.columns[df_features.isnull().any()].tolist()
    assert len(nan_columns) == 0, f"Found NaN values in columns: {nan_columns}"

    # Basic feature validation
    logger.info("Validating feature values")
    assert (df_features["RSI"] >= 0).all() and (
        df_features["RSI"] <= 100
    ).all(), "RSI values out of range"
    assert (
        df_features["BB_upper"] >= df_features["BB_lower"]
    ).all(), "Invalid Bollinger Bands"
    assert (df_features["volume_ma"] >= 0).all(), "Negative volume MA"

    # Test Moving Averages
    for window in [5, 10, 20]:
        assert f"SMA_{window}" in df_features.columns, f"Missing SMA_{window}"
        assert f"EMA_{window}" in df_features.columns, f"Missing EMA_{window}"

        # Verify MA values are between min and max of price
        price_min = df_features["$close"].min()
        price_max = df_features["$close"].max()
        assert (df_features[f"SMA_{window}"] >= price_min).all() and (
            df_features[f"SMA_{window}"] <= price_max
        ).all(), f"SMA_{window} values out of price range"
        assert (df_features[f"EMA_{window}"] >= price_min).all() and (
            df_features[f"EMA_{window}"] <= price_max
        ).all(), f"EMA_{window} values out of price range"


@pytest.fixture
def empty_df():
    """Create empty DataFrame for testing"""
    return pd.DataFrame()


@pytest.fixture
def invalid_df():
    """Create DataFrame with missing columns for testing"""
    return pd.DataFrame(
        {"$close": [100, 101, 102], "$volume": [1000, 1100, 1200]}
    )


@pytest.fixture
def nan_df(sample_data):
    """Create DataFrame with NaN values for testing"""
    df = sample_data.copy()
    df.loc[df.index[0], "$close"] = np.nan
    return df


def test_feature_generator_error_handling(empty_df, invalid_df, nan_df):
    """Test FeatureGenerator error handling"""
    generator = FeatureGenerator()

    # Test with empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        generator.generate_features(empty_df)

    # Test with missing columns
    with pytest.raises(ValueError, match="Missing required columns"):
        generator.generate_features(invalid_df)

    # Test with NaN values
    features_with_nan = generator.generate_features(nan_df)
    assert (
        not features_with_nan.isnull().any().any()
    ), "NaN values should be handled"


def test_simple_pipeline():
    """Test complete data pipeline with a small sample"""
    logger.info("Starting simple pipeline test")

    # 1. Load data
    loader = DataLoader(
        exchange_id="binance", symbol="BTC/USDT", timeframe="1h"
    )
    df = loader.fetch_data("2023-01-01", "2023-01-02")

    # Verify required columns exist
    required_columns = {"$open", "$high", "$low", "$close", "$volume"}
    missing_columns = required_columns - set(df.columns)
    assert (
        len(missing_columns) == 0
    ), f"Missing required columns: {missing_columns}"

    # 2. Generate features
    generator = FeatureGenerator()
    df_features = generator.generate_features(df)

    # Basic validations
    assert not df_features.empty, "Feature DataFrame is empty"
    assert len(df_features.columns) > len(
        df.columns
    ), "No new features generated"
    assert not df_features.isnull().any().any(), "Found NaN values in features"

    logger.info(f"Generated features: {list(df_features.columns)}")
    logger.debug(f"Sample data:\n{df_features.head()}")

    return df_features


if __name__ == "__main__":
    logger.info("Running basic pipeline test...")
    df = test_simple_pipeline()
    logger.info("Test completed successfully!")
