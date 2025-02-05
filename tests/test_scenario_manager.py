"""
Tests for the ScenarioManager class
"""

import pytest
import pandas as pd
import numpy as np
from training.backtesting.scenario_manager import ScenarioManager

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    data = pd.DataFrame({
        "$open": np.linspace(100, 120, 100),
        "$high": np.linspace(101, 121, 100),
        "$low": np.linspace(99, 119, 100),
        "$close": np.linspace(100, 120, 100),
        "$volume": np.full(100, 1000000.0)
    }, index=timestamps)
    return data

@pytest.fixture
def scenario_manager():
    """Create ScenarioManager instance."""
    return ScenarioManager()

def test_apply_scenario_none(sample_data, scenario_manager):
    """Test that 'none' scenario returns unmodified data."""
    result = scenario_manager.apply_scenario(sample_data, "none", {})
    pd.testing.assert_frame_equal(result, sample_data)

def test_apply_flash_crash(sample_data, scenario_manager):
    """Test flash crash scenario application."""
    params = {
        "crash_size": 30,  # 30% crash
        "crash_at": 50,    # At 50% of data
    }
    
    result = scenario_manager.apply_scenario(sample_data, "flash_crash", params)
    
    # Verify basic properties
    assert len(result) == len(sample_data)
    assert all(col in result.columns for col in sample_data.columns)
    
    # Verify crash occurred
    crash_idx = len(sample_data) // 2
    pre_crash_price = result["$close"].iloc[crash_idx - 1]
    crash_price = result["$close"].iloc[crash_idx]
    crash_size = (pre_crash_price - crash_price) / pre_crash_price
    
    assert 0.25 <= crash_size <= 0.35  # Allow some flexibility due to volatility

def test_apply_low_liquidity(sample_data, scenario_manager):
    """Test low liquidity scenario application."""
    params = {
        "volume_reduction": 80,  # 80% volume reduction
        "low_liq_start": 30,     # Start at 30% of data
        "low_liq_length": 20     # Duration in periods
    }
    
    result = scenario_manager.apply_scenario(sample_data, "low_liquidity", params)
    
    # Verify basic properties
    assert len(result) == len(sample_data)
    assert all(col in result.columns for col in sample_data.columns)
    
    # Verify volume reduction
    start_idx = int(len(sample_data) * 0.3)
    low_liq_volume = result["$volume"].iloc[start_idx:start_idx + 20].mean()
    normal_volume = sample_data["$volume"].mean()
    reduction = (normal_volume - low_liq_volume) / normal_volume
    
    assert 0.75 <= reduction <= 0.85  # Allow some flexibility

def test_invalid_scenario_type(sample_data, scenario_manager):
    """Test that invalid scenario type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scenario type"):
        scenario_manager.apply_scenario(sample_data, "invalid_scenario", {})

def test_missing_required_params(sample_data, scenario_manager):
    """Test that missing required parameters raise ValueError."""
    with pytest.raises(ValueError, match="Missing required parameters"):
        scenario_manager.apply_scenario(sample_data, "flash_crash", {})

def test_missing_columns(scenario_manager):
    """Test that missing required columns raise ValueError."""
    invalid_data = pd.DataFrame({
        "open": [100],  # Missing $ prefix
        "close": [101]
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        scenario_manager.apply_scenario(invalid_data, "flash_crash", {"crash_size": 30})

def test_percentage_conversion(sample_data, scenario_manager):
    """Test that percentages are properly converted to decimals."""
    params = {
        "crash_size": 30,    # Should be converted to 0.3
        "crash_at": 50,      # Should be converted to index
    }
    
    result = scenario_manager.apply_scenario(sample_data, "flash_crash", params)
    
    # Verify crash occurred at approximately the middle
    mid_point = len(result) // 2
    crash_range = result["$close"].iloc[mid_point-2:mid_point+2]
    assert crash_range.min() < sample_data["$close"].iloc[mid_point] * 0.8  # At least 20% drop somewhere in range 