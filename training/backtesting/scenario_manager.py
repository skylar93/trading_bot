"""
Scenario Manager Module
=====================

This module provides a unified interface for applying different market scenarios
to OHLCV data, whether real or synthetic.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from .scenario import (
    generate_flash_crash_data_deterministic,
    generate_low_liquidity_data,
    apply_flash_crash_to_real_data,
    apply_low_liquidity_to_real_data
)

logger = logging.getLogger(__name__)

class ScenarioManager:
    """Manages the application of different market scenarios to OHLCV data.
    
    This class provides a clean interface for transforming raw market data
    into scenario-specific datasets (e.g., flash crash, low liquidity).
    
    Features:
    - Unified interface for all scenario transformations
    - Validation of scenario parameters
    - Logging of scenario applications
    
    Implementation Notes:
    - Each scenario function should preserve the DataFrame index
    - All scenario functions should maintain OHLCV column naming ($-prefix)
    - Input validation ensures required parameters are present
    """
    
    REQUIRED_COLUMNS = ["$open", "$high", "$low", "$close", "$volume"]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def apply_scenario(
        self,
        raw_data: pd.DataFrame,
        scenario_type: str,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply a scenario to the raw data.
        
        Args:
            raw_data: The base data to apply the scenario to
            scenario_type: Type of scenario ("none", "flash_crash", "low_liquidity")
            params: Scenario-specific parameters
            
        Returns:
            Modified DataFrame with scenario applied
            
        Raises:
            ValueError: If required columns are missing or scenario type is invalid
        """
        try:
            # Validate required columns
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in raw_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Normalize scenario type to lowercase with underscores
            scenario_type = scenario_type.lower().replace(" ", "_")
            
            if scenario_type == "none":
                return raw_data
            elif scenario_type == "flash_crash":
                return self._apply_flash_crash(raw_data, params)
            elif scenario_type == "low_liquidity":
                return self._apply_low_liquidity(raw_data, params)
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
                
        except Exception as e:
            self.logger.error(f"Error applying scenario: {str(e)}", exc_info=True)
            raise
            
    def _apply_flash_crash(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply flash crash scenario to the data."""
        required_params = ["crash_size"]
        self._validate_params(params, required_params)
        
        # Convert percentage params to decimals
        crash_size = params["crash_size"] / 100.0 if params["crash_size"] > 1 else params["crash_size"]
        
        # Calculate crash position if given as percentage
        crash_at = params.get("crash_at")
        if crash_at is not None and crash_at > 1:  # Assume it's a percentage
            crash_at = int(len(data) * crash_at / 100)
            
        return apply_flash_crash_to_real_data(
            base_data=data,
            crash_size=crash_size,
            crash_at=crash_at,
            crash_duration=params.get("crash_duration", 5),
            recovery_duration=params.get("recovery_duration", 10)
        )
        
    def _apply_low_liquidity(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply low liquidity scenario to the data."""
        required_params = ["volume_reduction"]
        self._validate_params(params, required_params)
        
        # Convert percentage params to decimals
        volume_reduction = params["volume_reduction"] / 100.0 if params["volume_reduction"] > 1 else params["volume_reduction"]
        
        # Calculate start position if given as percentage
        low_liq_start = params.get("low_liq_start")
        if low_liq_start is not None and low_liq_start > 1:  # Assume it's a percentage
            low_liq_start = int(len(data) * low_liq_start / 100)
            
        return apply_low_liquidity_to_real_data(
            base_data=data,
            volume_reduction=volume_reduction,
            low_liq_start=low_liq_start,
            low_liq_length=params.get("low_liq_length", 100)
        )
        
    def _validate_params(
        self,
        params: Dict[str, Any],
        required_params: list
    ) -> None:
        """Validate that all required parameters are present."""
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}") 