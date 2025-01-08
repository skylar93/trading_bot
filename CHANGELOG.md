# Changelog

## [1.1.0] - 2024-01-XX

### Fixed
- Position size management in RiskAwareBacktester now strictly respects limits
  * Added 0.999 buffer for float precision
  * Implemented exact position size calculation
  * Double verification of final position size
- PnL calculation now matches test requirements
  * Formula: size * (price - entry_price) - size * price * fee
  * Simplified entry price tracking
  * Separate handling for buy/sell trades

### Technical Details
- Position Size Control:
  * Initial conservative sizing at 90% of limit
  * Exact limit calculation when needed
  * Final position verification with buffer
- Risk Management:
  * Integration with RiskManager for signal processing
  * Portfolio VaR monitoring and position adjustment
  * Multi-asset correlation tracking
- Trade Execution:
  * Improved logging for debugging
  * Cleaner position tracking
  * Automatic cleanup of zero positions

### Test Coverage
- Added comprehensive tests for:
  * Position size limits (10% with 0.1% tolerance)
  * PnL calculation accuracy (0.01 tolerance)
  * Risk management integration
  * Trade execution logging

## [1.0.0] - 2024-01-XX

### Added
- Initial implementation of RiskAwareBacktester
- Basic position management and risk assessment
- Integration with RiskManager
- Multi-asset support with correlation tracking
- Portfolio value and PnL calculation 