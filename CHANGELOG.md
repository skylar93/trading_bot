# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation system using Sphinx
- Detailed architecture documentation for backtesting systems
- Position history tracking in BacktestEngine
- Detailed trade logging across all backtester implementations

### Changed
- Improved transaction cost handling in BacktestEngine
- Enhanced portfolio value calculation accuracy
- Standardized logging format across all components
- Updated docstrings to support automatic documentation generation

### Fixed
- Position size validation in Backtester
- Dust position handling in multi-asset systems
- PnL calculation accuracy in trade execution
- Transaction cost consideration in position sizing

## [0.1.0] - 2024-01-08

### Added
- Initial implementation of Backtester for single-asset trading
- BacktestEngine implementation for multi-asset trading
- RiskAwareBacktester with advanced risk management
- Basic logging system for debugging and monitoring
- Performance metrics calculation (Sharpe, Sortino, Max DD)
- Trade execution with transaction cost consideration
- Position management with size limits
- Portfolio value tracking and history

### Changed
- Standardized OHLCV column naming with '$' prefix
- Improved position sizing logic
- Enhanced trade execution validation

### Fixed
- Initial bugs in PnL calculation
- Position tracking accuracy
- Transaction cost handling
- Trade history logging 