# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified training pipeline for both single-agent and multi-agent training
- Environment factory for creating both types of environments from configuration
- Robust evaluation functions for performance assessment
- Training and evaluation metrics tracking
- Unified configuration system for both single-agent and multi-agent training
- Enhanced ConfigManager with dot notation access and validation
- Configuration history tracking for reproducibility
- Support for UW Hyak SLURM parameters in configuration
- Example training script demonstrating unified configuration
- Snapshot system for experiment configuration preservation
- Hyperparameter optimization configuration structure
- Ray Tune integration for distributed hyperparameter optimization
- Detailed architecture documentation for backtesting systems
- Position history tracking in BacktestEngine
- Detailed trade logging across all backtester implementations

### Changed
- Improved training loop with checkpointing and evaluation
- Standardized environment creation process
- Enhanced agent creation with better configuration handling
- Refactored training.py to use the unified configuration system
- Improved transaction cost handling in BacktestEngine
- Enhanced portfolio value calculation accuracy
- Standardized logging format across all components
- Updated docstrings to support automatic documentation generation
- Organized training files: moved hyperopt_ray.py to training/hyperopt directory
- Marked deprecated training files with 'deprecated_' prefix for clarity

### Fixed
- Position size validation in Backtester
- Dust position handling in multi-asset systems
- PnL calculation accuracy in trade execution
- Transaction cost consideration in position sizing
- Fixed MultiAgentTradingEnv constructor parameters in env_factory.py to match actual implementation
- Added backward compatibility for data path specification in hyperopt_ray.py - now supports both paths.data and data.data_path configurations
- Fixed error handling in hyperparameter optimization when data paths are not correctly specified

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