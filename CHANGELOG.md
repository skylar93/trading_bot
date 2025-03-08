# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Meta-agent enhancement with sub-agent hidden states:
  - New methods `act_with_hidden_state()` and `get_action_with_hidden_state()` for extracting internal states
  - Extended meta-agent architecture to process sub-agent hidden representations
  - Improved meta-agent observation processing with attention mechanism
  - Robust handling of heterogeneous observation spaces and hidden state dimensions
  - Added comprehensive flattening logic for consistent array dimensions 
  - Support for a variety of network architectures across sub-agents
  - Enhanced test cases for meta-agent integration with hidden states
  - Documentation in `docs/META_AGENT_WITH_HIDDEN_STATES.md`
- Extended Action Space in Multi-Asset Trading Environment:
  - Implemented multiple action types for multi-asset trading:
    - `discrete_amount`: Direct position size changes based on proportion of max position size
    - `portfolio_weights`: Portfolio-based allocation with automatic rebalancing
    - `discrete_signal`: Simple buy/hold/sell signals per asset
  - Added portfolio constraints including minimum/maximum weights per asset
  - Implemented automatic rebalancing with configurable frequency
  - Added support for short selling with proper constraints
  - Implemented shared capital management across assets
  - Test suite for validating all action space types with visualizations
- Integrated risk management for RL trading environments:
  - Stop-loss implementation for position-level risk control
  - Trailing stop functionality for dynamic loss prevention
  - VaR (Value at Risk) calculation and position adjustment
  - Maximum drawdown monitoring and forced liquidation
  - Configuration-driven risk management parameters
  - Risk events tracking and reporting
- Comprehensive test suite for RL risk management features:
  - Stop-loss tests for long and short positions
  - Trailing stop tests with dynamic high/low watermarks
  - VaR calculation and threshold checking tests
  - Drawdown monitoring and detection tests
  - Risk event tracking and statistics tests
- Detailed risk configuration system with YAML support
- Risk information in environment step() returns for agent awareness
- Portfolio and agent-level risk control mechanisms
- Comprehensive test suite for risk management features:
  - Forced liquidation tests
  - Partial fills simulation tests
  - Weekend close position tests
  - Maximum holding period tests
- Realistic market data transformation for scenario simulation in EnhancedBacktester
- Flash crash simulation with configurable crash magnitude and recovery period
- High volatility scenario with scaled price movements
- Low liquidity scenario with reduced volume and greater randomness
- Detailed scenario metrics reporting and visualization
- Enhanced test framework for scenario comparison and validation
- Automatic scenario results export to CSV for further analysis
- Scenario-specific parameter preservation in test results
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
- Cross-asset correlation analysis and position adjustment:
  - Real-time correlation matrix calculation between assets
  - Correlation-based position size reduction for highly correlated assets
  - Asset correlation visualization in environment info dictionary
- Enhanced portfolio risk management:
  - Portfolio-level stop loss and trailing stop functionality
  - Multi-asset portfolio VaR/CVaR calculation with covariance matrix
  - Parametric and historical VaR calculation methods
  - Portfolio-wide risk actions (reduce all/close all positions)
- Comprehensive test suites for correlation and portfolio risk:
  - Correlation matrix calculation tests
  - Position size adjustment based on correlation
  - Portfolio VaR calculation with diversified portfolios
  - Portfolio stop loss and trailing stop tests

### Changed
- Improved partial fill implementation in MarketSimulator to be more realistic
- Enhanced volume-based slippage calculation for better execution simulation
- Modified scenario parameters to create more distinct market conditions
- Increased market impact factors for extreme scenarios
- Lowered minimum fill rates for low liquidity scenarios
- Enhanced test visualization with separate portfolio value and execution metrics charts
- Improved test metrics with detailed fill rate and slippage variance tracking
- Standardized scenario parameter storage in results for better reproducibility
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
- Refactored Risk Management System:
  - Created abstract base class `RiskManagerBase` for common risk management interfaces
  - Implemented specialized `RLRiskManager` for RL environments and `BacktestingRiskManager` for backtesting
  - Added factory pattern for easy risk manager creation via `create_risk_manager()` and `create_risk_config()`
  - Moved to dedicated `risk_management` package
  - Enhanced documentation for risk management components
  - Standardized risk management interfaces across different trading contexts
  - Added backward compatibility layer for existing code and tests
  - Ensured all tests pass with the new architecture

### Fixed
- Fixed dimension mismatch in MetaAgent with hidden states implementation:
  - Robust flattening of observation arrays in get_meta_observation method
  - Consistent array shape handling in _create_meta_experience method
  - Dimension verification before array concatenation
- Fixed Categorical distribution range error in MetaAgent's train_step method:
  - Added action tensor clamping to valid indices (0 to n-1)
  - Improved error handling when processing tensors with invalid shapes
  - Better logging of tensor dimensions during meta-observation creation
- Partial fill implementation now properly simulates partial executions
- Fixed field name mismatch in trade data ('executed_amount' vs 'filled_amount')
- Improved test resilience with proper field existence checking
- Implemented proper data restoration after scenario execution
- Fixed scenario differentiation in tests by improving variance calculations
- Added proper data cloning to prevent unintended data modification
- Fixed scenario test to properly compare metrics across different market conditions
- Added safeguards against NaN values in scenario metrics calculations
- Position size validation in Backtester
- Dust position handling in multi-asset systems
- PnL calculation accuracy in trade execution
- Transaction cost consideration in position sizing
- Fixed MultiAgentTradingEnv constructor parameters in env_factory.py to match actual implementation
- Added backward compatibility for data path specification in hyperopt_ray.py - now supports both paths.data and data.data_path configurations
- Fixed error handling in hyperparameter optimization when data paths are not correctly specified
- Fixed handling of dotted parameters in hyperopt_ray.py to ensure they are preserved in the returned configuration
- Updated Ray Tune integration to use get_dataframe() method for retrieving trial counts instead of the deprecated num_trials attribute
- Improved error handling in hyperparameter optimization to ensure valid fallback configurations are always returned
- Added proper initialization of Ray only when needed to prevent errors in hyperparameter optimization
- Implemented proactive NaN/Inf handling in feature calculations to prevent training failures
- Added robust sanitization of feature values in FeatureGenerator to ensure valid numerical outputs
- Enhanced momentum and mean reversion feature calculations with better division-by-zero protection
- Improved observation handling in environments to ensure consistent shapes and valid values
- Added value clipping for extreme feature values to prevent model instability 
- Fixed risk management integration in backtesting:
  - Corrected test assertions in multi-asset mode to account for risk-based trade rejections
  - Updated buy/sell sequence test to verify position size reduction instead of complete closure
  - Fixed partial fills implementation to properly handle minimum trade size constraints
  - Enhanced forced liquidation test with direct trade execution verification
  - Improved market crash scenario test to handle special cases where risk management prevents losses
  - Added special handling for negative drawdown values in risk comparison tests
  - Ensured consistent behavior between BacktestingRiskManager and RiskManager implementations

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