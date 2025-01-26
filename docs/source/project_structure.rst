Project Structure
================

The trading bot project follows a modular architecture with clear separation of concerns. Here's a detailed overview of the project structure:

Core Modules
-----------

training/
^^^^^^^^^

The training module contains all components related to agent training and evaluation:

* **agents/** - Reinforcement learning agents implementation
    - PPOAgent: Main trading agent using PPO algorithm
    - BaseNetwork: Neural network architecture definitions
    - Memory: Experience replay buffer implementation

* **environments/** - Trading environment implementations
    - TradingEnv: OpenAI Gym compatible environment
    - MultiAssetEnv: Environment for multiple asset trading

* **utils/** - Training utilities
    - ray_manager.py: Ray distributed computing integration
    - recovery_manager.py: Checkpoint and recovery handling
    - risk_management.py: Risk metrics during training
    - state_manager.py: Agent state management
    - trainer.py: Training loop implementation

trading/
^^^^^^^^

The trading module handles live and paper trading execution:

* **live/** - Live trading implementation
    - LiveTradingEnvironment: Real exchange integration
    - OrderManager: Order execution and tracking
    - NetworkManager: Connection handling and retry logic

* **paper/** - Paper trading simulation
    - PaperTradingEnvironment: Simulated trading
    - MockExchange: Exchange simulation

* **data/** - Data handling and processing
    - DataManager: OHLCV data processing
    - StreamManager: Real-time data streaming
    - IndicatorManager: Technical indicator calculation

risk/
^^^^^

Risk management and monitoring components:

* risk_manager.py: Core risk management implementation
    - Position sizing
    - Stop-loss management
    - Portfolio risk metrics
    - Multi-asset correlation tracking

tests/
^^^^^^

Comprehensive test suite:

* **test_agents/** - Agent and training tests
* **test_trading/** - Trading system tests
* **test_risk/** - Risk management tests

scripts/
^^^^^^^^

Executable scripts for different operations:

* train.py: Agent training script
* backtest.py: Backtesting script
* live_trade.py: Live trading script

Configuration
------------

* **requirements.txt**: Python package dependencies
* **.cursorrules**: Development and documentation standards
* **CHANGELOG.md**: Version history and changes
* **README.md**: Project overview and quick start

Documentation
------------

* **docs/**
    - architecture/: Detailed design documents
    - api/: API reference documentation
    - examples/: Usage examples
    - source/: Sphinx documentation source

Development Guidelines
--------------------

1. Follow PEP 8 style guide
2. Add docstrings to all classes and methods
3. Update CHANGELOG.md with significant changes
4. Write tests for new features
5. Document API changes

Recent Changes
-------------

See CHANGELOG.md for detailed version history and recent updates. 