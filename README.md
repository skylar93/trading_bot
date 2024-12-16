# Trading Bot Project

## Latest Achievements 🎯

- **Risk-Aware Backtesting System Implemented (2024-12-13)**:
  - Portfolio returns: +1.42%
  - Sharpe Ratio: 0.43
  - Sortino Ratio: 0.60
  - Max Drawdown: -4.26%
  - Tested with flash crash and low liquidity scenarios
  - Dynamic position sizing and stop-loss mechanisms integrated
  - Initiated hyperparameter optimization pipeline using Ray Tune
  - Unit tests for `hyperopt_env.py`, `hyperopt_agent.py`, `hyperopt_tuner.py` completed
  - Optimization pipeline separated for independent execution

- **Advanced Backtesting & Visualization**:
  - Scenario-based testing (flash crash, low liquidity)
  - Comprehensive metrics (Sharpe, Sortino, MDD)
  - Portfolio visualization and trade analysis
  - Real-time monitoring of simulated results
  - Risk management metrics and scenarios integrated

- **Hyperparameter Optimization**:
  - Modular structure under `training/hyperopt/`
  - Ray Tune integration for learning rate, batch size, network architecture tuning
  - MLflow for result tracking and reproducibility
  - Dedicated script `scripts/run_hyperopt.py` for full hyperparameter search

- **Resource Optimization System (New)**:
  - Under `training/monitoring/` (e.g., `metrics_collector.py`, `performance_analyzer.py`, `worker_manager.py`, `optimizer.py`)
  - Ray Actor model for distributed processing
  - Dynamic resource scaling (2 to 8 workers)
  - Automated optimization of batch sizes, worker allocation, and resource usage
  - Real-time performance monitoring (batch processing time, memory, GPU utilization)

- **Real-Time Trading Preparation**:
  - CCXT WebSocket integration for live data streaming
  - Paper trading environment for real-time testing (`training/paper_trading.py`)
  - `websocket_loader.py` for asynchronous data handling
  - Environment modifications in `trading_env.py` to support live trading scenarios

## Core Architecture Components

### A. Data Pipeline Layer

- **Frameworks**: TA-Lib + ccxt
- **Pipeline**:
  ```
  Raw Data (ccxt) → TA-Lib Pipeline → Feature Store → Training Data
  ```
- **Directory Structure**:
  ```
  /data
    /raw
    /processed
    /features
    /utils/data_loader.py
    /utils/feature_generator.py
    /utils/websocket_loader.py
  ```

### B. Training and Inference Layer

- **Frameworks**: RLlib + PPO Agent
- **Architecture**:
  ```
  Market Environment
    └── PPO Agent (Currently Active)
  ```
- **Key Files**:
  - `train.py` (Stable training pipeline)
  - `ppo_agent.py` (Core PPO agent)
  - `trading_env.py` (Stable single-agent environment)
  - `multi_agent_env.py` (Multi-agent environment)
  - `evaluation.py`, `backtest.py`, `advanced_backtest.py` (Evaluation & Backtesting)
  - `hyperopt/` (Hyperparameter tuning modules)
  
### C. Web Interface Layer

- **Frameworks**: FastAPI + Streamlit
- **Features**:
  ```
  Web UI (Streamlit)
    ├── Model Selection
    ├── Parameter Tuning
    ├── Live Monitoring
    └── Performance Visualization

  API Server (FastAPI)
    ├── Training Control
    ├── Model Management
    └── Data Pipeline Control
  ```
- **File**:
  - `deployment/web_interface/app.py`

## Completed Features

1. **Data Pipeline** ✅
   - ccxt for data ingestion
   - 44 technical indicators via TA-Lib
   - Data caching for efficiency

2. **Reinforcement Learning** ✅
   - PPO agent implementation
   - Stable training pipeline (`train.py`)
   - MLflow tracking integrated
   - Successful training runs validated

3. **Backtesting System** ✅
   - Performance metrics: Sharpe, Sortino, MDD
   - Portfolio visualization and trade analysis
   - Real-time monitoring of simulation results
   - `backtest.py`, `advanced_backtest.py` for scenario testing

4. **Visualization Tools** ✅
   - Portfolio value tracking
   - Returns distribution
   - Drawdown analysis
   - Risk metric visualizations (`visualization.py`)

5. **Advanced Scenario Testing** ✅
   - Flash crash scenarios (sudden price drops, partial recovery)
   - Low liquidity conditions (reduced volumes, high volatility)
   - Metrics for survival rate, recovery time, execution feasibility

6. **Risk-Aware Backtesting** ✅
   - Dynamic position sizing
   - Stop-loss mechanisms & drawdown limits
   - Real-time leverage monitoring
   - Enhanced risk metrics in backtesting results

7. **Hyperparameter Optimization** (Ongoing) ✅
   - `training/hyperopt/` for modular tuning environment
   - `hyperopt_env.py`, `hyperopt_agent.py`, `hyperopt_tuner.py` implemented
   - Ray Tune integration for parameter search
   - MLflow logging for reproducibility
   - `scripts/run_hyperopt.py` for dedicated optimization runs

8. **Real-Time Trading Preparation**
   - CCXT WebSocket integration for live data streams
   - `trading_env.py` adapted for on-the-fly action application
   - `paper_trading.py` for sandbox testing of live strategies

9. **Resource Optimization & Monitoring System**
   - Ray Actor model for distributed processing
   - Worker management and scaling (2 to 8 workers)
   - Real-time performance metrics (batch time, memory, GPU)
   - Automated optimization of batch sizes and resource allocation

## Integrated Architecture Workflow

```
Data Sources (Exchanges) → ccxt → Raw Data Storage
                                  ↓
                              TA-Lib Pipeline
                                  ↓
                              Feature Store
                                  ↓
                          RLlib Training Pipeline
                                  ↓
                          Model Store (MLflow)
                                  ↓
                       FastAPI Backend + Streamlit UI
```

## Development Guidelines

- **MVP First**: Focus on core functionality, test components individually, avoid unnecessary complexity.
- **Stable Core**: `train.py`, `trading_env.py`, `ppo_agent.py` are stable; extend rather than modify.
- **Testing and CI/CD**: Always test (`test_training.py`) before deploying changes. Consider GitHub Actions for automation.
- **Modular Design**: Add new features via separate modules (inheritance, composition) without altering stable core.
- **Documentation**: Thoroughly document changes and new modules.

For all development work, please refer to our [Development Guidelines](DEVELOPMENT_GUIDELINES.md). This document contains:

- Code standards and best practices
- Testing requirements
- Data validation rules
- Debugging procedures
- Common issues and solutions

Following these guidelines ensures consistent code quality and makes debugging easier.

## Running the Project

1. Setup
   ```bash
   cd Users/skylar/Desktop/trading_bot
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Test Core System
   ```bash
   python test_training.py
   ```

3. Launch Web UI
   ```bash
   streamlit run deployment/web_interface/app.py
   ```

4. Execute Hyperparameter Optimization
   ```bash
   python scripts/run_hyperopt.py
   ```
   - Monitor results via MLflow UI.

## Key Files & Components

```
trading_bot/
├─ config/
│  ├─ default_config.yaml         # Core configuration
│  ├─ env_settings.yaml           # Environment settings
│  └─ model_config.yaml           # Model architecture settings
│
├─ data/
│  ├─ raw/
│  │  └─ crypto/                  # Raw cryptocurrency data
│  ├─ processed/
│  │  ├─ features/               # Processed features
│  │  └─ cache/                  # Cached computations
│  └─ utils/
│     ├─ data_loader.py          # Data loading utilities
│     ├─ feature_generator.py    # Feature generation
│     └─ validation.py           # Data validation
│
├─ agents/
│  ├─ base/
│  │  ├─ base_agent.py          # Abstract base agent
│  │  └─ agent_factory.py       # Agent creation factory
│  ├─ strategies/
│  │  └─ ppo_agent.py          # PPO implementation
│  └─ models/
│     ├─ policy_network.py      # Policy network architectures
│     └─ value_network.py       # Value network architectures
│
├─ envs/
│  ├─ base_env.py              # Base trading environment
│  └─ trading_env.py           # Main trading environment
│
├─ training/
│  ├─ train.py                 # Single-agent training
│  ├─ train_multi_agent.py     # Multi-agent training
│  ├─ evaluation.py            # Performance evaluation
│  ├─ backtest.py             # Basic backtesting
│  ├─ advanced_backtest.py     # Advanced scenario testing
│  └─ utils/
│     ├─ metrics.py            # Performance metrics
│     └─ callbacks.py          # Training callbacks
│
├─ deployment/
│  ├─ web_interface/
│  │  ├─ app.py               # Streamlit main app
│  │  ├─ pages/              # Multi-page components
│  │  │  ├─ training.py      # Training interface
│  │  │  ├─ backtest.py      # Backtesting interface
│  │  │  └─ monitor.py       # Monitoring interface
│  │  └─ utils/
│  │     └─ state_management.py  # Session state management
│  └─ api/
│     ├─ main.py             # FastAPI main app
│     └─ routers/
│        ├─ training.py      # Training endpoints
│        └─ data.py          # Data endpoints
│
└─ tests/
   ├─ test_environment.py    # Environment tests
   ├─ test_agents.py        # Agent tests
   └─ test_training.py      # Training pipeline tests
```

## Dependencies and Roles
```
graph TD
    A[environment.py] --> B[agents.py]
    B --> C[train.py]
    B --> D[backtest.py]
    D --> E[evaluation.py]
    F[async_feature_generator.py] --> G[app.py]
    H[data_management.py] --> G
```

## Visual Results

- **Key Visualizations**:
  - Portfolio value tracking over time
  - Returns distribution with Sharpe and Sortino ratios
  - Drawdown analysis and maximum drawdown periods
  - Scenario annotations (flash crash markers, low liquidity notes)
  - Risk-adjusted metrics and trade-level details

Latest results in `training_viz/training_progress_ep0.png`:
- Positive returns (+1.42%)
- Stable Sharpe ratio (0.43)
- Controlled max drawdown (-4.26%)
- Effective risk management in volatile scenarios

## Testing Changes

1. **Core Testing**:
   - `python test_training.py` for baseline checks
   - Validate MLflow logs and metrics

2. **Scenario Testing**:
   - `tests/test_advanced_backtest.py` for scenario validations
   - Ensure scenario-specific metrics match expectations

3. **Risk Management Testing**:
   - `tests/test_risk_backtest.py` to ensure dynamic position sizing, stop-loss, and drawdown mechanisms function correctly

## Critical Notes

- **MVP Stability**:  
  - Do not modify stable core files (`train.py`, `trading_env.py`, `ppo_agent.py`)
  - Extend functionality via new modules

- **Code Changes**:  
  - Use inheritance and composition
  - Keep changes isolated and well-documented

- **Success Factors**:  
  - MVP bot runs successfully
  - Advanced scenarios and risk-aware features operational
  - Visualization and metrics tracking robust
  - CI/CD and resource optimization in progress

## Latest Updates 🔄

### Recent Improvements (2024-03-20)
- **Environment Wrapper GPU Support** ✅
  - Added GPU acceleration to observation normalization
  - Improved memory efficiency in observation stacking
  - NaN handling and numerical stability fixes
  - MLflow integration for environment metrics

- **Data Pipeline Optimization** ✅
  - Fixed feature generation issues
  - Improved PVT calculation
  - Updated naming conventions
  - All data pipeline tests passing

### Current Active Components
- **PPO Agent**: Located in `training/agents.py` (primary implementation)
- **Environment Wrapper**: Located in `envs/wrap_env.py` (GPU-enabled)
- **Base Environment**: Located in `envs/base_env.py`

### Deprecated Components ⚠️
- `agents/strategies/ppo_agent.py` is no longer in use
- Use `training/agents.py` for the current PPO implementation

### Next Steps 🎯
1. **GPU Support Enhancement**
   - Update PPO agent for efficient GPU utilization
   - Batch processing optimization
   - Memory management improvements

2. **MLflow Integration**
   - Enhanced metric logging
   - Hyperparameter tracking
   - Experiment management

3. **Multi-Agent System**
   - Parallel environment handling
   - Experience sharing mechanism
   - Distributed training support
