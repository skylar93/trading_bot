# Trading Bot Project

## Latest Achievement 🎯

Successfully implemented risk-aware backtesting system (Dec 13, 2024):

- Portfolio returns: +1.42%
- Sharpe Ratio: 0.43
- Sortino Ratio: 0.60
- Max Drawdown: -4.26%
- Flash Crash and Low Liquidity Scenarios tested successfully
- Risk-aware trading implemented with dynamic position sizing and stop-loss mechanisms
- Initiated hyperparameter optimization pipeline using Ray Tune
- Completed unit tests for `hyperopt_env.py`, `hyperopt_agent.py`, and `hyperopt_tuner.py`
- Actual optimization pipeline separated for independent execution

## Core Architecture Components

### A. Data Pipeline Layer

- **Frameworks**: TA-Lib + ccxt

- **Data Processing Pipeline**:

  ```
  Raw Data (ccxt) → TA-Lib Pipeline → Feature Store → Training Data
  ```

- **Storage Structure**:

  ```
  /data
    /raw          # Original data collected via ccxt
    /processed    # TA-Lib-processed data
    /features     # Final feature sets for training
  ```

### B. Training and Inference Layer

- **Frameworks**: RLlib + PPO Agent
- **Multi-Agent Architecture**:
  ```
  Market Environment
    └── PPO Agent (Currently Active)
  ```

### C. Web Interface Layer

- **Frameworks**: FastAPI + Streamlit
- **Components**:
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

## Completed Features

1. **Data Pipeline** ✅

   - ccxt data collection
   - 44 technical indicators
   - Data caching system

2. **Reinforcement Learning** ✅

   - PPO agent implementation
   - Trading environment
   - MLflow tracking
   - Successful training runs

3. **Backtesting System** ✅

   - Performance metrics (Sharpe, Sortino, MDD)
   - Portfolio visualization
   - Trade analysis
   - Real-time monitoring

4. **Visualization Tools** ✅

   - Portfolio value tracking
   - Returns distribution
   - Drawdown analysis
   - Trade and risk metric visualizations

5. **Advanced Backtesting Scenarios** ✅

   - Flash Crash Scenarios:
     - Tested with sudden price drops and partial recovery
     - Evaluated metrics such as survival rate and recovery time
   - Low Liquidity Scenarios:
     - Tested with reduced trade volumes and high volatility
     - Measured trade execution costs and feasibility

6. **Risk-Aware Backtesting** ✅

   - Integrated dynamic position sizing
   - Real-time leverage monitoring
   - Stop-loss mechanisms and drawdown limits
   - Risk metrics summary added to backtesting results

7. **Hyperparameter Optimization (Ongoing)**

   - Created `training/hyperopt/` directory with modular structure:
     - `hyperopt_env.py`: Simplified trading environment for tuning.
     - `hyperopt_agent.py`: Minimal agent for fast experimentation.
   - Integrated Ray Tune for automated optimization of learning rate, batch size, and network architecture.
   - Separated testing and full optimization for faster iteration.
   - `scripts/run_hyperopt.py`: Dedicated script for full optimization.

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

### MVP First

- Focus on core functionality
- Test components individually
- Keep changes minimal
- Avoid modifying working components

### Code Modification Rules

```
IMPORTANT: Make minimal changes to:
- train.py
- trading_env.py
- ppo_agent.py

These files are WORKING and STABLE.
Changes should be made through extension, not modification.
```

### Current Focus

1. **Advanced Backtesting System** (Priority)

   - Further scenario development
   - Test additional edge cases (e.g., extreme volatility, correlated assets)
   - Validate new metrics and visualizations

2. **Risk Management**

   - Position sizing
   - Stop-loss mechanisms
   - Portfolio constraints
   - Implement as separate modules

3. **Hyperparameter Optimization**

   - Complete `hyperopt_tuner.py` for Ray Tune integration.
   - Validate with simple experiments and existing training pipeline.
   - Track results with MLflow for reproducibility.
   - Run optimization with `scripts/run_hyperopt.py` for full hyperparameter search.

4. **Real-Time Trading Preparation** (Next Major Goal)

   - **CCXT WebSocket Integration:**
     - Set up real-time data streaming with CCXT.
     - Ensure low-latency data updates.
   - **Trading Environment Adaptation:**
     - Modify `trading_env.py` to support live action application.
     - Implement mechanisms to adjust portfolio positions and balances dynamically.
   - **Live Trading Verification:**
     - Add real-time validation features to the backtesting system for consistency checks.
   - **Paper Trading:**
     - Develop a sandbox environment to simulate live trades without real financial exposure.
   - **Performance Monitoring:**
     - Create real-time performance dashboards using Streamlit or Plotly.

5. **Resource Optimization**

   - Optimize multi-agent training resource usage.
   - Implement GPU utilization strategies (e.g., Ray Actor model).

6. **CI/CD Implementation**

   - Automate testing, backtesting, and deployment using GitHub Actions.
   - Integrate MLflow logging into the pipeline for automated result tracking.

## Current Ongoing part

### **Real-Time Trading Integration**

### **1. CCXT WebSocket Integration**

- **Implemented Features**:
  - Real-time data streaming:
    - **Trades**: Live trade data
    - **Orderbook**: Real-time order book updates
    - **Ticker**: Current market state information
  - Asynchronous data processing:
    - Utilized `asyncio` and `queue` for seamless data flow management
    - Combined real-time streaming with existing REST API data

- **Relevant Files**:
  ```
  /data/utils/websocket_loader.py
  /envs/trading_env.py (Added support for live trading environments)
  ```

---

### **2. Paper Trading Feature**

- **Implemented Functionalities**:
  - Support for paper trading in the `RealTimeTradingEnv` environment
  - Real-time testing of trading strategies:
    - Actions: `Buy`, `Sell`
    - Position sizing and balance management

- **Relevant Files**:
  ```
  /envs/trading_env.py
  /training/paper_trading.py
  ```


## Running the Project

### Local Directory
```
Users/skylar/Desktop/trading_bot
```

### Virtual Environment
```
venv
```

### GitHub Repository
```
https://github.com/skylar93/trading_bot
```

1. Setup

```bash
cd Users/skylar/Desktop/trading_bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Test Core System

```bash
python test_training.py  # Always start here!
```

3. Launch Web UI

```bash
streamlit run deployment/web_interface/app.py
```

4. Execute Hyperparameter Optimization

```bash
python scripts/run_hyperopt.py
```

   - Note: Full optimization may take significant time depending on num_samples and resource settings.
   - Results can be monitored and analyzed via MLflow UI.

## Key Files & Components

```
trading_bot/
├── agents/
│   └── ppo_agent.py         # Core PPO implementation - STABLE
├── data/
│   └── utils/
│       ├── data_loader.py   # ccxt integration
│       └── feature_generator.py
├── envs/
│   ├── base_env.py          # Base trading environment
│   ├── trading_env.py       # Single-agent trading environment - STABLE
│   └── multi_agent_env.py   # Multi-agent trading environment
├── training/
│   ├── train.py             # Training pipeline - STABLE
│   ├── train_multi_agent.py # Multi-agent training
│   ├── evaluation.py        # Performance metrics
│   ├── backtest.py          # Backtesting system
│   ├── advanced_backtest.py # Advanced scenario backtesting
│   ├── hyperopt/
│   │   ├── __init__.py         # Module initialization
│   │   ├── hyperopt_env.py     # Simplified environment for tuning
│   │   ├── hyperopt_agent.py   # Minimal agent for experimentation
│   │   ├── hyperopt_tuner.py   # Ray Tune-based tuning logic (pending)
│   ├── utils/
│   │   ├── visualization.py    # Visualization tools
│   │   ├── mlflow_manager.py   # MLflow tracking manager
│   │   ├── mlflow_utils.py     # MLflow helper utilities
│   │   ├── advanced_scenarios.py # Advanced scenario generation
│   │   ├── risk_management.py  # Risk management logic
│   │   └── risk_backtest.py    # Risk-aware backtesting system
├── deployment/
│   └── web_interface/
│       └── app.py           # Streamlit UI
├── scripts/ 
│   └── run_hyperopt.py   # Script for executing hyperparameter optimization

## Recommendations for Next Steps

1. **Real-Time Trading Setup**
   - Finalize CCXT WebSocket integration for live market data.
   - Modify the trading environment for seamless real-time portfolio adjustments.
   - Create validation systems for comparing live performance with historical backtesting results.
   - Design a comprehensive dashboard to monitor key metrics in real-time.

2. **Enhanced Risk Management**
   - Implement additional risk constraints, such as value-at-risk (VaR) and conditional VaR.
   - Introduce automated rebalancing strategies to optimize portfolio allocations under different market conditions.
   - Extend the stop-loss mechanisms to support trailing stops and dynamic thresholds.

3. **Visualization Improvements**
   - Transition to interactive visualizations with Plotly or Dash for enhanced data exploration.
   - Introduce comparative performance analysis to evaluate different agents or scenarios.

4. **Pipeline Scaling and Resource Optimization**
   - Test the current pipeline in a GPU-enabled environment to benchmark improvements.
   - Introduce batch processing for agent evaluation and backtesting.
   - Investigate distributed training frameworks (e.g., Horovod or PyTorch Lightning) to improve scalability.

5. **CI/CD Automation**
   - Expand the existing pipeline to support automatic deployment of trained agents to the trading environment.
   - Set up notification systems for alerting about significant deviations in live trading metrics.

6. **Algorithmic Enhancements**
   - Explore new architectures such as LSTMs, Transformers, or hybrid models for better time-series prediction.
   - Develop multi-strategy agents capable of combining momentum, mean reversion, and market-making techniques.

7. **Community Contributions and Documentation**
   - Publish the current codebase on GitHub with detailed usage instructions and examples.
   - Actively seek feedback and contributions from the open-source community to further improve the system.

8. **Future Goals**
   - Transition from paper trading to live trading in a controlled environment.
   - Implement advanced order execution strategies to minimize slippage and market impact.
   - Extend the trading bot's capabilities to other asset classes, such as commodities or options.


## Visual Results

The advanced backtesting system generates comprehensive visualizations:

- Portfolio value tracking
- Returns distribution with key metrics
- Drawdown analysis and maximum drawdown periods
- Scenario-specific annotations (e.g., flash crash markers)
- Trading metrics dashboard including Sharpe and Sortino ratios
- Risk-adjusted metrics and trade-level details

Latest results in `training_viz/training_progress_ep0.png` show:

- Positive returns (+1.42%)
- Stable Sharpe ratio (0.43)
- Controlled drawdown (-4.26%)
- Survived extreme market scenarios
- Effective risk management during volatile periods

## Testing Changes

1. **Core Testing**

   - Always start with test_training.py
   - Verify basic functionality first
   - Check MLflow logs for performance regression
   - Validate all metrics

2. **Scenario Testing**

   - Run tests in `tests/test_advanced_backtest.py`
   - Ensure all scenarios pass with expected results
   - Validate scenario-specific metrics (e.g., max drawdown recovery)

3. **Risk Management Testing**

   - Validate with `tests/test_risk_backtest.py`
   - Ensure dynamic position sizing is respected
   - Confirm stop-loss and drawdown mechanisms work as expected

## Critical Notes

1. **Core Principle**: MVP Stability

   - Don't modify working core components
   - Add features through new modules
   - Test thoroughly before integration
   - Keep the stable core stable

2. **Code Changes**

   - Create new files instead of modifying existing ones
   - Use inheritance and composition
   - Keep changes isolated
   - Document everything thoroughly

3. **Current Success Factors**

   - MVP bot runs successfully
   - Advanced scenarios tested and passed
   - Visualization system works
   - Training pipeline stable
   - Risk-aware system operational
   - Proper metrics tracking

Remember: The goal is to extend and enhance, not to modify what works.

