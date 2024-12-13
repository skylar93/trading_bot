# Trading Bot Project

## Latest Achievement 🎯

Successfully implemented backtesting & visualization system (Dec 13, 2024):

- Portfolio returns: +1.42%
- Sharpe Ratio: 0.43
- Sortino Ratio: 0.60
- Max Drawdown: -4.26%

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

1. **Backtesting System** (Priority)

   - Further visualization development
   - Additional metrics
   - Trade analysis tools
   - Add new files under training/utils/

2. **Risk Management**

   - Position sizing
   - Stop-loss mechanisms
   - Portfolio constraints
   - Implement as separate modules

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
│   └── utils/
│       ├── visualization.py # Visualization tools
│       ├── mlflow_manager.py # MLflow tracking manager
│       └── mlflow_utils.py   # MLflow helper utilities
├── deployment/
│   └── web_interface/
│       └── app.py           # Streamlit UI
```

## Visual Results

The backtesting system generates comprehensive visualizations:

- Portfolio value tracking
- Returns distribution with key metrics
- Drawdown analysis and maximum drawdown periods
- Trading metrics dashboard including Sharpe and Sortino ratios
- Real-time performance indicators

Latest results in `training_viz/training_progress_ep0.png` show:

- Positive returns (+1.42%)
- Stable Sharpe ratio (0.43)
- Controlled drawdown (-4.26%)
- Consistent risk-adjusted performance

## Testing Changes

1. **Core Testing**

   - Always start with test_training.py
   - Verify basic functionality first
   - Check MLflow logs for performance regression
   - Validate all metrics

2. **Visualization Testing**

   - Check training_viz directory
   - Verify all plots are generated
   - Validate metrics display
   - Ensure proper file saving

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
   - Shows positive returns
   - Visualization system works
   - Training pipeline stable
   - Proper metrics tracking

Remember: The goal is to extend and enhance, not to modify what works.

