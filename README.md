# AI-Powered Trading Bot

A comprehensive trading system that combines reinforcement learning, risk management, and real-time execution.

## Core Components

### 1. Reinforcement Learning
- PPO-based trading agent
- Custom trading environments
- Multi-agent support
- Hyperparameter optimization with Ray Tune
- MLflow experiment tracking

### 2. Risk Management
- Position size control
- Portfolio VaR monitoring
- Multi-asset correlation tracking
- Dynamic risk adjustment
- Advanced backtesting

### 3. Live Trading
- Real-time execution via CCXT
- Paper trading support
- Order types: limit, stop-limit, trailing-stop
- Rate limiting and error handling
- Network resilience

### 4. Data Pipeline
- OHLCV data processing
- Technical indicators (TA-Lib)
- Market scenario simulation
- Multi-asset data handling
- Real-time data streaming

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Training an Agent**
```python
from training.agents import PPOAgent
from training.environments import TradingEnv

# Initialize and train agent
agent = PPOAgent(config)
env = TradingEnv(data)
agent.train(env)
```

2. **Backtesting**
```python
from training.utils.risk_backtest import RiskAwareBacktester
from training.utils.risk_config import RiskConfig

# Run backtest
backtester = RiskAwareBacktester(data, risk_config)
results = backtester.run(agent)
```

3. **Live Trading**
```python
from trading.live import LiveTradingEnvironment

# Start live trading
live_env = LiveTradingEnvironment(
    exchange_config=config,
    risk_config=risk_config
)
live_env.run(agent)
```

## Project Structure

```
trading_bot/
├── training/
│   ├── agents/           # RL agents (PPO, etc.)
│   ├── environments/     # Trading environments
│   └── utils/           
│       ├── risk_backtest.py
│       └── hyperopt.py   # Ray Tune integration
├── trading/
│   ├── live/            # Live trading
│   ├── paper/           # Paper trading
│   └── data/            # Data handling
├── risk/
│   └── risk_manager.py  # Risk management
├── tests/
│   ├── test_agents/
│   ├── test_trading/
│   └── test_risk/
└── scripts/
    ├── train.py
    ├── backtest.py
    └── live_trade.py
```

## Development

### Testing
```bash
python -m pytest tests/
```

### Code Quality
- Follow PEP 8
- Add docstrings
- Update CHANGELOG.md

## Documentation

- See class/method docstrings
- Check CHANGELOG.md
- Review test files

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Update documentation
5. Submit pull request

## License

MIT License