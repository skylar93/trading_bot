# Trading Bot Project

## Overview
Multi-agent reinforcement learning based trading system with GPU acceleration and web interface.

## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
- Edit `config/default_config.yaml` for general settings
- Edit `config/env_settings.yaml` for environment settings

## Project Structure
- `config/`: Configuration files
- `data/`: Data management and processing
- `envs/`: Trading environment implementation
- `agents/`: RL agents and models
- `training/`: Training scripts and notebooks
- `deployment/`: Web interface and API
- `scripts/`: Utility scripts
- `tests/`: Unit tests

## Quick Start
1. Data Collection:
```python
from data.utils.data_loader import DataLoader
loader = DataLoader()
data = loader.fetch_ohlcv('BTC/USDT', '1h')
```

2. Training (Coming soon...)
3. Web Interface (Coming soon...)

## Development Roadmap
1. Basic data pipeline setup
2. Single agent implementation
3. Web dashboard development
4. Multi-agent system integration
5. GPU optimization