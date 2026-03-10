# Model Training UI

This document provides an overview of the Model Training UI for the Trading Bot, including how to use it, parameter explanations, and best practices.

## Overview

The Model Training UI allows you to train reinforcement learning models for trading strategies through a user-friendly web interface. It supports both single-agent and multi-agent training scenarios, with real-time progress tracking and visualization.

## Key Features

- **Environment Configuration**: Set up the trading environment, including symbol selection, timeframe, date range, and risk management parameters.
- **Agent Settings**: Configure the RL algorithm, network architecture, and LSTM/RNN usage.
- **Multi-Agent Support**: Create and configure multiple agents with different strategies and capital allocations.
- **Training Parameters**: Adjust hyperparameters such as learning rate, batch size, and discount factor.
- **Real-time Monitoring**: Track training progress, rewards, and evaluation metrics in real-time.
- **MLflow Integration**: All training runs are automatically logged to MLflow for experiment tracking.

## How to Use

### 1. Environment Settings

- **Symbol & Timeframe**: Select the trading pair (e.g., BTC/USDT) and timeframe (e.g., 1h).
- **Date Range**: Choose the historical data period for training.
- **Window Size**: Set the number of past timesteps to include in each observation.
- **Risk Management**: Configure stop-loss settings and other risk parameters.
- **Multi-Agent**: Enable/disable multi-agent training mode.

### 2. Agent Settings

- **Algorithm**: Choose the RL algorithm (PPO, SAC, DQN, A2C).
- **Network Structure**: Set hidden layer sizes for the neural network.
- **LSTM**: Enable LSTM layers for sequence modeling (optional).

### 3. Multi-Agent Configuration (if enabled)

- **Number of Agents**: Set how many agents will trade simultaneously.
- **Ensemble Method**: Choose how agent decisions are combined (weighted, majority vote, meta-agent).
- **Per-Agent Settings**: Configure each agent's type and capital allocation.

### 4. Training Parameters

- **Total Timesteps**: How many environment steps to train for.
- **Batch Size**: Size of batches used for training updates.
- **Learning Rate**: Step size for the optimizer.
- **Gamma**: Discount factor for future rewards.
- **Checkpoint Settings**: How often to evaluate and save model checkpoints.

### 5. Run Training

- Review the configuration summary.
- Click "Start Training" to begin.
- Monitor progress, metrics, and visualizations in real-time.
- Upon completion, you can directly navigate to the Backtest Results page to test your trained model.

## Best Practices

### Single-Agent Training

- Start with PPO algorithm and default hyperparameters.
- Use a window size of 20-50 for hourly data.
- Train for at least 100,000 timesteps initially.
- Enable stop-loss during training to learn risk management.

### Multi-Agent Training

- Start with 2-3 agents of different types (e.g., PPO, momentum, mean reversion).
- Distribute capital evenly at first, then adjust based on performance.
- The "weighted" ensemble method is generally a good starting point.
- For advanced users, try the "meta-agent" method with a PPO meta-agent.

### Hyperparameter Tips

- Learning rate: 3e-4 is a good default; try 1e-4 for more stability or 1e-3 for faster learning.
- Batch size: 64 or 128 is usually a good balance.
- Network size: Start with [64, 64] and increase if the model appears to be underfitting.
- LSTM: Can be helpful for capturing market patterns, but increases training time.

## Troubleshooting

- **Training too slow**: Reduce the window size or simplify the network architecture.
- **Poor performance**: Try increasing the total timesteps, adjusting the learning rate, or using LSTM.
- **Agent not learning**: Check if the reward function is appropriate; consider adjusting gamma.
- **Multi-agent issues**: Start with simpler configurations and gradually increase complexity.

## Advanced Usage

### Integration with HPC

For large-scale training runs, you can configure the training parameters here and then generate a SLURM job script for execution on an HPC cluster.

### Custom Agent Types

The UI supports various built-in agent types, but you can also implement custom agents by extending the agent factory in the codebase.

### Hyperparameter Optimization

While basic hyperparameter settings are available in the UI, for advanced hyperparameter optimization, consider using Ray Tune (available through the command line interface). 