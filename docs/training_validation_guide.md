# Training Pipeline Validation Guide

This guide explains how to validate and monitor your training pipeline before submitting long-running jobs to SLURM, and how to analyze the training performance using MLflow.

## Understanding the Training Pipeline

Our trading bot has several different training configurations:

1. **Single Asset-Single Agent**: One agent trading a single asset
2. **Single Asset-Multi Agent**: Multiple agents collaborating/competing to trade a single asset
3. **Multi Asset-Single Agent**: One agent trading multiple assets
4. **Multi Asset-Multi Agent**: Multiple agents trading multiple assets
5. **Meta Agent**: A special agent that coordinates other agents in multi-agent setups

## Quick Validation Before SLURM Submission

To avoid wasting computational resources on failed jobs, always validate your training pipeline with smaller timesteps first.

### Running the Validation Script

The validation script will test all pipeline configurations with minimal epochs:

```bash
python scripts/validate_training.py
```

This script:
- Validates all training pipeline modes
- Uses synthetic data for quick testing
- Reduces the timesteps to get faster feedback
- Verifies MLflow logging is functioning correctly
- Ensures checkpoint saving/loading works
- Tests that agent interactions work properly in multi-agent scenarios

### What to Look For

The validation script will produce a report with a status for each configuration:

- ✅ PASS: The configuration works as expected
- ❌ FAIL: The configuration has issues that need to be fixed

If any test fails, check the `validation.log` file for details about the error.

## Analyzing Training Performance with MLflow

MLflow is our primary tool for tracking and analyzing training performance.

### Starting MLflow UI

To view your training runs:

```bash
mlflow ui
```

This will start a web server at http://localhost:5000 where you can view all experiments.

### Key Metrics to Watch

When reviewing your training results, focus on these key metrics:

1. **episode_reward**: The reward the agent received in each episode
2. **eval_reward**: The reward during evaluation episodes
3. **best_eval_reward**: The highest evaluation reward achieved so far
4. **loss**: The loss value of the model during training

For multi-agent setups, these metrics will be tracked for each agent individually.

### Using the MLflow Visualization Tool

For more detailed analysis and visualization, use the MLflow visualization script:

```bash
python scripts/visualize_mlflow_results.py --experiment-filter "ppo" --metrics best_eval_reward episode_reward loss --show-agent-comparison
```

Command line options:
- `--experiment-filter`: Filter experiments by name (e.g., "ppo" or "multi_agent")
- `--experiment-id`: Analyze a specific experiment by ID
- `--metrics`: Specific metrics to visualize (defaults to best_eval_reward)
- `--group-by`: Group results by a parameter (e.g., "learning_rate")
- `--show-agent-comparison`: Show performance comparison for multi-agent runs
- `--output-dir`: Directory to save reports (default: mlflow_reports)

The script generates HTML reports with:
- Statistical summaries of metrics
- Visualizations of metric trends
- Comparisons between different runs
- Best performing parameter configurations

## Signs of Successful Training

### Single-Agent Training

A successful single-agent training should show:

1. **Increasing rewards**: episode_reward and eval_reward should trend upward
2. **Decreasing loss**: loss should generally decrease and stabilize
3. **Steady improvements**: best_eval_reward should increase and eventually plateau

Example pattern:
```
Episode 100: reward=-10.5, loss=2.3
Episode 200: reward=-5.2, loss=1.8
Episode 300: reward=2.7, loss=1.2
Episode 400: reward=7.3, loss=0.8
Episode 500: reward=9.5, loss=0.5
```

### Multi-Agent Training

For multi-agent training, look for:

1. **Balanced performance**: All agents should show improvement
2. **Specialization**: Different agents might specialize in different market conditions
3. **Meta-agent learning**: If using a meta-agent, it should learn to select the best agents

### Meta-Agent Performance

A successful meta-agent should:

1. **Outperform individual agents**: The meta-agent's decisions should result in better overall performance
2. **Adapt to market conditions**: The weights assigned to each agent should change with market conditions
3. **Reduce drawdowns**: The meta-agent should reduce portfolio volatility compared to individual agents

## Troubleshooting Training Issues

### Common Problems and Solutions

1. **Rewards not improving**
   - Check reward function implementation
   - Increase learning rate or reduce it if there are oscillations
   - Add more features to the observation space

2. **Loss exploding**
   - Reduce learning rate
   - Check for extreme values in the observation space
   - Implement gradient clipping

3. **Meta-agent not learning**
   - Ensure sub-agents have diverse strategies
   - Increase meta-agent learning rate
   - Check meta-agent observation space includes sufficient information

4. **Multi-agent performance unstable**
   - Implement experience sharing between agents
   - Add more structure to the reward function
   - Consider using shared parameters for some layers

## Next Steps After Validation

Once your validation passes and you understand the training metrics to watch for:

1. **Update configuration files** for full-scale training
2. **Submit to SLURM** using the provided scripts
3. **Monitor progress** using MLflow
4. **Analyze results** with the visualization tool once training completes

By following this workflow, you'll save time and computational resources by catching issues early, and you'll gain deeper insights into your training performance. 