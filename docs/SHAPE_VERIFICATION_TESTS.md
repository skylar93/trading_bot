# Shape Verification Tests

## Overview

Shape verification tests are a critical component of our testing infrastructure, designed to validate tensor dimensions and data flow within the trading bot codebase. These tests ensure that tensors maintain the expected shapes throughout the entire pipeline, from environment observations to agent actions and training.

## Purpose

The shape verification tests serve several important purposes:

1. **Detect Dimension Mismatches**: Identify when tensors don't match the expected dimensions, which can cause runtime errors or silent failures.
2. **Validate Data Flow**: Ensure that data flows correctly between environments, agents, and training pipelines.
3. **Ensure Compatibility**: Verify that different components (environments, agents, strategies) work together with compatible tensor shapes.
4. **Identify NaN Values**: Detect when NaN values appear in tensors, which can cause training instability.
5. **Verify Tensor Shape Consistency**: Ensure that tensor shapes remain consistent during training and inference.

## Test Suite

The shape verification test suite includes the following tests:

1. **test_single_agent_shapes**: Validates tensor shapes in a single-agent environment, checking observation and action shapes.
2. **test_multi_agent_shapes**: Verifies tensor shapes in a multi-agent environment, ensuring that observations and actions for each agent have the correct dimensions.
3. **test_meta_agent_ensemble**: Tests the meta-agent architecture, validating that the meta-agent can properly process inputs from sub-agents.
4. **test_train_pipeline_minimal**: Checks tensor shapes throughout the training pipeline for a single agent.
5. **test_multi_agent_train_pipeline_minimal**: Validates tensor shapes in the multi-agent training pipeline.

## Running the Tests

To run the shape verification tests, use the following command:

```bash
./scripts/run_shape_tests.sh
```

This script will execute all shape verification tests and generate a log file (`shape_test_debug.log`) with detailed information about tensor shapes and any errors encountered.

## Common Issues and Solutions

### NaN Values in Observations

If NaN values are detected in observations, the test will log a warning. This can be caused by:
- Missing data in the input dataset
- Division by zero in feature calculations
- Incorrect normalization

**Solution**: Add NaN checks and handling in the environment's observation generation.

### Inconsistent Parameter Names

Parameter naming inconsistencies (e.g., `initial_capital` vs. `initial_balance`) can cause initialization errors.

**Solution**: Standardize parameter names across all environments and agents.

### Agent Action Method Names

Different agents may use different method names for generating actions (e.g., `act()` vs. `get_action()`).

**Solution**: Standardize action method names or use appropriate adapters.

### Environment Initialization

Environments may require different initialization parameters.

**Solution**: Create factory functions that handle parameter differences.

### Division by Zero Errors

Setting `checkpoint_interval` to 0 can cause division by zero errors.

**Solution**: Ensure checkpoint intervals are at least 1.

## Known Dimension Mismatch Issues

During validation testing, several dimension mismatch issues were identified:

1. **Meta Agent Input Dimension Mismatches**:
   ```
   Input dimension mismatch: got 1650, expected 10. Reshaping input to match expected dimension.
   Input dimension mismatch: got 2190, expected 10. Reshaping input to match expected dimension.
   ```

2. **Action Shape Mismatches in MultiAgentMultiAssetEnv**:
   ```
   Action shape (1,) for agent agent1 doesn't match expected shape (3,). Adapting action.
   ```

3. **Multi-Agent Manager Training Step Errors**:
   ```
   Error in train_step for agent1: Cannot interpret 2D input torch.Size([1, 543]) as (batch_size, 13)
   ```

4. **Advantages Shape Mismatch in Meta Agent**:
   ```
   Advantages shape mismatch: advantages torch.Size([32]), log_probs torch.Size([32, 1])
   ```

For a detailed analysis of these issues and a plan to fix them, see [DIMENSION_MISMATCH_ISSUES.md](DIMENSION_MISMATCH_ISSUES.md).

## Adding New Tests

When adding new shape verification tests, follow these guidelines:

1. Use the `@pytest.mark.shape_verification` decorator to include the test in the shape verification suite.
2. Log tensor shapes at key points in the test using the logging module.
3. Include checks for NaN values in observations and actions.
4. Verify that tensor shapes match the expected dimensions.
5. Use small, synthetic datasets to keep tests fast.

Example:

```python
@pytest.mark.shape_verification
def test_new_agent_shapes():
    # Create environment and agent
    env = TradingEnvironment(...)
    agent = NewAgent(...)
    
    # Reset environment and get initial observation
    obs = env.reset()
    logger.info(f"Observation shape: {obs.shape}")
    
    # Check for NaN values
    if np.isnan(obs).any():
        logger.warning(f"NaN values detected in observation")
    
    # Get action from agent
    action = agent.get_action(obs)
    logger.info(f"Action shape: {action.shape}")
    
    # Verify shapes
    assert obs.shape == (expected_obs_shape), f"Expected {expected_obs_shape}, got {obs.shape}"
    assert action.shape == (expected_action_shape), f"Expected {expected_action_shape}, got {action.shape}"
```

## Troubleshooting

If a shape verification test fails, follow these steps:

1. Check the `shape_test_debug.log` file for detailed information about tensor shapes.
2. Look for warning messages about NaN values or dimension mismatches.
3. Verify that the environment and agent are compatible (e.g., observation space and action space).
4. Check for any reshaping operations that might be masking underlying issues.
5. Ensure that the test is using the correct environment and agent configurations.

Remember that shape verification tests are designed to catch issues early, before they cause problems in training or deployment. Fixing shape issues is essential for ensuring the stability and correctness of the trading bot.

## Key Features

- **Fast execution**: Tests run with minimal data (100 rows) and minimal steps (5-10)
- **Comprehensive coverage**: Tests all environment and agent combinations 
- **Detailed logging**: Records tensor shapes and NaN values for debugging
- **Pre-SLURM validation**: Run these tests before submitting to SLURM to catch shape errors early

## Running the Tests

### Quick Method (Recommended)

Run the provided script:

```bash
./scripts/run_shape_tests.sh
```

This script will:
1. Run each shape verification test individually
2. Display pass/fail status for each test
3. Write detailed logs to `shape_test_debug.log`
4. Return non-zero exit code if any test fails

### Using Pytest Directly

You can also run the tests directly with pytest:

```bash
# Run all shape tests
python -m pytest tests/test_small_integration.py -v

# Run a specific test
python -m pytest tests/test_small_integration.py::test_single_agent_shapes -v

# Run only shape verification tests across all files
python -m pytest -m shape_verification -v
```

## Test Descriptions

### `test_single_agent_shapes`

Tests a single agent with a single asset environment. Verifies:
- Observation tensor shapes
- Action tensor shapes
- Agent update process
- NaN detection in all tensors

### `test_multi_agent_shapes`

Tests multiple agents with the multi-asset environment. Verifies:

- Agent-specific observation shapes
- Action dictionary shapes for each agent
- Independent agent updates
- NaN detection across all agents

### `test_meta_agent_ensemble`

Tests the meta-agent ensemble method. Verifies:
- Sub-agent and meta-agent observation shapes
- Coordination between agents
- Meta-agent input/output dimension compatibility
- NaN detection in the ensemble process

### `test_train_pipeline_minimal`

Tests the complete training pipeline with minimal steps. Verifies:
- Environment creation
- Agent creation
- Training loop execution
- End-to-end shape compatibility

### `test_multi_agent_train_pipeline_minimal`

Tests the multi-agent training pipeline. Verifies:
- Multi-agent environment creation
- Agent interaction during training
- End-to-end shape compatibility with multiple agents

## Understanding Test Failures

When a test fails, check the `shape_test_debug.log` file for details. Common issues include:

1. **Dimension Mismatch**: Look for error messages like `Cannot interpret 2D input torch.Size([1, 243]) as (batch_size, 13)` which indicate tensor shapes don't match between components.

2. **NaN Values**: Look for `NaN in policy network input` or similar messages, which indicate NaN values in tensors.

3. **Tensor Shape Records**: The ShapeMonitor records all tensor shapes during execution. Check the log for lines like:
   ```
   === Recorded Tensor Shapes ===
   initial_obs_agent1: (100,) (no NaNs)
   action_agent1_step_0: (1,) (no NaNs)
   ```

## Extending the Tests

To add a new test case:

1. Follow the pattern in `test_small_integration.py`
2. Use the `ShapeMonitor` class to record shapes
3. Keep the test minimal but representative
4. Add the `@pytest.mark.shape_verification` decorator
5. Update the `scripts/run_shape_tests.sh` script to include your new test

## Best Practices

- Run these tests before submitting to SLURM or other compute clusters
- Run these tests after making changes to network architectures or agent policies
- Check the logs even when tests pass to identify potential issues
- Use the shape information to optimize your network dimensions

## Integration with CI/CD

These tests are designed to be fast enough for CI/CD pipelines. Add them to your workflow to prevent shape-related issues from reaching production.

```yaml
- name: Run Shape Verification Tests
  run: ./scripts/run_shape_tests.sh
``` 