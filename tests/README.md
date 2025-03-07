# Multi-Agent Trading System Test Suite

This directory contains tests for the multi-agent trading system, with a focus on validating the enhancements for collaborative trading strategies, ensemble decision making, and hierarchical reinforcement learning.

## Test Structure

The test suite is organized into several files:

- `test_multi_agent_env.py`: Tests for the multi-agent trading environment, including shared capital allocation and action correlation tracking.
- `test_multi_agent_manager.py`: Tests for the multi-agent manager, focusing on ensemble methods and meta-agent capabilities.
- `test_hierarchical_agent.py`: Tests for the hierarchical agent with manager and worker components.
- `test_multi_agent_integration.py`: Integration tests that verify different components work together correctly.

## Running Tests

### Using the Run Script

The easiest way to run the tests is using the provided run script:

```bash
python run_tests.py
```

This will run all tests and provide a summary of the results.

### Options

The run script supports several options:

- `--verbose` or `-v`: Show detailed test output
- `--coverage`: Generate a coverage report
- `--single=FILE`: Run a single test file (e.g., `test_multi_agent_env.py`)

Examples:

```bash
# Run with verbose output
python run_tests.py --verbose

# Generate coverage report
python run_tests.py --coverage

# Run a specific test file
python run_tests.py --single=test_multi_agent_env.py
```

### Using pytest Directly

You can also run the tests directly using pytest:

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_multi_agent_env.py

# Run a specific test function
pytest tests/test_multi_agent_env.py::test_shared_capital_initialization
```

## Test Dependencies

The tests require the following dependencies:

- pytest
- numpy
- pandas
- torch
- gymnasium

You can install these dependencies using pip:

```bash
pip install pytest numpy pandas torch gymnasium
```

For coverage reports, you'll also need:

```bash
pip install pytest-cov
```

## Test Design

### Mock Objects and Fallbacks

The tests are designed to work even if some components of the actual implementation are missing. Each test file includes fallback mechanisms to use mock objects when the real implementations are not available.

This allows you to run the tests during development, even before all components are fully implemented.

### Test Data Generation

Test data, including price series and observations, is generated synthetically using fixtures. This ensures reproducibility and avoids dependencies on external data sources.

### Test Scenarios

The tests cover various scenarios, including:

1. **Basic functionality**: Ensuring environment steps and agent actions work correctly
2. **Shared capital**: Testing capital allocation and reallocation mechanisms
3. **Ensemble methods**: Verifying different ensemble strategies (weighted, best, meta)
4. **Hierarchical behavior**: Testing the manager-worker architecture
5. **Strategy synergy**: Evaluating how multiple strategies work together

## Troubleshooting

### Import Errors

If you encounter import errors, the test suite will automatically fall back to using mock objects. However, if you want to test with the real implementations, make sure all required components are correctly implemented and accessible.

### Missing Dependencies

If you see errors about missing packages, make sure you have installed all the required dependencies.

### Failing Tests

If specific tests are failing:

1. Check the error message to identify the issue
2. If it's a functionality issue, ensure the corresponding component is correctly implemented
3. If it's a test expectation issue, check if the test needs to be updated to match the actual implementation

## Contributing

When adding new features to the multi-agent trading system, please also add corresponding tests to ensure the features work correctly and continue to work as the system evolves. 