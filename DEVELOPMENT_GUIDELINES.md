Below is a revised `development_guid.md` that incorporates the core principles, detailed naming and formatting rules, testing standards, and a recommended step-by-step refactoring workflow. This guide should serve as a standalone document that developers can refer to for maintaining consistency and quality across the codebase.

[MLflow Configuration & Best Practices]

1. **MLflow Directory Structure**:
   - Base directory: `./mlflow_runs/`
   - Experiment directories: `./mlflow_runs/<experiment_name>/`
   - Artifact format: Use parquet for DataFrames
   - Clean up MLflow directories on initialization:
     ```python
     mlflow_dirs = ["./mlruns", "./mlflow_runs"]
     ```

2. **Experiment Lifecycle**:
   - Always delete existing experiment before creating new one
   - Wait for operations to complete:
     ```python
     time.sleep(0.5)  # After experiment creation/deletion
     ```
   - Log dummy metric to ensure meta.yaml creation:
     ```python
     mlflow.log_metric("_dummy", 0.0)
     ```

3. **Run Management**:
   - End active run before starting new one (unless nested)
   - Verify run creation and status
   - Add wait times between operations
   - Log dummy metrics before artifact operations

4. **Critical Components (DO NOT MODIFY)**:
   - `MLflowManager.__init__()`: Experiment creation logic
   - `MLflowManager.start_run()`: Run initialization with meta.yaml creation
   - `MLflowManager.end_run()`: Run cleanup and verification
   - `MLflowManager.cleanup()`: Resource cleanup logic

5. **Testing Requirements**:
   - Use `mlflow_test_context` fixture for isolated tests
   - Clean up MLflow resources after each test
   - Verify experiment and run existence
   - Check for meta.yaml creation

[Current Progress & Next Steps]

1. **Completed**:
   - MLflow initialization and cleanup ✓
   - Meta.yaml creation fix ✓
   - Run lifecycle management ✓
   - Artifact logging stability ✓

2. **Pending**:
   - Risk backtesting column validation
   - Multi-agent environment fixes
   - Paper trading environment updates
   - Network initialization issues

3. **DO NOT MODIFY**:
   - `training/utils/mlflow_manager.py`
   - `training/hyperopt/hyperopt_tuner.py`
   - `tests/training/hyperopt/test_hyperopt_tuner.py`
   - `tests/test_mlflow_logger.py`

[Current State Analysis (2024-12-17)]

1. **Stabilized Components (DO NOT MODIFY)**:
   - MLflow Manager (`training/utils/mlflow_manager.py`)
   - Hyperopt Tuner (`training/hyperopt/hyperopt_tuner.py`)
   - Related Tests:
     - `tests/training/hyperopt/test_hyperopt_tuner.py`
     - `tests/test_mlflow_logger.py`

2. **Current Issues & Priorities**:

   a. **High Priority - Data & Environment Structure**:
      - Risk Backtesting Data Columns:
        ```python
        REQUIRED_COLUMNS = {
            '$open', '$high', '$low', '$close', '$volume'
        }
        ```
      - Multi-Agent Environment Reset:
        - Fix `env.reset()` return type
        - Ensure proper observation space format
      
   b. **Medium Priority - Environment Consistency**:
      - PaperTrading Environment:
        - Use `trading_fee` consistently (not `transaction_fee`)
        - Ensure `$`-prefixed columns
      - PolicyNetwork Initialization:
        - Resolve `shared` attribute expectation
        - Update tests or implement shared layer
      
   c. **Lower Priority - Logic & Tests**:
      - Risk Manager Tests:
        - Align "Should be invalid" expectations
        - Verify trade limit logic
      - Tuner/Hyperopt Details:
        - Handle `episodes` parameter
        - Implement or remove `run_optimization`

3. **Directory Management**:
   - Test Fixtures:
     ```python
     @pytest.fixture(scope="function")
     def test_dir():
         temp_dir = tempfile.mkdtemp()
         yield temp_dir
         if os.path.exists(temp_dir):
             shutil.rmtree(temp_dir)
     ```
   - MLflow Test Tracking:
     - Clean up `.trash` directories
     - Handle temporary experiment paths

4. **Action Plan**:
   a. **Data Column Standardization**:
      - Create/update data generation fixtures
      - Validate OHLCV columns in all scenarios
      - Add column validation to environment initialization
   
   b. **Environment Fixes**:
      - Fix Multi-Agent reset method
      - Update PaperTrading fee parameter
      - Resolve PolicyNetwork initialization
   
   c. **Test Alignment**:
      - Update Risk Manager test expectations
      - Clean up Tuner parameter handling
      - Improve directory management fixtures

5. **Testing Guidelines**:
   - Run tests after each component fix
   - Verify failure count decreases
   - Document any test expectation changes
   - Use proper test isolation and cleanup

```markdown
# Development Guide

This development guide defines the core principles, naming and formatting rules, testing standards, and a recommended workflow for refactoring and maintaining the trading bot codebase. By following these guidelines, we ensure code clarity, maintainability, and consistent behavior across all components.

---

## Core Principles

1. **MVP Stability**:  
   - Preserve the stability of core files (`train.py`, `trading_env.py`, `ppo_agent.py`).
   - Extend functionality via new modules or wrappers rather than directly modifying stable core components.

2. **Incremental Improvements**:  
   - Implement and test new features in isolation.
   - Only integrate into the main pipeline once fully validated.

3. **Testing and CI/CD**:  
   - Write tests for every new component (unit and integration tests).
   - Run tests before merging changes.
   - Utilize CI/CD pipelines (e.g., GitHub Actions) for automated testing and linting.

4. **Documentation & Transparency**:  
   - Document all code changes, function signatures, and parameter definitions.
   - Maintain comprehensive docstrings and comments.
   - Update this guide and other documentation files regularly.

---

## Naming and Formatting Rules

### Data Columns
- **OHLCV Columns**: Always use `$` prefix.
  - Required: `$open`, `$high`, `$low`, `$close`, `$volume`
  - Additional features: `$<feature_name>` (e.g., `$rsi`, `$macd_signal`)
- **Rationale**: Ensures a global standard that all data-related components recognize, simplifying debugging and validation.

### Environment Parameters
- **Data Parameter**: Use `df` for all pandas DataFrame inputs (no `data`, `dataframe`, etc.).
- **Fee Parameter**: Use `trading_fee` everywhere (no `transaction_fee` or `fee` alone).

### Observation Space
- **Shape**: `(window_size, n_features)` (2D array) from the environment.
- **Flattening**: If needed, flatten `(window_size, n_features)` inside the model/preprocessing stage, not in the environment.
- **Consistency**: All tests and components must agree on this shape.

### MLflow Configuration
- **Experiment Directory**: `./mlflow_runs/`
- **File Format**: Prefer parquet over CSV for logging artifacts.
- **Naming Pattern**: Use consistent experiment names (e.g., `trading_bot_dev` or `trading_bot_prod`).

### Async & Paper Trading
- **Async Methods**: Clearly name asynchronous functions (e.g., `async def initialize()`, `async def run_stream()`).
- **Mocking Real-Time Data**: Mock WebSocket or external calls in tests. Document how to do so consistently.
- **Cleanup**: Ensure every async component has a `cleanup()` or `teardown()` method.

### Hyperparameter Optimization (Ray Tune)
- **Ray Configuration**: Use `storage_path` for Ray's result directories (avoid `local_dir`).
- **Loggers**: Update code to the current Ray Tune API; do not revert to older parameters.
- **Version Control**: Document the Ray Tune version and features in use.

### General Code Style
- **Function/Variable Names**: `snake_case` for Python functions and variables.
- **Class Names**: `PascalCase` for classes.
- **Constants**: `UPPER_CASE` for constants.
- **Indentation**: 4 spaces, no tabs.
- **Line Length**: Aim for ≤100 characters per line.
- **Imports & PEP8**: Follow PEP8 and keep imports organized.

### Documentation Standards
- **Docstrings**: Every public class, method, or function must have a docstring.
- **Comments**: Explain non-obvious choices, avoid outdated comments.

---

## Testing Standards

### Test Requirements
1. **Unit Tests**: Each new function/module must have unit tests covering:
   - Success cases
   - Error handling
   - Edge cases (empty data, NaN values, boundary conditions)
   
2. **Integration Tests**: After unit tests pass, ensure the entire pipeline works together:
   - Data pipeline → Feature generation → Model → Backtesting  
   - Check for consistent naming, formatting, and data handling throughout.

3. **Performance & Stress Tests**:
   - Optional but recommended for long-running tasks.
   - Ensure memory usage, GPU utilization, and run times are within acceptable bounds.

### Testing Conventions
- Use `pytest` for Python tests.
- Use fixtures for reusable test data.
- Separate tests by functionality (e.g., `test_data_pipeline.py`, `test_agents.py`, `test_backtest.py`).
- Follow the Arrange-Act-Assert pattern in tests.

---

## Data Validation Rules

1. **Price Data**:
   - `$high >= $low`
   - `$volume >= 0`
   - No timestamp gaps or duplicates

2. **Technical Indicators**:
   - RSI: 0 ≤ RSI ≤ 100
   - Moving Averages within price range
   - Bollinger Bands: Upper ≥ Middle ≥ Lower
   - Volume indicators must be non-negative

3. **NaN Handling**:
   - Detect and fill NaN values with forward/backward fill.
   - Log warnings when NaNs are detected.

---

## Caching and Feature Generation

- **Cache Manager**:  
  - Store cached data consistently.
  - Validate cached data before use.
  - Document cache invalidation policies.

- **Feature Generation**:
  - Input must be `$`-prefixed OHLCV columns.
  - Output must have no NaNs and be within valid ranges.
  - Clearly document feature calculations.

---

## Refactoring & Migration Workflow

If the codebase requires a major refactoring to align with these guidelines, follow this suggested order of operations:

1. **Update Configuration & Constants**:  
   - Files: `config/*.yaml`  
   - Rename parameters (e.g., `transaction_fee`), ensure `$` columns in configs.

2. **Refactor Data Pipeline**:
   - Files: `data_loader.py`, `feature_generator.py`  
   - Apply `$`-prefixed columns, `df` parameter, and validate data.

3. **Standardize the Environment**:
   - Files: `trading_env.py`, `base_env.py`  
   - Use `df` and `transaction_fee`, ensure `(window_size, n_features)` observation.

4. **Align Agents & Models**:
   - Files: `ppo_agent.py`, `policy_network.py`, `value_network.py`  
   - Handle 2D input consistently, flatten only in model code.

5. **Unify Backtester & Scenarios**:
   - Files: `backtest.py`, `advanced_backtest.py`  
   - Standardize metrics, columns, and parameter names.

6. **MLflow & Logging Setup**:
   - Files: `mlflow_manager.py`, tests  
   - Ensure `mlflow_runs/` directory, parquet format, consistent experiment naming.

7. **Async & Paper Trading**:
   - Files: `paper_trading.py`, WebSocket mocks  
   - Fix async patterns, ensure cleanup, follow naming conventions.

8. **Hyperopt & Ray Tune**:
   - Files: `hyperopt_tuner.py`, tests  
   - Use `storage_path`, follow latest Ray API.

9. **Integration Tests & CI/CD**:
   - Run integration tests after each step.
   - Confirm consistent naming, formatting, and directory structures.

10. **Documentation**:
    - Update `README.md` and `DEVELOPMENT_GUIDELINES.md` to reflect all changes.
    - Add a changelog or migration guide.

---

## Debugging Checklist

1. **Logs**:
   - Check DEBUG logs for warnings and errors.
   - Look at execution times and memory usage logs.

2. **Data Validation**:
   - Ensure `$` prefix in columns.
   - Check for NaNs and invalid ranges.

3. **Component Isolation**:
   - Test individual components (data loader, feature generator, agent).
   - Run unit tests to pinpoint the issue.

4. **Integration Checks**:
   - Verify that data flows correctly through the pipeline.
   - Ensure no shape or naming mismatches occur between components.

5. **Performance Reviews**:
   - Check if caching is effective.
   - Profile memory and CPU/GPU usage.

---

## Common Issues & Solutions

- **Missing Features**:  
  Check feature generator output, log missing columns, update pipeline accordingly.

- **NaN Values**:  
  Fill forward/backward and log warnings.

- **Data Range Errors**:  
  Validate `$high >= $low`, correct any data anomalies.

- **Performance Slowdown**:  
  Monitor execution time, optimize batch sizes, leverage Ray for parallel processing.

---

## References

- [Project Repository](https://github.com/your-repo)
- [Issue Tracker](https://github.com/your-repo/issues)
- [README.md](./README.md)

By adhering to these principles, rules, and workflow steps, you ensure a consistent, robust, and maintainable trading bot codebase.