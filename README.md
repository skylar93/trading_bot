Below is a revised `README.md` that integrates the guidelines, rules, and refactoring steps discussed. It highlights the current project state, the development guidelines, and the workflow recommended. This document is meant to replace your current `README.md`.

```markdown
# Trading Bot Project

A reinforcement learning-based trading bot using PPO agents, comprehensive backtesting, and real-time integration for future live trading scenarios. The codebase follows strict development guidelines and naming conventions to ensure maintainability, reproducibility, and robust testing.

---

## Latest Achievements 🎯

- **Risk-Aware Backtesting System Implemented (2024-12-13)**:
  - Portfolio returns: +1.42%
  - Sharpe Ratio: 0.43, Sortino Ratio: 0.60, Max Drawdown: -4.26%
  - Dynamic position sizing and stop-loss integrated
  - Scenario-based testing (flash crash, low liquidity)
  - Initiated hyperparameter optimization with Ray Tune and MLflow tracking
  - Completed unit tests for `hyperopt_env.py`, `hyperopt_agent.py`, and `hyperopt_tuner.py`
  - Optimization pipeline decoupled for independent execution

- **Advanced Backtesting & Visualization**:
  - Flash crash and low liquidity scenario analysis
  - Comprehensive metrics (Sharpe, Sortino, MDD)
  - Portfolio visualization and trade analysis
  - Real-time simulation monitoring
  - Enhanced risk management metrics

- **Hyperparameter Optimization**:
  - Under `training/hyperopt/` modules
  - Ray Tune integration for tuning (learning rate, batch size, architectures)
  - MLflow for tracking experiments and reproducibility
  - Dedicated script `scripts/run_hyperopt.py` for full hyperparameter search

- **Resource Optimization System**:
  - Located in `training/monitoring/*` (e.g., `metrics_collector.py`, `worker_manager.py`)
  - Ray Actor model for distributed processing
  - Dynamic worker scaling (2 to 8 workers)
  - Performance metrics (batch time, memory, GPU utilization)
  - Automated optimization of batch sizes and resource usage

- **Real-Time Trading Preparation**:
  - CCXT WebSocket integration for live data streaming
  - Paper trading environment (`paper_trading.py`) for real-time strategy testing
  - `trading_env.py` adapted for on-the-fly decision-making

---

## Core Architecture Components

### Data Pipeline Layer
- **Tech Stack**: TA-Lib + ccxt
- **Flow**:  
  `Raw Data (ccxt) → TA-Lib Pipeline → Feature Store → Training Data`
- **Structure**:
  ```
  data/
    raw/
    processed/
    features/
    utils/data_loader.py
    utils/feature_generator.py
    utils/websocket_loader.py
  ```

### Training and Inference Layer
- **Frameworks**: RLlib + PPO Agent
- **Key Components**:
  - `train.py`: Stable single-agent training
  - `ppo_agent.py`: PPO agent
  - `trading_env.py`: Stable trading environment (single-agent)
  - `train_multi_agent.py`: Multi-agent environment
  - `evaluation.py`, `backtest.py`, `advanced_backtest.py`: Evaluation & scenario testing
  - `training/hyperopt/`: Hyperparameter tuning modules

### Web Interface Layer
- **Frameworks**: FastAPI + Streamlit
- **Features**:
  - Model selection, parameter tuning, monitoring
  - Live performance visualization
- **File**:
  - `deployment/web_interface/app.py`

---

## Completed Features ✅

1. **Data Pipeline**:  
   - ccxt ingestion, 44 TA-Lib indicators
   - `$`-prefixed column names for OHLCV (e.g., `$open`, `$close`)
   - Caching for performance

2. **Reinforcement Learning**:  
   - PPO agent implementation
   - Stable training (`train.py`)
   - MLflow integrated
   - Validated training runs

3. **Backtesting System**:  
   - Metrics: Sharpe, Sortino, MDD
   - Scenario tests (flash crash, low liquidity)
   - Real-time simulation monitoring

4. **Visualization Tools**:  
   - Portfolio value, returns distribution, drawdown
   - Risk metric visualizations

5. **Advanced Scenarios**:  
   - Flash crash and low liquidity tests
   - Risk management with dynamic sizing, stop-loss

6. **Risk-Aware Backtesting**:  
   - Leverage, drawdown control
   - Stop-loss mechanisms validated

7. **Hyperparameter Optimization**:  
   - `training/hyperopt/` modules
   - Ray Tune for parameter search
   - MLflow for reproducibility

8. **Real-Time Trading Preparation**:  
   - CCXT WebSocket integration
   - `paper_trading.py` for sandbox tests

9. **Resource Optimization & Monitoring**:  
   - Ray Actor model for distributed tasks
   - Worker scaling and performance metrics

---

## Development Guidelines

**Refer to**: [DEVELOPMENT_GUIDELINES.md](DEVELOPMENT_GUIDELINES.md) for full details.

### Key Naming & Formatting Rules

- **Data Columns**: Always use `$` prefix (e.g., `$open`, `$close`).
- **Parameters**: Use `df` for DataFrame parameters, `transaction_fee` for fees.
- **Observation Shape**: `(window_size, n_features)` 2D arrays from the environment. Flatten only inside the model if needed.
- **MLflow**: Use `./mlflow_runs/` for experiments and parquet format.
- **Async and Paper Trading**: Follow consistent async method naming, mock WebSocket in tests, ensure cleanup methods.

### Testing Standards

- Use pytest for unit and integration tests.
- Test success, error, and edge cases.
- Validate data formats, handle NaN with ffill/bfill.
- Integration tests check full pipeline (data → features → model → backtest).

### Refactoring and Migration Steps (if needed)

1. Update config files (`config/*.yaml`) to match naming conventions.
2. Refactor data pipeline (`data_loader.py`, `feature_generator.py`) to `$` columns.
3. Standardize `trading_env.py` parameters and observation shapes.
4. Align PPO agent and networks with `(window_size, n_features)` inputs.
5. Unify backtester metrics and scenario outputs.
6. Configure MLflow experiments under `./mlflow_runs/`.
7. Fix async patterns in paper trading and mock WebSocket in tests.
8. Update Hyperopt code to use `storage_path` for Ray Tune.
9. Re-run integration tests and CI/CD pipelines after each step.
10. Document changes and update guidelines.

---

## Running the Project

1. **Setup**
   ```bash
   cd Users/skylar/Desktop/trading_bot
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Test Core System**
   ```bash
   pytest tests/
   ```

3. **Launch Web UI**
   ```bash
   streamlit run deployment/web_interface/app.py
   ```

4. **Hyperparameter Optimization**
   ```bash
   python scripts/run_hyperopt.py
   ```
   - Monitor via MLflow UI.

---

## Additional Notes

- **MVP Stability**:  
  Focus on core stable files: `train.py`, `trading_env.py`, `ppo_agent.py`.
  
- **Code Changes**:  
  Use inheritance and composition. Follow naming conventions and formatting rules from the guidelines.

- **Experiment Tracking**:  
  Use MLflow for all runs and experiments. Stick to the documented directories and parameter naming.

- **CI/CD**:  
  Integrate GitHub Actions or similar for automated testing and linting.

- **Resource Monitoring**:  
  Ray Actor model for scaling workers and optimizing batch sizes. Validate performance with tests and logs.

## Additional Guidelines and Recommendations

### Environment Variables and Configuration

To maintain flexibility and security in different deployment contexts (development, staging, production), we employ environment variables and external configuration files:

- **Required Variables**:  
  - `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET` for CCXT-based data retrieval  
  - `MLFLOW_TRACKING_URI` for experiment logging  
  - `REDIS_URL`, `POSTGRES_URL` (if database or caching layers are employed)  
  - `WEB_SOCKET_ENDPOINT` for real-time data streams

- **Credential Management**:  
  - Store API keys and secrets in a `.env` file that is not committed to version control.  
  - For production or sensitive deployments, consider using a secure secrets management solution such as HashiCorp Vault or AWS Secrets Manager.  
  - The cursor and CI/CD pipelines may inject credentials as environment variables at runtime.

- **Configuration Environments**:
  - **Development**: Local `.env.development` file and local config overrides.  
  - **Staging**: `.env.staging` loaded on staging servers, possibly with test exchange keys.  
  - **Production**: `.env.production` managed by Ops team or secret vault integration.  
  - Switch between environments by setting a `ENV` variable or passing a command-line argument; the application’s startup script or Dockerfile can select the correct configuration.

### Dependency Versioning and Management

Consistent dependency management ensures reproducible builds and predictable behavior across different machines and stages:

- **Recommended Approach**:
  - Use a `requirements.txt` file pinned with exact versions for stability.  
  - Alternatively, adopt Poetry for handling dependencies and virtual environments, which generates a `poetry.lock` file for strict reproducibility.
  
- **Freezing Dependencies**:
  - If using `requirements.txt`, after installing or upgrading dependencies, run `pip freeze > requirements.txt` to lock versions.  
  - If using Poetry, rely on `poetry.lock` for version consistency.

### Contribution Guidelines

For teams working collaboratively or expecting external contributions, establish clear contribution policies:

- **Feature Proposals & Bug Fixes**:
  - Open a GitHub issue or pull request (PR) describing the proposed changes or bug fixes.  
  - Include tests and documentation updates with each PR.
  
- **Code Review & Branching**:
  - Use a standard branch naming convention: `feature/<short_description>`, `fix/<issue_number>`.  
  - Require at least one code review approval before merging into `main` or `master`.
  
- **Style Enforcement**:
  - Integrate `pre-commit` hooks to run linting (pylint), type checks (mypy), and tests (pytest) before commits.  
  - The cursor (or CI/CD) will reject merges that fail these checks.

### Known Limitations and Future Improvements

While the current system is robust, there are known areas for enhancement:

- **Known Edge Cases**:
  - Extremely low liquidity scenarios may not fully reflect real exchange order book microstructure.
  - High latency environments or rate-limiting behaviors from some exchanges need further testing.
  
- **Roadmap**:
  - Additional exchange integrations (e.g., futures, options).  
  - Advanced risk models (e.g., VaR-based constraints, regime detection).
  - UI enhancements for Streamlit dashboards (real-time order book visualization, portfolio heatmaps).

### Performance Benchmarks

Performance goals help guide optimization efforts and ensure scalability:

- **Targets**:
  - Training runtime: Complete a single training epoch within a set time (e.g., < 5 minutes for standard configuration).
  - Maximum acceptable latency for real-time updates: < 1 second end-to-end from data ingestion to action decision.
  - Memory/GPU usage: Keep GPU utilization high without causing out-of-memory errors, and log these metrics with MLflow or Ray Tune’s logging.

- **Profiling & Benchmarking**:
  - Use built-in Python profiling (`cProfile`, `line_profiler`) to identify bottlenecks.
  - Leverage Ray’s dashboard and MLflow logging for performance metrics at scale.
  - Document the results and optimizations in `performance_notes.md` or a dedicated section in the repo.

### Security and Compliance Considerations

For systems that may eventually handle real-world trades or sensitive financial data:

- **Security Requirements**:
  - Encrypt all secrets at rest and in transit.
  - Sanitize logs to avoid printing API keys or sensitive info.
  
- **Compliance**:
  - If operating in regulated environments, consider KYC/AML checks and regulatory reporting.
  - Use security scanning tools (like `bandit`) and type checking (`mypy`) regularly.
  
- **Recommended Tools**:
  - `bandit` for Python security linting.
  - Regular dependency vulnerability scans (e.g., `pip-audit`).
