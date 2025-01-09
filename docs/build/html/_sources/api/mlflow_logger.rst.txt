MLflow Logger API
==============

Overview
--------

The MLflow logger module provides a convenient wrapper for MLflow experiment tracking, with specialized support for trading models and backtest results.

Class Structure
-------------

MLflowLogger
^^^^^^^^^^

.. code-block:: python

    class MLflowLogger:
        """Wrapper for MLflow experiment tracking."""
        
        def __init__(self, experiment_name: str):
            """Initialize logger.
            
            Args:
                experiment_name: Name of MLflow experiment
            """
            self.experiment_name = experiment_name
            self._setup_experiment()

Key Methods
---------

Run Management
^^^^^^^^^^^

.. code-block:: python

    def start_run(self, run_name: Optional[str] = None):
        """Start new MLflow run.
        
        Args:
            run_name: Optional name for run
        """
        if not run_name:
            run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}"
        mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name
        )
        
    def end_run(self):
        """End current MLflow run."""
        mlflow.end_run()

Logging Methods
^^^^^^^^^^^^

.. code-block:: python

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        # Convert complex types to strings
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                params[key] = json.dumps(value)
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], 
                   step: Optional[int] = None):
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)

Artifact Logging
^^^^^^^^^^^^^

.. code-block:: python

    def log_model(self, model, artifact_path: str):
        """Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path for artifact
        """
        mlflow.pytorch.log_model(model, artifact_path)
        
    def log_figure(self, figure, artifact_path: str):
        """Log matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_path: Path for artifact
        """
        mlflow.log_figure(figure, artifact_path)
        
    def log_dict(self, dictionary: Dict, artifact_path: str):
        """Log dictionary as JSON.
        
        Args:
            dictionary: Dictionary to log
            artifact_path: Path for artifact
        """
        mlflow.log_dict(dictionary, artifact_path)

Backtest Results
^^^^^^^^^^^^^

.. code-block:: python

    def log_backtest_results(self, results: Dict,
                           artifact_path: str = 'backtest'):
        """Log backtest results.
        
        Args:
            results: Dictionary of backtest results
            artifact_path: Base path for artifacts
        """
        # Log trades as table
        if 'trades' in results:
            trades_df = pd.DataFrame(results['trades'])
            mlflow.log_table(
                trades_df,
                f"{artifact_path}/trades.csv"
            )
            
        # Log portfolio values
        if 'portfolio_values' in results:
            values_df = pd.DataFrame(results['portfolio_values'])
            mlflow.log_table(
                values_df,
                f"{artifact_path}/portfolio_values.csv"
            )
            
        # Log full results as JSON
        self.log_dict(results, f"{artifact_path}/results.json")

Best Run Retrieval
^^^^^^^^^^^^^^^

.. code-block:: python

    def get_best_run(self, metric_name: str,
                    mode: str = 'max') -> Dict:
        """Get best run by metric.
        
        Args:
            metric_name: Metric to sort by
            mode: 'max' or 'min'
            
        Returns:
            dict: Best run info
        """
        order = "DESC" if mode == 'max' else "ASC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"]
        )
        if len(runs) == 0:
            return None
            
        best_run = runs.iloc[0]
        return {
            'run_id': best_run.run_id,
            'metrics': best_run.metrics,
            'params': best_run.params
        }

Implementation Details
-------------------

Experiment Management
^^^^^^^^^^^^^^^^^

1. Setup:
   - Create/get experiment
   - Configure tracking URI
   - Handle experiment ID

2. Run Lifecycle:
   - Start/end runs
   - Run naming
   - Run metadata

Logging Features
^^^^^^^^^^^^^

1. Parameter Logging:
   - Type conversion
   - Complex object handling
   - Validation

2. Metric Logging:
   - Step tracking
   - Batch logging
   - History management

3. Artifact Management:
   - Model saving
   - Figure logging
   - JSON/CSV handling

Dependencies
----------

- MLflow
- Pandas (DataFrame operations)
- PyTorch (model logging)
- JSON (serialization)

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Create logger
    logger = MLflowLogger("trading_experiment")
    
    # Start run
    logger.start_run("training_001")
    
    # Log parameters and metrics
    logger.log_params({
        'learning_rate': 0.001,
        'batch_size': 64
    })
    
    logger.log_metrics({
        'loss': 0.5,
        'accuracy': 0.95
    }, step=1)
    
    # Log model
    logger.log_model(model, "model")
    
    # End run
    logger.end_run()

Backtest Logging
^^^^^^^^^^^^^

.. code-block:: python

    # Log backtest results
    logger.log_backtest_results({
        'trades': trades_list,
        'portfolio_values': values_list,
        'metrics': {
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.2
        }
    })
    
    # Get best run
    best_run = logger.get_best_run(
        metric_name='sharpe_ratio',
        mode='max'
    )

Best Practices
-----------

1. Experiment Organization
^^^^^^^^^^^^^^^^^^^

- Use consistent naming
- Group related runs
- Track experiment versions

2. Parameter Management
^^^^^^^^^^^^^^^^^

- Document parameter meanings
- Use consistent types
- Handle nested structures

3. Metric Tracking
^^^^^^^^^^^^

- Define key metrics
- Use appropriate steps
- Track training progress

4. Artifact Management
^^^^^^^^^^^^^^^^^

- Organize artifact paths
- Version artifacts
- Clean up unused artifacts 