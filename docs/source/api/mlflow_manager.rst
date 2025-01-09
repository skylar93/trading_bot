MLflow Manager API
================

Overview
--------

The ``MLflowManager`` class provides a convenient interface for managing MLflow experiments, runs, and logging capabilities in the training process.

Class Structure
-------------

.. code-block:: python

    class MLflowManager:
        def __init__(self):
            """Initialize MLflow manager."""
            self.logger = logging.getLogger(self.__class__.__name__)

Key Methods
---------

start_run
^^^^^^^^

.. code-block:: python

    def start_run(self, run_name=None, nested=False):
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            Context manager for the run
        """

log_metrics
^^^^^^^^^

.. code-block:: python

    def log_metrics(self, metrics, step=None):
        """Log metrics to current MLflow run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """

log_params
^^^^^^^^

.. code-block:: python

    def log_params(self, params):
        """Log parameters to current MLflow run.
        
        Args:
            params: Dictionary of parameter names and values
        """

end_run
^^^^^^

.. code-block:: python

    def end_run(self):
        """End the current MLflow run if one exists."""

Implementation Details
-------------------

Context Management
^^^^^^^^^^^^^^^

The class provides context management for MLflow runs:

1. Safe run handling:
   - Automatic run cleanup
   - Nested run support
   - Run name management

2. Logging safety:
   - Checks for active runs
   - Warning logs for missing runs
   - Parameter validation

Dependencies
----------

- MLflow
- Python logging

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    mlflow_manager = MLflowManager()
    
    # Using context manager
    with mlflow_manager.start_run("training_run"):
        # Log parameters
        mlflow_manager.log_params({
            "learning_rate": 0.001,
            "batch_size": 64
        })
        
        # Log metrics
        mlflow_manager.log_metrics({
            "loss": 0.5,
            "accuracy": 0.95
        }, step=1)

Test Environment
^^^^^^^^^^^^^

.. code-block:: python

    # Automatic test experiment detection
    with mlflow_manager.start_run("test_run"):
        mlflow_manager.log_metrics({
            "test_metric": 1.0
        })

Best Practices
-----------

1. Run Management
^^^^^^^^^^^^^

- Use context managers for runs
- Give descriptive run names
- Handle nested runs properly

2. Metric Logging
^^^^^^^^^^^^^

- Use consistent metric names
- Include step numbers
- Group related metrics

3. Parameter Logging
^^^^^^^^^^^^^^^

- Log all important parameters
- Use clear parameter names
- Include parameter types/units

4. Testing
^^^^^^^

- Use test experiment when available
- Validate logged values
- Clean up test runs 