MLflow Management Components
=========================

Overview
--------

The MLflow management components provide different implementations for experiment tracking, metric logging, and model management. Each implementation serves specific use cases while sharing common core functionality.

Component Structure
-----------------

MLflowManager (New)
^^^^^^^^^^^^^^^^^

Located in ``training/mlflow_manager_new.py``, this is the latest implementation with context manager support.

Key Features:
  * Context manager interface (``with`` statement support)
  * Experiment initialization and run management
  * Parameter, metric, and artifact logging
  * Backtest results logging with Parquet support
  * Cleanup functionality

.. code-block:: python

    from training.mlflow_manager_new import MLflowManager

    with MLflowManager(tracking_dir="experiments") as mlf:
        mlf.log_params({"learning_rate": 0.001})
        mlf.log_metrics({"loss": 0.5})
        mlf.log_model(model, "model_checkpoint")

MLflowManager (SQLite)
^^^^^^^^^^^^^^^^^^^

Located in ``training/mlflow_manager.py``, this version uses SQLite for experiment tracking.

Key Features:
  * SQLite-based tracking (``mlflow.db``)
  * DataFrame logging support
  * Backtest results management
  * Model loading utilities

.. code-block:: python

    from training.mlflow_manager import MLflowManager

    manager = MLflowManager(experiment_name="trading_experiment")
    manager.start_run()
    manager.log_dataframe(results_df, "backtest_results")
    manager.end_run()

MLflowManager (Utils)
^^^^^^^^^^^^^^^^^^

Located in ``training/utils/mlflow_utils.py``, this utility version focuses on training metrics.

Key Features:
  * Training/validation metrics logging
  * Simplified experiment setup
  * Basic model logging support

.. code-block:: python

    from training.utils.mlflow_utils import MLflowManager

    manager = MLflowManager()
    manager.start_run()
    manager.log_training_metrics({
        "loss": 0.5,
        "val_loss": 0.6
    })
    manager.end_run()

Implementation Details
--------------------

Common Functionality
^^^^^^^^^^^^^^^^^

All implementations share these core features:
  * Experiment management
  * Run lifecycle control
  * Basic metric and parameter logging
  * PyTorch model artifact support

Key Differences
^^^^^^^^^^^^

1. Context Management:
   * New version: Full context manager support
   * SQLite version: Basic context management
   * Utils version: No context management

2. Storage:
   * New version: File-based tracking
   * SQLite version: SQLite database
   * Utils version: Default MLflow storage

3. Special Features:
   * New version: Enhanced cleanup, Parquet support
   * SQLite version: DataFrame handling, model loading
   * Utils version: Training metrics convenience methods

Dependencies
-----------

Common Dependencies:
  * MLflow
  * PyTorch
  * Pandas
  * Python standard library (os, pathlib, shutil)

Version-Specific Dependencies:
  * New version: tempfile
  * SQLite version: sqlite3
  * Utils version: datetime

Usage Guidelines
--------------

When to Use Each Version
^^^^^^^^^^^^^^^^^^^^^

1. New Version (mlflow_manager_new.py):
   * Modern applications with context manager support
   * Projects requiring clean experiment tracking
   * Backtest result analysis with Parquet

2. SQLite Version (mlflow_manager.py):
   * Applications needing persistent storage
   * Projects with heavy DataFrame usage
   * Model loading/reuse scenarios

3. Utils Version (mlflow_utils.py):
   * Training-focused applications
   * Simple metric logging needs
   * Basic experiment tracking

Best Practices
------------

1. Version Selection
^^^^^^^^^^^^^^^^^
* Choose based on storage needs
* Consider context manager requirements
* Evaluate special feature requirements

2. Experiment Management
^^^^^^^^^^^^^^^^^^^^
* Use consistent naming conventions
* Clean up experiments when needed
* Handle nested runs appropriately

3. Metric Logging
^^^^^^^^^^^^^
* Group related metrics logically
* Use consistent metric names
* Consider logging frequency

4. Model Management
^^^^^^^^^^^^^^^
* Use appropriate artifact paths
* Include model metadata
* Implement version control

Recent Changes
------------

* Added context manager support in new version
* Enhanced backtest results logging
* Improved cleanup functionality
* Added training metrics convenience methods 