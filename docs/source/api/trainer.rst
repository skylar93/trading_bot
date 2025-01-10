Distributed Trainer
=================

Overview
--------

The Distributed Trainer module provides a comprehensive framework for distributed training of trading agents using Ray and MLflow.

Key Components
-------------

TrainingConfig
^^^^^^^^^^^^

Configuration dataclass for training parameters:

* ``num_epochs``: Number of training epochs (default: 100)
* ``batch_size``: Training batch size (default: 64)
* ``learning_rate``: Learning rate (default: 3e-4)
* ``gamma``: Discount factor (default: 0.99)
* ``hidden_size``: Network hidden size (default: 256)
* ``num_parallel``: Number of parallel workers (default: 4)
* ``initial_balance``: Initial portfolio balance (default: 10000.0)
* ``trading_fee``: Trading fee percentage (default: 0.001)
* ``window_size``: Observation window size (default: 20)
* ``tune_trials``: Number of tuning trials (default: 10)
* ``experiment_name``: MLflow experiment name (default: "trading_bot")

DistributedTrainer
^^^^^^^^^^^^^^^

Main trainer class for distributed training:

Core Methods:
    * ``__init__(config: TrainingConfig)``: Initialize trainer
    * ``create_env(data: pd.DataFrame)``: Create trading environment
    * ``create_agent(env)``: Create PPO agent
    * ``train(train_data, val_data)``: Train agent with validation
    * ``evaluate(agent, env)``: Evaluate agent performance
    * ``tune(train_data, val_data)``: Run hyperparameter tuning
    * ``save_model(path)``: Save trained model
    * ``load_model(path)``: Load trained model
    * ``cleanup()``: Clean up resources

Implementation Details
--------------------

Training Pipeline
^^^^^^^^^^^^^^

* MLflow experiment tracking
* Parallel training with Ray
* Validation during training
* Metrics logging and monitoring

Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^

* Ray Tune integration
* Search space definition
* Trial management
* Best configuration selection

Dependencies
-----------

* ``ray``: Distributed computing
* ``mlflow``: Experiment tracking
* ``torch``: Deep learning
* ``pandas``: Data handling
* ``numpy``: Numerical operations
* ``logging``: Error tracking

Usage Example
------------

.. code-block:: python

    # Initialize trainer
    config = TrainingConfig(
        num_epochs=100,
        batch_size=64,
        learning_rate=3e-4
    )
    trainer = DistributedTrainer(config)

    # Train agent
    metrics = trainer.train(train_data, val_data)

    # Tune hyperparameters
    best_config = trainer.tune(train_data, val_data)

    # Save model
    trainer.save_model("models/best_model.pt")

    # Cleanup
    trainer.cleanup()

Best Practices
-------------

1. Training Configuration
   * Set appropriate batch sizes
   * Configure learning parameters
   * Define validation strategy

2. Resource Management
   * Monitor GPU usage
   * Clean up Ray resources
   * Track experiment artifacts

3. Hyperparameter Tuning
   * Define reasonable search spaces
   * Set appropriate number of trials
   * Monitor tuning progress

4. Model Management
   * Save checkpoints regularly
   * Track best models
   * Document model versions

Recent Changes
-------------

* Added MLflow integration
* Enhanced hyperparameter tuning
* Improved validation pipeline
* Added model versioning 