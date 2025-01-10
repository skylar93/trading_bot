Hyperopt Tuner API
===============

Overview
--------

The ``MinimalTuner`` class provides a streamlined interface for hyperparameter optimization using Ray Tune, with optional MLflow integration for experiment tracking.

Class Structure
-------------

.. code-block:: python

    class MinimalTuner:
        def __init__(self, train_data, val_data, mlflow_tracking=True):
            """Initialize hyperparameter tuner.
            
            Args:
                train_data: Training dataset
                val_data: Validation dataset
                mlflow_tracking: Whether to use MLflow
            """
            self.train_data = train_data
            self.val_data = val_data
            self.mlflow_tracking = mlflow_tracking
            if mlflow_tracking:
                self.mlflow_manager = MLflowManager()

Key Methods
---------

Objective Function
^^^^^^^^^^^^^^^

.. code-block:: python

    def objective(config, session=None):
        """Objective function for Ray Tune.
        
        Args:
            config: Hyperparameter configuration
            session: Ray Tune session
            
        Returns:
            None (reports metrics via session)
        """
        # Train agent with config
        agent = train_agent(config)
        
        # Evaluate on validation set
        metrics = evaluate_config(config, agent)
        
        # Report results
        session.report(metrics)

Configuration Evaluation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def evaluate_config(config, agent=None):
        """Evaluate configuration on validation data.
        
        Args:
            config: Hyperparameter configuration
            agent: Optional pre-trained agent
            
        Returns:
            dict: Evaluation metrics
        """
        env = TradingEnvironment(self.val_data)
        
        # Run evaluation episodes
        returns = []
        for episode in range(20):
            episode_return = run_episode(env, agent)
            returns.append(episode_return)
            
        # Calculate metrics
        metrics = calculate_metrics(returns)
        return metrics

Optimization Process
^^^^^^^^^^^^^^^^^

.. code-block:: python

    def optimize(self, search_space=None, num_samples=10, 
                max_concurrent=4, **kwargs):
        """Run hyperparameter optimization.
        
        Args:
            search_space: Parameter search space
            num_samples: Number of trials
            max_concurrent: Max parallel trials
            **kwargs: Additional tuner config
            
        Returns:
            dict: Best trial results
        """
        # Setup tuner
        search_alg = OptunaSearch(
            metric="score",
            mode="max"
        )
        
        tuner = tune.Tuner(
            self.objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent
            )
        )
        
        # Run optimization
        results = tuner.fit()
        return results.get_best_result()

Implementation Details
-------------------

Search Space Configuration
^^^^^^^^^^^^^^^^^^^^^^

1. Default Parameters:
   - Learning rate (log uniform)
   - Hidden size (choice)
   - Batch size (choice)
   - Training steps (fixed)

2. Custom Parameters:
   - Support for user-defined spaces
   - Integration with Optuna
   - Constraint handling

Evaluation Process
^^^^^^^^^^^^^^^

1. Training Phase:
   - Configure agent with params
   - Train on training data
   - Track training metrics

2. Validation Phase:
   - Run multiple episodes
   - Calculate performance metrics
   - Compute composite score

3. Metric Reporting:
   - Performance metrics
   - Resource utilization
   - Training statistics

Dependencies
----------

- Ray (tune, air)
- MLflow (optional)
- Optuna
- Pandas

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Initialize tuner
    tuner = MinimalTuner(train_data, val_data)
    
    # Define search space
    space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([32, 64, 128])
    }
    
    # Run optimization
    results = tuner.optimize(
        search_space=space,
        num_samples=50,
        max_concurrent=4
    )
    
    # Get best configuration
    best_config = results.config
    best_metrics = results.metrics

MLflow Integration
^^^^^^^^^^^^^^^

.. code-block:: python

    # With MLflow tracking
    tuner = MinimalTuner(
        train_data, 
        val_data,
        mlflow_tracking=True
    )
    
    # Metrics and parameters automatically logged
    results = tuner.optimize(search_space)

Best Practices
-----------

1. Search Space Design
^^^^^^^^^^^^^^^^^^

- Use appropriate distributions
- Set reasonable bounds
- Consider parameter interactions

2. Resource Management
^^^^^^^^^^^^^^^^^

- Monitor memory usage
- Adjust concurrent trials
- Set appropriate timeouts

3. Evaluation Strategy
^^^^^^^^^^^^^^^^^

- Use multiple validation episodes
- Consider different metrics
- Balance exploration/exploitation

4. Result Analysis
^^^^^^^^^^^^^

- Analyze parameter importance
- Check for convergence
- Compare trial performances 