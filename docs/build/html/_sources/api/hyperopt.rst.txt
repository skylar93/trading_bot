Hyperopt API
===========

Overview
--------

The hyperopt module provides a minimal hyperparameter optimization framework with random search capabilities.

Class Structure
-------------

MinimalTuner
^^^^^^^^^^

.. code-block:: python

    class MinimalTuner:
        """Simple hyperparameter tuner using random search."""
        
        def __init__(self, train_fn):
            """Initialize tuner.
            
            Args:
                train_fn: Training function to optimize
                         Takes parameters dict, returns score
            """
            self.train_fn = train_fn
            self.best_score = float('-inf')
            self.best_params = None

Key Methods
---------

Optimization
^^^^^^^^^

.. code-block:: python

    def optimize(self, search_space: Dict[str, Tuple[float, float]], 
                n_trials: int = 10) -> Dict[str, float]:
        """Run hyperparameter optimization.
        
        Args:
            search_space: Parameter ranges {name: (min, max)}
            n_trials: Number of trials to run
            
        Returns:
            dict: Best parameters found
        """
        for _ in range(n_trials):
            # Sample parameters
            params = {
                name: np.random.uniform(low, high)
                for name, (low, high) in search_space.items()
            }
            
            # Evaluate parameters
            score = self.train_fn(params)
            
            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                
        return self.best_params

Implementation Details
-------------------

Search Process
^^^^^^^^^^^

1. Parameter Sampling:
   - Uniform sampling in ranges
   - Independent parameters
   - No parameter dependencies

2. Evaluation:
   - Call training function
   - Get performance score
   - Track best results

3. Result Management:
   - Store best parameters
   - Store best score
   - Optional history

Dependencies
----------

- NumPy (random sampling)
- typing (type hints)

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Define training function
    def train_and_evaluate(params):
        model = create_model(**params)
        score = train_model(model)
        return score
    
    # Create tuner
    tuner = MinimalTuner(train_and_evaluate)
    
    # Define search space
    space = {
        'learning_rate': (1e-4, 1e-2),
        'hidden_size': (32, 128),
        'dropout': (0.1, 0.5)
    }
    
    # Run optimization
    best_params = tuner.optimize(
        search_space=space,
        n_trials=50
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {tuner.best_score}")

Best Practices
-----------

1. Search Space Design
^^^^^^^^^^^^^^^^^

- Use appropriate ranges
- Consider parameter scales
- Set reasonable bounds

2. Trial Management
^^^^^^^^^^^^^

- Set sufficient trials
- Monitor convergence
- Consider time budget

3. Score Function
^^^^^^^^^^^^

- Use consistent metrics
- Handle failures gracefully
- Consider validation

4. Result Analysis
^^^^^^^^^^^^

- Check parameter distributions
- Analyze score trends
- Validate stability 