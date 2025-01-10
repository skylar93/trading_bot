Batch Strategy API
================

Overview
--------

The batch strategy module provides dynamic batch size adjustment strategies for training, with support for time-weighted and data-adaptive approaches.

Class Structure
-------------

BatchStats
^^^^^^^^

.. code-block:: python

    @dataclass
    class BatchStats:
        """Statistics for a single batch processing.
        
        Args:
            batch_size: Size of the batch
            processing_time: Time taken to process
            loss: Training loss value
            metrics: Additional metrics
        """
        batch_size: int
        processing_time: float
        loss: float
        metrics: Dict[str, float]

Base Strategy
^^^^^^^^^^^

.. code-block:: python

    class BatchStrategy:
        """Base class for dynamic batch processing strategies."""
        
        def __init__(self, target_time: float = 0.1):
            """Initialize batch strategy.
            
            Args:
                target_time: Target processing time per batch
            """
            self.target_time = target_time
            self.history = []
            
        def get_next_batch_size(self) -> int:
            """Calculate next batch size based on history.
            
            Returns:
                int: Next batch size to use
            """
            if len(self.history) < 5:
                return self.current_batch_size
                
            avg_time = np.mean([
                stat.processing_time 
                for stat in self.history[-5:]
            ])
            
            if avg_time > self.target_time:
                return int(self.current_batch_size * 0.8)
            else:
                return int(self.current_batch_size * 1.2)

Time-Sensitive Strategy
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class TimeSensitiveBatchStrategy(BatchStrategy):
        """Batch strategy with time-weighted history."""
        
        def get_next_batch_size(self) -> int:
            """Calculate next batch size with time weighting.
            
            Returns:
                int: Next batch size to use
            """
            base_size = super().get_next_batch_size()
            
            # Apply time weighting
            weights = np.linspace(0.5, 1.0, len(self.history[-5:]))
            weighted_time = np.average(
                [stat.processing_time for stat in self.history[-5:]],
                weights=weights
            )
            
            # Additional adjustment based on weighted time
            if weighted_time > self.target_time:
                return int(base_size * 0.9)
            else:
                return base_size

Adaptive Strategy
^^^^^^^^^^^^^

.. code-block:: python

    class AdaptiveBatchStrategy(BatchStrategy):
        """Data-aware batch size adjustment strategy."""
        
        def adjust_for_data(self, df: pd.DataFrame) -> int:
            """Adjust batch size based on data characteristics.
            
            Args:
                df: Input DataFrame
                
            Returns:
                int: Adjusted batch size
            """
            stats = self.analyze_data(df)
            base_size = self.get_next_batch_size()
            
            # Adjust for volatility
            if stats['volatility'] > 0.1:
                base_size *= 0.8
                
            # Adjust for missing data
            if stats['missing_ratio'] > 0.05:
                base_size *= 0.7
                
            # Adjust for unique values
            if stats['unique_ratio'] > 0.8:
                base_size *= 1.1
                
            return int(base_size)

Implementation Details
-------------------

Base Strategy Logic
^^^^^^^^^^^^^^^

1. History Management:
   - Track recent batch statistics
   - Use rolling window of 5 batches
   - Calculate average processing time

2. Size Adjustment:
   - Compare against target time
   - Increase/decrease by 20%
   - Maintain minimum size

Time-Sensitive Features
^^^^^^^^^^^^^^^^^^

1. Time Weighting:
   - Linear weights from 0.5 to 1.0
   - More emphasis on recent batches
   - Weighted average calculation

2. Additional Adjustment:
   - Based on weighted processing time
   - More conservative scaling
   - Smoother transitions

Data-Adaptive Features
^^^^^^^^^^^^^^^^^

1. Data Analysis:
   - Volatility calculation
   - Missing data ratio
   - Unique value ratio

2. Size Adjustments:
   - Reduce for high volatility
   - Reduce for missing data
   - Increase for high uniqueness

Dependencies
----------

- NumPy (array operations)
- Pandas (DataFrame analysis)
- Python dataclasses
- typing (type hints)

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Create strategy
    strategy = BatchStrategy(target_time=0.1)
    
    # Process batches
    while training:
        batch_size = strategy.get_next_batch_size()
        start_time = time.time()
        
        # Process batch
        loss = train_batch(batch_size)
        
        # Record statistics
        strategy.record_batch(BatchStats(
            batch_size=batch_size,
            processing_time=time.time() - start_time,
            loss=loss,
            metrics={}
        ))

Adaptive Usage
^^^^^^^^^^

.. code-block:: python

    # Create adaptive strategy
    strategy = AdaptiveBatchStrategy(target_time=0.1)
    
    # Adjust for data
    batch_size = strategy.adjust_for_data(training_data)
    
    # Use adjusted size
    loss = train_batch(batch_size)

Best Practices
-----------

1. Strategy Selection
^^^^^^^^^^^^^^^

- Use base strategy for simple cases
- Use time-sensitive for dynamic workloads
- Use adaptive for complex data

2. Parameter Tuning
^^^^^^^^^^^^^

- Set appropriate target time
- Monitor adjustment frequency
- Consider hardware constraints

3. Performance Monitoring
^^^^^^^^^^^^^^^^^^^

- Track batch statistics
- Monitor convergence
- Validate improvements

4. Error Handling
^^^^^^^^^^^^

- Handle empty history
- Validate batch sizes
- Check processing times 