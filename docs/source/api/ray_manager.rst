Ray Manager
===========

Overview
--------

The Ray Manager module provides a framework for distributed processing using Ray, with support for parallel training and evaluation operations.

Key Components
-------------

RayConfig
^^^^^^^^^

Configuration dataclass for Ray initialization:

* ``num_cpus``: Number of CPUs to use (None = all available)
* ``num_gpus``: Number of GPUs to use (None = all available)
* ``memory``: Memory limit in bytes
* ``object_store_memory``: Object store memory limit

BatchConfig
^^^^^^^^^^

Configuration dataclass for batch processing:

* ``batch_size``: Size of each batch (default: 128)
* ``num_parallel``: Number of parallel workers (default: 4)
* ``chunk_size``: Size of each chunk (default: 32)

RayActor
^^^^^^^^

Base actor class for parallel processing:

* Abstract base class defining the interface for parallel processing actors
* Requires implementation of ``process_batch`` method

TrainingActor
^^^^^^^^^^^^

Actor for parallel training operations:

* Inherits from ``RayActor``
* Initialized with agent configuration
* Processes training batches and returns metrics

EvaluationActor
^^^^^^^^^^^^^^

Actor for parallel evaluation operations:

* Inherits from ``RayActor``
* Initialized with environment configuration
* Processes evaluation batches and returns metrics

RayManager
^^^^^^^^^

Main manager class for Ray-based parallel processing:

Methods:
    * ``__init__(ray_config: RayConfig)``: Initialize Ray with configuration
    * ``create_actor_pool(actor_class, actor_config, num_actors)``: Create pool of actors
    * ``process_in_parallel(actor_pool, data, batch_config)``: Process data in parallel
    * ``shutdown()``: Shutdown Ray

Dependencies
-----------

* ``ray``: Core Ray functionality
* ``ray.util.actor_pool``: Actor pool management
* ``numpy``: Numerical operations
* ``dataclasses``: Configuration classes
* ``logging``: Error tracking

Usage Example
------------

.. code-block:: python

    # Initialize Ray manager
    ray_config = RayConfig(num_cpus=4)
    manager = RayManager(ray_config)

    # Create actor pool
    actor_pool = manager.create_actor_pool(
        TrainingActor,
        {"learning_rate": 0.001},
        num_actors=4
    )

    # Process data in parallel
    batch_config = BatchConfig(batch_size=128)
    results = manager.process_in_parallel(actor_pool, data, batch_config)

    # Cleanup
    manager.shutdown()

Best Practices
-------------

1. Resource Management
   * Configure Ray resources based on system capabilities
   * Monitor memory usage in object store
   * Clean up resources with shutdown()

2. Batch Processing
   * Choose appropriate batch sizes for workload
   * Balance parallelism with overhead
   * Handle empty batches gracefully

3. Error Handling
   * Implement proper exception handling
   * Log errors with appropriate detail
   * Clean up resources on failure

Recent Changes
-------------

* Added type hints for better code clarity
* Enhanced error handling in parallel processing
* Improved resource management
* Added batch configuration options 