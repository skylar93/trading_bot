State Manager
============

Overview
--------

The State Manager module provides state tracking and management functionality for distributed training actors using Ray's framework.

Key Components
-------------

ActorState
^^^^^^^^^

Dataclass for tracking individual actor state:

* ``actor_id``: Unique actor identifier
* ``status``: Current status ('idle', 'processing', 'failed')
* ``metrics``: Dictionary of performance metrics
* ``last_batch_size``: Size of last processed batch
* ``processing_time``: Time taken for last batch
* ``gpu_memory``: Optional GPU memory usage

StateManager
^^^^^^^^^^

Ray remote class for managing distributed actor states:

Methods:
    * ``__init__(max_history=100)``: Initialize state manager
    * ``register_actor(actor_id)``: Register new actor
    * ``update_actor_state(actor_id, status, metrics, ...)``: Update actor state
    * ``get_actor_state(actor_id)``: Get specific actor state
    * ``get_all_states()``: Get all actor states
    * ``get_metrics_history()``: Get historical metrics
    * ``get_system_stats()``: Get system-wide statistics
    * ``cleanup_actor(actor_id)``: Remove actor from tracking

Implementation Details
--------------------

State Management
^^^^^^^^^^^^^

* Thread-safe operations using locks
* Internal actor state dictionary
* Metrics history using deque
* System-wide statistics calculation

Metrics Tracking
^^^^^^^^^^^^^

* Performance metrics history
* Processing time tracking
* Batch size monitoring
* GPU memory tracking (optional)

Dependencies
-----------

* ``ray``: Distributed computing framework
* ``numpy``: Statistical calculations
* ``threading``: Concurrency control
* ``collections.deque``: Metrics history
* ``dataclasses``: State representation
* ``logging``: Error tracking

Usage Example
------------

.. code-block:: python

    # Initialize state manager
    state_manager = StateManager.remote(max_history=100)

    # Register actor
    actor_id = "actor_1"
    await state_manager.register_actor.remote(actor_id)

    # Update state
    await state_manager.update_actor_state.remote(
        actor_id=actor_id,
        status="processing",
        metrics={"loss": 0.5, "accuracy": 0.95},
        batch_size=64,
        processing_time=0.1
    )

    # Get system stats
    stats = await state_manager.get_system_stats.remote()
    print(f"Active actors: {stats['active_actors']}")

Best Practices
-------------

1. State Management
   * Use unique actor IDs
   * Handle actor registration/cleanup
   * Monitor actor status changes

2. Metrics Tracking
   * Configure appropriate history size
   * Track relevant performance metrics
   * Monitor system-wide statistics

3. Concurrency
   * Use thread-safe operations
   * Handle concurrent updates
   * Manage shared resources

4. Resource Management
   * Clean up inactive actors
   * Monitor GPU memory usage
   * Track processing times

Recent Changes
-------------

* Added thread-safe operations
* Enhanced metrics tracking
* Improved system statistics
* Added GPU memory monitoring 