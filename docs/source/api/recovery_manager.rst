Recovery Manager
===============

Overview
--------

The Recovery Manager module provides worker recovery and checkpoint management functionality using Ray's distributed framework.

Key Components
-------------

WorkerState
^^^^^^^^^^

Dataclass for tracking worker state:

* ``worker_id``: Unique worker identifier
* ``status``: Current status ('active', 'failed', 'recovered')
* ``last_checkpoint``: Path to most recent checkpoint
* ``failure_count``: Number of failures
* ``last_heartbeat``: Timestamp of last heartbeat
* ``current_batch``: Optional current batch state

RecoveryManager
^^^^^^^^^^^^^

Ray remote actor for managing worker recovery and checkpointing:

Methods:
    * ``__init__(heartbeat_interval, max_failures, checkpoint_dir)``: Initialize manager
    * ``start_monitoring()``: Start worker monitoring thread
    * ``stop_monitoring()``: Stop monitoring thread
    * ``register_worker(worker_id, initial_state)``: Register new worker
    * ``heartbeat(worker_id, current_state)``: Record worker heartbeat
    * ``save_checkpoint(worker_id, state)``: Save worker checkpoint
    * ``load_checkpoint(worker_id)``: Load worker checkpoint
    * ``get_failed_workers()``: Get list of failed workers
    * ``mark_worker_failed(worker_id)``: Mark worker as failed

Implementation Details
--------------------

Monitoring Thread
^^^^^^^^^^^^^^^

* Runs in background checking worker heartbeats
* Marks workers as failed if heartbeat timeout exceeded
* Uses thread-safe operations with lock

Checkpoint Management
^^^^^^^^^^^^^^^^^^

* JSON-based checkpoint storage
* Automatic checkpoint directory creation
* Timestamp-based checkpoint naming
* Error handling for I/O operations

Dependencies
-----------

* ``ray``: Distributed computing framework
* ``threading``: Thread management
* ``json``: Checkpoint serialization
* ``os``: File operations
* ``logging``: Error tracking
* ``datetime``: Timestamp management

Usage Example
------------

.. code-block:: python

    # Initialize recovery manager
    recovery_manager = RecoveryManager.remote(
        heartbeat_interval=5.0,
        max_failures=3,
        checkpoint_dir="checkpoints"
    )

    # Start monitoring
    await recovery_manager.start_monitoring.remote()

    # Register worker
    worker_id = "worker_1"
    initial_state = {"batch_id": 0}
    await recovery_manager.register_worker.remote(worker_id, initial_state)

    # Regular heartbeat
    while processing:
        current_state = {"batch_id": current_batch}
        should_continue = await recovery_manager.heartbeat.remote(
            worker_id, current_state
        )
        if not should_continue:
            break

    # Save checkpoint
    checkpoint_path = await recovery_manager.save_checkpoint.remote(
        worker_id, final_state
    )

Best Practices
-------------

1. Worker Management
   * Set appropriate heartbeat intervals
   * Configure reasonable max_failures limit
   * Handle worker registration failures

2. Checkpoint Strategy
   * Save checkpoints at meaningful intervals
   * Include sufficient state for recovery
   * Clean up old checkpoints

3. Error Handling
   * Monitor worker failures
   * Implement recovery procedures
   * Log relevant error information

4. Resource Cleanup
   * Stop monitoring before shutdown
   * Clean up checkpoint files
   * Release worker resources

Recent Changes
-------------

* Added thread-safe operations
* Enhanced checkpoint management
* Improved error handling
* Added worker state tracking 