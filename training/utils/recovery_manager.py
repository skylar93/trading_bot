"""Worker recovery and checkpoint management"""

import ray
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os
import json
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """Worker state information"""

    worker_id: str
    status: str  # 'active', 'failed', 'recovered'
    last_checkpoint: str
    failure_count: int
    last_heartbeat: float
    current_batch: Optional[Dict[str, Any]] = None


@ray.remote
class RecoveryManager:
    """Manages worker recovery and checkpointing"""

    def __init__(
        self,
        heartbeat_interval: float = 5.0,  # seconds
        max_failures: int = 3,
        checkpoint_dir: str = "checkpoints",
    ):
        self._workers: Dict[str, WorkerState] = {}
        self._heartbeat_interval = heartbeat_interval
        self._max_failures = max_failures
        self._checkpoint_dir = checkpoint_dir
        self._running = False
        self._lock = threading.Lock()

        os.makedirs(checkpoint_dir, exist_ok=True)

    def start_monitoring(self):
        """Start worker monitoring"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_workers, daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop worker monitoring"""
        self._running = False
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join()

    def register_worker(
        self, worker_id: str, initial_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a new worker

        Args:
            worker_id: Worker ID
            initial_state: Initial worker state

        Returns:
            True if registration successful
        """
        with self._lock:
            if worker_id in self._workers:
                return False

            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                status="active",
                last_checkpoint=self._get_checkpoint_path(
                    worker_id, "initial"
                ),
                failure_count=0,
                last_heartbeat=time.time(),
            )

            # Save initial state if provided
            if initial_state:
                self.save_checkpoint(worker_id, initial_state)

            return True

    def heartbeat(
        self, worker_id: str, current_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record worker heartbeat

        Args:
            worker_id: Worker ID
            current_state: Current worker state

        Returns:
            True if worker should continue
        """
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker = self._workers[worker_id]
            worker.last_heartbeat = time.time()

            if current_state:
                worker.current_batch = current_state

            # Check if worker should stop
            if worker.failure_count >= self._max_failures:
                return False

            return worker.status == "active"

    def save_checkpoint(
        self, worker_id: str, state: Dict[str, Any]
    ) -> Optional[str]:
        """Save worker checkpoint

        Args:
            worker_id: Worker ID
            state: State to save

        Returns:
            Checkpoint path if successful
        """
        with self._lock:
            if worker_id not in self._workers:
                return None

            worker = self._workers[worker_id]

            # Create checkpoint path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self._get_checkpoint_path(worker_id, timestamp)

            try:
                with open(checkpoint_path, "w") as f:
                    json.dump(state, f)

                worker.last_checkpoint = checkpoint_path
                return checkpoint_path

            except Exception as e:
                logger.error(
                    f"Error saving checkpoint: {str(e)}", exc_info=True
                )
                return None

    def load_checkpoint(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Load worker checkpoint

        Args:
            worker_id: Worker ID

        Returns:
            Checkpoint state if available
        """
        with self._lock:
            if worker_id not in self._workers:
                return None

            worker = self._workers[worker_id]

            try:
                with open(worker.last_checkpoint, "r") as f:
                    return json.load(f)

            except Exception as e:
                logger.error(
                    f"Error loading checkpoint: {str(e)}", exc_info=True
                )
                return None

    def get_failed_workers(self) -> List[str]:
        """Get list of failed workers

        Returns:
            List of failed worker IDs
        """
        with self._lock:
            return [
                worker_id
                for worker_id, state in self._workers.items()
                if state.status == "failed"
            ]

    def mark_worker_failed(self, worker_id: str):
        """Mark a worker as failed

        Args:
            worker_id: Worker ID
        """
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = "failed"
                worker.failure_count += 1

    def _monitor_workers(self):
        """Monitor worker heartbeats"""
        while self._running:
            try:
                current_time = time.time()

                with self._lock:
                    for worker_id, state in self._workers.items():
                        if state.status == "active":
                            # Check heartbeat timeout
                            if (
                                current_time - state.last_heartbeat
                                > self._heartbeat_interval * 2
                            ):
                                logger.warning(
                                    f"Worker {worker_id} missed heartbeat"
                                )
                                self.mark_worker_failed(worker_id)

            except Exception as e:
                logger.error(
                    f"Error in worker monitoring: {str(e)}", exc_info=True
                )

            time.sleep(self._heartbeat_interval)

    def _get_checkpoint_path(self, worker_id: str, identifier: str) -> str:
        """Get path for worker checkpoint

        Args:
            worker_id: Worker ID
            identifier: Checkpoint identifier

        Returns:
            Checkpoint file path
        """
        return os.path.join(
            self._checkpoint_dir, f"{worker_id}_{identifier}.json"
        )
