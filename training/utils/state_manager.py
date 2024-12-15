"""State management for distributed training"""

import ray
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class ActorState:
    """State information for a Ray actor"""
    actor_id: str
    status: str  # 'idle', 'processing', 'failed'
    metrics: Dict[str, float]
    last_batch_size: int
    processing_time: float
    gpu_memory: Optional[float] = None
    
@ray.remote
class StateManager:
    """Manages state for distributed training actors"""
    
    def __init__(self, max_history: int = 100):
        self._actors: Dict[str, ActorState] = {}
        self._metrics_history = deque(maxlen=max_history)
        self._lock = threading.Lock()
        
    def register_actor(self, actor_id: str) -> None:
        """Register a new actor
        
        Args:
            actor_id: Unique ID for the actor
        """
        with self._lock:
            self._actors[actor_id] = ActorState(
                actor_id=actor_id,
                status='idle',
                metrics={},
                last_batch_size=0,
                processing_time=0.0
            )
    
    def update_actor_state(
        self,
        actor_id: str,
        status: str,
        metrics: Dict[str, float],
        batch_size: int,
        processing_time: float,
        gpu_memory: Optional[float] = None
    ) -> None:
        """Update state for an actor
        
        Args:
            actor_id: Actor ID
            status: Current status
            metrics: Performance metrics
            batch_size: Size of last processed batch
            processing_time: Time taken to process batch
            gpu_memory: GPU memory usage (if applicable)
        """
        with self._lock:
            if actor_id not in self._actors:
                self.register_actor(actor_id)
                
            self._actors[actor_id] = ActorState(
                actor_id=actor_id,
                status=status,
                metrics=metrics,
                last_batch_size=batch_size,
                processing_time=processing_time,
                gpu_memory=gpu_memory
            )
            
            # Store metrics history
            self._metrics_history.append({
                'actor_id': actor_id,
                'timestamp': ray.get_runtime_context().get_time_since_start(),
                **metrics
            })
    
    def get_actor_state(self, actor_id: str) -> Optional[ActorState]:
        """Get current state of an actor
        
        Args:
            actor_id: Actor ID
            
        Returns:
            Actor state if found
        """
        with self._lock:
            return self._actors.get(actor_id)
    
    def get_all_states(self) -> Dict[str, ActorState]:
        """Get states of all actors
        
        Returns:
            Dictionary of actor states
        """
        with self._lock:
            return self._actors.copy()
    
    def get_metrics_history(self) -> list:
        """Get historical metrics
        
        Returns:
            List of metric records
        """
        with self._lock:
            return list(self._metrics_history)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            active_actors = sum(1 for state in self._actors.values() 
                              if state.status == 'processing')
            
            processing_times = [state.processing_time 
                              for state in self._actors.values()]
            
            avg_time = np.mean(processing_times) if processing_times else 0
            
            return {
                'total_actors': len(self._actors),
                'active_actors': active_actors,
                'idle_actors': len(self._actors) - active_actors,
                'avg_processing_time': avg_time,
                'total_batches_processed': len(self._metrics_history)
            }
    
    def cleanup_actor(self, actor_id: str) -> None:
        """Remove an actor from tracking
        
        Args:
            actor_id: Actor ID to remove
        """
        with self._lock:
            self._actors.pop(actor_id, None)