"""GPU resource management utilities"""

import ray
import numpy as np
import psutil
import threading
from typing import Dict, Any, List, Optional
import logging
import time
from dataclasses import dataclass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

@dataclass
class GPUStats:
    """GPU statistics"""
    device_id: int
    memory_used: float  # in MB
    memory_total: float  # in MB
    utilization: float  # in %
    temperature: float  # in Celsius
    power_usage: Optional[float] = None  # in Watts

@dataclass
class SystemStats:
    """System resource statistics"""
    cpu_percent: float
    memory_used: float  # in MB
    memory_total: float # in MB
    gpu_stats: Optional[List[GPUStats]] = None

class ResourceMonitor:
    """Monitors system resources"""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,  # seconds
        history_size: int = 3600  # 1 hour at 1 second intervals
    ):
        """Initialize monitor
        
        Args:
            monitoring_interval: How often to update stats
            history_size: How many records to keep
        """
        self.interval = monitoring_interval
        self.history_size = history_size
        
        self._stats_history = []
        self._running = False
        self._lock = threading.Lock()
        
        # Check GPU availability
        self.has_gpu = HAS_TORCH and torch.cuda.is_available()
        if self.has_gpu:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 0
    
    def start(self):
        """Start resource monitoring"""
        if self._running:
            return
            
        self._running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def stop(self):
        """Stop resource monitoring"""
        self._running = False
    
    def get_current_stats(self) -> SystemStats:
        """Get current system statistics
        
        Returns:
            Current system statistics
        """
        with self._lock:
            if self._stats_history:
                return self._stats_history[-1]
            return self._collect_stats()
    
    def get_stats_history(self) -> List[SystemStats]:
        """Get historical statistics
        
        Returns:
            List of historical statistics
        """
        with self._lock:
            return self._stats_history.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                stats = self._collect_stats()
                
                with self._lock:
                    self._stats_history.append(stats)
                    
                    # Maintain history size
                    if len(self._stats_history) > self.history_size:
                        self._stats_history = self._stats_history[-self.history_size:]
                
            except Exception as e:
                logger.error(f"Error collecting stats: {str(e)}", exc_info=True)
            
            time.sleep(self.interval)
    
    def _collect_stats(self) -> SystemStats:
        """Collect current system statistics
        
        Returns:
            Current system statistics
        """
        # Collect CPU and memory stats
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        stats = SystemStats(
            cpu_percent=cpu_percent,
            memory_used=memory.used / 1024 / 1024,  # Convert to MB
            memory_total=memory.total / 1024 / 1024
        )
        
        # Collect GPU stats if available
        if self.has_gpu:
            gpu_stats = []
            for i in range(self.num_gpus):
                with torch.cuda.device(i):
                    memory_stats = torch.cuda.get_device_properties(i).total_memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    utilization = torch.cuda.utilization_per_device()[i]
                    
                    gpu_stats.append(GPUStats(
                        device_id=i,
                        memory_used=memory_allocated / 1024 / 1024,  # Convert to MB
                        memory_total=memory_stats / 1024 / 1024,
                        utilization=utilization,
                        temperature=0.0,  # Would need nvidia-smi for this
                        power_usage=None  # Would need nvidia-smi for this
                    ))
            
            stats.gpu_stats = gpu_stats
        
        return stats

@ray.remote
class ResourceManager:
    """Manages resource allocation for distributed training"""
    
    def __init__(
        self,
        max_memory_percent: float = 90.0,  # Maximum memory usage percentage
        max_gpu_percent: float = 95.0  # Maximum GPU usage percentage
    ):
        self.monitor = ResourceMonitor()
        self.max_memory_percent = max_memory_percent
        self.max_gpu_percent = max_gpu_percent
        self._allocated_resources = {}
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitor.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitor.stop()
    
    def allocate_resources(
        self,
        actor_id: str,
        cpu_needed: float,
        memory_needed: float,
        gpu_needed: Optional[float] = None
    ) -> Dict[str, Any]:
        """Allocate resources to an actor
        
        Args:
            actor_id: Actor requesting resources
            cpu_needed: CPU cores needed
            memory_needed: Memory needed (MB)
            gpu_needed: GPU memory needed (MB)
            
        Returns:
            Allocation information
        """
        stats = self.monitor.get_current_stats()
        
        # Check CPU availability
        if stats.cpu_percent + cpu_needed > 100:
            raise ResourceError("Insufficient CPU available")
            
        # Check memory availability
        memory_percent = (stats.memory_used + memory_needed) / stats.memory_total * 100
        if memory_percent > self.max_memory_percent:
            raise ResourceError("Insufficient memory available")
        
        # Check GPU if needed
        gpu_device = None
        if gpu_needed and stats.gpu_stats:
            # Find GPU with most available memory
            available_gpus = []
            for gpu in stats.gpu_stats:
                memory_percent = (gpu.memory_used + gpu_needed) / gpu.memory_total * 100
                if memory_percent <= self.max_gpu_percent:
                    available_gpus.append((gpu.device_id, memory_percent))
            
            if available_gpus:
                # Select GPU with lowest current usage
                gpu_device = min(available_gpus, key=lambda x: x[1])[0]
            else:
                raise ResourceError("Insufficient GPU memory available")
        
        # Record allocation
        self._allocated_resources[actor_id] = {
            'cpu': cpu_needed,
            'memory': memory_needed,
            'gpu': gpu_needed,
            'gpu_device': gpu_device
        }
        
        return {
            'allocated': True,
            'gpu_device': gpu_device
        }
    
    def release_resources(self, actor_id: str):
        """Release resources allocated to an actor
        
        Args:
            actor_id: Actor releasing resources
        """
        self._allocated_resources.pop(actor_id, None)
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage
        
        Returns:
            Dictionary of resource usage statistics
        """
        stats = self.monitor.get_current_stats()
        
        usage = {
            'cpu_percent': stats.cpu_percent,
            'memory_percent': stats.memory_used / stats.memory_total * 100
        }
        
        if stats.gpu_stats:
            for i, gpu in enumerate(stats.gpu_stats):
                usage[f'gpu_{i}_percent'] = gpu.utilization
                usage[f'gpu_{i}_memory_percent'] = gpu.memory_used / gpu.memory_total * 100
        
        return usage

class ResourceError(Exception):
    """Exception raised for resource allocation failures"""
    pass