"""Tests for Ray Manager"""

import pytest
import numpy as np
from training.utils.ray_manager import (
    RayConfig,
    BatchConfig,
    RayManager,
    TrainingActor,
    EvaluationActor
)

class TestTrainingActor(TrainingActor):
    """Test implementation of TrainingActor"""
    
    def process_batch(self, batch_data: np.ndarray) -> dict:
        """Simple batch processing for testing"""
        return {
            'loss': float(np.mean(batch_data)),
            'metrics': {'batch_size': len(batch_data)}
        }

class TestEvaluationActor(EvaluationActor):
    """Test implementation of EvaluationActor"""
    
    def process_batch(self, batch_data: np.ndarray) -> dict:
        """Simple batch processing for testing"""
        return {
            'returns': float(np.sum(batch_data)),
            'metrics': {'batch_size': len(batch_data)}
        }

@pytest.fixture
def ray_manager():
    """Create RayManager instance for testing"""
    config = RayConfig(num_cpus=2, num_gpus=0)
    manager = RayManager(config)
    yield manager
    manager.shutdown()

def test_ray_initialization(ray_manager):
    """Test Ray initialization"""
    assert ray_manager is not None

def test_actor_pool_creation(ray_manager):
    """Test actor pool creation"""
    config = {'test_param': 1}
    pool = ray_manager.create_actor_pool(TestTrainingActor, config, num_actors=2)
    assert pool is not None

def test_parallel_processing(ray_manager):
    """Test parallel data processing"""
    # Create test data
    data = np.random.randn(100, 10)
    
    # Create actor pool
    config = {'test_param': 1}
    pool = ray_manager.create_actor_pool(TestTrainingActor, config, num_actors=2)
    
    # Process data
    batch_config = BatchConfig(batch_size=10, num_parallel=2, chunk_size=5)
    results = ray_manager.process_in_parallel(pool, data, batch_config)
    
    assert len(results) > 0
    assert all('loss' in result for result in results)
    assert all('metrics' in result for result in results)

def test_multiple_actor_types(ray_manager):
    """Test using multiple types of actors"""
    # Create test data
    data = np.random.randn(100, 10)
    
    # Create training pool
    train_config = {'test_param': 1}
    train_pool = ray_manager.create_actor_pool(
        TestTrainingActor, 
        train_config, 
        num_actors=2
    )
    
    # Create evaluation pool
    eval_config = {'test_param': 2}
    eval_pool = ray_manager.create_actor_pool(
        TestEvaluationActor,
        eval_config,
        num_actors=2
    )
    
    # Process data with both pools
    batch_config = BatchConfig(batch_size=10, num_parallel=2, chunk_size=5)
    
    train_results = ray_manager.process_in_parallel(
        train_pool,
        data,
        batch_config
    )
    eval_results = ray_manager.process_in_parallel(
        eval_pool,
        data,
        batch_config
    )
    
    assert len(train_results) > 0 and len(eval_results) > 0
    assert all('loss' in result for result in train_results)
    assert all('returns' in result for result in eval_results)

if __name__ == '__main__':
    pytest.main([__file__])