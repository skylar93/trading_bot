# Development Guidelines

## Core Development Principles

### 1. Data Format Standards
- OHLCV columns must use `$` prefix: `$open`, `$high`, `$low`, `$close`, `$volume`
- Maintain this convention throughout the entire pipeline
- Example:
```python
df = pd.DataFrame({
    '$open': [...],
    '$high': [...],
    '$low': [...],
    '$close': [...],
    '$volume': [...]
})
```

### 2. Logging Standards
```python
import logging
import os
from datetime import datetime

# Setup pattern
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler for debugging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(
    os.path.join(log_dir, f"{__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

# Console handler for important info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter('%(levelname)s - %(message)s')
)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

### 3. Error Handling Pattern
```python
def safe_execute(func: Callable, *args, **kwargs) -> Any:
    """Standard error handling wrapper"""
    try:
        result = func(*args, **kwargs)
        if isinstance(result, pd.Series) and result.isnull().any():
            logger.warning(f"NaN values found in {func.__name__}, applying forward/backward fill")
            result = result.ffill().bfill()
        return result
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
        raise
```

### 4. Testing Standards

#### Test Structure
```python
@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    return pd.DataFrame({
        '$open': np.random.randn(len(dates)) * 100 + 1000,
        '$high': np.random.randn(len(dates)) * 100 + 1000,
        '$low': np.random.randn(len(dates)) * 100 + 1000,
        '$close': np.random.randn(len(dates)) * 100 + 1000,
        '$volume': np.abs(np.random.randn(len(dates)) * 1000 + 5000)
    }, index=dates)

def test_functionality(sample_data):
    """Test description with clear purpose"""
    # Arrange
    expected_result = ...
    
    # Act
    actual_result = function_under_test(sample_data)
    
    # Assert
    assert actual_result == expected_result
```

#### Required Test Cases
1. Success cases with valid data
2. Error cases with invalid input
3. Edge cases (empty data, NaN values, boundary conditions)
4. Integration tests for full pipeline

### 5. Data Validation Rules

#### Price Data
- High prices must be >= Low prices
- Volume must be non-negative
- No gaps in timestamp sequence
- No duplicate timestamps

#### Technical Indicators
- RSI must be between 0 and 100
- Moving averages must be between min and max price
- Bollinger Bands: Upper > Middle > Lower
- Volume indicators must be non-negative

### 6. Cache Management
```python
class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, identifier: str) -> Path:
        """Generate consistent cache file path"""
        return self.cache_dir / f"{identifier}.csv"
    
    def save(self, df: pd.DataFrame, identifier: str):
        """Save data with proper column handling"""
        cache_path = self.get_cache_path(identifier)
        df.to_csv(cache_path)
        logger.info(f"Cached data to: {cache_path}")
    
    def load(self, identifier: str) -> Optional[pd.DataFrame]:
        """Load data with validation"""
        cache_path = self.get_cache_path(identifier)
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            if self._validate_cache(df):
                return df
        return None
```

### 7. Feature Generation Standards

#### Input Requirements
- Must have all required OHLCV columns with `$` prefix
- Timestamps must be sorted and unique
- No NaN values in input (or handled explicitly)

#### Output Requirements
- All features must be properly named
- No NaN values in output
- All features within valid ranges
- Documentation of feature calculations

### 8. Integration Testing Checklist

```python
def test_integration():
    """Full pipeline integration test"""
    # 1. Data Loading
    assert data is not None
    assert all required columns present
    assert no invalid values
    
    # 2. Feature Generation
    assert all features generated
    assert no NaN values
    assert features in valid ranges
    
    # 3. Model Integration
    assert model accepts features
    assert predictions in expected format
    
    # 4. Performance
    assert execution time within limits
    assert memory usage acceptable
```

### 9. Performance Guidelines

#### Data Processing
- Use caching for historical data
- Implement progress tracking
- Monitor memory usage
- Log execution times

#### Optimization Tips
```python
# Progress tracking
def process_with_progress(items):
    total = len(items)
    for i, item in enumerate(items, 1):
        logger.info(f"Processing {i}/{total}")
        yield process_item(item)

# Memory optimization
def process_in_chunks(df, chunk_size=1000):
    for chunk in np.array_split(df, len(df) // chunk_size + 1):
        process_chunk(chunk)
```

### 10. Documentation Standards

#### Function Documentation
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Clear description of function purpose
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        Exception1: Description of when this error occurs
        Exception2: Description of when this error occurs
    
    Example:
        ```python
        result = function_name(param1, param2)
        ```
    """
```

## Debugging Checklist

When debugging issues:

1. Check Logs First
   - Review DEBUG level logs
   - Look for warnings and errors
   - Check execution times

2. Validate Data
   - Verify column names ($ prefix)
   - Check for NaN values
   - Verify data ranges
   - Check timestamp sequence

3. Test Components
   - Isolate the failing component
   - Use unit tests
   - Check edge cases
   - Verify cache integrity

4. Performance Issues
   - Review memory usage
   - Check execution times
   - Verify cache usage
   - Look for bottlenecks

5. Integration Issues
   - Verify pipeline connections
   - Check data transformations
   - Validate feature generation
   - Test full pipeline

## Common Issues and Solutions

1. Missing Features
   ```python
   # Check feature generation
   expected_features = ['RSI', 'SMA_5', ...]
   missing = [f for f in expected_features if f not in df.columns]
   if missing:
       logger.error(f"Missing features: {missing}")
   ```

2. NaN Values
   ```python
   # Check for NaN values
   nan_cols = df.columns[df.isnull().any()].tolist()
   if nan_cols:
       logger.error(f"NaN values in columns: {nan_cols}")
   ```

3. Data Range Issues
   ```python
   # Validate price relationships
   if not (df['$high'] >= df['$low']).all():
       logger.error("Invalid price relationships")
   ```

4. Performance Problems
   ```python
   # Monitor execution time
   start_time = time.time()
   result = process_data(df)
   elapsed = time.time() - start_time
   if elapsed > threshold:
       logger.warning(f"Performance warning: {elapsed:.2f}s")
   ```

## References

- [Project Repository](https://github.com/your-repo)
- [Issue Tracker](https://github.com/your-repo/issues)
- [Documentation](https://your-docs-site.com) 